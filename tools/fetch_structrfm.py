"""Fetch the structRFM checkpoint, in parallel, with checksum verification.

WHY THIS EXISTS RATHER THAN `huggingface_hub`. The Hub client is the right tool and it
was tried first, twice. It pulled the five small files in seconds and then left a
0-byte `.incomplete` blob for `model.safetensors`. The installed huggingface_hub has
moved to Xet and says so:

    FutureWarning: The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated
    as 'hf_transfer' is not used anymore. Please use `HF_XET_HIGH_PERFORMANCE` instead

so the hf_transfer route no longer works, and Xet against a mirror does not move at all
-- hf-mirror.com does not speak the Xet protocol. Plain ranged GETs against the same
mirror do work, so the protocol was the problem, not the host.

WHY PARALLEL. One connection gets about 69 KB/s to hf-mirror.com *and* to github.com on
this network, measured over 40 s each, so the limit is the link and changing hosts buys
nothing. Ranged requests in parallel do help, measured on the same 344 MB file:

    1 connection    69 KB/s    (84 min)
    4 connections  224 KB/s
    8 connections  369 KB/s    (16 min)

Each worker owns one byte range and an ordinary file, so a dropped connection costs one
chunk and is resumed rather than restarted. The assembled file is checked against the
size and LFS sha256 the Hub publishes, and on mismatch it is set aside as `.BAD`
instead of being installed -- a 344 MB file that arrives quietly corrupt is worse than
one that does not arrive.

`tools/configure_deps.py` looks for the result under `TorusFold-tools/structRFM` next
to the repository, which is the default destination here.

Usage:
    python tools/fetch_structrfm.py                    # default destination
    python tools/fetch_structrfm.py --dest D:\\models\\structRFM
    python tools/fetch_structrfm.py --workers 16 --endpoint https://huggingface.co
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO.parent / "TorusFold-tools" / "structRFM"
REPO_ID = "heqin-zhu/structRFM"

# The five small files, plus the weights. README.md and example.py travel with the
# checkpoint because they document the loading convention this repository relies on:
# `AutoModel.from_pretrained(path)` WITHOUT trust_remote_code.
SMALL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "README.md", "example.py")
WEIGHTS = "model.safetensors"
CHUNK = 4 * 1024 * 1024


def hub_meta(filename: str, timeout: float = 60.0):
    """(size, sha256) as published by the Hub, or (None, None) if it says nothing."""
    url = "https://huggingface.co/api/models/%s?blobs=true" % REPO_ID
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.load(r)
    except Exception:                                         # noqa: BLE001
        return None, None
    for sib in data.get("siblings", []):
        if sib["rfilename"] == filename:
            return sib.get("size"), (sib.get("lfs") or {}).get("sha256")
    return None, None


def fetch_range(url: str, start: int, end: int, path: str, attempts: int = 6) -> str:
    """One byte range, resuming from whatever the part file already holds."""
    have = os.path.getsize(path) if os.path.isfile(path) else 0
    want = end - start + 1
    for attempt in range(attempts):
        if have >= want:
            return path
        req = urllib.request.Request(url, headers={
            "Range": "bytes=%d-%d" % (start + have, end),
            "User-Agent": "torusfold-checkpoint-fetch",
        })
        try:
            with urllib.request.urlopen(req, timeout=90) as r, open(path, "ab") as fh:
                while True:
                    buf = r.read(1 << 16)
                    if not buf:
                        break
                    fh.write(buf)
                    have += len(buf)
                    if have >= want:
                        break
        except Exception as exc:                              # noqa: BLE001
            if attempt == attempts - 1:
                raise
            print("    part %d: %s, retry %d/%d from %d B"
                  % (start, type(exc).__name__, attempt + 1, attempts, have), flush=True)
            time.sleep(3 * (attempt + 1))
    return path


def download_small(dest: pathlib.Path, endpoint: str) -> None:
    for name in SMALL_FILES:
        target = dest / name
        url = "%s/%s/resolve/main/%s" % (endpoint, REPO_ID, name)
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                blob = r.read()
        except Exception as exc:                              # noqa: BLE001
            print("  %-26s FAILED: %s: %s" % (name, type(exc).__name__, exc), flush=True)
            raise SystemExit(1)
        target.write_bytes(blob)
        print("  %-26s %8d bytes" % (name, len(blob)), flush=True)


def download_weights(dest: pathlib.Path, endpoint: str, workers: int) -> None:
    out = dest / WEIGHTS
    size, want_sha = hub_meta(WEIGHTS)
    if size is None:
        print("the Hub did not report a size for %s; refusing to guess" % WEIGHTS)
        raise SystemExit(2)
    print("\n%s  %d bytes  sha256 %s" % (WEIGHTS, size, want_sha), flush=True)
    url = "%s/%s/resolve/main/%s" % (endpoint, REPO_ID, WEIGHTS)

    parts = [(start, min(start + CHUNK, size) - 1, str(dest / (".part_%012d" % start)))
             for start in range(0, size, CHUNK)]

    t0 = time.time()
    done = [0]

    def run(item):
        fetch_range(url, item[0], item[1], item[2])
        return item[2]

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(run, p) for p in parts]
        for fut in concurrent.futures.as_completed(futures):
            done[0] += os.path.getsize(fut.result())
            el = time.time() - t0
            print("  %7.1f / %.1f MB  %6.0f KB/s  %5.0f s"
                  % (done[0] / 1e6, size / 1e6, done[0] / 1024 / max(el, 1e-9), el),
                  flush=True)

    h = hashlib.sha256()
    with open(out, "wb") as dst:
        for _s, _e, path in parts:
            with open(path, "rb") as src:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    h.update(buf)
                    dst.write(buf)
    got = h.hexdigest()
    print("\nassembled %d bytes, sha256 %s" % (out.stat().st_size, got), flush=True)
    if out.stat().st_size != size or (want_sha and got != want_sha):
        bad = out.with_suffix(out.suffix + ".BAD")
        os.replace(str(out), str(bad))
        raise SystemExit("CHECKSUM MISMATCH -- moved aside to %s, do not use" % bad)
    for _s, _e, path in parts:
        os.remove(path)
    print("checksum OK -- %s is complete" % out, flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(DEFAULT_DEST),
                    help="destination directory (default: %s)" % DEFAULT_DEST)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel ranged connections for the weights (default 8)")
    ap.add_argument("--endpoint", default=os.environ.get("TF_HF_ENDPOINT",
                                                         "https://hf-mirror.com"),
                    help="Hub base URL. The default mirror exists because the Hub "
                         "itself was slow or unreachable; pass https://huggingface.co "
                         "where the real Hub is fast.")
    args = ap.parse_args(argv)

    dest = pathlib.Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print("destination: %s" % dest)
    print("endpoint   : %s" % args.endpoint)
    print("\nsmall files")
    download_small(dest, args.endpoint)
    download_weights(dest, args.endpoint, max(1, args.workers))
    print("\nNow:  python tools\\configure_deps.py discover   (the structRFM row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
