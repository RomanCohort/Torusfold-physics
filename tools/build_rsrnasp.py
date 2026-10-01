# -*- coding: utf-8 -*-
"""Build rsRNASP1 under WSL, or report why it cannot be built.

rsRNASP1 is C++ with a Makefile and ships no Windows binary and no pip package,
so on Windows the only route is to compile it inside WSL. That is a handful of
steps with two non-obvious traps, and this script exists so they are not
rediscovered:

  1. `git clone` may fail inside WSL on a network that allows ordinary HTTPS but
     not the git wire protocol. The source is fetched as a GitHub tarball from the
     Windows side instead, where codeload.github.com is reachable, and copied into
     WSL over the \\\\wsl$ share.

  2. The documented `-d` flag for pointing the binary at its energy files ABORTS
     it: "std::logic_error: basic_string: construction from null", exit 134. Use
     the rsRNASP_RNA_HOME environment variable. rsrnasp_quality.py already does.

Usage
-----
    python tools/build_rsrnasp.py              # build if missing, then verify
    python tools/build_rsrnasp.py --check      # verify only, build nothing
    python tools/build_rsrnasp.py --wsl-root /home/you/tools/rsRNASP1

After a successful build, set the path for the pipeline:

    set TORUSFOLD_RSRNASP=/home/you/tools/rsRNASP1

(or leave it unset to use the default ~/tools/rsRNASP1, which the wrapper looks
for inside WSL — note that is WSL's home, not the Windows one.)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

TARBALL_URL = "https://codeload.github.com/Tan-group/rsRNASP1/tar.gz/refs/heads/main"
DEFAULT_WSL_ROOT = "~/tools/rsRNASP1"


def _wsl(args, timeout=600.0) -> Tuple[bool, str]:
    try:
        p = subprocess.run(["wsl.exe", "-e", "bash", "-lc", args],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timed out"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)[:200]
    return p.returncode == 0, ((p.stdout or "") + (p.stderr or "")).strip()


def _distro() -> Optional[str]:
    ok, out = _wsl("echo $WSL_DISTRO_NAME", timeout=60)
    if not ok:
        return None
    return (out.strip().splitlines() or [""])[-1] or None


def _win_home_in_wsl(distro: str) -> Optional[str]:
    """The \\\\wsl$ share for a distribution, for copying files in."""
    return r"\\wsl$\%s" % distro


def _download(url: str, dest: str) -> bool:
    """Fetch a URL on the Windows side.

    urllib before anything fancier: it needs no extra dependency and the tarball is
    a plain GET. Reports the size so a truncated download is visible.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as exc:                                  # noqa: BLE001
        print("   download failed: %s" % exc)
        return False
    size = os.path.getsize(dest)
    print("   downloaded %.1f MB" % (size / 1048576.0))
    return size > 1_000_000


def check(wsl_root: str) -> bool:
    from torusfold.scheme2 import rsrnasp_quality as rq
    probe = rq.available()
    if probe.get("available"):
        print("rsRNASP1 is usable at %s (via %s)"
              % (probe.get("root"), probe.get("distro")))
        return True
    print("rsRNASP1 is NOT usable: %s" % probe.get("why", "unknown reason"))
    return False


def build(wsl_root: str) -> bool:
    distro = _distro()
    if not distro:
        print("WSL is not available. rsRNASP1 ships Linux builds only; there is no")
        print("Windows binary and no pip package, so it cannot be installed without WSL.")
        return False
    print("WSL distribution: %s" % distro)

    # Resolve ~ against WSL's home, which is where the tarball will be unpacked.
    ok, home = _wsl("echo $HOME", timeout=60)
    if not ok or not home:
        print("could not determine the WSL home directory")
        return False
    home = home.strip().splitlines()[-1]
    root = wsl_root
    if root.startswith("~"):
        root = home + root[1:]
    print("target directory: %s" % root)

    share = _win_home_in_wsl(distro)
    if not os.path.isdir(share):
        print("cannot reach the WSL filesystem at %s" % share)
        print("copy the source in manually and re-run with --check")
        return False

    tmp = tempfile.mkdtemp(prefix="rsrnasp_")
    tarball = os.path.join(tmp, "rsRNASP1.tar.gz")
    print("fetching source (about 108 MB)...")
    if not _download(TARBALL_URL, tarball):
        return False

    # Copy into WSL: extraction is much faster on the Linux filesystem than over
    # the 9p mount from Windows.
    parent = os.path.dirname(root)
    ok, _ = _wsl('mkdir -p "%s"' % parent, timeout=120)
    if not ok:
        print("could not create %s inside WSL" % parent)
        return False
    # A WSL distribution is mounted at \\wsl$\<distro>\ for its whole filesystem,
    # not just the home directory, so any absolute target works. (Earlier this
    # assumed the home directory and refused anything outside it, which ruled out
    # a location like /opt.)
    dest_dir = os.path.join(_win_home_in_wsl(distro),
                            parent.lstrip("/").replace("/", os.sep))
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as exc:
        print("cannot write to %s: %s" % (dest_dir, exc))
        print("copy rsRNASP1.tar.gz into %s inside WSL by hand and re-run with --check"
              % parent)
        return False
    dest_tar = os.path.join(dest_dir, "rsRNASP1.tar.gz")
    try:
        import shutil
        shutil.copyfile(tarball, dest_tar)
    except OSError as exc:
        print("copy into WSL failed: %s" % exc)
        return False
    print("   copied into WSL")

    # Extract, build, and validate in one pass.
    steps = (
        'set -e; cd "%s"; rm -rf rsRNASP1 rsRNASP1-*; '
        'tar xzf rsRNASP1.tar.gz; mv rsRNASP1-* rsRNASP1; '
        'cd rsRNASP1 && make -j"$(nproc)" >/tmp/rsrnasp_build.log 2>&1; '
        'echo BUILT; ls -l bin/rsRNASP1'
    ) % parent
    print("building...")
    ok, out = _wsl(steps, timeout=1800)
    if not ok or "BUILT" not in out:
        print("build failed. Last lines of WSL /tmp/rsrnasp_build.log:")
        _ok, log = _wsl("tail -25 /tmp/rsrnasp_build.log", timeout=120)
        print(log or out)
        return False
    print("   %s" % out.splitlines()[-1])

    # Validate against the reference scores the paper's own examples produce. A
    # build that compiles but scores wrong is worse than one that fails, so this is
    # the check that matters.
    print("validating against the upstream example scores...")
    expected = {"1a9nR.pdb": -3146.575662, "1h4sT.pdb": -7757.550000}
    ok, out = _wsl('cd "%s/rsRNASP1" && export rsRNASP_RNA_HOME="$PWD" && '
                   './bin/rsRNASP1 example/*.pdb' % parent, timeout=300)
    if not ok:
        print("could not run the examples: %s" % out)
        return False
    failures = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name = os.path.basename(parts[0])
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        want = expected.get(name)
        if want is None:
            continue
        if abs(value - want) > 1e-6:
            failures.append("%s scored %s, expected %s" % (name, value, want))
        else:
            print("   %s -> %s (matches)" % (name, value))
    if failures:
        print("reference scores do not match:")
        for f in failures:
            print("   " + f)
        return False

    print()
    print("Built and validated. Now set the path the pipeline looks for:")
    print("   set TORUSFOLD_RSRNASP=%s" % root)
    print("For this session only, or add it to .env.local to keep it:")
    print("   TORUSFOLD_RSRNASP=%s" % root)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="only verify that rsRNASP1 is usable")
    ap.add_argument("--wsl-root", default=DEFAULT_WSL_ROOT,
                    help="where to build it inside WSL (default: %s)" % DEFAULT_WSL_ROOT)
    args = ap.parse_args()

    # The wrapper reads TORUSFOLD_RSRNASP at import time, so honour the flag here
    # too — otherwise --check would look at the default even when told otherwise.
    if args.wsl_root != DEFAULT_WSL_ROOT:
        os.environ["TORUSFOLD_RSRNASP"] = args.wsl_root

    if args.check:
        return 0 if check(args.wsl_root) else 1

    if check(args.wsl_root):
        print("Already built. Re-run with --check to verify only.")
        return 0
    return 0 if build(args.wsl_root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
