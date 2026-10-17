"""Check a model bundle against models/manifest.json.

The large third-party checkouts and weights are deliberately not part of the
repository.  This checker keeps the on-disk layout and the environment-variable
layout in sync, whether the bundle arrives from a cloud drive, a release
archive, or a shared laboratory disk.

Examples:

    python tools/check_model_assets.py --root C:\torusfold-models
    python tools/check_model_assets.py --root /models --json
    python tools/check_model_assets.py --root /models --sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "models" / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_asset(root: Path, asset: dict, verify_sha256: bool = False) -> dict:
    asset_root = root / asset["path"]
    result = {
        "id": asset["id"],
        "title": asset.get("title", asset["id"]),
        "root": str(asset_root),
        "required": bool(asset.get("required", False)),
        "ok": True,
        "missing": [],
        "too_small": [],
        "sha256": [],
    }
    if asset.get("platform") == "windows" and os.name != "nt":
        result["skipped"] = "Windows-only asset"
        return result

    if not asset_root.is_dir():
        result["ok"] = False
        result["missing"].append(".")
        return result

    min_bytes = asset.get("min_bytes", {})
    for rel in asset.get("required_files", []):
        path = asset_root / rel
        if not path.is_file():
            result["ok"] = False
            result["missing"].append(rel)
            continue
        if rel in min_bytes and path.stat().st_size < int(min_bytes[rel]):
            result["ok"] = False
            result["too_small"].append(
                "%s (%d < %d bytes)" % (rel, path.stat().st_size, int(min_bytes[rel]))
            )

    if verify_sha256:
        for rel, expected in asset.get("sha256", {}).items():
            path = asset_root / rel
            if not path.is_file():
                result["ok"] = False
                result["sha256"].append("%s missing" % rel)
                continue
            actual = _sha256(path)
            if actual.lower() != expected.lower():
                result["ok"] = False
                result["sha256"].append("%s mismatch" % rel)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.environ.get("TORUSFOLD_MODELS", str(REPO / "models")),
        help="model bundle root (default: TORUSFOLD_MODELS or ./models)",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--sha256", action="store_true", help="verify recorded SHA-256 values")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error("cannot read manifest %s: %s" % (manifest_path, exc))

    results = [
        inspect_asset(root, asset, verify_sha256=args.sha256)
        for asset in manifest.get("assets", [])
    ]
    if args.json:
        print(json.dumps({"root": str(root), "assets": results}, indent=2))
    else:
        print("Model bundle: %s" % root)
        print("-" * 78)
        for item in results:
            if item.get("skipped"):
                status = "SKIP"
            else:
                status = "OK  " if item["ok"] else "MISS"
            print("%s %-12s %s" % (status, item["id"], item["root"]))
            if item["missing"]:
                print("     missing: %s" % ", ".join(item["missing"]))
            if item["too_small"]:
                print("     too small: %s" % ", ".join(item["too_small"]))
            if item["sha256"]:
                print("     sha256: %s" % ", ".join(item["sha256"]))
        required_bad = [r for r in results if r["required"] and not r["ok"] and not r.get("skipped")]
        print()
        print("%d required asset(s) unresolved" % len(required_bad))

    required_bad = [r for r in results if r["required"] and not r["ok"] and not r.get("skipped")]
    return 1 if required_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
