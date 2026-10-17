"""Checks for the external model-bundle contract and trRNA2 adapter."""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_model_assets as assets  # noqa: E402
import trrna2_runner as trrna2  # noqa: E402


def test_asset_checker_finds_a_complete_required_file(tmp_path):
    asset = {
        "id": "demo",
        "path": "Demo",
        "required": True,
        "required_files": ["weights.bin"],
        "min_bytes": {"weights.bin": 4},
    }
    root = tmp_path / "bundle"
    target = root / "Demo" / "weights.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"1234")

    result = assets.inspect_asset(root, asset)

    assert result["ok"] is True
    assert result["missing"] == []
    assert result["too_small"] == []


def test_asset_checker_rejects_a_truncated_weight_file(tmp_path):
    asset = {
        "id": "demo",
        "path": "Demo",
        "required": True,
        "required_files": ["weights.bin"],
        "min_bytes": {"weights.bin": 4},
    }
    root = tmp_path / "bundle"
    target = root / "Demo" / "weights.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"12")

    result = assets.inspect_asset(root, asset)

    assert result["ok"] is False
    assert result["too_small"]


def test_manifest_is_valid_json_and_names_each_asset():
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    ids = [asset["id"] for asset in manifest["assets"]]
    assert len(ids) == len(set(ids))
    assert {"rhofold", "rnabpflow", "trrna2"} <= set(ids)


def test_trrna2_runner_reads_the_first_sequence(tmp_path):
    msa = tmp_path / "query.a3m"
    msa.write_text(">query\nacgu-TT\n>other\nGGGG\n", encoding="utf-8")

    assert trrna2._query_sequence(msa) == "ACGUUU"
