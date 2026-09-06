"""Deterministic, hash-verified manifest builder for the local library."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import (
    SCHEMA_VERSION, LibraryEntry, SchemaError, canonical_json_bytes,
    safe_relpath, sha256_file,
)

BUILDER_VERSION = "1"


def _canonical(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _root_path(root: str | os.PathLike[str]) -> Path:
    return Path(root).expanduser().resolve()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _relative(root: Path, path: Path) -> str:
    if path.is_symlink():
        raise SchemaError(f"symlink artifacts are not allowed: {path}")
    resolved = path.resolve()
    if not _inside(root, resolved):
        raise SchemaError(f"artifact escapes library root: {path}")
    return resolved.relative_to(root).as_posix()


def _validate_artifact(root: Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise SchemaError("artifact reference must be an object")
    rel = safe_relpath(artifact.get("path", ""))
    path = root / Path(rel)
    if not path.exists() or not path.is_file():
        raise SchemaError(f"missing artifact: {rel}")
    if path.is_symlink() or not _inside(root, path):
        raise SchemaError(f"unsafe artifact: {rel}")
    expected_size = artifact.get("size_bytes", artifact.get("size"))
    actual_size = path.stat().st_size
    if expected_size is not None and int(expected_size) != actual_size:
        raise SchemaError(f"artifact size mismatch: {rel}")
    expected_hash = artifact.get("sha256", "")
    actual_hash = sha256_file(path)
    if expected_hash and str(expected_hash).lower() != actual_hash:
        raise SchemaError(f"artifact hash mismatch: {rel}")
    result = dict(artifact)
    result["path"] = rel
    result["sha256"] = actual_hash
    result["size_bytes"] = actual_size
    return result


def _iter_artifacts(entry: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    raw = entry.get("artifacts", ())
    if isinstance(raw, Mapping):
        raw = raw.values()
    if isinstance(raw, (list, tuple)):
        values.extend(x for x in raw if isinstance(x, Mapping))
    # Older ingest outputs use named refs or don't include refs at all.  The
    # builder validates named ref fields when present, without inventing hashes.
    for container_name in ("structure", "sequence", "mapping", "qc", "source"):
        container = entry.get(container_name)
        if isinstance(container, Mapping):
            for key in ("artifact", "artifact_ref", "ref"):
                value = container.get(key)
                if isinstance(value, Mapping):
                    values.append(value)
    return values


def _entry_artifacts(root: Path, entry: Mapping[str, Any], entry_path: Path) -> tuple[dict[str, Any], ...]:
    refs = []
    seen = set()
    for ref in _iter_artifacts(entry):
        try:
            checked = _validate_artifact(root, ref)
        except (KeyError, TypeError, ValueError, OSError, SchemaError):
            raise
        key = checked["path"]
        if key not in seen:
            seen.add(key)
            refs.append(checked)
    # Always include the entry object itself, so manifest validation covers it.
    entry_rel = _relative(root, entry_path)
    entry_bytes = entry_path.read_bytes()
    refs.insert(0, {"path": entry_rel, "sha256": hashlib.sha256(entry_bytes).hexdigest(),
                    "size_bytes": len(entry_bytes), "media_type": "application/json", "role": "entry"})
    return tuple(refs)


def _load_curation(path: Path | None) -> tuple[dict[str, Any], dict[str, str]]:
    decisions: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    if path is None:
        return {}, errors
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("curation line must be an object")
                entry_id = str(value.get("entry_id", ""))
                raw_sha = str(value.get("raw_sha256", ""))
                if not entry_id or not raw_sha:
                    raise ValueError("curation requires entry_id and raw_sha256")
                decisions.setdefault(f"{entry_id}\0{raw_sha}", []).append(dict(value))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                errors[f"line:{line_no}"] = str(exc)
    selected: dict[str, Any] = {}
    for key, values in decisions.items():
        signatures = {canonical_json_bytes(v) for v in values}
        if len(signatures) != 1:
            errors[key] = "conflicting active curation decisions"
        else:
            selected[key] = values[0]
    return selected, errors


def _apply_curation(entry: dict[str, Any], decisions: Mapping[str, Any], errors: Mapping[str, str]) -> tuple[dict[str, Any], str | None]:
    entry_id = str(entry.get("entry_id", ""))
    raw_sha = str(entry.get("raw_sha256", ""))
    if not raw_sha:
        raw_sha = str(entry.get("source", {}).get("raw_sha256", "")) if isinstance(entry.get("source"), Mapping) else ""
    key = f"{entry_id}\0{raw_sha}"
    if key in errors:
        return entry, errors[key]
    decision = decisions.get(key)
    if decision is None:
        return entry, None
    result = dict(entry)
    circularity = dict(result.get("circularity", {})) if isinstance(result.get("circularity"), Mapping) else {}
    for field in ("is_circular", "evidence_tier", "basis", "references", "bsj_index", "curator", "curated_at"):
        if field in decision:
            circularity[field] = decision[field]
    result["circularity"] = circularity
    result["curation"] = {k: decision[k] for k in decision if k not in {"entry_id", "raw_sha256"}}
    return result, None


def _write_atomic(path: Path, data: bytes, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite existing manifest: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _entry_status(entry: Mapping[str, Any]) -> str:
    value = entry.get("status", "candidate")
    return getattr(value, "value", str(value))


def build_manifest(library_root: str | os.PathLike[str], output: str | os.PathLike[str], *,
                   curation_path: str | os.PathLike[str] | None = None,
                   include_statuses: Iterable[str] = ("accepted", "candidate"),
                   strict: bool = True, replace: bool = False) -> dict[str, Any]:
    """Build a stable JSONL manifest and metadata sidecar.

    ``entry.json`` files are the source of truth.  A malformed or incomplete
    entry is recorded in ``rejections.jsonl``; strict mode raises after the
    scan so callers cannot accidentally treat a partial manifest as complete.
    """
    root = _root_path(library_root)
    out = Path(output).expanduser()
    if not out.is_absolute():
        out = root / out
    out = out.resolve()
    if not _inside(root, out.parent):
        raise SchemaError("manifest output must be inside library root")
    statuses = {str(s) for s in include_statuses}
    decisions, curation_errors = _load_curation(Path(curation_path) if curation_path else None)
    entries: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    derived = root / "derived"
    entry_paths = sorted(derived.glob("*/entry.json")) if derived.exists() else []
    for entry_path in entry_paths:
        try:
            if entry_path.is_symlink():
                raise SchemaError("entry.json is a symlink")
            entry = json.loads(entry_path.read_text(encoding="utf-8"))
            if not isinstance(entry, Mapping):
                raise SchemaError("entry.json must contain an object")
            if str(entry.get("schema_version", SCHEMA_VERSION)) != str(SCHEMA_VERSION):
                raise SchemaError("unsupported schema version")
            entry = dict(entry)
            # Validate the complete entry contract, not only its top-level keys.
            LibraryEntry.from_dict(entry)
            entry, curation_error = _apply_curation(entry, decisions, curation_errors)
            if curation_error:
                raise SchemaError(curation_error)
            status = _entry_status(entry)
            if status not in statuses:
                continue
            refs = _entry_artifacts(root, entry, entry_path)
            entry["artifacts"] = list(refs)
            # Keep full nested payload available for a local manifest consumer,
            # but coordinates remain in referenced residues artifacts when the
            # producer supplied them that way.
            entries.append(entry)
        except (OSError, ValueError, TypeError, SchemaError) as exc:
            rejections.append({"path": _relative(root, entry_path), "reason": str(exc)})
    entries.sort(key=lambda item: str(item.get("entry_id", "")))
    manifest_bytes = b"".join(_canonical(entry) + b"\n" for entry in entries)
    rejection_bytes = b"".join(_canonical(item) + b"\n" for item in sorted(rejections, key=lambda x: x["path"]))
    if strict and rejections:
        # Do not write a misleading partial manifest in strict mode.
        raise ValueError(f"manifest contains {len(rejections)} rejected entries")
    _write_atomic(out, manifest_bytes, replace=replace)
    rejection_path = out.with_name("rejections.jsonl")
    _write_atomic(rejection_path, rejection_bytes, replace=replace)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    meta = {
        "schema_version": SCHEMA_VERSION, "library_version": "1", "builder_version": BUILDER_VERSION,
        "manifest_sha256": manifest_hash, "entry_count": len(entries),
        "rejected_count": len(rejections), "included_statuses": sorted(statuses),
        "python_version": platform.python_version(), "options": {"strict": strict, "replace": replace},
        "options_hash": hashlib.sha256(_canonical({"strict": strict, "statuses": sorted(statuses)})).hexdigest(),
    }
    meta_bytes = _canonical(meta) + b"\n"
    _write_atomic(out.with_name("manifest.meta.json"), meta_bytes, replace=replace)
    return {"manifest_path": str(out), "meta_path": str(out.with_name("manifest.meta.json")),
            "rejections_path": str(rejection_path), "entry_count": len(entries),
            "rejected_count": len(rejections), "manifest_sha256": manifest_hash,
            "entries": entries, "rejections": rejections}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Build a deterministic circRNA library manifest")
    parser.add_argument("library_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--curation", type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    args = parser.parse_args(argv)
    result = build_manifest(args.library_root, args.output, curation_path=args.curation,
                            replace=args.replace, strict=not args.non_strict)
    print(json.dumps({k: v for k, v in result.items() if k not in {"entries", "rejections"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["build_manifest"]
