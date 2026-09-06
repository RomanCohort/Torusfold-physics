"""Optional-Gemmi adapter for the isolated circRNA structure library.

The adapter deliberately has no import-time dependency on Gemmi or on the
TorusFold prediction stack.  It converts one PDB/mmCIF source into one entry
per RNA chain and selected model, while preserving source bytes by digest.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional

from .circular_qc import assess
from .schema import (
    ArtifactRef, ChainSnapshot, CircularityEvidence, CircularityStatus,
    EntryStatus, LibraryEntry, Provenance, ResidueMapping, ResidueRecord,
    SequenceRecord, StructureSnapshot, AtomRecord, canonical_json_bytes,
    sha256_file,
)

_MODIFIED_BASES = {
    "A": "A", "C": "C", "G": "G", "U": "U",
    "ADE": "A", "CYT": "C", "GUA": "G", "URI": "U",
    "RA": "A", "RC": "C", "RG": "G", "RU": "U",
    "1MA": "A", "2MA": "A", "6MA": "A", "MIA": "A", "AMP": "A",
    "5MC": "C", "OMC": "C", "M5C": "C", "CCC": "C", "CMP": "C",
    "7MG": "G", "M2G": "G", "OMG": "G", "GTP": "G", "GDP": "G",
    "PSU": "U", "H2U": "U", "5MU": "U", "OMU": "U", "U34": "U",
    "UMP": "U", "1MU": "U", "4SU": "U", "5BU": "U",
}
_DNA_NAMES = {"DA", "DC", "DG", "DT", "DI", "THY", "T"}
_RNA_ATOMS = {"P", "OP1", "OP2", "OP3", "O3'", "O3*", "O2'", "C1'", "C1*"}


@dataclass(frozen=True)
class IngestOptions:
    model_policy: str = "first"
    include_non_rna: bool = False
    source_id: Optional[str] = None
    source_uri: Optional[str] = None
    accession: Optional[str] = None
    title: Optional[str] = None
    sequence_path: Optional[str] = None
    record_type: str = "experimental"

    def __post_init__(self) -> None:
        if self.model_policy not in {"first", "all"}:
            raise ValueError("model_policy must be 'first' or 'all'")
        if self.record_type not in {"experimental", "predicted", "physics_generated", "constraint"}:
            raise ValueError("unsupported record_type")

    @classmethod
    def from_value(cls, value: "IngestOptions | Mapping[str, Any] | None") -> "IngestOptions":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            names = {field for field in cls.__dataclass_fields__}
            return cls(**{k: v for k, v in value.items() if k in names})
        raise TypeError("options must be IngestOptions, a mapping, or None")


@dataclass
class IngestResult:
    source_path: str
    source_format: str
    raw_sha256: str
    raw_size: int
    raw_bytes: bytes
    normalized_mmcif: str
    metadata: dict[str, Any]
    entries: list[LibraryEntry]
    warnings: list[str]
    errors: list[str]
    rejections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        result = {
            "source_path": self.source_path, "source_format": self.source_format,
            "raw_sha256": self.raw_sha256, "raw_size": self.raw_size,
            "normalized_mmcif": self.normalized_mmcif,
            "metadata": self.metadata,
            "entries": [entry.to_dict() for entry in self.entries],
            "warnings": list(self.warnings), "errors": list(self.errors),
        }
        if include_raw:
            result["raw_bytes"] = self.raw_bytes
        return result


def _load_gemmi():
    try:
        import gemmi  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Gemmi is required for PDB/mmCIF ingestion; install it with "
            "'pip install gemmi'."
        ) from exc
    return gemmi


def _read_bytes(path: Path) -> bytes:
    """Read the exact source bytes; gzip decompression is parse-only."""
    return path.read_bytes()


def _parse_bytes(path: Path, raw: bytes) -> tuple[Path, Optional[Path]]:
    """Return a path Gemmi can parse and an optional temporary path to remove."""
    if not path.name.lower().endswith(".gz"):
        return path, None
    suffix = ".cif" if path.name.lower().endswith((".cif.gz", ".mmcif.gz")) else ".pdb"
    fd, name = tempfile.mkstemp(prefix="torusfold_ingest_", suffix=suffix)
    with os.fdopen(fd, "wb") as handle:
        handle.write(gzip.decompress(raw))
    return Path(name), Path(name)


def _format(path: Path, raw: bytes) -> str:
    suffix = path.name.lower()
    if suffix.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz")):
        return "mmcif"
    if suffix.endswith((".pdb", ".ent", ".pdb.gz")):
        return "pdb"
    text = raw[:4096].decode("utf-8", errors="ignore")
    return "mmcif" if re.search(r"(^|\n)\s*data_", text, re.I) else "pdb"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    try:
        value = getattr(obj, name)
    except (AttributeError, TypeError):
        return default
    try:
        return value() if callable(value) else value
    except TypeError:
        return default


def _num(value: Any, default: Any = None) -> Any:
    try:
        result = float(value)
        if not result == result or abs(result) == float("inf"):
            return default
        return int(result) if result.is_integer() else result
    except (TypeError, ValueError):
        return default


def _base(name: str) -> Optional[str]:
    key = re.sub(r"[^A-Z0-9]", "", name.upper().strip())
    if key in _DNA_NAMES:
        return None
    return _MODIFIED_BASES.get(key)


def _models(structure: Any) -> list[Any]:
    try:
        return list(structure)
    except TypeError:
        return list(_get(structure, "models", ()) or ())


def _chains(model: Any) -> list[Any]:
    try:
        return list(model)
    except TypeError:
        return list(_get(model, "chains", ()) or ())


def _residues(chain: Any) -> list[Any]:
    try:
        return list(chain)
    except TypeError:
        return list(_get(chain, "residues", ()) or ())


def _atoms(residue: Any) -> list[Any]:
    try:
        return list(residue)
    except TypeError:
        return list(_get(residue, "atoms", ()) or ())


def _seq_fields(residue: Any, fallback: int) -> tuple[Optional[int], str, Optional[int]]:
    seqid = _get(residue, "seqid", None)
    auth = _num(_get(seqid, "num", None), None)
    icode = str(_get(seqid, "icode", "") or "").strip()
    label = _num(_get(residue, "label_seq", _get(residue, "label_seq_id", None)), auth)
    if auth is None:
        auth = fallback
    if label is None:
        label = auth
    return int(auth) if isinstance(auth, (int, float)) else None, icode, int(label) if isinstance(label, (int, float)) else None


def _atom_candidates(atom: Any) -> tuple[str, float, str, float, float, float, str, Optional[int]]:
    pos = _get(atom, "pos", None)
    element = _get(atom, "element", "")
    element = _get(element, "name", element) or ""
    return (
        str(_get(atom, "name", "") or "").strip(),
        float(_num(_get(atom, "occ", _get(atom, "occupancy", 1.0)), 1.0) or 1.0),
        str(_get(atom, "altloc", "") or "").strip(),
        float(_num(_get(pos, "x", None), 0.0) or 0.0),
        float(_num(_get(pos, "y", None), 0.0) or 0.0),
        float(_num(_get(pos, "z", None), 0.0) or 0.0),
        str(element).strip(),
        _num(_get(atom, "serial", None), None),
    )


def _choose_atoms(residue: Any, chain_id: str, model_number: int, resname: str,
                  auth_seq: Optional[int], icode: str, uid: str) -> tuple[AtomRecord, ...]:
    chosen: dict[str, tuple[Any, ...]] = {}
    for atom in _atoms(residue):
        candidate = _atom_candidates(atom)
        name, occ, altloc, x, y, z, element, serial = candidate
        if not name:
            continue
        previous = chosen.get(name)
        # Blank altloc wins; otherwise highest occupancy and lexicographically
        # smallest alternative are deterministic tie breakers.
        rank = (0 if not altloc else 1, -occ, altloc)
        if previous is None or rank < previous[0]:
            chosen[name] = (rank, candidate)
    records = []
    for name in sorted(chosen):
        _, (_, occ, altloc, x, y, z, element, serial) = chosen[name]
        records.append(AtomRecord(
            atom_name=name, element=element or name[0], x=x, y=y, z=z,
            chain_id=chain_id, residue_name=resname, residue_number=auth_seq,
            insertion_code=icode, altloc=altloc, occupancy=occ,
            model_number=model_number, auth_atom_id=name, label_atom_id=name,
            serial=int(serial) if isinstance(serial, (int, float)) else None,
        ))
    return tuple(records)


def _chain_is_rna(residues: list[Any]) -> bool:
    known = sum(_base(str(_get(r, "name", ""))) is not None for r in residues)
    nucleic_atoms = sum(
        any(str(_get(a, "name", "")).strip().upper() in _RNA_ATOMS for a in _atoms(r))
        for r in residues
    )
    return known >= 2 or (known >= 1 and nucleic_atoms >= 1)


def _safe_id(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return result or "entry"


def _external_sequence(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8")
    return "".join(line.strip() for line in text.splitlines() if line.strip() and not line.startswith(">"))


def _declared_chain_sequence(chain: Any) -> Optional[str]:
    """Read a declared polymer sequence exposed by a parser/test double."""
    candidates = [
        _get(chain, "declared_sequence", None),
        _get(chain, "polymer_sequence", None),
        _get(chain, "sequence", None),
    ]
    entity = _get(chain, "entity", None)
    if entity is not None:
        candidates.extend((_get(entity, "full_sequence", None),
                          _get(entity, "sequence", None)))
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        value = candidate.upper().replace("T", "U")
        value = re.sub(r"[^ACGUNRYKMSWBDHV]", "", value)
        if value:
            return value
    return None


def _align_declared_observed(declared: str, observed: str) -> tuple[tuple[Optional[int], ...], tuple[int, ...], float, float, tuple[dict[str, Any], ...]]:
    """Globally align a declared sequence to observed residues.

    The output maps declared positions to observed residue indices.  A bounded
    prefix fallback avoids quadratic memory for unusually large polymers while
    still recording the limitation as an event.
    """
    n, m = len(declared), len(observed)
    if n == m and declared == observed:
        return tuple(range(m)), (), 1.0, 1.0, ()
    if n * m > 4_000_000:
        mapped = tuple(range(min(n, m))) + (None,) * max(0, n - m)
        events = ({"type": "alignment_bounded", "declared_length": n, "observed_length": m},)
        return mapped, tuple(i for i, item in enumerate(mapped) if item is None), min(n, m) / max(1, n), 0.0, events
    score = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0], trace[i][0] = -2 * i, "up"
    for j in range(1, m + 1):
        score[0][j], trace[0][j] = -2 * j, "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + (2 if declared[i - 1] == observed[j - 1] else -1)
            up, left = score[i - 1][j] - 2, score[i][j - 1] - 2
            best = max((diag, "diag"), (up, "up"), (left, "left"), key=lambda item: (item[0], item[1] == "diag", item[1] == "up"))
            score[i][j], trace[i][j] = best
    mapping: list[Optional[int]] = [None] * n
    aligned = 0
    matches = 0
    events: list[dict[str, Any]] = []
    i, j = n, m
    while i or j:
        step = trace[i][j]
        if step == "diag":
            i -= 1; j -= 1
            mapping[i] = j
            aligned += 1
            matches += declared[i] == observed[j]
            if declared[i] != observed[j]:
                events.append({"type": "mismatch", "sequence_index": i, "declared": declared[i], "observed": observed[j]})
        elif step == "up":
            i -= 1
            events.append({"type": "missing_observed_residue", "sequence_index": i})
        else:
            j -= 1
            events.append({"type": "extra_observed_residue", "observed_index": j})
    coverage = aligned / max(1, n)
    identity = matches / max(1, aligned)
    return tuple(mapping), tuple(i for i, item in enumerate(mapping) if item is None), coverage, identity, tuple(reversed(events))


def _structure_links(structure: Any) -> tuple[tuple[str, ...], ...]:
    """Extract explicit covalent links without exposing Gemmi objects."""
    links = _get(structure, "connections", None)
    if links is None:
        links = _get(structure, "covalent_links", None)
    if links is None:
        links = _get(structure, "struct_conn", ())
    result: list[tuple[str, ...]] = []
    for link in links or ():
        values: list[str] = []
        if isinstance(link, Mapping):
            values = [str(link.get(key, "")) for key in ("left", "right", "atom1", "atom2", "residue1", "residue2") if link.get(key) not in (None, "")]
        else:
            for name in ("partner1", "partner2", "atom1", "atom2"):
                part = _get(link, name, None)
                if part is not None:
                    values.append(str(_get(part, "name", part)))
            if not values and isinstance(link, (tuple, list)):
                values = [str(x) for x in link]
        if values:
            result.append(tuple(values))
    return tuple(result)


def _chain_entry(source_stem: str, model: Any, model_index: int, chain: Any,
                 model_count: int, source_metadata: Mapping[str, Any],
                 external_sequence: Optional[str], record_type: str,
                 explicit_links: tuple[tuple[str, ...], ...] = (),
                 declared_sequence: Optional[str] = None) -> Optional[LibraryEntry]:
    raw_residues = _residues(chain)
    if not raw_residues or not _chain_is_rna(raw_residues):
        return None
    auth_chain = str(_get(chain, "name", "") or "A").strip() or "A"
    label_chain = str(_get(chain, "label_asym_id", auth_chain) or auth_chain).strip() or auth_chain
    model_num = _num(_get(model, "num", _get(model, "name", model_index + 1)), model_index + 1)
    model_num = int(model_num) if isinstance(model_num, (int, float)) else model_index + 1
    residues: list[ResidueRecord] = []
    sequence: list[str] = []
    residue_to_sequence: dict[str, int] = {}
    for observed_index, residue in enumerate(raw_residues):
        resname = str(_get(residue, "name", "UNK") or "UNK").strip().upper()
        base = _base(resname)
        has_nucleic_atom = any(str(_get(a, "name", "")).strip().upper() in _RNA_ATOMS for a in _atoms(residue))
        if base is None and not has_nucleic_atom:
            continue
        base = base or "N"
        auth_seq, icode, label_seq = _seq_fields(residue, observed_index + 1)
        uid = f"m{model_num}|l{label_chain}|a{auth_chain}|{label_seq}|{auth_seq}{icode}|{resname}"
        atoms = _choose_atoms(residue, auth_chain, model_num, resname, auth_seq, icode, uid)
        metadata = {"residue_uid": uid, "label_asym_id": label_chain,
                    "auth_asym_id": auth_chain}
        residues.append(ResidueRecord(
            residue_name=resname, sequence_index=len(sequence), residue_uid=uid,
            chain_id=label_chain,
            auth_seq_id=auth_seq, label_seq_id=label_seq, insertion_code=icode,
            one_letter_code=base, atoms=atoms, is_rna=True,
            modified=(resname not in {"A", "C", "G", "U"}), metadata=metadata,
        ))
        sequence.append(base)
        residue_to_sequence[uid] = len(sequence) - 1
    if not sequence:
        return None
    observed = "".join(sequence)
    declared = external_sequence or observed
    if external_sequence and external_sequence.upper() != external_sequence:
        declared = external_sequence.upper().replace("T", "U")
    seq_record = SequenceRecord(
        sequence=declared, sequence_id=f"{source_stem}_{label_chain}",
        chain_id=label_chain, circular=None,
        description="external sequence" if external_sequence else "observed polymer sequence",
        metadata={"sequence_origin": "external" if external_sequence else "observed"},
    )
    # The common case is a complete observed chain.  For an external sequence,
    # retain a conservative prefix mapping; callers can replace it with a
    # curated mapping when the deposited construct uses a different rotation.
    if len(declared) == len(residues):
        seq_to_residue = tuple(range(len(residues)))
        unmapped: tuple[int, ...] = ()
    else:
        seq_to_residue = tuple(range(len(residues))) + (None,) * max(0, len(declared) - len(residues))
        seq_to_residue = seq_to_residue[:len(declared)]
        unmapped = tuple(i for i, value in enumerate(seq_to_residue) if value is None)
    mapping = ResidueMapping(
        sequence_id=seq_record.sequence_id, chain_id=label_chain,
        sequence_length=len(declared), sequence_to_residue=seq_to_residue,
        residue_to_sequence=residue_to_sequence, unmapped_sequence_indices=unmapped,
        confidence=tuple(1.0 if x is not None else 0.0 for x in seq_to_residue),
        notes="observed_only" if not external_sequence else "external_sequence_prefix_mapping",
    )
    chain_snapshot = ChainSnapshot(
        chain_id=label_chain, auth_chain_id=auth_chain, label_chain_id=label_chain,
        residues=tuple(residues), sequence=observed, entity_type="RNA",
        metadata={"model_number": model_num, "observed_sequence": observed,
                  "declared_sequence": declared},
    )
    structure = StructureSnapshot(
        structure_id=f"{source_stem}_{label_chain}_m{model_num}",
        chains=(chain_snapshot,), model_number=model_num, model_count=model_count,
        source_format=str(source_metadata.get("source_format", "")),
        metadata=dict(source_metadata),
        model_policy=str(source_metadata.get("model_policy", "first")),
    )
    entry_id = _safe_id(f"{source_stem}_{label_chain}_m{model_num}_{source_metadata['raw_sha256'][:12]}")
    evidence = CircularityEvidence(status=CircularityStatus.NOT_ASSESSED, is_circular=None, evidence_tier="C")
    provenance = Provenance(
        source=str(source_metadata.get("source_filename", source_stem)),
        source_sha256=str(source_metadata["raw_sha256"]),
        source_format=str(source_metadata.get("source_format", "")),
        tool="gemmi", tool_version="", metadata={"model_policy": source_metadata.get("model_policy", "first")},
    )
    return LibraryEntry(
        entry_id=entry_id, sequence=seq_record, record_type=record_type,
        structure=structure, mapping=mapping,
        circularity=evidence, provenance=provenance, status=EntryStatus.CANDIDATE,
        raw_sha256=str(source_metadata["raw_sha256"]), metadata=dict(source_metadata),
        schema_version=1,
    )


def _normalised_text(structure: Any, raw: bytes, source_format: str) -> str:
    if source_format == "mmcif":
        try:
            document = structure.make_mmcif_document()
            text = document.as_string()
            if text:
                return text if text.endswith("\n") else text + "\n"
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace") if raw else "data_torusfold\n#\n"


def ingest_file(path: str | os.PathLike[str], options: IngestOptions | Mapping[str, Any] | None = None) -> IngestResult:
    """Parse a local PDB/mmCIF source into schema records.

    This function performs no network access and writes nothing.  Call
    :func:`write_artifacts` to archive the immutable source and derived files.
    """
    opts = IngestOptions.from_value(options)
    source = Path(path)
    raw = _read_bytes(source)
    digest = hashlib.sha256(raw).hexdigest()
    source_format = _format(source, raw)
    gemmi = _load_gemmi()
    parse_path, temp_path = _parse_bytes(source, raw)
    try:
        structure = gemmi.read_structure(str(parse_path))
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    models_all = _models(structure)
    models = models_all[:1] if opts.model_policy == "first" else models_all
    metadata: dict[str, Any] = {
        "source_filename": source.name, "source_format": source_format,
        "raw_sha256": digest, "raw_size": len(raw),
        "model_policy": opts.model_policy, "accession": opts.accession or source.stem,
    }
    for attr, key in (("title", "title"), ("name", "structure_name"),
                      ("resolution", "resolution"), ("experimental_method", "experimental_method")):
        value = _get(structure, attr, None)
        if value not in (None, ""):
            metadata[key] = str(value)
    if opts.source_id:
        metadata["source_id"] = opts.source_id
    if opts.source_uri:
        metadata["source_uri"] = opts.source_uri
    if opts.title:
        metadata["title"] = opts.title
    external_sequence = _external_sequence(opts.sequence_path)
    entries: list[LibraryEntry] = []
    warnings: list[str] = []
    for model_index, model in enumerate(models):
        for chain in _chains(model):
            entry = _chain_entry(source.stem, model, model_index, chain, len(models_all),
                                 metadata, external_sequence, opts.record_type)
            if entry is not None:
                entries.append(entry)
    if not models_all:
        warnings.append("structure contains no models")
    if not entries:
        warnings.append("no RNA chains found")
    return IngestResult(
        source_path=str(source), source_format=source_format, raw_sha256=digest,
        raw_size=len(raw), raw_bytes=raw, normalized_mmcif=_normalised_text(structure, raw, source_format),
        metadata=metadata, entries=entries, warnings=warnings, errors=[],
    )


def _atomic_write(path: Path, data: bytes, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite and path.read_bytes() != data:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
        return
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


def _ref(root: Path, path: Path, role: str, media_type: str) -> ArtifactRef:
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    return ArtifactRef(path=rel, sha256=sha256_file(path), size_bytes=path.stat().st_size,
                       media_type=media_type, role=role)


def write_artifacts(result: IngestResult, library_root: str | os.PathLike[str], *, overwrite: bool = False) -> dict[str, Any]:
    """Archive raw bytes and write validated JSON-derived entry artifacts."""
    root = Path(library_root).expanduser().resolve()
    ext = ".cif" if result.source_format == "mmcif" else ".pdb"
    raw_path = root / "raw" / result.raw_sha256[:2] / f"{result.raw_sha256}{ext}"
    _atomic_write(raw_path, result.raw_bytes, overwrite=False)
    output: dict[str, Any] = {"raw": str(raw_path), "entries": [], "normalized": []}
    for original in result.entries:
        entry_dir = root / "derived" / original.entry_id
        normalized_path = root / "normalized" / original.entry_id / "structure.cif"
        _atomic_write(normalized_path, result.normalized_mmcif.encode("utf-8"), overwrite=overwrite)
        sequence_path = entry_dir / "sequence.fasta"
        residues_path = entry_dir / "residues.jsonl"
        mapping_path = entry_dir / "mapping.json"
        qc_path = entry_dir / "qc.json"
        entry_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(sequence_path, (f">{original.sequence.sequence_id or original.entry_id}\n"
                                      f"{original.sequence.sequence}\n").encode("utf-8"), overwrite=overwrite)
        residue_lines = []
        for residue in original.structure.residues:
            residue_lines.append(canonical_json_bytes(residue.to_dict()) + b"\n")
        _atomic_write(residues_path, b"".join(residue_lines), overwrite=overwrite)
        _atomic_write(mapping_path, canonical_json_bytes(original.mapping.to_dict()) + b"\n", overwrite=overwrite)
        qc = assess(original.structure, original.mapping, original.circularity, entry_id=original.entry_id)
        _atomic_write(qc_path, canonical_json_bytes(qc.to_dict()) + b"\n", overwrite=overwrite)
        refs = (
            _ref(root, raw_path, "raw_source", "chemical/x-pdb" if ext == ".pdb" else "chemical/x-mmcif"),
            _ref(root, normalized_path, "normalized_structure", "chemical/x-mmcif"),
            _ref(root, sequence_path, "sequence", "text/x-fasta"),
            _ref(root, residues_path, "residues", "application/jsonl"),
            _ref(root, mapping_path, "mapping", "application/json"),
            _ref(root, qc_path, "qc", "application/json"),
        )
        entry = replace(original, artifacts=refs, qc=qc)
        entry_path = entry_dir / "entry.json"
        _atomic_write(entry_path, canonical_json_bytes(entry.to_dict()) + b"\n", overwrite=overwrite)
        output["entries"].append(str(entry_path))
        output["normalized"].append(str(normalized_path))
    return output


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Ingest a local PDB/mmCIF into the circRNA library")
    parser.add_argument("source", type=Path)
    parser.add_argument("--library-root", type=Path)
    parser.add_argument("--model-policy", choices=("first", "all"), default="first")
    parser.add_argument("--sequence", type=Path)
    parser.add_argument("--record-type", choices=("experimental", "predicted", "physics_generated", "constraint"), default="experimental")
    args = parser.parse_args(argv)
    try:
        result = ingest_file(args.source, IngestOptions(
            model_policy=args.model_policy, sequence_path=str(args.sequence) if args.sequence else None,
            record_type=args.record_type,
        ))
    except ImportError as exc:
        parser.error(str(exc))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.library_root:
        written = write_artifacts(result, args.library_root)
        print(json.dumps({"entries": len(result.entries), "written": written}, indent=2))
    else:
        print(json.dumps({"entries": len(result.entries), "warnings": result.warnings,
                          "raw_sha256": result.raw_sha256}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["IngestOptions", "IngestResult", "ingest_file", "write_artifacts", "main"]
