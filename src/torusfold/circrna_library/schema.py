"""Immutable, dependency-free data contracts for the circRNA structure library.

The library deliberately keeps its interchange format boring: JSON-compatible
frozen dataclasses, tuples instead of lists, and explicit schema versioning.
Parsers (Gemmi, ViennaRNA, OpenMM, and friends) belong in separate modules;
this file only describes and validates the data crossing those boundaries.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints

SCHEMA_VERSION = 1

T = TypeVar("T", bound="SchemaModel")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NUCLEOTIDES = frozenset("ACGUNRYKMSWBDHV")


class SchemaError(ValueError):
    """Raised when a serialized library object is malformed."""


class ModelPolicy(str, Enum):
    FIRST = "first"
    ALL = "all"


class QCStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_ASSESSED = "not_assessed"


class EntryStatus(str, Enum):
    ACCEPTED = "accepted"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


class CircularityStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_ASSESSED = "not_assessed"


class RejectionCode(str, Enum):
    INVALID_SCHEMA = "invalid_schema"
    INVALID_ARTIFACT = "invalid_artifact"
    MALFORMED_STRUCTURE = "malformed_structure"
    NON_RNA = "non_rna"
    QC_FAILED = "qc_failed"
    DUPLICATE = "duplicate"
    UNSUPPORTED = "unsupported"
    MISSING_DATA = "missing_data"


def _finite(value: Any, label: str = "value") -> float:
    """Return a real finite number, rejecting bool, NaN and infinities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError(f"{label} must be finite")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{label} must be a non-negative integer")
    return value


def _string(value: Any, label: str, *, empty: bool = True) -> str:
    if not isinstance(value, str) or (not empty and not value):
        suffix = "" if empty else " and non-empty"
        raise SchemaError(f"{label} must be a string{suffix}")
    if "\x00" in value:
        raise SchemaError(f"{label} must not contain NUL")
    return value


def _freeze(value: Any) -> Any:
    """Recursively make common JSON values immutable and finite."""
    if isinstance(value, SchemaModel):
        return value
    if isinstance(value, Enum):
        return value
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaError("mapping keys must be strings")
            out[key] = _freeze(item)
        return MappingProxyType(out)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaError("numeric values must be finite")
    return value


def _to_primitive(value: Any) -> Any:
    if isinstance(value, SchemaModel):
        return {f.name: _to_primitive(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _to_primitive(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_primitive(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaError("numeric values must be finite")
        return value
    raise SchemaError(f"unsupported value for JSON serialization: {type(value).__name__}")


def _coerce(value: Any, annotation: Any, label: str) -> Any:
    """Strictly coerce JSON input according to a dataclass annotation."""
    if annotation is Any or annotation is object:
        return _freeze(value)
    if value is None:
        if type(None) in get_args(annotation):
            return None
        raise SchemaError(f"{label} must not be null")

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union,):
        errors = []
        for option in args:
            if option is type(None):
                continue
            try:
                return _coerce(value, option, label)
            except (SchemaError, TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise SchemaError(f"{label} has invalid type ({'; '.join(errors)})")
    # PEP 604 unions (X | Y) expose types.UnionType as origin.
    if str(origin) == "<class 'types.UnionType'>":
        for option in args:
            try:
                return _coerce(value, option, label)
            except (SchemaError, TypeError, ValueError):
                pass
        raise SchemaError(f"{label} has invalid type")

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            return value
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{label} must be one of {[x.value for x in annotation]}") from exc

    if isinstance(annotation, type) and issubclass(annotation, SchemaModel):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, Mapping):
            raise SchemaError(f"{label} must be an object")
        return annotation.from_dict(value)

    if origin in (tuple, Tuple):
        if not isinstance(value, (list, tuple)):
            raise SchemaError(f"{label} must be an array")
        if not args:
            return tuple(_freeze(x) for x in value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(x, args[0], f"{label}[{i}]") for i, x in enumerate(value))
        if len(value) != len(args):
            raise SchemaError(f"{label} must contain {len(args)} items")
        return tuple(_coerce(x, t, f"{label}[{i}]") for i, (x, t) in enumerate(zip(value, args)))

    if origin in (list, set, frozenset, Sequence):
        if not isinstance(value, (list, tuple)):
            raise SchemaError(f"{label} must be an array")
        subtype = args[0] if args else Any
        return tuple(_coerce(x, subtype, f"{label}[{i}]") for i, x in enumerate(value))

    if origin in (dict, Dict, Mapping):
        if not isinstance(value, Mapping):
            raise SchemaError(f"{label} must be an object")
        key_type = args[0] if args else str
        value_type = args[1] if len(args) > 1 else Any
        out = {}
        for key, item in value.items():
            if key_type is str and not isinstance(key, str):
                raise SchemaError(f"{label} keys must be strings")
            out[_coerce(key, key_type, f"{label} key")] = _coerce(item, value_type, f"{label}.{key}")
        return MappingProxyType(out)

    if annotation is str:
        if not isinstance(value, str):
            raise SchemaError(f"{label} must be a string")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise SchemaError(f"{label} must be a boolean")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaError(f"{label} must be an integer")
        return value
    if annotation is float:
        return _finite(value, label)
    if annotation is Path:
        if not isinstance(value, str):
            raise SchemaError(f"{label} must be a path string")
        return Path(value)
    return _freeze(value)


def _normalize(self: "SchemaModel") -> None:
    hints = get_type_hints(type(self))
    for f in dataclasses.fields(self):
        value = getattr(self, f.name)
        try:
            value = _coerce(value, hints.get(f.name, f.type), f.name)
        except (SchemaError, TypeError, ValueError) as exc:
            if isinstance(exc, SchemaError):
                raise
            raise SchemaError(f"invalid {f.name}: {exc}") from exc
        object.__setattr__(self, f.name, value)
    self._validate()


class SchemaModel:
    """Mixin implementing strict immutable JSON conversion."""

    schema_version: ClassVar[int] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _normalize(self)

    def _validate(self) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return _to_primitive(self)

    @classmethod
    def from_dict(cls: Type[T], data: Mapping[str, Any]) -> T:
        if not isinstance(data, Mapping):
            raise SchemaError(f"{cls.__name__} must be decoded from an object")
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(data) - set(fields)
        if unknown:
            raise SchemaError(f"{cls.__name__} contains unknown field(s): {', '.join(sorted(map(str, unknown)))}")
        missing = [f.name for f in fields.values() if f.init and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING and f.name not in data]
        if missing:
            raise SchemaError(f"{cls.__name__} is missing required field(s): {', '.join(missing)}")
        # Let dataclass defaults apply to omitted optional fields, while every
        # supplied field is recursively validated by __post_init__.
        return cls(**dict(data))  # type: ignore[arg-type]


@dataclass(frozen=True)
class ArtifactRef(SchemaModel):
    """A content-addressed file relative to a library root."""
    path: str
    sha256: str
    size_bytes: int
    media_type: str = "application/octet-stream"
    role: str = ""
    format: str = ""

    def _validate(self) -> None:
        object.__setattr__(self, "path", safe_relpath(self.path))
        if not _SHA256_RE.fullmatch(self.sha256):
            raise SchemaError("sha256 must be a 64-character hexadecimal digest")
        _nonnegative_int(self.size_bytes, "size_bytes")


@dataclass(frozen=True)
class Provenance(SchemaModel):
    source: str = ""
    source_path: str = ""
    source_sha256: str = ""
    source_format: str = ""
    tool: str = ""
    tool_version: str = ""
    created_at: str = ""
    parent_refs: Tuple[ArtifactRef, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def _validate(self) -> None:
        if self.source_sha256 and not _SHA256_RE.fullmatch(self.source_sha256):
            raise SchemaError("source_sha256 must be a 64-character hexadecimal digest")
        if self.source_path:
            # Source paths may be absolute before ingestion, but must not carry
            # shell/control characters. ArtifactRef is the portable path type.
            _string(self.source_path, "source_path")
        object.__setattr__(self, "parent_refs", tuple(
            ref if isinstance(ref, ArtifactRef) else ArtifactRef.from_dict(ref)
            for ref in self.parent_refs
        ))


@dataclass(frozen=True)
class AtomRecord(SchemaModel):
    atom_name: str
    element: str
    x: float
    y: float
    z: float
    chain_id: str = ""
    residue_name: str = ""
    residue_number: Optional[int] = None
    insertion_code: str = ""
    altloc: str = ""
    occupancy: float = 1.0
    b_factor: float = 0.0
    model_number: int = 1
    serial: Optional[int] = None
    auth_atom_id: str = ""
    label_atom_id: str = ""

    def _validate(self) -> None:
        _string(self.atom_name, "atom_name", empty=False)
        _string(self.element, "element", empty=False)
        for name in ("x", "y", "z", "occupancy", "b_factor"):
            _finite(getattr(self, name), name)
        if self.residue_number is not None:
            if isinstance(self.residue_number, bool) or not isinstance(self.residue_number, int):
                raise SchemaError("residue_number must be an integer or null")
        _nonnegative_int(self.model_number, "model_number")

    @property
    def name(self) -> str:
        return self.atom_name

    @property
    def coords(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def resseq(self) -> Optional[int]:
        return self.residue_number


@dataclass(frozen=True)
class ResidueRecord(SchemaModel):
    residue_name: str
    sequence_index: int
    residue_uid: str = ""
    chain_id: str = ""
    auth_seq_id: Optional[int] = None
    label_seq_id: Optional[int] = None
    insertion_code: str = ""
    one_letter_code: str = ""
    atoms: Tuple[AtomRecord, ...] = field(default_factory=tuple)
    is_rna: bool = True
    modified: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def _validate(self) -> None:
        _string(self.residue_name, "residue_name", empty=False)
        _string(self.residue_uid, "residue_uid")
        _nonnegative_int(self.sequence_index, "sequence_index")
        if self.auth_seq_id is not None and (isinstance(self.auth_seq_id, bool) or not isinstance(self.auth_seq_id, int)):
            raise SchemaError("auth_seq_id must be an integer or null")
        if self.label_seq_id is not None and (isinstance(self.label_seq_id, bool) or not isinstance(self.label_seq_id, int)):
            raise SchemaError("label_seq_id must be an integer or null")
        if self.one_letter_code and self.one_letter_code.upper() not in _NUCLEOTIDES:
            raise SchemaError("one_letter_code must be an RNA/IUPAC nucleotide")

    @property
    def name(self) -> str:
        return self.residue_name

    @property
    def index(self) -> int:
        return self.sequence_index

    @property
    def atoms_by_name(self) -> Mapping[str, AtomRecord]:
        return MappingProxyType({a.atom_name: a for a in self.atoms})


@dataclass(frozen=True)
class ChainSnapshot(SchemaModel):
    chain_id: str
    residues: Tuple[ResidueRecord, ...] = field(default_factory=tuple)
    sequence: str = ""
    entity_type: str = "RNA"
    auth_chain_id: str = ""
    label_chain_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def _validate(self) -> None:
        _string(self.chain_id, "chain_id", empty=False)
        if self.sequence and any(base.upper() not in _NUCLEOTIDES for base in self.sequence):
            raise SchemaError("sequence contains non-RNA/IUPAC symbols")
        indices = [r.sequence_index for r in self.residues]
        if len(indices) != len(set(indices)):
            raise SchemaError("residues contain duplicate sequence_index values")

    @property
    def chain_name(self) -> str:
        return self.chain_id


@dataclass(frozen=True)
class StructureSnapshot(SchemaModel):
    structure_id: str = ""
    chains: Tuple[ChainSnapshot, ...] = field(default_factory=tuple)
    model_number: int = 1
    model_count: int = 1
    source_format: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    explicit_links: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple)
    model_policy: str = "first"

    def _validate(self) -> None:
        _nonnegative_int(self.model_number, "model_number")
        _nonnegative_int(self.model_count, "model_count")
        if self.model_count and self.model_number > self.model_count:
            raise SchemaError("model_number cannot exceed model_count")
        ids = [c.chain_id for c in self.chains]
        if len(ids) != len(set(ids)):
            raise SchemaError("structure contains duplicate chain IDs")

    @property
    def residues(self) -> Tuple[ResidueRecord, ...]:
        return tuple(r for c in self.chains for r in c.residues)

    @property
    def atoms(self) -> Tuple[AtomRecord, ...]:
        return tuple(a for r in self.residues for a in r.atoms)


@dataclass(frozen=True)
class SequenceRecord(SchemaModel):
    sequence: str
    sequence_id: str = ""
    chain_id: str = ""
    alphabet: str = "RNA"
    description: str = ""
    circular: Optional[bool] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def _validate(self) -> None:
        _string(self.sequence, "sequence")
        normalized = self.sequence.upper().replace("T", "U")
        if any(base not in _NUCLEOTIDES for base in normalized):
            raise SchemaError("sequence contains non-RNA/IUPAC symbols")
        object.__setattr__(self, "sequence", normalized)

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass(frozen=True)
class ResidueMapping(SchemaModel):
    sequence_id: str = ""
    chain_id: str = ""
    sequence_length: int = 0
    sequence_to_residue: Tuple[Optional[int], ...] = field(default_factory=tuple)
    residue_to_sequence: Mapping[str, int] = field(default_factory=dict)
    unmapped_sequence_indices: Tuple[int, ...] = field(default_factory=tuple)
    ambiguous_sequence_indices: Tuple[int, ...] = field(default_factory=tuple)
    extra_residue_uids: Tuple[str, ...] = field(default_factory=tuple)
    confidence: Tuple[float, ...] = field(default_factory=tuple)
    coverage: float = 0.0
    identity: float = 0.0
    method: str = ""
    events: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    notes: str = ""

    def _validate(self) -> None:
        _nonnegative_int(self.sequence_length, "sequence_length")
        if self.sequence_to_residue and self.sequence_length and len(self.sequence_to_residue) != self.sequence_length:
            raise SchemaError("sequence_to_residue length must equal sequence_length")
        for i, residue in enumerate(self.sequence_to_residue):
            if residue is not None and (isinstance(residue, bool) or not isinstance(residue, int) or residue < 0):
                raise SchemaError(f"sequence_to_residue[{i}] must be a non-negative integer or null")
        for i in self.unmapped_sequence_indices + self.ambiguous_sequence_indices:
            if isinstance(i, bool) or not isinstance(i, int) or i < 0:
                raise SchemaError("mapping indices must be non-negative integers")
        for i, confidence in enumerate(self.confidence):
            value = _finite(confidence, f"confidence[{i}]")
            if not 0.0 <= value <= 1.0:
                raise SchemaError("mapping confidence must be in [0, 1]")
        for key, value in self.residue_to_sequence.items():
            _string(key, "residue_to_sequence key", empty=False)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaError("residue_to_sequence values must be non-negative integers")
        for name in ("coverage", "identity"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise SchemaError(f"{name} must be in [0, 1]")
        object.__setattr__(self, "extra_residue_uids", tuple(self.extra_residue_uids))
        object.__setattr__(self, "events", tuple(self.events))

    @property
    def sequence_to_structure(self) -> Tuple[Optional[int], ...]:
        return self.sequence_to_residue

    @property
    def structure_to_sequence(self) -> Mapping[str, int]:
        return self.residue_to_sequence


@dataclass(frozen=True)
class BSJSpec(SchemaModel):
    bsj_index: Optional[int] = None
    sequence_length: int = 0
    window_nt: int = 0
    left_index: Optional[int] = None
    right_index: Optional[int] = None
    chain_id: str = ""
    expected_link: bool = True
    label: str = "BSJ"

    def _validate(self) -> None:
        _nonnegative_int(self.sequence_length, "sequence_length")
        _nonnegative_int(self.window_nt, "window_nt")
        for name in ("bsj_index", "left_index", "right_index"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise SchemaError(f"{name} must be a non-negative integer or null")
        if self.bsj_index is not None and self.sequence_length and self.bsj_index >= self.sequence_length:
            raise SchemaError("bsj_index must be smaller than sequence_length")
        if self.sequence_length and self.left_index is not None and self.left_index >= self.sequence_length:
            raise SchemaError("left_index must be smaller than sequence_length")
        if self.sequence_length and self.right_index is not None and self.right_index >= self.sequence_length:
            raise SchemaError("right_index must be smaller than sequence_length")

    @property
    def known(self) -> bool:
        return self.bsj_index is not None


@dataclass(frozen=True)
class CircularityEvidence(SchemaModel):
    status: CircularityStatus = CircularityStatus.NOT_ASSESSED
    is_circular: Optional[bool] = None
    evidence_tier: str = "unclassified"
    basis: Tuple[str, ...] = field(default_factory=tuple)
    references: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    bsj_index: Optional[int] = None
    closure_observed: Optional[bool] = None
    closure_method: str = ""
    endpoint_distance: Optional[float] = None
    endpoint_tolerance: Optional[float] = None
    explicit_link: Optional[bool] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def _validate(self) -> None:
        if self.is_circular is not None and not isinstance(self.is_circular, bool):
            raise SchemaError("is_circular must be boolean or null")
        if self.evidence_tier not in {"A", "B", "C", "D", "unclassified"}:
            raise SchemaError("invalid evidence_tier")
        for name in ("endpoint_distance", "endpoint_tolerance"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
                if value < 0:
                    raise SchemaError(f"{name} must be non-negative")
        if self.bsj_index is not None and (isinstance(self.bsj_index, bool) or not isinstance(self.bsj_index, int) or self.bsj_index < 0):
            raise SchemaError("bsj_index must be a non-negative integer or null")
        if self.references:
            object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "basis", tuple(self.basis))

    @property
    def assessed(self) -> bool:
        return self.status is not CircularityStatus.NOT_ASSESSED


@dataclass(frozen=True)
class QCPolicy(SchemaModel):
    min_sequence_length: int = 1
    require_rna: bool = True
    require_circularity: bool = False
    max_nonfinite_atoms: int = 0
    max_duplicate_atoms: int = 0
    min_rna_fraction: float = 0.5
    max_internal_gap: int = 0
    max_bond_distance: float = 12.0
    closure_distance_tolerance: float = 12.0
    warn_on_unmapped: bool = True
    strict: bool = True
    minimum_mapping_coverage: float = 0.8
    minimum_mapping_identity: float = 0.9
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def _validate(self) -> None:
        for name in ("min_sequence_length", "max_nonfinite_atoms", "max_duplicate_atoms", "max_internal_gap"):
            _nonnegative_int(getattr(self, name), name)
        for name in ("min_rna_fraction",):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise SchemaError(f"{name} must be in [0, 1]")
        for name in ("max_bond_distance", "closure_distance_tolerance"):
            if _finite(getattr(self, name), name) < 0:
                raise SchemaError(f"{name} must be non-negative")


@dataclass(frozen=True)
class QCCheck(SchemaModel):
    name: str
    status: QCStatus
    message: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    severity: str = ""
    code: str = ""

    def _validate(self) -> None:
        _string(self.name, "name", empty=False)
        _string(self.message, "message")
        _string(self.severity, "severity")
        _string(self.code, "code")

    @property
    def passed(self) -> bool:
        return self.status in (QCStatus.PASS, QCStatus.WARN)


@dataclass(frozen=True)
class QCReport(SchemaModel):
    status: QCStatus = QCStatus.NOT_ASSESSED
    checks: Tuple[QCCheck, ...] = field(default_factory=tuple)
    summary: Mapping[str, Any] = field(default_factory=dict)
    entry_id: str = ""
    policy: Optional[QCPolicy] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    options_hash: str = ""

    def _validate(self) -> None:
        if self.status is QCStatus.PASS and any(c.status is QCStatus.FAIL for c in self.checks):
            raise SchemaError("a passing QCReport cannot contain a failing check")

    @property
    def overall_status(self) -> QCStatus:
        return self.status

    @property
    def passed(self) -> bool:
        return self.status in (QCStatus.PASS, QCStatus.WARN)


@dataclass(frozen=True)
class Rejection(SchemaModel):
    code: RejectionCode
    message: str
    entry_id: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def _validate(self) -> None:
        _string(self.message, "message", empty=False)


@dataclass(frozen=True)
class LibraryEntry(SchemaModel):
    entry_id: str
    sequence: SequenceRecord
    record_type: str = "experimental"
    structure: Optional[StructureSnapshot] = None
    mapping: Optional[ResidueMapping] = None
    bsj: Optional[BSJSpec] = None
    circularity: Optional[CircularityEvidence] = None
    qc: Optional[QCReport] = None
    provenance: Provenance = field(default_factory=Provenance)
    artifacts: Tuple[ArtifactRef, ...] = field(default_factory=tuple)
    status: EntryStatus = EntryStatus.CANDIDATE
    raw_sha256: str = ""
    schema_version: int = int(SCHEMA_VERSION)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    rejection: Optional[Rejection] = None

    def _validate(self) -> None:
        _string(self.entry_id, "entry_id", empty=False)
        if self.raw_sha256 and not _SHA256_RE.fullmatch(self.raw_sha256):
            raise SchemaError("raw_sha256 must be a 64-character hexadecimal digest")
        if int(self.schema_version) != int(SCHEMA_VERSION):
            raise SchemaError(f"unsupported schema_version: {self.schema_version!r}")
        if self.status is EntryStatus.REJECTED and self.rejection is None:
            raise SchemaError("rejected entries require rejection")
        if self.record_type not in {"experimental", "predicted", "physics_generated", "constraint"}:
            raise SchemaError("invalid record_type")


    @property
    def entry_status(self) -> EntryStatus:
        return self.status


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a schema object using deterministic, UTF-8 JSON bytes.

    Mapping keys are sorted, insignificant whitespace is removed, and NaN or
    infinity is rejected rather than emitting non-standard JSON.
    """
    primitive = _to_primitive(value)
    try:
        text = json.dumps(primitive, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"cannot encode canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def sha256_file(path: Union[str, os.PathLike[str]], *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(str(file_path))
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def safe_relpath(path: Union[str, os.PathLike[str]], root: Optional[Union[str, os.PathLike[str]]] = None) -> str:
    """Normalize a portable relative path and reject traversal/absolute paths.

    If ``root`` is supplied, ``path`` may be absolute and is accepted only
    when it resolves below ``root``; the returned path is still POSIX-style.
    """
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise SchemaError("path must be a non-empty string")
    if root is not None:
        root_path = Path(root).resolve()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root_path / candidate
        try:
            relative = candidate.resolve().relative_to(root_path)
        except ValueError as exc:
            raise SchemaError("path escapes root") from exc
        raw = relative.as_posix()
    # Test both POSIX and Windows interpretations. This catches drive paths
    # even when running on a different host OS.
    if PurePosixPath(raw).is_absolute() or PureWindowsPath(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise SchemaError("path must be relative")
    if "\\" in raw:
        raise SchemaError("path must use '/' separators")
    normalized = raw
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise SchemaError("path traversal is not allowed")
    result = "/".join(parts)
    if result.startswith("../") or result == "..":
        raise SchemaError("path traversal is not allowed")
    return result


__all__ = [
    "SCHEMA_VERSION", "SchemaError", "ModelPolicy", "QCStatus", "EntryStatus",
    "CircularityStatus", "RejectionCode", "ArtifactRef", "Provenance",
    "AtomRecord", "ResidueRecord", "ChainSnapshot", "StructureSnapshot",
    "SequenceRecord", "ResidueMapping", "BSJSpec", "CircularityEvidence",
    "QCPolicy", "QCCheck", "QCReport", "LibraryEntry", "Rejection",
    "canonical_json_bytes", "sha256_file", "safe_relpath",
]
