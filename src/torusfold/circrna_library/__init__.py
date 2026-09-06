"""Auditable local data layer for experimental and generated circRNA structures.

The package deliberately has no dependency on the TorusFold prediction pipeline.
Import :mod:`ingest_gemmi` only when Gemmi-backed structure parsing is needed.
"""

from .schema import (
    SCHEMA_VERSION,
    ArtifactRef,
    AtomRecord,
    BSJSpec,
    ChainSnapshot,
    CircularityEvidence,
    LibraryEntry,
    Provenance,
    QCCheck,
    QCPolicy,
    QCReport,
    Rejection,
    ResidueMapping,
    ResidueRecord,
    SequenceRecord,
    StructureSnapshot,
    canonical_json_bytes,
    safe_relpath,
    sha256_file,
)

__all__ = [
    "SCHEMA_VERSION", "ArtifactRef", "AtomRecord", "BSJSpec",
    "ChainSnapshot", "CircularityEvidence", "LibraryEntry", "Provenance",
    "QCCheck", "QCPolicy", "QCReport", "Rejection", "ResidueMapping",
    "ResidueRecord", "SequenceRecord", "StructureSnapshot",
    "canonical_json_bytes", "safe_relpath", "sha256_file",
]
