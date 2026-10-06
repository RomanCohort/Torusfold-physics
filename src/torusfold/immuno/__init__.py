"""Immune fingerprint: turning a predicted 3D structure into quantities that a
downstream immunogenicity / PK model can consume.

The literature basis, including which features are mechanistically anchored and
which are not, is in docs/immuno_fingerprint_literature.md. Read that before
adding a feature here.

Layers, bottom up:

    pair_graph_from_coords   base pairs from coordinates alone
    fingerprint              the features themselves

Nothing in this package reads a base-pair annotation, because the structures
this project ships do not carry one.
"""
from __future__ import annotations

from .pair_graph_from_coords import (
    BasePair,
    PairGraph,
    Residue,
    build_pair_graph,
    ideal_aform_helix,
    parse_pdb_residues,
)

__all__ = [
    "BasePair",
    "PairGraph",
    "Residue",
    "build_pair_graph",
    "ideal_aform_helix",
    "parse_pdb_residues",
]
