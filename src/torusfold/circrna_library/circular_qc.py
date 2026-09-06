"""Dependency-free quality checks for circular RNA structure snapshots.

The checks operate on schema records only.  They intentionally do not import
Gemmi, NumPy, OpenMM, ViennaRNA, or any TorusFold prediction module.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Optional, Sequence

from .schema import (
    CircularityEvidence, CircularityStatus, QCCheck, QCPolicy, QCReport,
    QCStatus, ResidueMapping, StructureSnapshot, canonical_json_bytes,
)


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_snapshot(value: Any) -> Any:
    if isinstance(value, StructureSnapshot):
        return value
    if isinstance(value, Mapping):
        try:
            return StructureSnapshot.from_dict(value)
        except Exception:
            # Compact/legacy snapshots are still inspectable through duck typing.
            return value
    return value


def _as_mapping(value: Any) -> Any:
    if isinstance(value, ResidueMapping):
        return value
    if isinstance(value, Mapping):
        try:
            return ResidueMapping.from_dict(value)
        except Exception:
            return value
    return value


def _as_circularity(value: Any) -> Any:
    if value is None:
        return CircularityEvidence()
    if isinstance(value, CircularityEvidence):
        return value
    if isinstance(value, Mapping):
        try:
            return CircularityEvidence.from_dict(value)
        except Exception:
            return value
    return value


def _chains(snapshot: Any) -> list[Any]:
    chains = _value(snapshot, "chains", ())
    if chains:
        return list(chains)
    residues = _value(snapshot, "residues", ())
    if residues:
        return [{
            "chain_id": _value(snapshot, "chain_id", "A"),
            "label_chain_id": _value(snapshot, "label_asym_id", _value(snapshot, "chain_id", "A")),
            "auth_chain_id": _value(snapshot, "auth_asym_id", _value(snapshot, "chain_id", "A")),
            "entity_type": "RNA", "residues": residues,
        }]
    return []


def _residues(chain: Any) -> list[Any]:
    return list(_value(chain, "residues", ()) or ())


def _atoms(residue: Any) -> list[Any]:
    atoms = _value(residue, "atoms", ())
    return list(atoms.values()) if isinstance(atoms, Mapping) else list(atoms or ())


def _atom_name(atom: Any) -> str:
    return str(_value(atom, "atom_name", _value(atom, "name", "")) or "").strip()


def _coords(atom: Any) -> Optional[tuple[float, float, float]]:
    raw = _value(atom, "coords", None)
    if raw is None:
        raw = (_value(atom, "x", None), _value(atom, "y", None), _value(atom, "z", None))
    try:
        xyz = tuple(float(x) for x in raw)
    except (TypeError, ValueError):
        return None
    if len(xyz) != 3 or not all(math.isfinite(x) for x in xyz):
        return None
    return xyz  # type: ignore[return-value]


def _representative(residue: Any) -> tuple[Optional[tuple[float, float, float]], str]:
    # P is preferred for coarse-grained and deposited structures.  The
    # fallbacks are explicitly reported because they are not equivalent.
    preferred = ("P", "C4'", "C4*", "C1'", "C1*")
    by_name = {_atom_name(atom).upper(): atom for atom in _atoms(residue)}
    for name in preferred:
        atom = by_name.get(name.upper())
        if atom is not None:
            xyz = _coords(atom)
            if xyz is not None:
                return xyz, name
    return None, ""


def _sequence_index(residue: Any, fallback: int) -> Optional[int]:
    value = _value(residue, "sequence_index", _value(residue, "seq_index", fallback))
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _uid(residue: Any, fallback: str) -> str:
    return str(_value(residue, "residue_uid", fallback) or fallback)


def _chain_id(chain: Any) -> str:
    return str(_value(chain, "chain_id", _value(chain, "label_chain_id", "")) or "")


def _is_rna(chain: Any) -> bool:
    entity = str(_value(chain, "entity_type", _value(chain, "polymer_type", "")) or "").upper()
    if "RNA" in entity or entity in {"NUCLEIC", "POLYNUCLEOTIDE"}:
        return True
    residues = _residues(chain)
    if not residues:
        return False
    recognized = sum(
        str(_value(r, "one_letter_code", _value(r, "canonical_base", "N"))).upper() in "ACGUN"
        for r in residues
    )
    return recognized >= max(1, len(residues) // 2)


def _check(name: str, status: QCStatus, message: str = "", **metrics: Any) -> QCCheck:
    return QCCheck(name=name, status=status, message=message, metrics=metrics)


def _worst(checks: Sequence[QCCheck]) -> QCStatus:
    statuses = {c.status for c in checks}
    if QCStatus.FAIL in statuses:
        return QCStatus.FAIL
    if QCStatus.WARN in statuses:
        return QCStatus.WARN
    if QCStatus.PASS in statuses:
        return QCStatus.PASS
    return QCStatus.NOT_ASSESSED


def _covalent_match(snapshot: Any, left_uid: str, right_uid: str) -> bool:
    links = _value(snapshot, "explicit_links", ()) or _value(snapshot, "covalent_links", ()) or ()
    wanted = {left_uid, right_uid}
    for link in links:
        if isinstance(link, Mapping):
            values = [str(link.get(k, "")) for k in
                      ("left", "right", "atom1", "atom2", "residue1", "residue2")]
        elif isinstance(link, (list, tuple, set)):
            values = [str(x) for x in link]
        else:
            values = [str(link)]
        if wanted.issubset(set(values)):
            return True
    return False


def _options_hash(policy: QCPolicy) -> str:
    return hashlib.sha256(canonical_json_bytes(policy.to_dict())).hexdigest()


def assess(snapshot: StructureSnapshot | Mapping[str, Any],
           mapping: ResidueMapping | Mapping[str, Any] | None = None,
           circularity: CircularityEvidence | Mapping[str, Any] | None = None,
           policy: QCPolicy | Mapping[str, Any] | None = None,
           *, entry_id: str = "") -> QCReport:
    """Assess internal geometry and an explicitly supplied BSJ.

    ``bsj_index`` is a boundary index: index 0 means residue ``L-1`` to
    residue 0.  The function never infers a BSJ from file order.
    """
    snapshot = _as_snapshot(snapshot)
    mapping = _as_mapping(mapping)
    circularity = _as_circularity(circularity)
    if policy is None:
        policy = QCPolicy()
    elif isinstance(policy, Mapping):
        policy = QCPolicy.from_dict(policy)
    options_hash = _options_hash(policy)

    checks: list[QCCheck] = []
    chains = _chains(snapshot)
    rna_chains = [chain for chain in chains if _is_rna(chain)]
    mapping_chain = str(_value(mapping, "chain_id", "") or "")
    selected_chains = [c for c in rna_chains if _chain_id(c) == mapping_chain] if mapping_chain else []
    if not selected_chains:
        selected_chains = rna_chains

    if not chains:
        checks.append(_check("structure_present", QCStatus.FAIL, "no chains found"))
        return QCReport(status=QCStatus.FAIL, checks=tuple(checks), summary={"n_chains": 0},
                        entry_id=entry_id, policy=policy, options_hash=options_hash)
    if not rna_chains and policy.require_rna:
        checks.append(_check("rna_chain", QCStatus.FAIL, "no RNA chain found"))
    else:
        checks.append(_check(
            "rna_chain", QCStatus.PASS if rna_chains else QCStatus.WARN,
            "RNA chain present" if rna_chains else "RNA chain not identified",
            n_rna_chains=len(rna_chains), n_chains=len(chains),
        ))

    all_res = [residue for chain in selected_chains for residue in _residues(chain)]
    uid_values = [_uid(residue, f"residue-{i}") for i, residue in enumerate(all_res)]
    duplicate_uids = len(uid_values) - len(set(uid_values))
    checks.append(_check(
        "residue_identity", QCStatus.FAIL if duplicate_uids else QCStatus.PASS,
        "duplicate residue UIDs" if duplicate_uids else "residue UIDs are unique",
        duplicates=duplicate_uids, count=len(uid_values),
    ))

    atom_count = 0
    nonfinite = 0
    duplicate_atoms = 0
    for residue_index, residue in enumerate(all_res):
        names: set[str] = set()
        uid = uid_values[residue_index]
        for atom in _atoms(residue):
            atom_count += 1
            if _coords(atom) is None:
                nonfinite += 1
            name = _atom_name(atom)
            if name in names:
                duplicate_atoms += 1
            names.add(name)
    finite_status = QCStatus.FAIL if nonfinite > policy.max_nonfinite_atoms else QCStatus.PASS
    checks.append(_check(
        "finite_coordinates", finite_status,
        f"{nonfinite} non-finite or malformed atom coordinates" if nonfinite else "all coordinates finite",
        n_atoms=atom_count, nonfinite_atoms=nonfinite,
    ))
    duplicate_status = QCStatus.FAIL if duplicate_atoms > policy.max_duplicate_atoms else QCStatus.PASS
    checks.append(_check(
        "atom_identity", duplicate_status,
        f"{duplicate_atoms} duplicate atom names" if duplicate_atoms else "atom names are unique per residue",
        duplicate_atoms=duplicate_atoms,
    ))

    sequence_length = int(_value(mapping, "sequence_length", 0) or 0)
    if not sequence_length:
        sequence_length = max(
            (_sequence_index(residue, i) or 0) for i, residue in enumerate(all_res)
        ) + (1 if all_res else 0)
    length_status = QCStatus.PASS if sequence_length >= policy.min_sequence_length else QCStatus.FAIL
    checks.append(_check("sequence_length", length_status,
                         f"sequence length={sequence_length}", length=sequence_length,
                         minimum=policy.min_sequence_length))

    coverage = _value(mapping, "coverage", None)
    if coverage is None or float(coverage) == 0.0 and not _value(mapping, "sequence_to_residue", ()):
        seq_to_res = _value(mapping, "sequence_to_residue", ()) or ()
        unmapped = _value(mapping, "unmapped_sequence_indices", ()) or ()
        coverage = (len(seq_to_res) - len(unmapped)) / len(seq_to_res) if seq_to_res else (1.0 if all_res else 0.0)
    coverage = float(coverage)
    identity = _value(mapping, "identity", None)
    identity_value = float(identity) if identity is not None else None
    mapping_status = QCStatus.PASS
    if coverage < getattr(policy, "minimum_mapping_coverage", 0.0):
        mapping_status = QCStatus.WARN if policy.warn_on_unmapped else QCStatus.FAIL
    if identity_value is not None and identity_value < getattr(policy, "minimum_mapping_identity", 0.0):
        mapping_status = QCStatus.WARN if policy.warn_on_unmapped else QCStatus.FAIL
    checks.append(_check("residue_mapping", mapping_status,
                         f"mapping coverage={coverage:.3f}", coverage=coverage,
                         identity=identity_value, n_residues=len(all_res)))

    # Keep each chain's index space separate.  RNA/protein complexes and
    # multi-chain RNA structures commonly restart sequence_index at zero.
    indexed_by_chain: dict[str, dict[int, tuple[Any, Optional[tuple[float, float, float]], str]]] = {}
    internal_distances: list[float] = []
    internal_missing: list[int] = []
    for chain in selected_chains:
        chain_key = _chain_id(chain) or f"chain-{len(indexed_by_chain)}"
        indexed: dict[int, tuple[Any, Optional[tuple[float, float, float]], str]] = {}
        for i, residue in enumerate(_residues(chain)):
            idx = _sequence_index(residue, i)
            if idx is not None:
                indexed.setdefault(idx, (residue, *_representative(residue)))
        indexed_by_chain[chain_key] = indexed
        if len(indexed) < 2:
            continue
        for idx in range(min(indexed), max(indexed)):
            left, right = indexed.get(idx), indexed.get(idx + 1)
            if left is None or right is None or left[1] is None or right[1] is None:
                internal_missing.append(idx)
                continue
            distance = math.sqrt(sum((left[1][axis] - right[1][axis]) ** 2 for axis in range(3)))
            internal_distances.append(distance)

    if internal_distances:
        mean_gap = sum(internal_distances) / len(internal_distances)
        max_gap = max(internal_distances)
        over_distance = sum(d > policy.max_bond_distance for d in internal_distances)
        if over_distance or len(internal_missing) > policy.max_internal_gap:
            gap_status = QCStatus.FAIL if policy.strict else QCStatus.WARN
        elif internal_missing:
            gap_status = QCStatus.WARN
        else:
            gap_status = QCStatus.PASS
        checks.append(_check("internal_backbone", gap_status,
                             "internal representative-atom gaps assessed",
                             n_bonds=len(internal_distances), mean_gap_angstrom=mean_gap,
                             median_gap_angstrom=sorted(internal_distances)[len(internal_distances) // 2],
                             max_gap_angstrom=max_gap, missing_bonds=len(internal_missing),
                             over_distance=over_distance))
    else:
        mean_gap = 0.0
        checks.append(_check("internal_backbone", QCStatus.NOT_ASSESSED,
                             "fewer than two residues with representative coordinates"))

    is_circular = _value(circularity, "is_circular", None)
    if is_circular is None:
        status = _value(circularity, "status", CircularityStatus.NOT_ASSESSED)
        is_circular = status in (CircularityStatus.PASS, CircularityStatus.PASS.value, "pass")
    bsj_index = _value(circularity, "bsj_index", None)
    if bsj_index is None:
        bsj_obj = _value(circularity, "bsj", None)
        bsj_index = _value(bsj_obj, "bsj_index", None)
    if policy.require_circularity and is_circular is not True:
        checks.append(_check("circularity_evidence", QCStatus.FAIL,
                             "circularity is required but not explicitly established"))
    elif is_circular is True:
        checks.append(_check("circularity_evidence", QCStatus.PASS,
                             "circularity explicitly supplied"))
    else:
        checks.append(_check("circularity_evidence", QCStatus.NOT_ASSESSED,
                             "circularity evidence not supplied"))

    bsj_metrics: dict[str, Any] = {"sequence_length": sequence_length}
    if is_circular is not True or bsj_index is None:
        checks.append(_check("bsj_closure", QCStatus.NOT_ASSESSED,
                             "circularity or BSJ index not explicitly provided", **bsj_metrics))
    else:
        try:
            bsj_index = int(bsj_index)
        except (TypeError, ValueError):
            bsj_index = -1
        target_chain = selected_chains[0] if selected_chains else None
        target_key = _chain_id(target_chain) if target_chain is not None else ""
        indexed = indexed_by_chain.get(target_key, {})
        if bsj_index < 0 or bsj_index >= sequence_length:
            checks.append(_check("bsj_closure", QCStatus.FAIL,
                                 "BSJ index outside sequence length", bsj_index=bsj_index, **bsj_metrics))
        else:
            left_idx, right_idx = (bsj_index - 1) % sequence_length, bsj_index
            left, right = indexed.get(left_idx), indexed.get(right_idx)
            bsj_metrics.update({"bsj_index": bsj_index, "left_index": left_idx, "right_index": right_idx})
            if left is None or right is None or left[1] is None or right[1] is None:
                checks.append(_check("bsj_closure", QCStatus.NOT_ASSESSED,
                                     "BSJ endpoint coordinates are unavailable", **bsj_metrics))
            else:
                distance = math.sqrt(sum((left[1][axis] - right[1][axis]) ** 2 for axis in range(3)))
                explicit = _covalent_match(snapshot, _uid(left[0], str(left_idx)), _uid(right[0], str(right_idx)))
                tolerance = policy.closure_distance_tolerance
                status = QCStatus.PASS if distance <= tolerance else QCStatus.WARN
                bsj_metrics.update({
                    "bsj_distance_angstrom": distance,
                    "closure_tolerance_angstrom": tolerance,
                    "closure_ratio": distance / max(tolerance, 1e-12),
                    "endpoint_atom_method": f"{left[2]}-{right[2]}",
                    "explicit_covalent_link": explicit,
                    "closure_basis": "explicit_covalent" if explicit else "geometric_only",
                    "median_internal_gap_angstrom": sorted(internal_distances)[len(internal_distances) // 2] if internal_distances else None,
                })
                checks.append(_check("bsj_closure", status,
                                     "BSJ closure assessed from explicit boundary", **bsj_metrics))

    summary = {
        "n_chains": len(chains), "n_rna_chains": len(rna_chains),
        "n_residues": len(all_res), "n_atoms": atom_count,
        "sequence_length": sequence_length, "mapping_coverage": coverage,
        "internal_mean_gap_angstrom": mean_gap,
        "bsj_assessed": any(c.name == "bsj_closure" and c.status != QCStatus.NOT_ASSESSED for c in checks),
    }
    return QCReport(status=_worst(checks), checks=tuple(checks), summary=summary,
                    entry_id=entry_id, policy=policy, metrics=summary,
                    options_hash=options_hash)


assess_circular_qc = assess

__all__ = ["assess", "assess_circular_qc"]
