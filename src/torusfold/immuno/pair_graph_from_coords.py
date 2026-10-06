"""Base-pair graph derived from 3D coordinates.

This is the middle layer the immune fingerprint sits on. It reads a structure
and answers, per residue, "what is this base paired to, and with what geometry"
-- without ever consulting a base-pair annotation, because the structures this
project has do not carry one.

WHY GEOMETRY AND NOT A DOT-BRACKET
----------------------------------
A sequence-based predictor returns the pairs the molecule *would* form in
isolation. The immune fingerprint has to answer what a receptor can actually
touch, so it needs the pairs the molecule *does* form in the model that got
delivered. For the 2013 nt artifact that is not a preference: the PDB carries a
single REMARK line and no BPOS records, and provenance.json records that it was
decoded from a viewer payload rather than re-run, so no pairing annotation was
ever available. See docs/immuno_fingerprint_literature.md.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
No Leontis-Westhof family classification, no DSSR parity. DW/WW/Hoogsteen
sub-types need base-frame constructions that would be unverifiable without a
reference implementation, and no receptor in the fingerprint's scope is known to
read the difference. Pairs are classified into wc / wobble / mismatch / unknown
by base identity plus cis/trans, and nothing finer. Claiming more resolution than
can be validated would be worse than claiming less.

CIRCULAR TOPOLOGY
-----------------
Helix runs wrap around the origin. A caller that knows the junction must pass
`bsj_index`, whose convention matches torusfold.circrna_library.circular_qc:
index 0 means residue L-1 is bonded to residue 0. This module never infers a
junction from file order -- 2OIU is the counterexample that proves why: the
deposited file has a TER record between U71 and G1, yet _struct_conn records a
1.598 A O3'(71)-P(1) covalent bond, so the molecule is closed and the file order
says otherwise.

CRITERIA, AND WHICH OF THEM ACTUALLY DISCRIMINATE
------------------------------------------------
Three gates plus a base-specific edge test. The thresholds are set from 1QC0,
whose pairing is not in doubt, because two of the gates turned out to mean the
opposite of what was first assumed:

Distance: C1'--C1' <= 11.5 A. Necessary, and alone nearly useless. 1QC0 is 19
stacked pairs, so its C1'--C1' matrix is dense with 7.4-8.0 A contacts between
*neighbouring* pairs, all inside any reasonable distance gate. The 19 real pairs
measure 10.06-11.14 A on this structure.

Glycosidic orientation: the two C1'->N vectors must be PARALLEL (< 60 deg). The
first version of this test rejected near-parallel arrangements on the theory that
parallel meant stacking. That was backwards, and it rejected all 19 real pairs:
in an A-form duplex both glycosidic bonds point out to the same side and the
measured angle is 3-7 deg. Antiparallel is the anomaly, not the signal.

Coplanarity: base normals closer to parallel than perpendicular (< 75 deg). Real
Watson-Crick pairs here measure 9-28 deg. Deliberately loose -- a cheap
prefilter, not the decision.

Base-pair edges: the load-bearing criterion. Each entry is a set of donor and
acceptor contacts that must all lie within HBOND_TOL of IDEAL_HBOND. This is what
separates a real pair from a backbone neighbour, and it is what DSSR and 3DNA
classify on. Contacts verified directly on 1QC0: G-C gives O6-N4 2.85,
N1-N3 2.78, N2-O2 2.66; A-U gives N1-N3 2.79, N6-O4 2.86.

Everything a gate rejects is recorded in PairGraph.rejected with its reason, so a
miss is visible rather than silently absent.
"""
from __future__ import annotations

import dataclasses
import math
import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Residue",
    "BasePair",
    "PairGraph",
    "parse_pdb_residues",
    "build_pair_graph",
    "ideal_aform_helix",
]

# Bases whose ring nitrogens carry the glycosidic bond; the C1'--N vector picks
# out the glycosidic bond direction.
_GLYCOSIDIC_N = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}

# Ring atoms used to fit the base plane, per base. Any subset present is enough;
# three non-collinear atoms define the plane.
_PLANE_ATOMS = {
    "A": ("N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"),
    "G": ("N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"),
    "C": ("N1", "C2", "N3", "C4", "C5", "C6"),
    "U": ("N1", "C2", "N3", "C4", "C5", "C6"),
}

_CANONICAL = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
_WOBBLE = {("G", "U"), ("U", "G")}

# --- base-pair edges -------------------------------------------------------
#
# The geometric tests above (C1' distance, glycosidic orientation, coplanarity)
# are necessary but NOT sufficient, and the failure mode is instructive: in an
# A-form duplex the C1' atoms of residues two to four apart along one strand sit
# 10-11 A from each other with near-parallel base planes, so those tests happily
# "pair" a residue with its own backbone neighbour. Measured on 1QC0, a duplex
# whose pairing is not in doubt: 19 of 24 detections were intra-strand
# pseudo-pairs like C101-C103, every one of them inside the accepted A-form
# distance range.
#
# What separates a real pair from a backbone neighbour is base-specific hydrogen
# bonding, which is what DSSR/3DNA actually classify on. Each entry below is an
# edge: a set of donor/acceptor atom contacts that must all be present. A pair
# counts as wc / wobble only if one edge's contacts are all satisfied; the name
# records the edge, so which edge matched is visible in the output.
#
# Contact distances are checked against an ideal ~2.95 A with a generous band,
# because 2.6 A crystal structures of RNA carry coordinate error comparable to
# the difference between a good and a mediocre hydrogen bond.
PAIR_EDGES = {
    "wc": {
        # Watson-Crick edges. Both bases are named in the 5'->3' sense.
        ("G", "C"): (("O6", "N4"), ("N1", "N3"), ("N2", "O2")),
        ("C", "G"): (("N4", "O6"), ("N3", "N1"), ("O2", "N2")),
        ("A", "U"): (("N1", "N3"), ("N6", "O4")),
        ("U", "A"): (("N3", "N1"), ("O4", "N6")),
    },
    "wobble": {
        # G-U wobble has two registers; both are real and both are listed.
        ("G", "U"): (("O6", "N3"), ("N1", "O2")),
        ("U", "G"): (("N3", "O6"), ("O2", "N1")),
    },
}

IDEAL_HBOND = 2.95
# Set by sweep against 1QC0's 19 known pairs. At 0.50-0.60 the detector returns
# 18 true pairs and 0 false ones; below 0.50 it starts dropping real pairs (an
# O2-N2 contact measuring 2.50 A, i.e. 0.45 off ideal, is the first casualty).
HBOND_TOL = 0.55


def _hbond_ok(a: Residue, b: Residue,
              contacts: Sequence[Tuple[str, str]]) -> Optional[float]:
    """Max deviation from the ideal contact distance, or None if any atom is absent."""
    worst = 0.0
    for atom_a, atom_b in contacts:
        pa = a.base_atoms.get(atom_a)
        pb = b.base_atoms.get(atom_b)
        if pa is None or pb is None:
            return None
        dev = abs(float(np.linalg.norm(pa - pb)) - IDEAL_HBOND)
        if dev > HBOND_TOL:
            return None
        worst = max(worst, dev)
    return worst


def _edge_match(a: Residue, b: Residue):
    """Best matching (pair_type, worst_deviation) for this base combination."""
    best = None
    for pair_type, table in PAIR_EDGES.items():
        contacts = table.get((a.name, b.name))
        if contacts is None:
            continue
        dev = _hbond_ok(a, b, contacts)
        if dev is None:
            continue
        if best is None or dev < best[1]:
            best = (pair_type, dev)
    return best

DEFAULT_MAX_C1_DIST = 11.5
DEFAULT_MIN_C1_DIST = 8.5                # measured: real pairs in 1QC0 are 10.06+
DEFAULT_MAX_GLYCOSIDIC_ANGLE = 60.0     # measured: real WC pairs in 1QC0 sit at 3-7
DEFAULT_MAX_PLANE_ANGLE = 75.0          # measured: real WC pairs in 1QC0 sit at 9-28


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _fit_plane_normal(points: np.ndarray) -> Optional[np.ndarray]:
    """Normal of the best-fit plane through >= 3 points (smallest singular vector).

    The SIGN of the returned vector is arbitrary -- SVD has no preferred
    orientation. That matters twice over: the coplanarity test must compare
    min(theta, 180-theta), and the cis/trans dihedral is built from a cross
    product whose sign flips with it. Callers fix the sign with
    `_orient_normal` before relying on either.
    """
    if points.shape[0] < 3:
        return None
    centred = points - points.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    return _unit(vt[-1])


def _orient_normal(normal: Optional[np.ndarray], c1p: np.ndarray,
                   atoms: Dict[str, np.ndarray],
                   glyco_atom: str) -> Optional[np.ndarray]:
    """Give the plane normal an intrinsic sign, so it is file-order independent.

    The reference direction is (C1'->glycosidic N) x (C1'->a second base atom):
    both vectors lie in or near the base plane, so their cross product is
    perpendicular to it and its direction is fixed by the base's own chemistry
    rather than by whichever ring atom the SVD happened to see first.

    Without this, two bases whose planes are genuinely parallel can come out with
    opposite normals, and every downstream test that mixes normal direction with
    magnitude then reports nonsense -- measured on 1QC0, eleven of nineteen real
    pairs were rejected as "base planes 170 deg apart".
    """
    if normal is None:
        return None
    glyco = atoms.get(glyco_atom)
    if glyco is None:
        return normal
    ref = None
    for name, pos in atoms.items():
        if name in (glyco_atom, "C1'") or name.startswith("H") or name in ("P", "OP1", "OP2"):
            continue
        v = pos - c1p
        if float(np.linalg.norm(v)) < 1e-6:
            continue
        ref = v
        break
    if ref is None:
        return normal
    cross = np.cross(glyco - c1p, ref)
    if float(np.dot(cross, normal)) < 0.0:
        return -normal
    return normal


@dataclasses.dataclass(frozen=True)
class Residue:
    """One nucleotide, reduced to what the pair geometry needs."""

    index: int                      # author residue number, as an int
    name: str                       # A / C / G / U
    chain: str
    c1p: np.ndarray                 # C1' position
    glyco: Optional[np.ndarray]     # unit vector C1' -> glycosidic N
    normal: Optional[np.ndarray]    # unit normal of the fitted base plane
    base_atoms: Dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    @property
    def sort_key(self) -> Tuple[str, int]:
        """Total order used to canonicalise a pair's two ends.

        Sorting by (chain, index) means a pair is always presented and measured
        the same way round, so results do not depend on the order atoms appear
        in the file. It matters: the signed normal dihedral flips sign under a
        swap, so a cis/trans call taken before canonicalisation would depend on
        which atom line happened to come first.
        """
        return (self.chain, self.index)

    @property
    def plane_ok(self) -> bool:
        return self.glyco is not None and self.normal is not None


@dataclasses.dataclass(frozen=True)
class BasePair:
    """A detected pair.

    `key`/`partner` are residue indices; `key_chain`/`partner_chain` say which
    strand each came from. For a single-chain model the chains are equal and the
    pair is intramolecular. Ordering is by (chain, index) so that a two-strand
    duplex always reports the same strand as `key` regardless of file order.
    """

    key: int
    partner: int
    base_key: str
    base_partner: str
    c1_dist: float
    glyco_angle_deg: float          # 0 = parallel, 180 = antiparallel
    plane_angle_deg: float          # 0 = coplanar (supplement taken, so in 0..90)
    pair_type: str                  # "wc" | "wobble" | "mismatch" -- edge matched
    key_chain: str = ""
    partner_chain: str = ""
    hbond_dev: float = 0.0          # worst contact deviation from ideal, in A
    normal_dihedral_deg: float = 0.0  # raw, for debugging only; see _classify

    @property
    def is_inter_chain(self) -> bool:
        return self.key_chain != self.partner_chain

    @property
    def span(self) -> int:
        """Separation along one chain. Only meaningful when intra-chain."""
        return abs(self.partner - self.key)

    def as_dict(self) -> dict:
        return {
            "key": f"{self.key_chain}:{self.key}" if self.key_chain else self.key,
            "partner": (f"{self.partner_chain}:{self.partner}"
                        if self.partner_chain else self.partner),
            "base_key": self.base_key,
            "base_partner": self.base_partner,
            "c1_dist": round(self.c1_dist, 3),
            "glyco_angle_deg": round(self.glyco_angle_deg, 2),
            "plane_angle_deg": round(self.plane_angle_deg, 2),
            "normal_dihedral_deg": round(self.normal_dihedral_deg, 1),
            "pair_type": self.pair_type,
            "hbond_dev": self.hbond_dev,
        }


@dataclasses.dataclass
class PairGraph:
    """Pairs plus the residue table they were derived from."""

    residues: Dict[int, Residue]
    pairs: List[BasePair]
    length: int
    is_circular: bool
    bsj_index: Optional[int]
    params: dict
    rejected: List[dict] = dataclasses.field(default_factory=list)

    # -- lookups ------------------------------------------------------------
    def partner_of(self, index: int) -> Optional[int]:
        for p in self.pairs:
            if p.key == index:
                return p.partner
            if p.partner == index:
                return p.key
        return None

    def paired_indices(self) -> set:
        out = set()
        for p in self.pairs:
            out.add(p.key)
            out.add(p.partner)
        return out

    def by_type(self, pair_type: str) -> List[BasePair]:
        return [p for p in self.pairs if p.pair_type == pair_type]

    # -- helices ------------------------------------------------------------
    def helix_runs(self, max_gap: int = 0) -> List[dict]:
        """Maximal runs of consecutive pairs that continue in one direction.

        Two pairs (i, j) and (i', j') continue each other when they advance the
        same way along both strands, which is what makes a double helix rather
        than a set of isolated contacts. Runs may be intra-chain (a hairpin stem)
        or inter-chain (a duplex), never a mix.

        `max_gap` allows the run to step over up to that many unpaired residues
        on the first strand, which bridges a single bulge without inventing a pair.
        """
        runs: List[dict] = []
        by_strands: Dict[Tuple[str, str], List[BasePair]] = {}
        for p in self.pairs:
            by_strands.setdefault((p.key_chain, p.partner_chain), []).append(p)

        for strands, group in by_strands.items():
            runs.extend(self._runs_within(group, max_gap))

        # longer runs first: the immune features are all about the long ones
        runs.sort(key=lambda r: -r["n_pairs"])
        return runs

    def _runs_within(self, pairs: List[BasePair], max_gap: int) -> List[dict]:
        lookup: Dict[Tuple[str, int], BasePair] = {}
        for p in pairs:
            lookup[(p.key_chain, p.key)] = p
            lookup[(p.partner_chain, p.partner)] = p

        ordered = sorted(
            ((min(p.key, p.partner), max(p.key, p.partner), p) for p in pairs),
            key=lambda t: (t[0], t[1]),
        )

        used = set()
        runs: List[dict] = []
        for a, b, pair in ordered:
            if id(pair) in used:
                continue
            # The direction of travel on the partner strand cannot be inferred
            # from one pair, so try both and keep the longer run.
            best: List[BasePair] = []
            for direction in (-1, +1):
                chain_pairs = [pair]
                cur_a, cur_b = a, b
                while True:
                    nxt = None
                    for step in range(1, max_gap + 2):
                        cand = lookup.get((pair.key_chain,
                                           self._wrap(cur_a + step, self.length)))
                        if cand is None:
                            continue
                        ca = min(cand.key, cand.partner)
                        cb = max(cand.key, cand.partner)
                        if ca == cur_a + step and cb == cur_b + direction * step:
                            nxt = (ca, cb, cand)
                            break
                    if nxt is None:
                        break
                    cur_a, cur_b, p2 = nxt
                    if id(p2) in used:
                        break
                    chain_pairs.append(p2)
                if len(chain_pairs) > len(best):
                    best = chain_pairs
            if len(best) < 2:
                continue
            for p in best:
                used.add(id(p))
            runs.append(self._run_record(best))
        return runs

    def _wrap(self, i: int, length: int) -> int:
        """Residue indices start at 1 here; wrap within 1..length."""
        return (i - 1) % length + 1

    def _run_record(self, run: List[BasePair]) -> dict:
        keys = [p.key for p in run]
        partners = [p.partner for p in run]
        return {
            "n_pairs": len(run),
            "key_start": min(keys),
            "key_end": max(keys),
            "partner_start": min(partners),
            "partner_end": max(partners),
            "mean_c1_dist": round(float(np.mean([p.c1_dist for p in run])), 3),
            "types": sorted({p.pair_type for p in run}),
            "pairs": [p.as_dict() for p in run],
        }

    def summary(self) -> dict:
        runs = self.helix_runs()
        return {
            "length": self.length,
            "n_residues_parsed": len(self.residues),
            "n_pairs": len(self.pairs),
            "n_paired_residues": len(self.paired_indices()),
            "paired_fraction": round(len(self.paired_indices()) / max(1, self.length), 4),
            "by_type": {
                t: len(self.by_type(t))
                for t in sorted({p.pair_type for p in self.pairs})
            },
            "n_helices": len(runs),
            "longest_helix_pairs": runs[0]["n_pairs"] if runs else 0,
            "is_circular": self.is_circular,
            "bsj_index": self.bsj_index,
            "params": self.params,
        }


# ---------------------------------------------------------------------------
# parsing


def parse_pdb_residues(pdb_text: str) -> Dict[int, Residue]:
    """Residues keyed by author residue number, in first-seen order.

    Only A/C/G/U are kept. Residue numbers must be positive integers; a
    structure that numbers from zero cannot be wrapped safely and is rejected
    loudly rather than silently renumbered.

    One-letter residue names and the chain ID share columns 18-21: the PDB
    format writes resName as a 4-character right-justified field, so a
    one-character name leaves its first column blank and that is where the chain
    goes (column 18). Depositions that use three-character names (GUA, CYT) fill
    18-20 instead, pushing the chain to column 22. Both conventions occur in the
    wild -- our own 2OIU and 1QC0 files use the three-letter form -- so the chain
    is taken from whichever of the two columns carries a non-blank.
    """
    raw: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    order: List[Tuple[str, int]] = []
    names: Dict[Tuple[str, int], str] = {}

    def identify(line: str) -> Optional[Tuple[str, int, str]]:
        """(chain, resseq, base) for one coordinate line, or None."""
        short = line[17:20].strip().upper()
        if short in _GLYCOSIDIC_N:
            base, chain = short, line[21]
        else:
            base, chain = line[18].upper(), line[17]
        if base not in _GLYCOSIDIC_N:
            return None
        try:
            resseq = int(line[22:26])
        except ValueError:
            return None
        return chain, resseq, base

    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        ident = identify(line)
        if ident is None:
            continue
        chain, resseq, base = ident
        key = (chain, resseq)
        if key not in raw:
            raw[key] = {}
            order.append(key)
            names[key] = base
        atom = line[12:16].strip()
        try:
            xyz = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
        except ValueError:
            continue
        raw[key][atom] = xyz

    if order and min(k[1] for k in order) < 1:
        raise ValueError(
            "residue numbering starts below 1; circular wrapping assumes 1..L"
        )

    residues: Dict[int, Residue] = {}
    for chain, resseq in order:
        atoms = raw[(chain, resseq)]
        c1p = atoms.get("C1'")
        if c1p is None:
            continue
        name = names[(chain, resseq)]
        glyco = None
        n_atom = atoms.get(_GLYCOSIDIC_N[name])
        if n_atom is not None:
            glyco = _unit(n_atom - c1p)
        plane_pts = [atoms[a] for a in _PLANE_ATOMS[name] if a in atoms]
        normal = None
        if len(plane_pts) >= 3:
            normal = _fit_plane_normal(np.vstack(plane_pts))
            normal = _orient_normal(normal, c1p, atoms, _GLYCOSIDIC_N[name])
        residues[resseq] = Residue(
            index=resseq, name=name, chain=chain, c1p=c1p, glyco=glyco,
            normal=normal, base_atoms=atoms,
        )
    return residues


# ---------------------------------------------------------------------------
# detection


def _pair_geometry(a: Residue, b: Residue):
    """Return (c1_dist, glyco_angle_deg, plane_angle_deg, orientation) or None."""
    if not a.plane_ok or not b.plane_ok:
        return None
    axis = _unit(b.c1p - a.c1p)
    if axis is None:
        return None

    # Glycosidic angle, measured in the plane perpendicular to the C1'--C1' axis
    # so that the along-axis component cannot masquerade as an angle.
    pa = a.glyco - float(np.dot(a.glyco, axis)) * axis
    pb = b.glyco - float(np.dot(b.glyco, axis)) * axis
    ua, ub = _unit(pa), _unit(pb)
    if ua is None or ub is None:
        return None
    cos_g = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    glyco_angle = math.degrees(math.acos(cos_g))

    # Dihedral of the two base normals about the same axis: distinguishes cis
    # from trans, and pins down "which edge pairs with which".
    na = a.normal - float(np.dot(a.normal, axis)) * axis
    nb = b.normal - float(np.dot(b.normal, axis)) * axis
    va, vb = _unit(na), _unit(nb)
    if va is None or vb is None:
        return None
    x = float(np.dot(va, vb))
    y = float(np.dot(np.cross(va, vb), axis))
    dihedral = math.degrees(math.atan2(y, x))

    # Coplanarity takes the supplement: two parallel planes may present normals
    # pointing the same way or opposite ways, and both mean "coplanar".
    cos_plane = abs(float(np.clip(np.dot(a.normal, b.normal), -1.0, 1.0)))
    plane_angle = math.degrees(math.acos(cos_plane))
    c1_dist = float(np.linalg.norm(b.c1p - a.c1p))
    return c1_dist, glyco_angle, plane_angle, dihedral


def _classify(b1: str, b2: str) -> str:
    """Pair type from base identity alone, via the edge that matched.

    There is deliberately no cis/trans label here. An earlier version carried
    one, derived from the sign of the dihedral between the two base normals.
    On 1QC0 that label came out `trans` for all 18 detected Watson-Crick pairs,
    which is the wrong answer for a WC helix -- so the sign convention behind it
    was not right. Rather than ship a field that is confidently wrong and unused,
    it is gone. Re-adding it means first establishing the convention against a
    structure whose pair orientations are tabulated, which this repository does
    not have.
    """
    if (b1, b2) in _CANONICAL:
        return "wc"
    if (b1, b2) in _WOBBLE:
        return "wobble"
    return "mismatch"


def build_pair_graph(
    pdb_text: str,
    *,
    chains: Optional[Sequence[str]] = None,
    is_circular: bool = False,
    bsj_index: Optional[int] = None,
    max_c1_dist: float = DEFAULT_MAX_C1_DIST,
    min_c1_dist: float = DEFAULT_MIN_C1_DIST,
    max_glyco_angle: float = DEFAULT_MAX_GLYCOSIDIC_ANGLE,
    max_plane_angle: float = DEFAULT_MAX_PLANE_ANGLE,
    allowed_types: Sequence[str] = ("wc", "wobble"),
) -> PairGraph:
    """Detect base pairs in `pdb_text` from coordinates alone.

    Args:
        chains: if given, only these chain IDs contribute. Default None takes
            every chain. Pairs within one chain are "intra"; pairs between two
            chains are "inter". The distinction is not cosmetic -- a two-strand
            duplex in one deposited file is normally inter-chain, and dropping
            cross-chain pairs would silently return zero pairs for it. Conversely
            a file that holds several independent copies of a molecule needs
            `chains` to avoid pairing copy A against copy B.
        is_circular: topology of the molecule. Only affects helix runs (and the
            reported length); pair detection is topology-free.
        bsj_index: boundary index for a circular molecule. If given, `is_circular`
            is forced True. Convention matches circular_qc: 0 means residue L-1
            is bonded to residue 0, so with 1-based numbering it is the residue
            number after which the chain closes.
    """
    if bsj_index is not None:
        is_circular = True

    all_residues = parse_pdb_residues(pdb_text)
    if not all_residues:
        raise ValueError("no A/C/G/U residues found in the PDB text")

    if chains is not None:
        wanted = set(chains)
        residues = {k: v for k, v in all_residues.items() if v.chain in wanted}
        missing = wanted - {v.chain for v in all_residues.values()}
        if missing:
            raise ValueError(f"requested chains not present: {sorted(missing)}")
        if not residues:
            raise ValueError(f"no residues in requested chains {sorted(wanted)}")
    else:
        residues = dict(all_residues)

    # Circular wrapping needs one contiguous chain numbered 1..L. Two chains are
    # two molecules as far as this module is concerned.
    chain_ids = sorted({r.chain for r in residues.values()})
    if len(chain_ids) > 1 and is_circular:
        raise ValueError(
            "is_circular with more than one chain selected is ambiguous; "
            "restrict `chains` to the circular molecule"
        )

    length = max(residues)
    if bsj_index is not None and not (0 <= bsj_index < length):
        raise ValueError(
            f"bsj_index {bsj_index} outside 0..{length - 1} for length {length}"
        )

    idx = sorted(residues)
    coords = np.vstack([residues[i].c1p for i in idx])

    # KD-tree when scipy is around (it is, via the [ml] extra); brute force
    # otherwise. At 2013 residues the squared-distance matrix is ~32 MB, so the
    # fallback is survivable; at 10k+ it would not be, hence the tree.
    neighbours: Iterable[Tuple[int, int]]
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(coords)
        raw = tree.query_pairs(r=max_c1_dist, output_type="ndarray")
        neighbours = ((int(i), int(j)) for i, j in raw)
    except Exception:
        n = len(idx)
        d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1)
        iu, ju = np.triu_indices(n, k=1)
        near = d2[iu, ju] <= max_c1_dist ** 2
        neighbours = ((int(a), int(b)) for a, b in zip(iu[near], ju[near]))

    pairs: List[BasePair] = []
    rejected: List[dict] = []
    for i, j in neighbours:
        ri, rj = residues[idx[i]], residues[idx[j]]
        if rj.sort_key < ri.sort_key:
            ri, rj = rj, ri
        geo = _pair_geometry(ri, rj)
        if geo is None:
            # Missing C1', glycosidic N, or too few ring atoms to fit a plane.
            # Recorded rather than dropped: a silently absent candidate looks
            # exactly like "no pair here", which is the failure mode this whole
            # rejection list exists to prevent.
            rejected.append({
                "key": f"{ri.chain}:{ri.index}", "partner": f"{rj.chain}:{rj.index}",
                "base_key": ri.name, "base_partner": rj.name,
                "c1_dist": round(float(np.linalg.norm(rj.c1p - ri.c1p)), 3),
                "reason": "missing C1'/glycosidic N/ring atoms for the geometry",
            })
            continue
        c1_dist, glyco_angle, plane_angle, dihedral = geo

        def reject(reason: str) -> None:
            rejected.append({
                "key": f"{ri.chain}:{ri.index}", "partner": f"{rj.chain}:{rj.index}",
                "base_key": ri.name, "base_partner": rj.name,
                "c1_dist": round(c1_dist, 3), "reason": reason,
            })

        # Lower bound first: nothing real pairs below ~9 A of C1'--C1'. Without
        # this, A-form stack neighbours inside one strand (5.3-5.6 A here) match
        # a wobble or wc edge by coincidence and appear as intra-chain pairs.
        if c1_dist < min_c1_dist:
            reject(f"C1'--C1' {c1_dist:.2f} A below {min_c1_dist}")
            continue
        # The two glycosidic bonds must be parallel. Measured 3-7 deg on the 19
        # real pairs of 1QC0; an antiparallel arrangement is a coaxial stack.
        if glyco_angle > max_glyco_angle:
            reject(f"glycosidic bonds {glyco_angle:.0f} deg apart (not parallel)")
            continue
        if plane_angle > max_plane_angle:
            reject(f"base planes {plane_angle:.0f} deg apart")
            continue

        # The deciding test. Without it, A-form backbone neighbours pass every
        # check above -- see the note on PAIR_EDGES.
        match = _edge_match(ri, rj)
        if match is None:
            # Say which atoms were missing, if any. A pair rejected for a genuine
            # geometric mismatch and one rejected because the file lacks the
            # atoms are different problems and must not look the same.
            absent = []
            for table in PAIR_EDGES.values():
                for (ba, bb) in table.get((ri.name, rj.name), ()):
                    if ba not in ri.base_atoms:
                        absent.append(f"{ri.name}.{ba}")
                    if bb not in rj.base_atoms:
                        absent.append(f"{rj.name}.{bb}")
            note = f" (missing {sorted(set(absent))})" if absent else ""
            reject("no base-pair edge satisfied" + note)
            continue
        pair_type, hbond_dev = match
        if pair_type not in allowed_types:
            reject(f"{pair_type} edge, not in allowed_types")
            continue

        pair_type = _classify(ri.name, rj.name)
        pairs.append(
            BasePair(
                key=ri.index, partner=rj.index,
                base_key=ri.name, base_partner=rj.name,
                c1_dist=c1_dist, glyco_angle_deg=glyco_angle,
                plane_angle_deg=plane_angle, pair_type=pair_type,
                normal_dihedral_deg=dihedral,
                key_chain=ri.chain, partner_chain=rj.chain,
                hbond_dev=round(hbond_dev, 3),
            )
        )

    pairs.sort(key=lambda p: (p.key_chain, p.key, p.partner_chain, p.partner))
    return PairGraph(
        residues=residues, pairs=pairs, length=length,
        is_circular=is_circular, bsj_index=bsj_index,
        rejected=rejected,
        params={
            "chains": list(chain_ids),
            "max_c1_dist": max_c1_dist,
            "min_c1_dist": min_c1_dist,
            "max_glyco_angle": max_glyco_angle,
            "max_plane_angle": max_plane_angle,
            "allowed_types": list(allowed_types),
            "hbond_tol": HBOND_TOL,
        },
    )


# ---------------------------------------------------------------------------
# synthetic ground truth


def ideal_aform_helix(n_pairs: int = 12, *, rise: float = 2.81,
                      twist_deg: float = 32.7) -> str:
    """Build an idealised A-form RNA duplex as PDB text.

    DO NOT USE THIS AS A VALIDATION ANCHOR. It cannot be one, and the reason is
    structural rather than a matter of tuning. A one-parameter helix places both
    strands' C1' atoms on a single cylinder at 180 degrees phase, so the paired
    C1'--C1' distance is fixed by the radius and the stacked distance by the
    radius plus the rise. Real A-form needs those two to be ~10.4 A and ~6.2 A
    respectively, and no single radius delivers both: r = 5.2 gives 10.4 A paired
    but 3.5 A stacked, r = 6.5 gives 6.2 A stacked but 13.0 A paired. A real
    duplex gets out of this because the glycosidic bonds are not radial, which is
    a second free parameter this generator does not have.

    Consequently the detector reports no pairs on this output, and the unit tests
    assert exactly that. It exists to exercise the parser, the column layout and
    the helix-run walker -- not the pair criterion. Anchor validation on 1QC0
    instead; see tools/validate_pair_graph.py.

    Only C1', the glycosidic nitrogen and a ring of base atoms are emitted, since
    those are all the pair detector reads. The ring is a regular polygon in the
    fitted plane. This is a synthetic fixture, not a stereochemically valid
    model, and must never be used as anything else.
    """
    c1_radius = 5.2          # 2 * 5.2 * sin(32.7/2 deg) = 2.93 A in-plane chord;
                             # with the 2.81 A rise the paired distance
                             # sqrt(2.93^2 + 2.81^2) = 4.06 A, which is NOT the
                             # ~10.4 A a real Watson-Crick pair shows.
    base_offset = 2.0
    one2three = {"A": "A", "C": "C", "G": "G", "U": "U"}
    s1 = ("GCGCGCGCGCGC" * 8)[:n_pairs]
    comp = {"G": "C", "C": "G", "A": "U", "U": "A"}
    s2 = "".join(comp[b] for b in s1)

    lines: List[str] = []
    serial = 1

    def emit(chain: str, resseq: int, resname: str, atom: str,
             pos: np.ndarray) -> None:
        # Build the record by explicit column placement instead of guessing a
        # format string. This generator was wrong here five times, every time by
        # one column, and the symptom was always the same: a blank chain ID, both
        # strands merged, "no pairs found". Laying the fields out on a fixed-width
        # canvas makes the column numbers the actual code rather than a comment
        # that can drift away from them.
        #
        # PDB columns, 1-based: record 1-6, serial 7-11, atom name 13-16,
        # altLoc 17, resName 18-20, chainID 22, resSeq 23-26, x 31-38,
        # y 39-46, z 47-54, occupancy 55-60, B 61-66, element 77-78.
        nonlocal serial
        buf = [" "] * 80

        def put(start: int, text: str) -> None:
            for i, ch in enumerate(text):
                idx = start - 1 + i
                if 0 <= idx < len(buf):
                    buf[idx] = ch

        put(1, "ATOM")
        put(7, f"{serial:>5d}")
        put(13, f"{atom:^4s}" if len(atom) < 4 else atom[:4])
        put(18, resname)
        put(22, chain)
        put(23, f"{resseq:>4d}")
        for start, value in ((31, pos[0]), (39, pos[1]), (47, pos[2])):
            put(start, f"{value:>8.3f}")
        put(55, f"{1.0:>6.2f}")
        put(61, f"{0.0:>6.2f}")
        put(77, atom[0] if atom[0].isalpha() else "C")
        lines.append("".join(buf))
        serial += 1

    for k in range(n_pairs):
        theta = math.radians(twist_deg) * k
        z = rise * k
        # positions of the two C1' atoms in this base-pair plane
        c1_positions = []
        for strand in (0, 1):
            phi = theta + (0.0 if strand == 0 else math.pi)
            c1_positions.append(
                np.array([c1_radius * math.cos(phi),
                          c1_radius * math.sin(phi), z], dtype=float)
            )

        for strand in (0, 1):
            resname = s1[k] if strand == 0 else s2[k]
            c1 = c1_positions[strand]
            partner = c1_positions[1 - strand]
            chain = "A" if strand == 0 else "B"
            # Strand 2 is numbered above strand 1 so the two strands never share
            # a residue index. Residues are keyed by index alone, so overlapping
            # numbering silently drops one strand -- measured: a 12 bp fixture
            # parsed as 12 residues instead of 24.
            resseq = k + 1 if strand == 0 else n_pairs + k + 1

            # Glycosidic bond direction. In a real A-form duplex the two
            # glycosidic bonds are PARALLEL (3-7 deg measured on 1QC0), not
            # pointing at each other -- pointing them at the partner gives an
            # antiparallel arrangement the detector correctly refuses. Both
            # bases' C1'->N vectors therefore run the same way around the helix,
            # with the base centre displaced along that same tangential sense.
            radial = np.array([math.cos(phi), math.sin(phi), 0.0], dtype=float)
            tangent = np.array([-math.sin(phi), math.cos(phi), 0.0], dtype=float)
            glyco_dir = _unit(tangent)
            # A-form bases lie in planes perpendicular to the helix axis
            normal = np.array([0.0, 0.0, 1.0], dtype=float)

            assert glyco_dir is not None
            emit(chain, resseq, one2three[resname], "C1'", c1)
            centre = c1 + base_offset * glyco_dir
            perp = _unit(np.cross(normal, glyco_dir))
            assert perp is not None
            for r in range(6):          # ring polygon in the base plane
                ang = 2.0 * math.pi * r / 6.0
                pos = centre + 1.4 * (
                    math.cos(ang) * glyco_dir + math.sin(ang) * perp
                )
                emit(chain, resseq, one2three[resname], f"C{r + 2}", pos)
            emit(chain, resseq, one2three[resname],
                 _GLYCOSIDIC_N[resname], centre + 1.5 * glyco_dir)

    return "\n".join(lines) + "\nEND\n"
