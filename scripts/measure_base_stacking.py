"""Base-level stacking, measured -- the one day of measurement before spending weeks on a new field.

THE QUESTION. docs/NOTES.md and Part 12 of docs/ibi_loop_and_oxrna_findings.md establish that stacking in
this architecture is derived three times over (K_STACK = 0; the scored stack coordinate is the algebraic
P(i)-P(i+2) distance; and reconstruct_all_atom places every base from a 1EHZ template plus an axis
heuristic, so the CG base beads never reach the product). The open question is what that costs, and where
the loss lives:

    (A) the TEMPLATE: reconstruction loses stacking even when the trace is perfect;
    (B) the TRACE: reconstruction loses stacking as the trace moves away from the crystal;
    (C) the refinement: how much of the crystal's stacking survives our own CG refinement.

WHAT IS MEASURED, and why these observables. Two bases stack when their rings are parallel and one sits
over the other, so for a base pair of planes the natural coordinates are: the centroid separation
|dc|, the angle between the normals theta, the RISE (the separation along the mean normal) and the TWIST
(the in-plane rotation between the two glycosidic directions). A "helical step" is a WC pair (i, j) whose
partner pair (i+1, j-1) is ALSO paired -- that is the definition that matters, because stacking is only
expected where a helix continues; sequence-adjacent bases across a bulge are not supposed to stack. A
step counts as STACKED when |dc| <= 4.5 A and theta <= 30 deg.

WHY GEOMETRIC PAIR DETECTION rather than a pair list from the loader: the same code then applies to the
crystal, to a reconstruction and to a CG product, with no second source of truth about which bases pair.
The WC atom triples used are the standard ones (A N1-U N3 / A N6-U O4, G N1-C N3 / G O6-C N4 / G N2-C O2)
with a 3.6 A cutoff on all of a pair's key contacts.

MODES
    --crystal  <pdb> [<pdb> ...]   measure the deposited structure itself (the reference)
    --fragments  N                 N fragments from _cgdata/combined, each measured as crystal,
                                   as reconstruction at trace RMSD 0, and under trace noise
    --reconstruct <pdb>            crystal trace -> reconstruct -> measure (isolates (A))
    --noise 0.5,1.0,1.5,2.0,3.0    trace noise in Angstrom -> reconstruct -> measure (gives (B))
    --trace <pdb> [...]            P-only CG products -> reconstruct -> measure (gives (C))
"""
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from torusfold.scheme2.aform_from_template import (                       # noqa: E402
    reconstruct_all_atom, reconstruct_all_atom_from_beads)

# Ring atoms that define each base plane. Purines: the fused bicyclic (9 atoms, planar). Pyrimidines:
# the six-membered ring. Four atoms is the minimum a plane fit needs; a residue missing more is skipped
# and counted, so a poor number can never be an artefact of silent skipping.
BASE_ATOMS = {
    "A": ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"),
    "G": ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4"),
    "C": ("N1", "C2", "N3", "C4", "C5", "C6"),
    "U": ("N1", "C2", "N3", "C4", "C5", "C6"),
}
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}
RESNAME2BASE = {
    "A": "A", "ADE": "A", "RA": "A", "DA": "A",  # RA/RU/RG/RC = Amber RNA names
    "G": "G", "GUA": "G", "RG": "G", "DG": "G",
    "C": "C", "CYT": "C", "RC": "C", "DC": "C",
    "U": "U", "URA": "U", "RU": "U", "DT": "U", "T": "U",
}
WC_TRIPLES = {
    ("A", "U"): (("N1", "N3"), ("N6", "O4")),
    ("G", "C"): (("N1", "N3"), ("O6", "N4"), ("N2", "O2")),
}


def base_of(resname):
    n = resname.strip().upper()
    if n in RESNAME2BASE:
        return RESNAME2BASE[n]
    if n.startswith("R") and len(n) == 2:
        return {"A": "A", "U": "U", "G": "G", "C": "C"}.get(n[1])
    return None


def parse_pdb(path, chain=None, min_res=10):
    """(sequence, [atom dict per residue], [P per residue]) in Angstrom, for one RNA chain."""
    chains = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            alt = line[16]
            if alt not in (" ", "A"):
                continue
            name = line[12:16].strip()
            resn = line[17:20].strip()
            ch = line[21]
            try:
                seqno = int(line[22:26])
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except ValueError:
                continue
            b = base_of(resn)
            if b is None:
                continue
            chains.setdefault(ch, {})
            key = (seqno, line[26])
            chains[ch].setdefault(key, {"base": b, "atoms": {}})
            chains[ch][key]["atoms"][name] = xyz
    if chain is not None:
        order = [chain] if chain in chains else []
    else:
        order = sorted(chains, key=lambda c: -len(chains[c]))
    for ch in order:
        res = [chains[ch][k] for k in sorted(chains[ch])]
        if len(res) >= min_res:
            seq = "".join(r["base"] for r in res)
            p = np.array([r["atoms"].get("P", np.full(3, np.nan)) for r in res])
            return seq, res, p, ch
    return None, None, None, None


def plane(res):
    """(centroid, unit normal, in-plane glycosidic direction) of one base, or None."""
    atoms = res["atoms"]
    names = BASE_ATOMS[res["base"]]
    pts = np.array([atoms[n] for n in names if n in atoms])
    if len(pts) < 4:
        return None
    c = pts.mean(0)
    _u, _s, vt = np.linalg.svd(pts - c)
    n = vt[2]
    gly = atoms.get(GLY[res["base"]])
    if gly is None:
        return None
    v = gly - c
    v = v - n * float(v @ n)
    nv = np.linalg.norm(v)
    if nv < 1e-6:
        return None
    return c, n / np.linalg.norm(n), v / nv


def wc_pairs(seq, residues, planes, cutoff=3.6, min_sep=3):
    """Geometric WC pairs, greedy: candidates ranked by their key-contact distance sum, and a base is
    allowed only ONE partner. Without the one-partner rule a helix's ends pick up cross pairs
    ((3,47) and (4,47) both surviving), which then makes the helical-step test below ambiguous."""
    cands = []
    L = len(seq)
    for i in range(L):
        for j in range(i + min_sep, L):
            trip = WC_TRIPLES.get((seq[i], seq[j])) or WC_TRIPLES.get((seq[j], seq[i]))
            if trip is None:
                continue
            ai, aj = residues[i]["atoms"], residues[j]["atoms"]
            ok = True
            keys = []
            for a, b in trip:
                if seq[i] + seq[j] in WC_TRIPLES:
                    keys.append((a, b))
                else:
                    keys.append((b, a))
            for a, b in keys:
                if a not in ai or b not in aj:
                    ok = False
                    break
                if float(np.linalg.norm(ai[a] - aj[b])) > cutoff:
                    ok = False
                    break
            if ok and planes[i] and planes[j]:
                d = float(sum(np.linalg.norm(ai[a] - aj[b]) for a, b in keys)) / len(keys)
                cands.append((d, i, j))
    cands.sort()
    taken = set()
    out = []
    for _d, i, j in cands:
        if i in taken or j in taken:
            continue
        taken.add(i)
        taken.add(j)
        out.append((i, j))
    return sorted(out)


def measure(seq, residues, pairs, planes):
    """Rise, twist, separation and normal angle over every helical step."""
    pset = set(pairs)
    rows = []
    skipped = 0
    # A HELICAL STEP is a pair whose NEXT pair along the helix also exists: (i, j) and (i+1, j-1).
    # Stacking is measured between the two bases that continue the helix -- i with i+1, and j with j-1.
    for (i, j) in pairs:
        if (i + 1, j - 1) not in pset:
            continue
        for (a, b) in ((i, i + 1), (j - 1, j)):
            if a < 0 or b < 0 or a >= len(seq) or b >= len(seq):
                continue
            if planes[a] is None or planes[b] is None:
                skipped += 1
                continue
            ca, na, va = planes[a]
            cb, nb, vb = planes[b]
            if a > b:
                ca, na, va, cb, nb, vb = cb, nb, vb, ca, na, va
            # THE NORMAL'S SIGN IS ARBITRARY (it comes out of an SVD), so a perfectly stacked pair can
            # read as theta = 180 degrees and a degenerate mean normal. Measured on 2OIU before this
            # line existed: 33 percent of the helical steps "stacked", several of the failures at
            # theta 174-179 deg with rise 0.15 A -- i.e. the criterion was reading the sign, not the
            # geometry. Align b's normal to a's first, then everything downstream is sign-free.
            if float(na @ nb) < 0.0:
                nb = -nb
            nb_m = na + nb
            nn = np.linalg.norm(nb_m)
            if nn < 1e-6:
                skipped += 1
                continue
            nb_m = nb_m / nn
            dc = cb - ca
            rise = abs(float(dc @ nb_m))
            sep = float(np.linalg.norm(dc))
            theta = math.degrees(math.acos(max(-1.0, min(1.0, float(na @ nb)))))
            w = vb - nb_m * float(vb @ nb_m)
            if np.linalg.norm(w) < 1e-6:
                skipped += 1
                continue
            w = w / np.linalg.norm(w)
            # twist = the in-plane rotation between the two glycosidic directions, taken from the
            # signed angle about the mean normal so its sign convention is the structure's, not SVD's.
            twist = abs(math.degrees(math.atan2(float(np.cross(va, w) @ nb_m),
                                                max(-1.0, min(1.0, float(va @ w))))))
            # STACKED is judged on the rise and the normal angle, not on the centroid separation: in
            # A-form RNA the centroids sit 4.2-4.8 A apart along a helix whose rise is 3.4 A, so a
            # separation cutoff rejects perfectly stacked steps (measured: 2OIU's crystal scored 50
            # percent with a 4.5 A cutoff, on steps whose mean rise is 3.40 +- 0.13 A).
            rows.append({"i": a, "j": b, "sep": sep, "rise": rise, "theta": theta, "twist": twist,
                         "stacked": (2.5 <= rise <= 4.0 and theta <= 30.0)})
    return rows, skipped


def perturb(p, sd, rng):
    return p + rng.normal(0.0, sd, size=p.shape)


def kabsch_rmsd(a, b):
    """RMSD of a onto b after optimal rotation (both (L,3), matched row by row)."""
    ca, cb = a.mean(0), b.mean(0)
    A, Bm = a - ca, b - cb
    u, _s, vt = np.linalg.svd(Bm.T @ A)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return float(np.sqrt(((Bm @ r - A) ** 2).sum(1).mean()))


def measure_allatom(seq, residues):
    """(rows, skipped, n_pairs) for a set of residues with their own geometry."""
    planes = [plane(r) for r in residues]
    pairs = wc_pairs(seq, residues, planes)
    rows, skipped = measure(seq, residues, pairs, planes)
    return rows, skipped, len(pairs)


def measure_reconstruction(seq, p_ang, pairs, tag):
    """Reconstruct from a P trace (Angstrom) and measure the result. pairs are 0-based (i, j).

    Returns (rows, skipped, n_wc_contacts_kept); rows is empty when the structure could not be
    reconstructed at all, and the reason is PRINTED rather than dropped -- a P trace with a missing atom
    makes the anchor SVD non-convergent, and a caller that silently skipped those would report a stacking
    fraction over a biased subset of the database.
    """
    if not np.all(np.isfinite(p_ang)):
        print("  %-34s skipped: trace has a non-finite P coordinate (missing atom in the deposit)" % tag)
        return [], 0, 0
    try:
        st = reconstruct_all_atom(p_ang, seq, pairs=[(i, j) for (i, j) in pairs])
    except Exception as exc:                                   # noqa: BLE001 - reported, not hidden
        print("  %-34s skipped: reconstruction failed: %s" % (tag, exc))
        return [], 0, 0
    L = len(seq)
    residues = []
    for i in range(L):
        idx = st.residue_atom_index[i]
        atoms = {}
        for nm, atom in idx.items():
            atoms[nm] = np.asarray(st.atoms[atom].xyz, dtype=float)
        residues.append({"base": seq[i], "atoms": atoms})
    # THE STEP LIST IS THE CRYSTAL'S, NOT THE RECONSTRUCTION'S. Measuring the reconstruction's own
    # geometrically detected pairs (which is the natural thing to write) compares a varying, biased
    # subset: on 4QK9 the reconstruction keeps only 2 of the crystal's 22 WC contacts within the 3.6 A
    # key-contact cutoff, so a re-detected comparison scores stacking on whichever 2 steps survived a
    # different failure. Both numbers are reported; only the crystal-pair one is a stacking result, the
    # other is a pairing-geometry result in its own right.
    planes = [plane(r) for r in residues]
    detected = wc_pairs(seq, residues, planes)
    rows, skipped = measure(seq, residues, pairs, planes)
    return rows, skipped, len(detected)


def structure_to_residues(st, seq):
    """An AllAtomStructure -> the residue list this module measures on."""
    out = []
    for i in range(len(seq)):
        idx = st.residue_atom_index[i]
        out.append({"base": seq[i],
                    "atoms": {nm: np.asarray(st.atoms[a].xyz, dtype=float)
                              for nm, a in idx.items()}})
    return out


def crystal_beads(seq, residues):
    """(L, 3, 3) Angstrom: the CG model's own beads (P, C4', N9/N1) read off a crystal."""
    out = np.zeros((len(seq), 3, 3), dtype=float)
    for i, base in enumerate(seq):
        atoms = residues[i]["atoms"]
        gly = GLY[base]
        for k, nm in enumerate(("P", "C4'", gly)):
            if nm not in atoms:
                return None
            out[i, k] = atoms[nm]
    return out


def trace_frames(p):
    """Per-residue (tangent, in-plane, normal) frame of a P trace, wrapped for a circle."""
    L = len(p)
    out = []
    for i in range(L):
        nxt, prv = p[(i + 1) % L], p[(i - 1) % L]
        b = nxt - prv
        nb = np.linalg.norm(b)
        b = b / nb if nb > 1e-9 else np.array([1.0, 0.0, 0.0])
        n = np.cross(nxt - p[i], prv - p[i])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            tmp = np.array([0.0, 0.0, 1.0]) if abs(b[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            n = np.cross(b, tmp)
            nn = np.linalg.norm(n)
        n = n / nn
        r = np.cross(n, b)
        out.append((b, r, n / np.linalg.norm(n) if False else n))
    return out


def beads_via_frames(p_new, beads_ref, p_ref):
    """Carry each residue's rigid unit from p_ref to p_new, keeping its roll in the LOCAL trace frame.

    THIS IS AN EMULATION, and the honest label matters. The CG solver's sampled beads are not available
    from a round (the loop writes histograms, and torch_gpu_refine returns only P), so the question "what
    would a base-frame reconstruction recover" is asked by moving the crystal's rigid units exactly the
    way the model's own constraints would: intra-residue geometry is rigid (K_INTRA_PN plus the K_LINK_*
    links, SHAKE/RATTLE), so a sampled bead set is the crystal's unit carried by the backbone frame, and
    the only thing a trace perturbation changes is that frame.
    """
    fr_new, fr_ref = trace_frames(p_new), trace_frames(p_ref)
    out = np.zeros_like(beads_ref)
    for i in range(len(p_new)):
        b0, r0, n0 = fr_ref[i]
        b1, r1, n1 = fr_new[i]
        for k in range(3):
            d = beads_ref[i, k] - p_ref[i]
            comp = np.array([float(d @ b0), float(d @ r0), float(d @ n0)])
            out[i, k] = p_new[i] + comp[0] * b1 + comp[1] * r1 + comp[2] * n1
    return out


def measure_bead_reconstruction(beads, seq, pairs, tag):
    """Reconstruct from the SAMPLED beads (the new path) and measure with the crystal's step list."""
    if beads is None or not np.all(np.isfinite(beads)):
        print("  %-34s skipped: beads missing" % tag)
        return [], 0, 0
    try:
        st = reconstruct_all_atom_from_beads(beads, seq)
    except Exception as exc:                                   # noqa: BLE001
        print("  %-34s skipped: %s" % (tag, exc))
        return [], 0, 0
    residues = structure_to_residues(st, seq)
    planes = [plane(r) for r in residues]
    detected = wc_pairs(seq, residues, planes)
    rows, skipped = measure(seq, residues, pairs, planes)
    return rows, skipped, len(detected)


def summarize(rows, n_pairs, skipped, tag, extra=""):
    if not rows:
        print("  %-34s no helical steps" % tag)
        return {}
    arr = {k: np.array([r[k] for r in rows]) for k in ("sep", "rise", "theta", "twist")}
    st = np.array([r["stacked"] for r in rows])
    out = {"n": len(rows), "stacked_frac": float(st.mean()), "sep": float(arr["sep"].mean()),
           "rise": float(arr["rise"].mean()), "rise_sd": float(arr["rise"].std()),
           "theta": float(arr["theta"].mean()), "twist": float(arr["twist"].mean()),
           "twist_sd": float(arr["twist"].std())}
    print("  %-34s steps %4d  stacked %5.1f%%  sep %5.2f A  rise %5.2f+-%.2f A  "
          "theta %5.1f deg  twist %5.1f+-%.1f deg %s"
          % (tag, out["n"], 100 * out["stacked_frac"], out["sep"], out["rise"], out["rise_sd"],
             out["theta"], out["twist"], out["twist_sd"], extra))
    return out


def do_one_crystal(path, pairs_only=False):
    seq, residues, p, ch = parse_pdb(path)
    if seq is None:
        print("  %-34s skipped (no RNA chain)" % Path(path).name)
        return None
    rows, skipped, n_pairs = measure_allatom(seq, residues)
    extra = "(L=%d, chain %s, %d WC pairs, %d skipped)" % (len(seq), ch, n_pairs, skipped)
    res = summarize(rows, n_pairs, skipped, "crystal " + Path(path).name, extra)
    if pairs_only or res is None:
        return res
    return {"seq": seq, "p": p, "pairs": wc_pairs(seq, residues, [plane(r) for r in residues]),
            "summary": res}


def main():
    args = sys.argv[1:]
    mode = args[0] if args else "--fragments"
    rng = np.random.default_rng(20261005)

    if mode == "--crystal":
        for path in args[1:]:
            do_one_crystal(path)
        return

    if mode == "--reconstruct":
        for path in args[1:]:
            info = do_one_crystal(path, pairs_only=True)
            seq, residues, p, ch = parse_pdb(path)
            planes = [plane(r) for r in residues]
            pairs = wc_pairs(seq, residues, planes)
            rows, skipped, _det = measure_reconstruction(seq, p, pairs, "recon")
            summarize(rows, len(pairs), skipped, "recon(trace=exact) " + Path(path).name,
                      "(skipped %d, %d/%d WC contacts kept)" % (skipped, det, len(pairs)))
        return

    if mode == "--noise":
        sds = [float(x) for x in args[1].split(",")]
        path = args[2] if len(args) > 2 else str(REPO / "artifacts" / "2oiu" / "2OIU.pdb")
        seq, residues, p, ch = parse_pdb(path)
        planes = [plane(r) for r in residues]
        pairs = wc_pairs(seq, residues, planes)
        print("%s: L=%d, %d WC pairs" % (Path(path).name, len(seq), len(pairs)))
        summarize(measure(seq, residues, pairs, planes)[0], len(pairs), 0, "crystal (reference)")
        rows, skipped, det = measure_reconstruction(seq, p, pairs, "recon")
        summarize(rows, len(pairs), skipped, "recon at trace RMSD 0.000 A",
                  "(%d/%d WC contacts kept)" % (det, len(pairs)))
        for sd in sds:
            pp = perturb(p, sd, rng)
            rms = kabsch_rmsd(pp, p)
            rows, skipped, det = measure_reconstruction(seq, pp, pairs, "recon")
            summarize(rows, len(pairs), skipped, "recon at trace RMSD %.3f A" % rms,
                      "(noise sd %.2f A, %d/%d WC contacts kept)" % (sd, det, len(pairs)))
        return

    if mode == "--trace":
        for path in args[1:]:
            seq, residues, p, ch = parse_pdb(path)
            if seq is None:
                # a P-only CG product: sequence comes from the residue NAMES, coords from P
                seq, residues, p, ch = parse_pdb(path, min_res=5)
            if seq is None:
                print("  %-34s skipped" % Path(path).name)
                continue
            planes = [plane(r) for r in residues]
            pairs = wc_pairs(seq, residues, planes)
            rows, skipped, _det = measure_reconstruction(seq, p, pairs, "recon")
            summarize(rows, len(pairs), skipped, "product " + Path(path).name,
                      "(L=%d, %d pairs, %d WC contacts kept)" % (len(seq), len(pairs), det))
        return

    if mode == "--beads":
        # The upper bound of this path: the crystal's OWN rigid units, so the fit has the right roll.
        for path in args[1:]:
            seq, residues, p, ch = parse_pdb(path)
            planes = [plane(r) for r in residues]
            pairs = wc_pairs(seq, residues, planes)
            beads = crystal_beads(seq, residues)
            rows, skipped, det = measure_bead_reconstruction(beads, seq, pairs,
                                                             "beads " + Path(path).name)
            summarize(rows, len(pairs), skipped, "beads(exact) " + Path(path).name,
                      "(%d/%d WC contacts kept)" % (det, len(pairs)))
        return

    if mode == "--beads-noise":
        # Same ladder as --noise, but the rigid units are carried by the perturbed frame instead of the
        # P trace being re-guessed by the axis heuristic. Same structure, same pairs, same criterion.
        sds = [float(x) for x in args[1].split(",")]
        path = args[2] if len(args) > 2 else str(REPO / "artifacts" / "2oiu" / "2OIU.pdb")
        seq, residues, p, ch = parse_pdb(path)
        planes = [plane(r) for r in residues]
        pairs = wc_pairs(seq, residues, planes)
        beads0 = crystal_beads(seq, residues)
        print("%s: L=%d, %d WC pairs" % (Path(path).name, len(seq), len(pairs)))
        summarize(measure(seq, residues, pairs, planes)[0], len(pairs), 0, "crystal (reference)")
        rows, skipped, det = measure_bead_reconstruction(beads0, seq, pairs, "beads")
        summarize(rows, len(pairs), skipped, "beads at trace RMSD 0.000 A",
                  "(%d/%d WC contacts kept)" % (det, len(pairs)))
        for sd in sds:
            pp = perturb(p, sd, rng)
            rms = kabsch_rmsd(pp, p)
            beads = beads_via_frames(pp, beads0, p)
            rows, skipped, det = measure_bead_reconstruction(beads, seq, pairs, "beads")
            summarize(rows, len(pairs), skipped, "beads at trace RMSD %.3f A" % rms,
                      "(noise sd %.2f A, %d/%d WC contacts kept)" % (sd, det, len(pairs)))
        return

    if mode == "--products":
        # --products <crystal.pdb> <product.pdb> [product.pdb ...]
        # The crystal supplies the sequence, the pair list AND the trace the RMSD is measured against,
        # so a product's distance from the deposit and the stacking it reconstructs to are reported
        # together -- the noise sweep above is the curve that connects them.
        cpath = args[1]
        seq, residues, p_ref, ch = parse_pdb(cpath)
        planes = [plane(r) for r in residues]
        pairs = wc_pairs(seq, residues, planes)
        print("crystal %s: L=%d, %d WC pairs, %.1f%% stacked"
              % (Path(cpath).name, len(seq), len(pairs),
                 100 * np.mean([r["stacked"] for r in measure(seq, residues, pairs, planes)[0]])))
        for path in args[2:]:
            seq2, residues2, p, ch2 = parse_pdb(path, min_res=5)
            if seq2 is None or len(seq2) != len(seq):
                print("  %-40s skipped (L=%s)" % (Path(path).name, None if seq2 is None else len(seq2)))
                continue
            rms = kabsch_rmsd(p, p_ref)
            rows, skipped, det = measure_reconstruction(seq, p, pairs, "recon")
            summarize(rows, len(pairs), skipped, "%-30s" % ("%s rms %.2f A" % (Path(path).stem, rms)),
                      "(%d/%d WC contacts kept)" % (det, len(pairs)))
        return

    if mode == "--fragments":
        n = int(args[1]) if len(args) > 1 else 40
        files = sorted((REPO / "_cgdata" / "combined").glob("*.pdb"))
        print("fragments: %d files, measuring the first %d with an RNA chain of <= 300 nt" % (len(files), n))
        done = 0
        tot = {"crystal": [], "recon": [], "beads": []}
        for f in files:
            if done >= n:
                break
            seq, residues, p, ch = parse_pdb(f)
            if seq is None or len(seq) > 300:
                continue
            planes = [plane(r) for r in residues]
            pairs = wc_pairs(seq, residues, planes)
            if len(pairs) < 4:
                continue
            rows_c, _ = measure(seq, residues, pairs, planes)
            rows_r, skipped, det = measure_reconstruction(seq, p, pairs, "recon")
            beads = crystal_beads(seq, residues)
            rows_b, _skb, detb = measure_bead_reconstruction(beads, seq, pairs, "beads")
            if not rows_c or not rows_r or not rows_b:
                continue
            sc = summarize(rows_c, len(pairs), 0, "crystal " + f.stem)
            sr = summarize(rows_r, len(pairs), skipped, "recon   " + f.stem,
                           "(skipped %d, %d/%d WC contacts kept)" % (skipped, det, len(pairs)))
            sb = summarize(rows_b, len(pairs), 0, "beads   " + f.stem,
                           "(%d/%d WC contacts kept)" % (detb, len(pairs)))
            if sc and sr and sb:
                tot["crystal"].append(sc)
                tot["recon"].append(sr)
                tot["beads"].append(sb)
                done += 1
        if tot["crystal"]:
            for key in ("stacked_frac", "sep", "rise", "theta", "twist"):
                a = np.array([d[key] for d in tot["crystal"]])
                b = np.array([d[key] for d in tot["recon"]])
                c = np.array([d[key] for d in tot["beads"]])
                print("MEAN %-12s crystal %8.3f +- %-6.3f   P-trace %8.3f +- %-6.3f   "
                      "bead-frame %8.3f +- %-6.3f   n=%d"
                      % (key, a.mean(), a.std(), b.mean(), b.std(), c.mean(), c.std(), len(a)))
        return

    raise SystemExit("unknown mode " + mode)


if __name__ == "__main__":
    main()
