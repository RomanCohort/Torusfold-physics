"""Does the RCM pair weight identify the real Watson-Crick pairs?

The question. src/torusfold/scheme2/isrnaclong.py:773 replaces the restraint weights that
lines 738-760 built from pairing-method agreement (1.0 / 0.8 / 0.6) with
compute_rcm_score(upstream_flank, downstream_flank)['confidence']. That confidence is
crossing / (crossing + within_up + within_down), a ratio of reverse-complement kmer match
counts. scripts/test_sequence_dependence.py measured that zeroing pair_w moves only two of
twelve energy terms and leaves the other ten bit-identical with zero force components
moving, so this one number is the entire sequence channel of the 3-bead force field. This
script asks whether it carries information about the pairs it is attached to, and whether
an obvious cheap alternative does better.

Data. The same 191 PDB files the bonded tables are fitted on, loaded through
boltzmann_bonded._chain_residues(..., with_names=True) so the residue selection, the
Watson-Crick set (B.WCP, which includes G-U) and the C1'-C1' 9.0-11.5 A pair criterion are
the project's, not a reimplementation. C1'-C1' distances are also needed for candidate
pairs _chain_residues never returns, because it keeps only P/C4'/glycan in the bead array,
so c1_coords() re-parses the deposited file. It is not trusted: for every chain the script
asserts that c1_coords() reproduces _chain_residues()'s bead array, base letters AND
accepted pair set exactly, and raises otherwise. The pair criterion is therefore verified
equal to the project's by construction on all 126 chains.

Positives. Every pair _chain_residues accepts: Watson-Crick compatible, at least 3 residues
apart, C1'-C1' inside the band.

Negatives, two matched sets, both keeping the >=3 residue separation and neither one
trivially easy:
  N1 "band-matched": pairs in the same 9.0-11.5 A band whose base pair is NOT
     Watson-Crick compatible, separation matched to its anchor within --tol. Geometry is
     held fixed and the question is whether the flank sequence content recovers the
     chemistry the base-identity test rejects. Base complementarity alone scores 1.0 here
     BY CONSTRUCTION, because the positives are exactly the WCP in-band pairs; it is
     reported as a ceiling reference, not as a competing score.
  N2 "separation-matched": pairs with the same >=3 separation matched within --tol, not
     themselves accepted pairs, with any base identity and any distance. Here base
     complementarity is a real predictor and far from perfect, because most
     WCP-compatible candidate pairs in a chain are nowhere near each other in 3D. This is
     the set on which "is the existing score worse than base complementarity alone" has a
     non-degenerate answer.

Control. Every score is recomputed with the chain's letters uniformly permuted, labels
left as the real structure defines them. AUC(real) - AUC(shuffled) is the share of the
separation that is sequence-specific rather than positional/geometric.

Scores compared (all oriented so larger = more pair-like):
  rcm_confidence     the score the pipeline installs (compute_rcm_score)
  rcm_crossing       its numerator alone (crossing_total)
  rcm_density        crossing matches per crossing comparison (rcm_density_score)
  rcm_confidence_T   the same confidence after rewriting U as T, which is the alphabet
                     rcm.py's _COMPLEMENT table (A->T) is consistent with; see score_pair()
  method_agree       the weights the pipeline actually installs when the RCM reweight is off:
                     isrnaclong.py:641-763's tiers, rebuilt on this chain's own letters with the
                     pipeline's own ViennaRNA settings. Hard (>=2 of {PF high, MFE}) is 1.0, MFE
                     soft is 0.8, PF medium is its own probability p, and anything not predicted
                     is 0.0. DivideFold is unavailable here, so the 0.6 tier has no members --
                     which is the state the pipeline itself is in when that subprocess fails.
                     This is the score the comment at isrnaclong.py:780 says had NOT been
                     measured as discriminative.
  wc_identity        base complementarity alone, 1 if (base_i, base_j) in B.WCP
  wc_x_geom          complementarity plus geometry: wc_identity x gaussian in C1'-C1'
                     with centre 10.25 A and sigma = band width / 4
  geom_in_band       geometry alone, 1 if C1'-C1' is inside 9.0-11.5 A; a control whose
                     AUC on N1 must be exactly 0.5 (all rows in band) and whose N2 AUC is
                     the criterion's own diagnostic power
  flank_up           upstream flank length; a diagnostic for length confounds, not a
                     proposal

AUC is the Mann-Whitney rank statistic with midranks (ties worth half). 0.5 is chance.
Confidence intervals are 95 percent cluster bootstrap over chains, the clusters being the
126 chains rather than the pairs, because pairs inside one chain share a sequence. The
paired difference between two scores is bootstrapped on the same resampled chains, and p is
the share of replicates in which the difference is <= 0.

Run: python scripts/measure_pair_weight_quality.py [--files N] [--per-pos K] [--tol T]
                                                 [--shuffles R] [--boot B] [--seed S]
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import boltzmann_bonded as B                                        # noqa: E402
from torusfold.scheme2.rcm import compute_rcm_score, rcm_density_score   # noqa: E402

import _cgdata
DATA = _cgdata.rsrnasp()
WCP = B.WCP
BAND = (9.0, 11.5)
BAND_CENTRE = 0.5 * (BAND[0] + BAND[1])
BAND_SIGMA = (BAND[1] - BAND[0]) / 4.0
MIN_SEP = 3
FLANK_CAP = 200          # isrnaclong.py:765
FLANK_MIN = 5            # isrnaclong.py:771
SCORES = ("rcm_confidence", "rcm_crossing", "rcm_density", "rcm_confidence_T",
          "rcm_crossing_T", "method_agree", "ma_x_wc", "wc_identity", "wc_x_geom",
          "geom_in_band", "flank_up")

# ------------------------------------------------- method-agreement (isrnaclong.py:641-763)
_MA_CACHE = {}


def method_agreement_map(seq):
    """{(i, j): weight} for one sequence, by the pipeline's own tier rules.

    Built from ViennaRNA with md.circ = 1, which is the setting isrnaclong.py:643-645 uses, and
    with its tier boundaries (P > 0.9 high; 0.5 < P <= 0.9 medium; the MFE dot-bracket for the
    rest). The order matters and is the pipeline's: hard first, then MFE softs at 0.8, then PF
    mediums only for pairs neither of those already claimed -- isrnaclong.py:750-757 mutates
    hard_set as it appends, so a pair in both MFE and the PF medium tier keeps 0.8, not p.
    """
    cached = _MA_CACHE.get(seq)
    if cached is not None:
        return cached
    import RNA
    md = RNA.md()
    md.circ = 1
    fc = RNA.fold_compound(seq, md)
    _ss, _e = fc.pf()
    plist = fc.plist_from_probs(0.01)
    pf_high = {(ep.i - 1, ep.j - 1) for ep in plist if ep.p > 0.9}
    pf_mid = {(ep.i - 1, ep.j - 1): float(ep.p) for ep in plist if 0.5 < ep.p <= 0.9}
    try:
        ss_mfe, _me = fc.mfe()
        stack, mfe_pairs = [], set()
        for k, c in enumerate(ss_mfe):
            if c == "(":
                stack.append(k)
            elif c == ")" and stack:
                mfe_pairs.add((stack.pop(), k))
    except Exception:
        mfe_pairs = set(pf_high) | set(pf_mid)

    votes = collections.Counter()
    for src in (pf_high, mfe_pairs):
        for p in src:
            votes[p] += 1
    hard = {p for p, v in votes.items() if v >= 2} | set(pf_high)

    w = {}
    for p in hard:
        w[p] = 1.0
    seen = set(hard)
    for p in mfe_pairs:
        if p not in seen:
            w[p] = 0.8
            seen.add(p)
    for p, prob in pf_mid.items():
        if p not in seen:
            w[p] = prob
    _MA_CACHE[seq] = w
    return w


# --------------------------------------------------------------- chain loading
def c1_coords(pdb):
    """_chain_residues()'s chain selection, keeping C1' which that function discards.

    Same file filters, same four required atoms, same consecutive-numbering run split and
    same first-run-in-range rule as boltzmann_bonded._chain_residues. Returns
    [(beads_nm, pairs, names, c1_angstrom)]; load_chains() verifies it against the real
    loader, so this copy never silently diverges.
    """
    ch = collections.OrderedDict()
    for line in open(pdb):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        rname = line[17:20].strip()
        if rname not in ("A", "C", "G", "U"):
            continue
        aname = line[12:16].strip()
        gly = "N9" if rname in ("A", "G") else "N1"
        if aname not in ("P", "C4'", gly, "C1'"):
            continue
        try:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        rec = ch.setdefault(line[21], collections.OrderedDict())
        rec.setdefault((line[22:27].strip(), rname), {})[aname] = xyz

    out = []
    for rec in ch.values():
        kept = []
        for (rid, rname), at in rec.items():
            gly = "N9" if rname in ("A", "G") else "N1"
            if not all(k in at for k in ("P", "C4'", gly, "C1'")):
                continue
            kept.append((rid, rname, at))

        runs, cur, prev = [], None, None
        for rid, rname, at in kept:
            try:
                n = int(rid)
            except ValueError:
                n = None
            if cur is not None and n is not None and prev is not None and n == prev + 1:
                cur.append((rname, at))
            else:
                if cur:
                    runs.append(cur)
                cur = [(rname, at)]
            prev = n
        if cur:
            runs.append(cur)

        lst = None
        for run in runs:
            if B.MIN_L < len(run) <= B.MAX_L:
                lst = run
                break
        if lst is None:
            continue

        beads = np.array([[at["P"], at["C4'"], at["N9" if r in ("A", "G") else "N1"]]
                          for r, at in lst]) / 10.0
        c1 = np.array([at["C1'"] for _, at in lst], dtype=float)
        names = [r for r, _ in lst]

        pairs = []
        for a in range(len(lst)):
            for b in range(a + MIN_SEP, len(lst)):
                if (lst[a][0], lst[b][0]) not in WCP:
                    continue
                d = float(np.linalg.norm(np.array(lst[a][1]["C1'"])
                                         - np.array(lst[b][1]["C1'"])))
                if BAND[0] <= d <= BAND[1]:
                    pairs.append((a, b))
        out.append((beads, pairs, names, c1))
    return out


def load_chains(limit=None):
    """Chain records, each verified against boltzmann_bonded._chain_residues."""
    chains = []
    files = sorted(DATA.glob("*.pdb"))
    for f in files:
        mine = c1_coords(f)
        theirs = B._chain_residues(f, with_names=True)
        if len(mine) != len(theirs):
            raise RuntimeError(f"{f.name}: re-parse found {len(mine)} chains, loader "
                               f"found {len(theirs)}")
        for (beads, pairs, names, c1), (tb, tp, tn) in zip(mine, theirs):
            if list(names) != list(tn) or list(pairs) != list(tp) \
                    or not np.allclose(beads, tb, atol=1e-9):
                raise RuntimeError(f"{f.name}: C1' re-parse disagrees with "
                                   f"boltzmann_bonded._chain_residues")
            chains.append({"file": f.stem, "names": names, "beads": beads,
                           "pairs": pairs, "c1": c1})
        if limit is not None and len(chains) >= limit:
            break
    return chains


# ------------------------------------------------------------------ statistics
def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie), Mann-Whitney with midranks."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    mid = starts + (counts - 1) * 0.5 + 1.0
    r = mid[inv][:pos.size].sum()
    return float((r - pos.size * (pos.size + 1) * 0.5) / (pos.size * neg.size))


def auc_ci(labels, values, chain_ids, n_boot, rng):
    """95 percent cluster bootstrap CI over chains, plus the point estimate."""
    y = np.asarray(labels, dtype=bool)
    v = np.asarray(values, dtype=float)
    cid = np.asarray(chain_ids)
    groups = [np.where(cid == c)[0] for c in np.unique(cid)]
    point = auc(v[y], v[~y])
    d = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[p] for p in pick])
        yy, vv = y[idx], v[idx]
        if yy.any() and (~yy).any():
            d.append(auc(vv[yy], vv[~yy]))
    d = np.array(d)
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), d


def paired_boot(labels, a, b, chain_ids, n_boot, rng):
    """Bootstrap the AUC difference a - b on the same resampled chains."""
    y = np.asarray(labels, dtype=bool)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cid = np.asarray(chain_ids)
    groups = [np.where(cid == c)[0] for c in np.unique(cid)]
    point = auc(a[y], a[~y]) - auc(b[y], b[~y])
    diff = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[p] for p in pick])
        yy = y[idx]
        if yy.any() and (~yy).any():
            diff.append(auc(a[idx][yy], a[idx][~yy]) - auc(b[idx][yy], b[idx][~yy]))
    diff = np.array(diff)
    return point, float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)), \
        float((diff <= 0).mean())


def pct(v, qs=(5, 25, 50, 75, 95)):
    v = np.asarray(v, dtype=float)
    return " ".join(f"{np.percentile(v, q):.4g}" for q in qs)


# --------------------------------------------------------------- negative sets
def flank_ok(L, i, j):
    """isrnaclong.py:765-771's window rule and its >=5 nt guard, verbatim."""
    f = min(FLANK_CAP, L // 4)
    return (i - max(0, i - f)) >= FLANK_MIN and (min(L, j + f) - j) >= FLANK_MIN


def build_rows(chains, per_pos, tol, rng):
    """Positive pool plus two matched negative sets, one row per (chain, i, j)."""
    rows, drop_flank, drop_nomatch = [], 0, 0
    n_pos_total = 0
    for ci, ch in enumerate(chains):
        L = len(ch["names"])
        names = ch["names"]
        c1 = ch["c1"]
        acc = set(ch["pairs"])
        n_pos_total += len(ch["pairs"])

        cands = []
        for i in range(L):
            for j in range(i + MIN_SEP, L):
                if (i, j) in acc or not flank_ok(L, i, j):
                    continue
                d = float(np.linalg.norm(c1[i] - c1[j]))
                cands.append((i, j, j - i, d, (names[i], names[j]) in WCP))

        kept = []
        for a, b in sorted(ch["pairs"]):
            if not flank_ok(L, a, b):
                drop_flank += 1
                continue
            s = b - a
            n1 = [c for c in cands if abs(c[2] - s) <= tol and not c[4]
                  and BAND[0] <= c[3] <= BAND[1]]
            n2 = [c for c in cands if abs(c[2] - s) <= tol]
            # N3: the only set that can test whether the folding-derived channel adds anything
            # beyond chemistry AND geometry. Both are held fixed here -- base-complementary like a
            # positive, inside the same C1'-C1' band like a positive -- so wc_identity reads 1.0
            # for every row and geom_in_band reads 1.0 for every row, and each is therefore an
            # uninformative 0.5. Whatever separates positives from N3 is sequence content in the
            # prediction itself, which is the thing isrnaclong.py:780 says was never measured.
            n3 = [c for c in cands if abs(c[2] - s) <= tol and c[4]
                  and BAND[0] <= c[3] <= BAND[1]]
            if not n1 or not n2:
                drop_nomatch += 1
                continue
            kept.append((a, b, s, float(np.linalg.norm(c1[a] - c1[b])), n1, n2, n3))

        for a, b, s, d, n1, n2, n3 in kept:
            rows.append({"chain": ci, "label": 1, "set": "pos", "i": a, "j": b,
                         "sep": s, "d": d})
            for sname, pool in (("N1", n1), ("N2", n2), ("N3", n3)):
                if not pool:
                    continue
                take = rng.choice(len(pool), size=min(per_pos, len(pool)), replace=False)
                for t in take:
                    i, j, sep, dd, is_wc = pool[int(t)]
                    rows.append({"chain": ci, "label": 0, "set": sname, "i": i, "j": j,
                                 "sep": sep, "d": dd})
    return rows, n_pos_total, drop_flank, drop_nomatch


# -------------------------------------------------------------------- scoring
def score_pair(seq, names, i, j, d):
    f = min(FLANK_CAP, len(seq) // 4)
    up = seq[max(0, i - f):i]
    dn = seq[j:min(len(seq), j + f)]
    if len(up) < FLANK_MIN or len(dn) < FLANK_MIN:
        raise RuntimeError(f"row ({i},{j}) has no valid flank; it should not have been kept")
    r = compute_rcm_score(up, dn)
    dd = rcm_density_score(up, dn)
    if abs(r["confidence"] - dd["confidence"]) > 1e-12:
        raise RuntimeError("rcm_density_score and compute_rcm_score disagree on confidence")
    # rcm.py's _COMPLEMENT is a DNA table (A -> T). Its prefilter is RNA-consistent (U maps
    # to the same complex value as T), but _validate_rcm compares against _COMPLEMENT, so
    # an A-U match is accepted in the (up=A, down=U) orientation and silently rejected in
    # (up=U, down=A), and the module's own __main__ example -- a true reverse complement --
    # returns crossing_total 0. Rewriting U as T makes the alphabet the validator assumes
    # consistent; nothing inside rcm.py is changed.
    rt = compute_rcm_score(up.replace("U", "T"), dn.replace("U", "T"))
    wc = 1.0 if (names[i], names[j]) in WCP else 0.0
    ma = method_agreement_map(seq).get((min(i, j), max(i, j)), 0.0)
    return {
        "method_agree": ma,
        "rcm_confidence": r["confidence"],
        "rcm_crossing": float(r["crossing_total"]),
        "rcm_density": dd["crossing_density"],
        "rcm_confidence_T": rt["confidence"],
        "rcm_crossing_T": float(rt["crossing_total"]),
        "wc_identity": wc,
        # the two channels multiplied: does the folding prediction still separate anything once
        # the base-chemistry lookup is applied on top of it
        "ma_x_wc": ma * wc,
        "wc_x_geom": wc * float(np.exp(-0.5 * ((d - BAND_CENTRE) / BAND_SIGMA) ** 2)),
        "geom_in_band": 1.0 if BAND[0] <= d <= BAND[1] else 0.0,
        "flank_up": float(len(up)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=None)
    ap.add_argument("--per-pos", type=int, default=4)
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--shuffles", type=int, default=3)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    chains = load_chains(args.files)
    print(f"=== data ===")
    print(f"PDB files read         : {len({c['file'] for c in chains})}")
    print(f"chains (verified ==B)  : {len(chains)}")
    Ls = [len(c["names"]) for c in chains]
    print(f"chain length           : min {min(Ls)}  median {int(np.median(Ls))}  max {max(Ls)}")
    print(f"flank window min(200,L//4): min {min(min(FLANK_CAP, L // 4) for L in Ls)}  "
          f"median {int(np.median([min(FLANK_CAP, L // 4) for L in Ls]))}  "
          f"max {max(min(FLANK_CAP, L // 4) for L in Ls)}")

    rows, n_pos_total, drop_flank, drop_nomatch = build_rows(
        chains, args.per_pos, args.tol, rng)
    print(f"accepted WC pairs      : {n_pos_total}")
    print(f"  dropped, flank <5 nt : {drop_flank}")
    print(f"  dropped, no matched negative within tol={args.tol}: {drop_nomatch}")
    print(f"positives used         : {sum(1 for r in rows if r['label'] == 1)}")
    print(f"negative matching      : separation matched within +/-{args.tol}, "
          f"{MIN_SEP} residue minimum for all sets")
    print()

    # score every row on the real sequence and on --shuffles permutations
    seqs = {}
    for ci, ch in enumerate(chains):
        s = "".join(ch["names"])
        seqs[(ci, -1)] = s
        for r in range(args.shuffles):
            seqs[(ci, r)] = "".join(rng.permutation(list(s)))

    variants = [("real", -1)] + [(f"shuf{k}", k) for k in range(args.shuffles)]
    vals = {v: {s: np.zeros(len(rows)) for s in SCORES} for v, _ in variants}
    for n, row in enumerate(rows):
        ci = row["chain"]
        for vname, vkey in variants:
            seq = seqs[(ci, vkey)]
            got = score_pair(seq, list(seq), row["i"], row["j"], row["d"])
            for s in SCORES:
                vals[vname][s][n] = got[s]

    labels = np.array([r["label"] for r in rows])
    sets = np.array([r["set"] for r in rows])
    cid = np.array([r["chain"] for r in rows])
    print(f"scored {len(rows)} rows x {len(variants)} sequences "
          f"({args.shuffles} shuffles) in {time.time() - t0:.1f}s")
    print()

    for sname in ("N1", "N2", "N3"):
        m = (sets == sname) | (labels == 1)
        lab = labels[m]
        desc = {"N1": "band-matched, non-WC -- geometry held fixed, chemistry removed",
                "N2": "separation-matched, any identity, any distance -- geometry dominates",
                "N3": "band-matched AND base-complementary -- chemistry and geometry BOTH held "
                      "fixed, so only the prediction can separate"}[sname]
        print(f"=== negative set {sname} ({desc}) ===")
        print(f"  n_pos={int(lab.sum())}  n_neg={int((~lab.astype(bool)).sum())}")
        sub = [r for r, k in zip(rows, m) if k]
        if not sub:
            print(f"=== negative set {sname}: no rows ===")
            print()
            continue
        ps = [r["sep"] for r in sub if r["label"] == 1]
        ns = [r["sep"] for r in sub if r["label"] == 0]
        pd = [r["d"] for r in sub if r["label"] == 1]
        nd = [r["d"] for r in sub if r["label"] == 0]
        if not ns or not nd:
            print(f"  n_neg=0. The pool is empty, and empty BY CONSTRUCTION rather than by a thin")
            print(f"  database: boltzmann_bonded._chain_residues accepts a pair exactly when it is")
            print(f"  Watson-Crick compatible, at least {MIN_SEP} residues apart, and inside the")
            print(f"  C1'-C1' band {BAND}. N3 asks for the same three conditions and NOT accepted,")
            print(f"  so it has no members anywhere, on any number of chains.")
            print(f"  Consequence: no score can be tested for information BEYOND chemistry and")
            print(f"  geometry on this database, because the label is that conjunction.")
            print()
            continue
        print(f"  separation  pos [5/25/50/75/95]: {pct(ps)}")
        print(f"  separation  neg [5/25/50/75/95]: {pct(ns)}")
        print(f"  C1'-C1' A   pos [5/25/50/75/95]: {pct(pd)}")
        print(f"  C1'-C1' A   neg [5/25/50/75/95]: {pct(nd)}")
        neg_comp = float(np.mean(vals["real"]["wc_identity"][m][lab == 0]))
        print(f"  fraction of negatives that are base-complementary: {neg_comp:.4f}")
        print()
        print(f"  {'score':16s} {'pos p5/25/50/75/95':>34s} {'neg p5/25/50/75/95':>34s}")
        for s in SCORES:
            print(f"  {s:16s} {pct(vals['real'][s][m][lab == 1]):>34s} "
                  f"{pct(vals['real'][s][m][lab == 0]):>34s}")
        print()
        print("  note: wc_identity is the definition of the positives on N1 and wc_x_geom is")
        print("        the definition on N2, so their perfect AUCs are construction, not skill;")
        print("        geom_in_band is fixed at 0.5 on N1 for the same reason (all rows in band).")
        print(f"  {'score':16s} {'AUC real':>9s} {'[95% CI]':>18s} {'AUC shuf':>9s} "
              f"{'real-shuf':>10s}")
        for s in SCORES:
            p, lo, hi, _ = auc_ci(labels[m], vals["real"][s][m], cid[m], args.boot, rng)
            sh = np.mean([auc(vals[v][s][m][lab == 1], vals[v][s][m][lab == 0])
                          for v, _ in variants[1:]])
            print(f"  {s:16s} {p:9.4f} {'[' + format(lo, '.4f') + ', ' + format(hi, '.4f') + ']':>18s} "
                  f"{sh:9.4f} {p - sh:10.4f}")
        print()
        print(f"  paired cluster bootstrap (AUC difference, 95% CI, p = share <= 0):")
        comps = [("rcm_confidence", "wc_identity"), ("rcm_confidence", "wc_x_geom"),
                 ("rcm_confidence", "rcm_density"),
                 ("rcm_confidence_T", "rcm_confidence"),
                 ("rcm_confidence_T", "wc_identity"), ("rcm_density", "wc_identity"),
                 ("method_agree", "wc_identity"), ("method_agree", "rcm_confidence"),
                 ("method_agree", "wc_x_geom"),
                 ("ma_x_wc", "wc_identity"), ("ma_x_wc", "method_agree")]
        # on N3 both reference scores are 1.0 for every row, so those three comparisons are
        # degenerate there and only the rcm one is shown
        if sname == "N3":
            comps = [("method_agree", "rcm_confidence")]
        for a, b in comps:
            p, lo, hi, pv = paired_boot(labels[m], vals["real"][a][m], vals["real"][b][m],
                                        cid[m], args.boot, rng)
            print(f"    {a} - {b}: {p:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={pv:.3f}")
        if args.shuffles >= 1:
            print("  real vs shuffled, same rows (the sequence-specific component):")
            for s in ("rcm_confidence", "rcm_confidence_T", "method_agree", "ma_x_wc",
                      "wc_identity"):
                p, lo, hi, pv = paired_boot(labels[m], vals["real"][s][m],
                                            vals["shuf0"][s][m], cid[m], args.boot, rng)
                print(f"    {s}: {p:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={pv:.3f}")
        print()

    # confidence is a ratio of small integers; show what values it actually takes
    conf = vals["real"]["rcm_confidence"][labels == 1]
    u, c = np.unique(np.round(conf, 6), return_counts=True)
    order = np.argsort(-c)[:8]
    print("=== rcm_confidence, values taken by the positives (real sequence) ===")
    print(f"  n={conf.size}  distinct={len(u)}  "
          f"frac==0 {np.mean(conf == 0):.4f}  frac==1 {np.mean(conf == 1):.4f}")
    print("  most frequent: " + "  ".join(f"{u[k]:.4f}x{c[k]}" for k in order))
    print(f"  median {np.median(conf):.4f}  <0.05 {np.mean(conf < 0.05):.4f}  "
          f"<0.10 {np.mean(conf < 0.10):.4f}  <0.20 {np.mean(conf < 0.20):.4f}")
    print(f"  rcm_crossing of positives: " + pct(vals["real"]["rcm_crossing"][labels == 1]))
    print(f"  rcm_density  of positives: " + pct(vals["real"]["rcm_density"][labels == 1]))
    print(f"  flank_up     of positives: " + pct(vals["real"]["flank_up"][labels == 1]))
    print(f"\ntotal {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
