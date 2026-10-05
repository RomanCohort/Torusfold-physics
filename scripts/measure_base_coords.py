"""The two sides of a BASE-LEVEL IBI: what the crystals hold, and what the field produces.

WHY THIS EXISTS. Part 14 measured that reconstructing the all-atom structure from the SAMPLED base frame
recovers half the stacking the P-trace heuristic loses (43.9 -> 75.7 percent over 20 fragments) -- and
that on a real CG state it does nothing, because the sampled base frames are not oriented relative to one
another: K_STACK = 0, there is no base-orientation term, and the roll is held only by the K_LINK_* links
against a pair potential that does not care. So the reconstruction change and a base-level term are a
pair, and this script measures the pair's INPUT: the base-level coordinate distributions on both sides.

BEAD-ONLY COORDINATES, deliberately. A CG potential can only be a function of the beads, so every
coordinate here is computed from the three points the model carries per residue -- P, C4', and N9 (purine)
or N1 (pyrimidine):

    nb_dist   |N_i - N_{i+1}|                    the base-base separation
    cc_dist   |C4'_i - C4'_{i+1}|               the sugar-sugar separation, for reference
    rise      (N_{i+1} - N_i) . n_mean          separation along the mean triangle normal
    theta     angle(n_i, n_{i+1})               triangle-normal angle (sign-aligned first)
    twist     in-plane rotation of P->N about n_mean

with n_i = normalize((C4'_i - P_i) x (N_i - P_i)). The triangle P-C4'-N is not the base plane, so the
target side also reports HOW GOOD that proxy is: the angle between the triangle normal and the true
base-plane normal (from the ring atoms, measure_base_stacking.plane) on the same residues.

TARGET SIDE: every consecutive pair in N crystal fragments of _cgdata/combined plus 2OIU. Report both the
marginal over ALL consecutive pairs (the honest reference distribution of the coordinate) and the subset
that is a helical step (the physically stacked ones, Part 13's criterion).
SAMPLED SIDE: run_round on the loader's own 3-bead chains (ibi_core, the same sampler the loop uses,
production tables installed) with collect_positions=True, and bin every frame.

Output: per-coordinate mean +- sd and the total-variation distance between the two normalized histograms
on a shared grid, plus the pooled samples in results/plan_c/_base_coords_*.npz so a fit can consume them.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("TORUSFOLD_RSRNASP", str(REPO / "_cgdata" / "combined"))

import boltzmann_bonded as B               # noqa: E402
import ibi_bonded as I                     # noqa: E402
import ibi_core as IC                      # noqa: E402
import measure_base_stacking as M          # noqa: E402
import cg_potentials as CP                 # noqa: E402
from torusfold.scheme2 import torch_cgsim as C                            # noqa: E402

REF = REPO / "results" / "refit_smooth5.npz"
PROD = REPO / "results" / "production_tables.npz"
OUT = REPO / "results" / "plan_c"
COORDS = ("nb_dist", "cc_dist", "rise", "theta", "twist")
GLY = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}


def frames_from_beads(p):
    """(L, 3, 3) nm -> [(N, n, v, P)] with n the triangle normal and v the in-plane P->N direction."""
    p = np.asarray(p, dtype=float)
    out = []
    for i in range(p.shape[0]):
        P, C4, N = p[i, 0], p[i, 1], p[i, 2]
        n = np.cross(C4 - P, N - P)
        nn = np.linalg.norm(n)
        n = n / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])
        v = N - P
        v = v - n * float(v @ n)
        nv = np.linalg.norm(v)
        v = v / nv if nv > 1e-12 else np.array([1.0, 0.0, 0.0])
        out.append((N, n, v, P))
    return out


def pair_coords(frames, i, j):
    Ni, ni, vi, _Pi = frames[i]
    Nj, nj, vj, _Pj = frames[j]
    if float(ni @ nj) < 0.0:
        nj = -nj
    nb = ni + nj
    nn = np.linalg.norm(nb)
    nb = nb / nn if nn > 1e-12 else ni
    dc = Nj - Ni
    sep = float(np.linalg.norm(dc))
    rise = float(dc @ nb)
    theta = float(np.degrees(np.arccos(max(-1.0, min(1.0, float(ni @ nj))))))
    w = vj - nb * float(vj @ nb)
    nw = np.linalg.norm(w)
    if nw < 1e-12:
        twist = float("nan")
    else:
        w = w / nw
        twist = float(abs(np.degrees(np.arctan2(float(np.cross(vi, w) @ nb),
                                                max(-1.0, min(1.0, float(vi @ w)))))))
    return sep, rise, theta, twist


# ---------------------------------------------------------------- target side
def target_samples(n_frag, want="all"):
    """Pooled (nb_dist, cc_dist, rise, theta, twist) over consecutive pairs of the crystal database."""
    rows = {k: [] for k in COORDS}
    proxy = []
    steps = set()
    files = sorted((REPO / "_cgdata" / "combined").glob("*.pdb"))
    done = 0
    for f in files:
        if done >= n_frag:
            break
        seq, residues, pP, ch = M.parse_pdb(f)
        if seq is None or len(seq) > 300 or len(seq) < 8:
            continue
        try:
            beads = np.stack([[r["atoms"]["P"], r["atoms"]["C4'"], r["atoms"][GLY[r["base"]]]]
                              for r in residues]) / 10.0                      # A -> nm
        except KeyError:
            continue
        planes = [M.plane(r) for r in residues]
        pairs = M.wc_pairs(seq, residues, planes)
        pset = set(pairs)
        for (i, j) in pairs:
            if (i + 1, j - 1) in pset:
                steps.add((i, i + 1))
                steps.add((j - 1, j))
        frames = frames_from_beads(beads)
        for i in range(len(seq) - 1):
            if want == "steps" and (i, i + 1) not in steps:
                continue
            a, b, th, tw = pair_coords(frames, i, i + 1)
            rows["nb_dist"].append(a)
            rows["cc_dist"].append(float(np.linalg.norm(beads[i + 1, 1] - beads[i, 1])))
            rows["rise"].append(b)
            rows["theta"].append(th)
            rows["twist"].append(tw)
            if planes[i] is not None:
                _c, ntrue, _v = planes[i]
                proxy.append(float(np.degrees(np.arccos(
                    max(-1.0, min(1.0, abs(float(frames[i][1] @ ntrue))))))))
        done += 1
    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}, np.asarray(proxy), done


# ---------------------------------------------------------------- sampled side
def small_pool():
    """The 24-34 residue band the loop's "small" pool uses, SORTED BY NAME.

    Sorted because the loader's order is not stable across processes (measured 2026-10-05: two calls in
    the same session returned different members first), which makes a chain index meaningless -- and the
    per-chain subprocess below addresses chains by index.
    """
    pool = [s for s in B.load_structures(limit=5000) if len(s["pairs"]) >= 8
            and 24 <= len(s["pos"]) <= 34]
    return sorted(pool, key=lambda s: s["name"])


def sampled_frames(n_chains=8, nsteps=5000, burn=1000, stride=25, blocks=4, threads=4,
                   chain_idx=None):
    # ONE CHAIN PER CALL IS ALSO THE SAFETY PROPERTY. Measured 2026-10-05: sampling the pool in one
    # process hangs on the sixth chain (3MJA) -- twice, at the same chain, after five finished -- so a
    # single-process loop loses everything that came before it. subprocess_chain() below runs each chain
    # in its own process under a timeout and keeps the rest.
    torch.set_num_threads(threads)
    pool = small_pool()
    if chain_idx is not None:
        pool = [pool[chain_idx]]
    tab = IC.load_tables(str(REF))
    with np.load(PROD) as z:
        for c in ("bb_bond", "angle", "dihedral"):
            U = np.asarray(z[f"{c}__U"], dtype=float)
            tab[c] = dict(tab[c], U=U)
    _pots, pot_kw = CP.build_potential_kwargs(str(PROD), wall_k=2000.0,
                                             coords=("bb_bond", "angle", "dihedral"))
    out = []
    for k, s in enumerate(pool[:n_chains]):
        pos_np = np.asarray(s["pos"], dtype=np.float64)
        L = pos_np.shape[0]
        pos = torch.tensor(pos_np.reshape(1, 3 * L, 3), dtype=torch.float64)
        vel = torch.zeros_like(pos)
        temps = torch.full((1,), 300.0, dtype=torch.float64)
        pairs = list(s["pairs"])
        ij = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
        pw = torch.ones(len(pairs), dtype=torch.float32)
        con = C.make_intra_constraints(L)
        t0 = time.time()
        res = IC.run_round(pos=pos, vel=vel, ij=ij, pw=pw, temps=temps, tab=tab,
                           nsteps=nsteps, burn=burn, stride=stride, blocks=blocks,
                           friction=1.0, force_cap=5000.0, pot_kw=pot_kw, seed=20261005 + k,
                           nrep=1, progress=False, constraints=con,
                           collect_positions=True, log=lambda *a, **kk: None)
        frames = res.positions                      # (nframes, 1, 3L, 3) nm
        print("  chain %-9s L=%2d  %4d frames in %.0f s"
              % (s["name"], L, frames.shape[0], time.time() - t0), flush=True)
        for fr in range(frames.shape[0]):
            p = frames[fr][0].reshape(L, 3, 3).numpy()
            out.append(p)
    return out


def sampled_frames_with_stack(eps, n_ch=2, nsteps=5000, burn=1000, stride=25, blocks=4, threads=4,
                              form="reward", weights=(1.0, 1.0, 1.0)):
    """Sample with the base-level stacking term installed. Returns (frames, trace joint J).

    Everything else is the same protocol as sampled_frames: production tables, the loader's 3-bead chains,
    the loop's own run_round. The only difference is the extra potential in pot_kw.
    """
    from torusfold.scheme2 import base_stacking as BS
    torch.set_num_threads(threads)
    pool = small_pool()[:n_ch]
    tab = IC.load_tables(str(REF))
    with np.load(PROD) as z:
        for c in ("bb_bond", "angle", "dihedral"):
            tab[c] = dict(tab[c], U=np.asarray(z[f"{c}__U"], dtype=float))
    _pots, pot_kw = CP.build_potential_kwargs(str(PROD), wall_k=2000.0,
                                             coords=("bb_bond", "angle", "dihedral"))
    if eps != 0.0:
        pot_kw = dict(pot_kw, base_stack_potential=BS.make_base_stack_potential(
            eps=eps, form=form, w_d=weights[0], w_r=weights[1], w_t=weights[2]))
    else:
        pot_kw = dict(pot_kw, base_stack_potential=None)
    out = []
    js = []
    for k, s in enumerate(pool):
        pos_np = np.asarray(s["pos"], dtype=np.float64)
        L = pos_np.shape[0]
        pos = torch.tensor(pos_np.reshape(1, 3 * L, 3), dtype=torch.float64)
        vel = torch.zeros_like(pos)
        temps = torch.full((1,), 300.0, dtype=torch.float64)
        pairs = list(s["pairs"])
        ij = torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)
        pw = torch.ones(len(pairs), dtype=torch.float32)
        con = C.make_intra_constraints(L)
        t0 = time.time()
        res = IC.run_round(pos=pos, vel=vel, ij=ij, pw=pw, temps=temps, tab=tab,
                           nsteps=nsteps, burn=burn, stride=stride, blocks=blocks,
                           friction=1.0, force_cap=5000.0, pot_kw=pot_kw, seed=20261005 + k,
                           nrep=1, progress=False, constraints=con,
                           collect_positions=True, log=lambda *a, **kk: None)
        skip = B.CONSTRAINED
        vals, j = IC.simref(res.acc, tab, skip)
        js.append(float(j))
        print("  chain %-9s L=%2d  %4d frames in %.0f s  J %.4f"
              % (s["name"], L, res.positions.shape[0], time.time() - t0, float(j)), flush=True)
        chain_frames = [res.positions[fr][0].reshape(L, 3, 3).numpy()
                        for fr in range(res.positions.shape[0])]
        out.append((s["name"], chain_frames))
    return out, float(np.mean(js))


def coords_from_frame_list(frame_list):
    """Bead-only coordinates for each sampled frame, with the base plane taken from the RIGID TEMPLATE
    MAP (`base_frames`) rather than from the raw P-C4'-N triangle.

    The triangle normal is 20.2 deg (median 16.2) off the true base-plane normal, so a table built on it
    measures a different quantity from the one a term can target: the map is what the model can express
    (17.6 deg off the ring planes on crystals, 6.6 deg on the template itself), and both sides of the
    comparison now use it.
    """
    from torusfold.scheme2 import base_frames as BF
    coef_A = BF.pooled_coef(1.0)                 # beads below are Angstrom, as base_frames expects
    rows = {k: [] for k in COORDS}
    for p in frame_list:
        beads_A = np.asarray(p, dtype=float) * 10.0
        n = BF.normals_np(beads_A, coef_A)       # (L, 3)
        for i in range(p.shape[0] - 1):
            a, b = n[i].copy(), n[i + 1].copy()
            if float(a @ b) < 0.0:
                b = -b
            nm = a + b
            nn = np.linalg.norm(nm)
            nm = nm / nn if nn > 1e-9 else a
            dc = (beads_A[i + 1, 2] - beads_A[i, 2]) / 10.0
            rows["nb_dist"].append(float(np.linalg.norm(dc)))
            rows["cc_dist"].append(float(np.linalg.norm(beads_A[i + 1, 1] - beads_A[i, 1]) / 10.0))
            rows["rise"].append(float(dc @ nm))
            rows["theta"].append(float(np.degrees(np.arccos(max(-1.0, min(1.0, float(a @ b)))))))
            # wj must use the SIGN-ALIGNED normal (b), not n[i+1]: with the raw one, every pair whose
            # map-normal was flipped contributes ~180 degrees and the twist column reads 122 +- 84 deg
            # with a 5th percentile of 0. Found in the first scan, where the other three columns were fine
            # -- which is exactly how a one-line error in one coordinate announces itself.
            wi = a - nm * float(a @ nm)
            wj = b - nm * float(b @ nm)
            nwi, nwj = np.linalg.norm(wi), np.linalg.norm(wj)
            if nwi < 1e-9 or nwj < 1e-9:
                continue
            wi, wj = wi / nwi, wj / nwj
            _cr = float(np.cross(wi, wj) @ nm)
            _dt = max(-1.0, min(1.0, float(wi @ wj)))
            rows["twist"].append(abs(float(np.degrees(np.arctan2(_cr, _dt)))))
    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}


def tv_distance(a, b, lo, hi, nbins=80):
    ha, _ = np.histogram(a[np.isfinite(a)], bins=nbins, range=(lo, hi))
    hb, _ = np.histogram(b[np.isfinite(b)], bins=nbins, range=(lo, hi))
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    return 0.5 * float(np.abs(ha - hb).sum()), ha, hb


def report(tag, d):
    for k in COORDS:
        v = d[k][np.isfinite(d[k])]
        if v.size == 0:
            continue
        print("  %-10s %-9s n=%7d  mean %8.4f  sd %7.4f  p5 %8.4f  p50 %8.4f  p95 %8.4f"
              % (tag, k, v.size, v.mean(), v.std(), np.percentile(v, 5),
                 np.percentile(v, 50), np.percentile(v, 95)))


def main():
    # --sampled-only: just the CG run (long: 8 chains x 5000 steps at ~100 s each) and its own npz.
    # Splitting it from the target side means a process sweep costs a re-run of the sampler rather than
    # of everything, and the target side is deterministic.
    if len(sys.argv) > 2 and sys.argv[1] == "--one-chain":
        k = int(sys.argv[2])
        frames = sampled_frames(chain_idx=k)
        smp = coords_from_frame_list(frames)
        report("chain%d" % k, smp)
        np.savez(OUT / f"_base_coords_chain{k}.npz", **{f"sampled_{key}": smp[key] for key in COORDS})
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--sampled-only":
        import subprocess
        print("=== SAMPLED: production tables, the loader's own 3-bead chains, 5000 steps ==="
              "\n    one process per chain, 420 s timeout: a chain that hangs costs only itself",
              flush=True)
        t0 = time.time()
        # The band holds 31 chains (with two duplicated names), each ~100 s, so a cap is part of the
        # protocol rather than an optimisation. Eight sorted chains give ~800 frames and ~23 000
        # consecutive-pair samples, which is what the comparison needs.
        n_pool = min(8, len(small_pool()))
        print("    chains: " + ", ".join(s["name"] for s in small_pool()[:n_pool]), flush=True)
        for k in range(n_pool):
            dest = OUT / f"_base_coords_chain{k}.npz"
            tag = small_pool()[k]["name"]
            try:
                cp = subprocess.run([sys.executable, "-u", __file__, "--one-chain", str(k)],
                                    timeout=420, capture_output=True, text=True)
                tail = (cp.stdout or "").strip().splitlines()
                print("  chain %d %-9s rc=%s  %s" % (k, tag, cp.returncode,
                                                     tail[-1] if tail else "(no output)"), flush=True)
                if cp.returncode != 0:
                    print("     stderr tail: " + (cp.stderr or "").strip().splitlines()[-1][:160]
                          if cp.stderr else "     (no stderr)", flush=True)
            except subprocess.TimeoutExpired:
                print("  chain %d %-9s TIMED OUT after 420 s -- skipped. A chain takes ~100 s, so"
                      " this is a hang of the sampler on this chain, not slowness" % (k, tag), flush=True)
        parts = []
        for k in range(n_pool):
            f = OUT / f"_base_coords_chain{k}.npz"
            if f.exists():
                with np.load(f) as z:
                    parts.append({key: z[f"sampled_{key}"] for key in COORDS})
        if not parts:
            raise SystemExit("no chain produced samples")
        smp = {key: np.concatenate([p[key] for p in parts]) for key in COORDS}
        report("sampled", smp)
        np.savez(OUT / "_base_coords_sampled.npz", **{f"sampled_{key}": smp[key] for key in COORDS})
        print("wrote results/plan_c/_base_coords_sampled.npz from %d/%d chains (%.0f s)"
              % (len(parts), n_pool, time.time() - t0), flush=True)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        tf = OUT / "_base_coords_two_sides.npz"
        sf = OUT / "_base_coords_sampled.npz"
        tz = np.load(tf)
        tgt = {k: tz[f"target_{k}"] for k in COORDS}
        if sf.exists():
            with np.load(sf) as sz:
                smp = {k: sz[f"sampled_{k}"] for k in COORDS}
        else:
            # Pool whatever per-chain files exist: a chain that hung or was cut for wall-clock costs
            # only itself, which is the whole reason each chain runs in its own process.
            parts = []
            for f in sorted(OUT.glob("_base_coords_chain*.npz")):
                with np.load(f) as z:
                    parts.append({k: z[f"sampled_{k}"] for k in COORDS})
            if not parts:
                raise SystemExit("no sampled data: run --sampled-only first")
            smp = {k: np.concatenate([p[k] for p in parts]) for k in COORDS}
            print("sampled side pooled from %d chain files: %s"
                  % (len(parts), ", ".join(f.stem.split("chain")[-1] for f in
                                           sorted(OUT.glob("_base_coords_chain*.npz")))), flush=True)
        report("target", tgt)
        report("sampled", smp)
        grid = {"nb_dist": (0.3, 1.6), "cc_dist": (0.3, 1.6), "rise": (-1.0, 1.0),
                "theta": (0.0, 180.0), "twist": (0.0, 180.0)}
        print("\n%-10s %6s %12s %12s %12s %12s" % ("coord", "TV", "target mean", "target sd",
                                                    "sampled mean", "sampled sd"), flush=True)
        for k in COORDS:
            tv, _ha, _hb = tv_distance(tgt[k], smp[k], *grid[k])
            tv_ref, _a, _b = tv_distance(tgt[k], np.asarray(tz[f"target_steps_{k}"]), *grid[k])
            print("%-10s %6.3f %12.4f %12.4f %12.4f %12.4f   (target all-pairs vs helical subset: TV %.3f)"
                  % (k, tv, np.nanmean(tgt[k]), np.nanstd(tgt[k]),
                     np.nanmean(smp[k]), np.nanstd(smp[k]), tv_ref), flush=True)
        return

    if len(sys.argv) > 2 and sys.argv[1] == "--stack-scan2":
        # --stack-scan2 eps1,eps2,... [n_chains] [nsteps]  -- the PENALTY form, eps as the strength
        from torusfold.scheme2 import base_stacking as BS
        eps_list = [float(x) for x in sys.argv[2].split(",")]
        n_ch = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        nst = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
        tz = np.load(OUT / "_base_coords_planes.npz")
        tgt = {k: tz[f"impl_{k}"] for k in ("nb_dist", "rise", "theta", "twist")}
        grid = {"nb_dist": (0.3, 1.6), "rise": (-1.0, 1.0), "theta": (0.0, 180.0), "twist": (0.0, 180.0)}
        w = (1.0, 1.0, 1.0)
        if len(sys.argv) > 5:
            w = tuple(float(x) for x in sys.argv[5].split(","))
        for eps in eps_list:
            print("\n===== SUM form, eps = %.2f kJ/mol (%.2f kBT) per pair, weights"
                  " (d, rise, theta) = %s =====" % (eps, eps / 2.494, w), flush=True)
            per_chain, jval = sampled_frames_with_stack(eps, n_ch=n_ch, nsteps=nst, form="sum",
                                                          weights=w)
            # PER-CHAIN coordinates as well as pooled ones: the settings are compared on the SAME chains
            # and the same seeds, so a paired test across chains is available and a pooled mean would only
            # be the average of it.
            per = {nm: coords_from_frame_list(frames) for nm, frames in per_chain}
            smp = {k: np.concatenate([per[nm][k] for nm, _f in per_chain]) for k in COORDS}
            for k in ("nb_dist", "rise", "theta"):
                v = smp[k][np.isfinite(smp[k])]
                tv, _ha, _hb = tv_distance(tgt[k], v, *grid[k])
                print("  %-8s mean %8.4f  sd %7.4f  p5 %8.4f   TV vs target %.3f  (target mean %.4f sd %.4f)"
                      % (k, v.mean(), v.std(), np.percentile(v, 5), tv,
                         np.nanmean(tgt[k]), np.nanstd(tgt[k])), flush=True)
            print("  trace joint J = %.4f" % jval, flush=True)
            payload = {f"sampled_{k}": smp[k] for k in COORDS}
            for nm, _f in per_chain:
                for k in COORDS:
                    payload[f"chain_{nm}_{k}"] = per[nm][k]
            payload["j"] = np.asarray([jval])
            payload["eps"] = np.asarray([eps])
            payload["weights"] = np.asarray(w)
            np.savez(OUT / ("_stack_sum_w%g_eps%g.npz" % (w[0], eps)), **payload)
        return

    if len(sys.argv) > 2 and sys.argv[1] == "--stack-scan":
        # --stack-scan eps1,eps2,... [n_chains] [nsteps]
        # The base-level term, scanned in strength, measured against the expressible target and against
        # the TRACE marginals at the same time: a base-level term that fixes stacking while pushing the
        # backbone out of its own fitted marginals would be a trade, and this is where that shows up.
        from torusfold.scheme2 import base_stacking as BS
        eps_list = [float(x) for x in sys.argv[2].split(",")]
        n_ch = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        nst = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
        tz = np.load(OUT / "_base_coords_planes.npz")
        tgt = {k: tz[f"impl_{k}"] for k in ("nb_dist", "rise", "theta", "twist")}
        grid = {"nb_dist": (0.3, 1.6), "rise": (-1.0, 1.0), "theta": (0.0, 180.0), "twist": (0.0, 180.0)}
        for eps in eps_list:
            print("\n===== stacking strength eps = %.2f kJ/mol (%.2f kBT) =====" % (eps, eps / 2.494),
                  flush=True)
            frame_list, jval = sampled_frames_with_stack(eps, n_ch=n_ch, nsteps=nst)
            smp = coords_from_frame_list(frame_list)
            rows = {}
            for k in ("nb_dist", "rise", "theta", "twist"):
                v = smp[k][np.isfinite(smp[k])]
                tv, _ha, _hb = tv_distance(tgt[k], v, *grid[k])
                rows[k] = tv
                print("  %-8s mean %8.4f  sd %7.4f  p5 %8.4f   TV vs target %.3f"
                      % (k, v.mean(), v.std(), np.percentile(v, 5), tv), flush=True)
            print("  trace joint J = %.4f  (the field's own scoreboard, unchanged coordinates)"
                  % jval, flush=True)
            np.savez(OUT / ("_base_coords_stack_eps%g.npz" % eps),
                     **{f"sampled_{k}": smp[k] for k in COORDS}, j=np.asarray([jval]))
        return

    if len(sys.argv) > 2 and sys.argv[1] == "--planes":
        # The TARGET a bead-based term can actually express: the same coordinates, but with the plane taken
        # from the beads through the rigid template (base_frames.normals_np) instead of from the ring atoms.
        # Reported beside the ring-atom version, so the cost of the three-bead representation is visible.
        from torusfold.scheme2 import base_frames as BF
        n_frag = int(sys.argv[2])
        coef = BF.pooled_coef(1.0)      # beads below are Angstrom
        true_rows = {k: [] for k in ("rise", "theta", "twist", "nb_dist")}
        impl_rows = {k: [] for k in ("rise", "theta", "twist", "nb_dist")}
        ang = []
        files = sorted((REPO / "_cgdata" / "combined").glob("*.pdb"))
        done = 0
        for f in files:
            if done >= n_frag:
                break
            seq, residues, _pP, _ch = M.parse_pdb(f)
            if seq is None or len(seq) > 300 or len(seq) < 8:
                continue
            try:
                beads = np.stack([[r["atoms"]["P"], r["atoms"]["C4'"], r["atoms"][GLY[r["base"]]]]
                                  for r in residues])
            except KeyError:
                continue
            planes = [M.plane(r) for r in residues]
            if any(q is None for q in planes):
                continue
            # SAME SIGN CONVENTION ON BOTH SIDES, or the rise is meaningless: M.plane returns an SVD
            # normal whose sign is arbitrary, so the ring-atom normals are flipped to point along
            # +e3 = (C4'-P) x (N-P) exactly as base_frames does for the bead ones. Measured before this
            # line existed: ring rise -0.349 A against bead rise +3.28 A for the same pairs, i.e. the two
            # columns disagreed in sign because the normals did.
            planes = []
            for _i, _r in enumerate(residues):
                _c, _n, _v = M.plane(_r)
                _e3 = np.cross(beads[_i, 1] - beads[_i, 0], beads[_i, 2] - beads[_i, 0])
                if float(_n @ _e3) < 0:
                    _n = -_n
                planes.append((_c, _n, _v))
            # The map is fed ANGSBROM (the template's unit) and the reported distances are nm, because
            # the coefficients are scale-sensitive: a cross product scales with length squared.
            n_impl = BF.normals_np(beads, coef)
            beads = beads / 10.0                      # A -> nm for the reported table
            for i in range(len(seq) - 1):
                ci, ni_t, vi = planes[i]
                cj, nj_t, vj = planes[i + 1]
                ang.append(float(np.degrees(np.arccos(max(-1.0, min(1.0,
                           abs(float(n_impl[i] @ ni_t))))))))
                for tag, rows, (na, nb) in (("true", true_rows, (ni_t, nj_t)),
                                            ("impl", impl_rows, (n_impl[i], n_impl[i + 1]))):
                    a, b = na.copy(), nb.copy()
                    if float(a @ b) < 0:
                        b = -b
                    nm = a + b
                    nm = nm / np.linalg.norm(nm) if np.linalg.norm(nm) > 1e-9 else a
                    dc = beads[i + 1, 2] - beads[i, 2]
                    rows["nb_dist"].append(float(np.linalg.norm(dc)))
                    rows["rise"].append(float(dc @ nm))
                    rows["theta"].append(float(np.degrees(np.arccos(max(-1.0, min(1.0, float(a @ b)))))))
                    wi = vi - nm * float(vi @ nm)
                    wj = vj - nm * float(vj @ nm)
                    wi = wi / np.linalg.norm(wi) if np.linalg.norm(wi) > 1e-9 else wi
                    wj = wj / np.linalg.norm(wj) if np.linalg.norm(wj) > 1e-9 else wj
                    _cr = float(np.cross(wi, wj) @ nm)
                    _dt = max(-1.0, min(1.0, float(wi @ wj)))
                    rows["twist"].append(abs(float(np.degrees(np.arctan2(_cr, _dt)))))
            done += 1
        print("fragments: %d; implied-plane normal vs RING-plane normal: mean %.1f deg, median %.1f deg"
              % (done, np.mean(ang), np.median(ang)), flush=True)
        for tag, rows in (("ring atoms", true_rows), ("from beads", impl_rows)):
            for k in ("nb_dist", "rise", "theta", "twist"):
                v = np.asarray(rows[k], dtype=float)
                v = v[np.isfinite(v)]
                print("  %-10s %-8s n=%5d  mean %8.4f  sd %7.4f  p5 %8.4f  p50 %8.4f  p95 %8.4f"
                      % (tag, k, v.size, v.mean(), v.std(), np.percentile(v, 5),
                         np.percentile(v, 50), np.percentile(v, 95)), flush=True)
        np.savez(OUT / "_base_coords_planes.npz",
                 **{f"ring_{k}": np.asarray(true_rows[k]) for k in true_rows},
                 **{f"impl_{k}": np.asarray(impl_rows[k]) for k in impl_rows},
                 implied_vs_ring_angle=np.asarray(ang))
        print("wrote results/plan_c/_base_coords_planes.npz", flush=True)
        return

    if len(sys.argv) > 2 and sys.argv[1] == "--target-only":
        n_frag = int(sys.argv[2])
        print("=== TARGET: %d crystal fragments + 2OIU, bead-only coordinates ===" % n_frag, flush=True)
        tgt_all, proxy, done = target_samples(n_frag, want="all")
        tgt_steps, _p2, _d2 = target_samples(n_frag, want="steps")
        print("fragments used: %d; triangle-normal vs TRUE base-plane normal: mean %.1f deg, median %.1f"
              % (done, proxy.mean(), np.median(proxy)), flush=True)
        report("all pairs", tgt_all)
        report("helical", tgt_steps)
        np.savez(OUT / "_base_coords_two_sides.npz",
                 **{f"target_{k}": tgt_all[k] for k in COORDS},
                 **{f"target_steps_{k}": tgt_steps[k] for k in COORDS},
                 proxy_angle=proxy)
        print("wrote results/plan_c/_base_coords_two_sides.npz", flush=True)
        return

    n_frag = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    print("=== TARGET: %d crystal fragments + 2OIU, bead-only coordinates ===" % n_frag, flush=True)
    tgt_all, proxy, done = target_samples(n_frag, want="all")
    tgt_steps, _p2, _d2 = target_samples(n_frag, want="steps")
    print("fragments used: %d; triangle-normal vs TRUE base-plane normal: mean %.1f deg, median %.1f deg"
          % (done, proxy.mean(), np.median(proxy)), flush=True)
    report("all pairs", tgt_all)
    report("helical", tgt_steps)

    seq, residues, pP, ch = M.parse_pdb(REPO / "artifacts" / "2oiu" / "2OIU.pdb")
    beads2 = np.stack([[r["atoms"]["P"], r["atoms"]["C4'"], r["atoms"][GLY[r["base"]]]]
                       for r in residues]) / 10.0
    f2 = frames_from_beads(beads2)
    d2 = {k: [] for k in COORDS}
    for i in range(len(seq) - 1):
        a, b, th, tw = pair_coords(f2, i, i + 1)
        d2["nb_dist"].append(a)
        d2["cc_dist"].append(float(np.linalg.norm(beads2[i + 1, 1] - beads2[i, 1])))
        d2["rise"].append(b)
        d2["theta"].append(th)
        d2["twist"].append(tw)
    report("2OIU", {k: np.asarray(v) for k, v in d2.items()})

    print("\n=== SAMPLED: production tables, the loader's own 3-bead chains, 5000 steps ===", flush=True)
    frame_list = sampled_frames()
    smp = coords_from_frame_list(frame_list)
    report("sampled", smp)

    print("\n=== how far apart are they (total variation on a shared grid) ===", flush=True)
    grid = {"nb_dist": (0.3, 1.6), "cc_dist": (0.3, 1.6), "rise": (-1.0, 1.0),
            "theta": (0.0, 180.0), "twist": (0.0, 180.0)}
    summary = {}
    for k in COORDS:
        tv, ha, hb = tv_distance(tgt_all[k], smp[k], *grid[k])
        summary[k] = {"tv": tv, "target_mean": float(np.nanmean(tgt_all[k])),
                      "sampled_mean": float(np.nanmean(smp[k])),
                      "target_sd": float(np.nanstd(tgt_all[k])),
                      "sampled_sd": float(np.nanstd(smp[k]))}
        print("  %-10s TV %.3f   target mean %8.4f (sd %7.4f)   sampled mean %8.4f (sd %7.4f)"
              % (k, tv, summary[k]["target_mean"], summary[k]["target_sd"],
                 summary[k]["sampled_mean"], summary[k]["sampled_sd"]), flush=True)

    np.savez(OUT / "_base_coords_two_sides.npz",
             **{f"target_{k}": tgt_all[k] for k in COORDS},
             **{f"target_steps_{k}": tgt_steps[k] for k in COORDS},
             **{f"sampled_{k}": smp[k] for k in COORDS},
             proxy_angle=proxy)
    (OUT / "_base_coords_two_sides.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwrote results/plan_c/_base_coords_two_sides.npz and .json", flush=True)


if __name__ == "__main__":
    main()
