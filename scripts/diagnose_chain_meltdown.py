"""Why do 12 chains own 99.48 percent of the round-0 bb_bond excursions?

Measured on results/ibi_full/round0.json (867 chains, 8 replicas' worth of samples pooled per
chain, 65 ps each): the pooled bb_bond out-of-support fraction is 1.0204 percent, above the 1
percent that ibi_bonded.plan_update allows, so the bond's update was REFUSED and the round
iterated two of its three coordinates. The excursion is not a property of the field:

    12 of 867 chains carry 99.48 percent of every out-of-support count
    drop those 12 and the pooled fraction is 0.0027 percent
    41.2 percent of chains never leave the support at all
    L <= 60: 0.0015 percent        L 60-120: 1.40    L 120-300: 2.38    L > 300: 0.54

The twelve are 9JHD_3(60), 8D8K_31(114/162/164), 7OYB_1(138), 7QVP_7(119/713), 9AXT_1(195),
9BH5_7(192), 8I9W_2(233), 6XU7_62(595), 9J9I_1(440), and their joint J is 1.31-2.05 against a
pool median of 0.17: those runs are not "a bond a little too wide", they are chains that come
apart.

TWO EXPLANATIONS WERE TESTED AND BOTH ARE DEAD, which is why this script measures rather than
asserts:

  * the numbering-gap pseudo-bond the loader docstring warns about (adjacent-numbered residues
    that a cryo-EM entry still has after the numbering split, so the "bond" is a multi-nm gap).
    The database DOES contain 17 such chains -- worst 7R6Q_10 at 7.224 nm -- but they are NOT
    these twelve. All twelve load clean: canonical ACGU only, adjacent P-P 0.711-0.766 nm max,
    P-C4' median 0.383-0.391 against 0.390, C4'-N median 0.337-0.347 against 0.335.
  * a modified-nucleotide base atom: every one of the twelve is ACGU, so the N9/N1 choice never
    differs.

So this script takes the twelve (or a length-matched sample of them) and a length-matched CLEAN
control for each, and runs the loop's own sampling core on them with the loop's own round-0
table, collecting three things the aggregated round json cannot show:

  1. per-block bb_bond outside fraction -- WHEN the chain leaves the support, not just that it
     did. Blocks are 5 ps each over the full 65 ps window, burn 0, so the onset is visible.
  2. per-block bb_bond sigma_sim/sigma_ref -- a chain that is merely wide shows a constant
     ratio; a chain that comes apart shows it climbing.
  3. the closest bead-bead approach per sampled frame (run_round already records it, and the
     whole run's minimum is the cheapest signature of a blown-up chain) plus max|F| at the
     deposited geometry, which is the other candidate: an initial clash the 5000 kJ/mol/nm cap
     turns into a constant force and a heating source.

The fourth argument is the entry-relaxation arm: 0 reproduces the meltdown exactly as the
867-chain round 0 sampled it, nonzero descends the injected Hamiltonian first
(ibi_core.relax_positions). Both arms share the seed and the protocol, so the difference between
them is the relaxation and nothing else -- which is the only way to say whether the fix works.

Run: python scripts/diagnose_chain_meltdown.py [nsteps] [blocks] [n_pairs] [relax_steps]
     python scripts/diagnose_chain_meltdown.py 32500 13 3 0      # as round 0 saw it
     python scripts/diagnose_chain_meltdown.py 32500 13 3 1500   # with the entry relaxation
     python scripts/diagnose_chain_meltdown.py 6000 6 1 1500     # one pair, quick

MELTDOWN_SERIAL=1 runs the chains in this process instead of an mp.Pool. A sandbox that denies
named pipes cannot build a Pool at all, and a one-chain arm should not need one.
"""
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B              # noqa: E402
import cg_potentials as P                 # noqa: E402
import ibi_core as IC                     # noqa: E402
import torusfold.scheme2.torch_cgsim as C  # noqa: E402

# The run under diagnosis. Round 0 sampled under tables_r0.npz, which is the reference table set
# written back out by ibi_loop.write_round_file -- so this reproduces round 0's Hamiltonian
# exactly, wall included.
RUN_DIR = REPO / "results" / "ibi_full"
ROUND_NPZ = RUN_DIR / "tables_r0.npz"
ROUND_JSON = RUN_DIR / "round0.json"
ROUND_LOG = RUN_DIR / "run.log"

WALL_K = 2000.0        # ibi_loop.WALL_K for the 867-chain run
FRICTION = 1.0         # ibi_loop's friction for the 867-chain run
FORCE_CAP = 5000.0
SEED_BASE = 20260914   # ibi_loop.main's seed; chain i therefore ran at SEED_BASE + i
DT_PS = 0.002
STRIDE = 5

NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 32500
BLOCKS = int(sys.argv[2]) if len(sys.argv) > 2 else 13
N_PAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
# Entry-relaxation arm: 0 reproduces the meltdown as the 867-chain round 0 sampled it, nonzero
# descends the injected Hamiltonian first (ibi_core.relax_positions). The two arms share a seed
# and a protocol, so the difference between them is the relaxation and nothing else.
RELAX = int(sys.argv[4]) if len(sys.argv) > 4 else 0
# MELTDOWN_SERIAL=1 runs the tasks in this process instead of an mp.Pool. It exists because a
# sandbox that denies named pipes cannot build a Pool at all (WinError 5 in
# multiprocessing.connection.Pipe), and a one-chain arm should not need a process pool anyway.
SERIAL = os.environ.get("MELTDOWN_SERIAL", "") == "1"


def _round0():
    """The 867 chains as (index, name, L, J, bb_bond out-of-support fraction).

    Names come from the log's per-chain lines and the counts from the round json; the two are
    joined by INDEX, and the join is checked rather than assumed. A silent misalignment here
    would compare the wrong chain's geometry against the wrong chain's excursion.
    """
    d = json.loads(ROUND_JSON.read_text(encoding="utf-8"))
    ps = d["per_structure"]
    names = []
    for ln in ROUND_LOG.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s+(\S+)\s+L=\s*(\d+)\s+J\s+([\d.]+)\s", ln)
        if m:
            names.append((m.group(1), int(m.group(2)), float(m.group(3))))
    if len(names) != len(ps):
        raise SystemExit(f"{ROUND_LOG} has {len(names)} per-chain lines, {ROUND_JSON} has "
                         f"{len(ps)} entries; the join would be wrong and nothing here would "
                         f"notice")
    out = []
    for i, (name, L, J) in enumerate(names):
        p = ps[i]
        if p["residues"] != L:
            raise SystemExit(f"chain {i}: log says {name} L={L}, json says L={p['residues']}")
        n = max(1, p["n_total"]["bb_bond"])
        out.append({"idx": i, "name": name, "L": L, "J": J,
                    "outside": p["n_outside"]["bb_bond"] / n})
    return out


def _pick(chains, n_pairs):
    """n_pairs bad chains (smallest first) and a length-matched clean control for each."""
    bad = sorted([c for c in chains if c["outside"] > 0.10], key=lambda c: c["L"])
    clean = [c for c in chains if c["outside"] == 0.0]
    pairs = []
    for b in bad[:n_pairs]:
        ctl = min(clean, key=lambda c: (abs(c["L"] - b["L"]), c["J"]))
        pairs.append((b, ctl))
    return pairs


def _geometry(pos_nm, pairs_ij):
    """Deposited-geometry numbers that would explain a meltdown before any dynamics runs."""
    P_atoms = pos_nm[:, 0::3, :]
    L = P_atoms.shape[1]
    d = np.linalg.norm(P_atoms[0, 1:] - P_atoms[0, :-1], axis=1)
    beads = pos_nm.reshape(-1, 3)
    dd = np.linalg.norm(beads[:, None, :] - beads[None, :, :], axis=-1)
    idx = np.arange(len(beads))
    # |i - j| < 3 is a bonded or angle neighbour and is supposed to be short
    far = np.abs(idx[:, None] - idx[None, :]) >= 3
    np.fill_diagonal(far, False)
    nb_min = float(dd[far].min()) if far.any() else float("nan")
    return {"pp_max": float(d.max()), "pp_min": float(d.min()), "nonbonded_min": nb_min,
            "paired_min": float(dd[pairs_ij[:, 0], pairs_ij[:, 1]].min()) if len(pairs_ij) else float("nan")}


def _initial_force(pos_nm, pairs_ij):
    """max|F| at the deposited geometry, under round 0's own Hamiltonian (with the wall)."""
    P.use_table_file(str(ROUND_NPZ))
    pots = []
    for coord in ("bb_bond", "angle", "dihedral"):
        spec = P.resolve_spec(f"table_wall:{WALL_K:g}", coord) if coord == "bb_bond" \
            else P.resolve_spec("table", coord)
        pots.append((coord, spec, P.make_potential(coord, spec)))
    pot_kw = P.potential_kwargs(pots)
    L = pos_nm.shape[1] // 3
    x = torch.tensor(pos_nm, dtype=torch.float64)
    ij = torch.tensor(pairs_ij, dtype=torch.long).reshape(-1, 2)
    pw = torch.ones(len(ij), dtype=torch.float32)
    with torch.no_grad():
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(x)
        e, f = C.cg_energy_forces(x, ij, pw, cell_list=cl, force_cap=FORCE_CAP, **pot_kw)
    return float(torch.linalg.norm(f.reshape(-1, 3), dim=-1).max()), float(e.reshape(-1)[0]), pot_kw


def _one(task):
    """One chain: deposit geometry, then the loop's own 65 ps sampling core on it."""
    label, seed, pos_np, pairs, nsteps, blocks = task
    torch.set_num_threads(1)
    t0 = time.time()
    L = pos_np.shape[0]
    ij_np = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    geo = _geometry(pos_np.reshape(1, 3 * L, 3), ij_np)
    f0, e0, pot_kw = _initial_force(pos_np.reshape(1, 3 * L, 3), ij_np)
    tab = IC.load_tables(str(ROUND_NPZ))
    pos = torch.tensor(pos_np.reshape(1, 3 * L, 3), dtype=torch.float64)
    vel = torch.zeros_like(pos)
    temps = torch.full((1,), 300.0, dtype=torch.float64)
    ij = torch.tensor(ij_np, dtype=torch.long)
    pw = torch.ones(len(ij_np), dtype=torch.float32)
    res = IC.run_round(pos=pos, vel=vel, ij=ij, pw=pw, temps=temps, tab=tab,
                       nsteps=nsteps, burn=0, stride=STRIDE, blocks=blocks, friction=FRICTION,
                       force_cap=FORCE_CAP, pot_kw=pot_kw, seed=seed, nrep=1,
                       progress=False, collect_values=True, relax=RELAX,
                       constraints=C.make_intra_constraints(L),
                       log=lambda *a, **k: None)
    vals = res.values["bb_bond"].reshape(blocks, -1) if res.values is not None else None
    per_block = []
    for b in range(blocks):
        n_frames = res.b_frames[b]
        n_obs = n_frames * max(L - 1, 1)
        in_support = int(res.b_counts[b]["bb_bond"].sum())
        ratio = IC.sim_ref_ratio(res.b_acc[b], "bb_bond", tab)
        obs = vals[b] if vals is not None and vals[b].size else np.array([np.nan])
        per_block.append({"block": b, "ps": (b + 1) * (nsteps / blocks) * DT_PS,
                          "frames": n_frames, "obs": n_obs,
                          "outside_frac": (1.0 - in_support / n_obs) if n_obs else float("nan"),
                          "sigma_ratio": ratio,
                          "min": float(obs.min()), "max": float(obs.max()),
                          "median": float(np.median(obs)), "mean": float(obs.mean())})
    return {"label": label, "L": L, "seed": seed, "seconds": time.time() - t0,
            "geometry": geo, "max_force_0": f0, "energy_0": e0, "relax": res.relax,
            "clash_min": min(res.clash_min) if res.clash_min else float("nan"),
            "steps_per_s": res.steps_per_s,
            "n_outside_total": int(res.n_outside["bb_bond"]),
            "n_total": int(res.n_total["bb_bond"]),
            "per_block": per_block}


def _report(r):
    """One run's block table. Shared by the serial and pooled paths so the two cannot diverge."""
    print(f"== {r['label']}")
    g = r["geometry"]
    print(f"   deposited: P-P {g['pp_min']:.3f}-{g['pp_max']:.3f} nm   "
          f"non-bonded min {g['nonbonded_min']:.3f}   paired min {g['paired_min']:.3f}   "
          f"E {r['energy_0']:.1f} kJ/mol   max|F| {r['max_force_0']:.1f}")
    x = r.get("relax")
    if x:
        print(f"   relaxed first: {x['accepted']} accepted / {x['rejected']} rejected of "
              f"{x['steps']} steps, {x['evals']} field evaluations   "
              f"E {x['energy_start']:.1f} -> {x['energy_end']:.1f} kJ/mol   "
              f"max|F| {x['max_force_start']:.1f} -> {x['max_force_end']:.1f}"
              + ("   LEFT THE CAP" if x["hit_cap"] and x["left_cap"] else ""))
    print(f"   sampled {NSTEPS} steps in {r['seconds']:.0f} s "
          f"({r['steps_per_s']:.3f} steps/s), whole-run outside "
          f"{r['n_outside_total']}/{r['n_total']} = "
          f"{100 * r['n_outside_total'] / max(1, r['n_total']):.4f}%, "
          f"closest bead approach {r['clash_min']:.3f} nm")
    print(f"   {'block':>5} {'ps':>6} {'outside%':>9} {'sigma_sim/ref':>14} "
          f"{'bb min':>8} {'bb median':>10} {'bb max':>8}")
    for b in r["per_block"]:
        print(f"   {b['block']:>5} {b['ps']:>6.1f} {100 * b['outside_frac']:>8.3f}% "
              f"{b['sigma_ratio']:>14.3f} {b['min']:>8.3f} {b['median']:>10.3f} "
              f"{b['max']:>8.3f}")
    print()


def main():
    print(f"round 0: {ROUND_NPZ}")
    chains = _round0()
    pairs = _pick(chains, N_PAIRS)
    if not pairs:
        raise SystemExit("no chain in round0.json is above 10 percent out of support; nothing to "
                         "diagnose, and re-running this would only re-measure a clean pool")
    pool = B.load_structures(limit=5000)
    tasks = []
    for bad, ctl in pairs:
        for role, c in (("BAD", bad), ("control", ctl)):
            if c["idx"] >= len(pool):
                raise SystemExit(f"chain {c['idx']} is out of the loaded pool ({len(pool)})")
            s = pool[c["idx"]]
            tasks.append((f"{role} {c['name']} L={c['L']} round0_out={100 * c['outside']:.2f}% "
                          f"J={c['J']:.4f}", SEED_BASE + c["idx"],
                          np.asarray(s["pos"], dtype=np.float64), s["pairs"], NSTEPS, BLOCKS))
    print(f"{len(tasks)} runs: {NSTEPS} steps = {NSTEPS * DT_PS:.0f} ps, burn 0, {BLOCKS} blocks "
          f"of {NSTEPS / BLOCKS * DT_PS:.1f} ps, friction {FRICTION}, wall {WALL_K:g}, "
          f"constraints ON, entry relaxation {RELAX} steps"
          + ("   [serial]\n" if SERIAL else "\n"))
    if SERIAL or len(tasks) == 1:
        for t in tasks:
            _report(_one(t))
    else:
        with mp.get_context("spawn").Pool(processes=min(len(tasks), os.cpu_count() or 1)) as procs:
            for r in procs.imap_unordered(_one, tasks):
                _report(r)
    print("done")


if __name__ == "__main__":
    main()
