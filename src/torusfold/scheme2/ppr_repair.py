"""
PPR (Post-hoc Pair Repair) - Base pair hydrogen bond repair.

Fixes mismatches between ViennaRNA pairing predictions and 3D structure:
pulls hydrogen bond donor/acceptor atoms to form correct pairs.
"""
import numpy as np
import time
from typing import List, Tuple, Optional


WC_HB_PAIRS = {
    ("A", "U"): [("N1", "N3"), ("N6", "O4")],
    ("U", "A"): [("N3", "N1"), ("O4", "N6")],
    ("G", "C"): [("N1", "N3"), ("O6", "N4"), ("N2", "O2")],
    ("C", "G"): [("N3", "N1"), ("N4", "O6"), ("O2", "N2")],
    ("G", "U"): [("N1", "N3"), ("O6", "N3")],
    ("U", "G"): [("N3", "N1"), ("N3", "O6")],
}

HB_TARGET = 2.95
HB_SAT_CUTOFF = 3.6
PPR_BB_K = 50000.0
PPR_BB_R0 = 5.9
PPR_HB_K_INIT = 5000.0
PPR_ANCHOR_K = 1000.0


def get_mfe_pairs(sequence):
    import RNA
    md = RNA.md(); md.circ = 1
    fc = RNA.fold_compound(sequence, md)
    ss, mfe = fc.mfe()
    pairs = []; stack = []
    for i, c in enumerate(ss):
        if c == "(": stack.append(i)
        elif c == ")" and stack: pairs.append((stack.pop(), i))
    return pairs, mfe, ss


def detect_hb_pairs(positions_A, res_atoms, keys, sequence, min_sep=4, hb_cut=3.6):
    all_pairs_set = set(WC_HB_PAIRS.keys())
    candidates = []
    n = len(keys)
    for i in range(n):
        ri = res_atoms.get(keys[i], {})
        for j in range(i + min_sep, n):
            rj = res_atoms.get(keys[j], {})
            ai, aj = sequence[i], sequence[j]
            if (ai, aj) not in all_pairs_set: continue
            best = 999.0
            for a1, a2 in WC_HB_PAIRS.get((ai, aj), []):
                if a1 in ri and a2 in rj:
                    d = np.linalg.norm(ri[a1] - rj[a2])
                    if d < best: best = d
                if a2 in ri and a1 in rj:
                    d = np.linalg.norm(ri[a2] - rj[a1])
                    if d < best: best = d
            if best < hb_cut: candidates.append((i, j, best))
    candidates.sort(key=lambda x: x[2])
    used = set(); pairs = []
    for gi, gj, dist in candidates:
        if gi in used or gj in used: continue
        pairs.append((gi, gj, dist)); used.add(gi); used.add(gj)
    return pairs


def ppr_repair(pdb_path, output_path, sequence, pairs=None,
               max_rounds=5, hb_k_init=PPR_HB_K_INIT, verbose=True):
    from openmm.app import PDBFile, Simulation
    import openmm as mm
    import openmm.unit as u

    if verbose:
        print(f"\n{'='*60}")
        print("PPR (Post-hoc Pair Repair) - hydrogen bond repair")
        print(f"{'='*60}")

    pdb = PDBFile(pdb_path)
    n_atoms = sum(1 for _ in pdb.topology.atoms())
    if verbose: print(f"  Atoms: {n_atoms}")

    res_p = {}
    for atom in pdb.topology.atoms():
        if atom.name.strip() == "P": res_p[atom.residue.index] = atom.index

    pos_list = pdb.positions.value_in_unit(u.nanometers)
    pos_A = np.array([[p.x, p.y, p.z] for p in pos_list]) * 10.0
    L = len(res_p)

    if pairs is None:
        pairs, mfe, ss = get_mfe_pairs(sequence)
        if verbose: print(f"  ViennaRNA MFE: {mfe:.2f}, {len(pairs)} pairs")

    # Initial H-bond detection
    res_atoms = {}
    for atom in pdb.topology.atoms():
        ri = atom.residue.index; an = atom.name.strip()
        if ri not in res_atoms: res_atoms[ri] = {}
        res_atoms[ri][an] = pos_A[atom.index]

    hb_init = detect_hb_pairs(pos_A, res_atoms, list(range(L)), sequence)
    hb_set = set((i, j) for i, j, _ in hb_init)
    n_sat_init = len(hb_init); n_total = len(pairs)
    n_mfe_in = sum(1 for p in pairs if (p[0], p[1]) in hb_set)

    if verbose:
        print(f"  Pairs: {n_total}")
        print(f"  H-bond satisfied (<{HB_SAT_CUTOFF}A): {n_sat_init}")
        print(f"  MFE pairs with correct H-bond: {n_mfe_in}/{n_total}")

    current_positions = pdb.positions; hb_k = hb_k_init

    for rnd in range(max_rounds):
        # Rebuild atom index map for current positions
        res_atoms_now = {}
        for atom in pdb.topology.atoms():
            ri = atom.residue.index; an = atom.name.strip()
            if ri not in res_atoms_now: res_atoms_now[ri] = {}
            res_atoms_now[ri][an] = pos_A[atom.index]

        pos_now = np.array([[p.x, p.y, p.z]
                             for p in current_positions.value_in_unit(u.nanometers)]) * 10.0
        hb_now = detect_hb_pairs(pos_now, res_atoms_now, list(range(L)), sequence)
        hb_set_now = set((i, j) for i, j, _ in hb_now)
        unsat = [(p[0], p[1]) for p in pairs if (p[0], p[1]) not in hb_set_now]

        if verbose:
            print(f"\n  Round {rnd+1}/{max_rounds}: unsat={len(unsat)}, hb_k={hb_k:.0f}")
        if not unsat:
            if verbose: print("  All MFE pairs satisfied!")
            break

        system = mm.System()
        for _ in range(n_atoms): system.addParticle(12.0)
        system.setDefaultPeriodicBoxVectors(mm.Vec3(1000,0,0), mm.Vec3(0,1000,0), mm.Vec3(0,0,1000))

        # Backbone bonds
        bf = mm.HarmonicBondForce()
        for ri in range(L-1):
            if ri in res_p and ri+1 in res_p:
                bf.addBond(res_p[ri], res_p[ri+1], PPR_BB_R0/10.0, PPR_BB_K)
        if 0 in res_p and L-1 in res_p:
            bf.addBond(res_p[L-1], res_p[0], PPR_BB_R0/10.0, PPR_BB_K)
        system.addForce(bf)

        # H-bond springs on donor/acceptor atoms
        hf = mm.CustomBondForce("0.5*k*(r-r0)^2")
        hf.addPerBondParameter("k"); hf.addPerBondParameter("r0")
        for i, j in unsat:
            ai, aj = sequence[i], sequence[j]
            for a1, a2 in WC_HB_PAIRS.get((ai, aj), []):
                idx1 = idx2 = None
                for atom in pdb.topology.atoms():
                    if atom.residue.index==i and atom.name.strip()==a1: idx1=atom.index
                    if atom.residue.index==j and atom.name.strip()==a2: idx2=atom.index
                    if idx1 is not None and idx2 is not None: break
                if idx1 is not None and idx2 is not None:
                    hf.addBond(idx1, idx2, [hb_k, HB_TARGET/10.0])
        system.addForce(hf)

        # Anchor satisfied pairs
        af = mm.CustomBondForce("0.5*k*(r-r0)^2")
        af.addPerBondParameter("k"); af.addPerBondParameter("r0")
        for i, j in list(hb_now)[:50]:
            ai, aj = sequence[i], sequence[j]
            for a1, a2 in WC_HB_PAIRS.get((ai, aj), []):
                idx1 = idx2 = None
                for atom in pdb.topology.atoms():
                    if atom.residue.index==i and atom.name.strip()==a1: idx1=atom.index
                    if atom.residue.index==j and atom.name.strip()==a2: idx2=atom.index
                    if idx1 is not None and idx2 is not None: break
                if idx1 is not None and idx2 is not None:
                    d0 = np.linalg.norm(pos_A[idx1]-pos_A[idx2])
                    af.addBond(idx1, idx2, [PPR_ANCHOR_K, d0/10.0])
        system.addForce(af)

        it = mm.LangevinIntegrator(300*u.kelvin, 1.0/u.picosecond, 0.002*u.picoseconds)
        sim = Simulation(pdb.topology, system, it, mm.Platform.getPlatformByName("CPU"))
        sim.context.setPositions(current_positions)
        t0=time.time(); sim.minimizeEnergy(maxIterations=5000)
        current_positions = sim.context.getState(getPositions=True).getPositions()
        hb_k *= 1.5
        if verbose: print(f"  Min: {time.time()-t0:.1f}s")

    with open(output_path, "w") as f:
        PDBFile.writeFile(pdb.topology, current_positions, f)

    # Final stats
    pos_final = np.array([[p.x, p.y, p.z]
                           for p in current_positions.value_in_unit(u.nanometers)]) * 10.0
    res_atoms_f = {}
    for atom in pdb.topology.atoms():
        ri = atom.residue.index; an = atom.name.strip()
        if ri not in res_atoms_f: res_atoms_f[ri] = {}
        res_atoms_f[ri][an] = pos_final[atom.index]
    hb_final = detect_hb_pairs(pos_final, res_atoms_f, list(range(L)), sequence)
    hb_set_f = set((i,j) for i,j,_ in hb_final)
    n_sat_f = sum(1 for p in pairs if (p[0], p[1]) in hb_set_f)
    if verbose:
        print(f"  H-bond before: {n_sat_init}/{n_total}")
        print(f"  H-bond after:  {n_sat_f}/{n_total}")
        print(f"  Improvement:   +{n_sat_f - n_sat_init}")
    return {"before": n_sat_init, "after": n_sat_f, "total": n_total, "output": output_path}
