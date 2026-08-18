"""
Post-hoc Pair Repair (PPR) v3 - with backbone constraints.
Forces: backbone bonds + pair springs + anchor springs.
"""

import RNA
import numpy as np
import time
import json

from openmm.app import PDBFile, Simulation
import openmm as mm
import openmm.unit as u

PDB_PATH = r'D:\weixin\xwechat_files\wxid_6detf9si8jb622_4946\msg\file\2026-08\isrnaclong_final_allatom_v2(1).pdb'
OUT_PATH = r'C:\Users\LENOVO\Desktop\torusfold-paper\repaired_v3.pdb'

THRESHOLD = 15.0
TARGET = 10.0
MAX_ROUNDS = 5
MIN_ITERS = 5000
BB_K = 50000.0       # backbone bond strength (kJ/mol/nm^2)
BB_R0 = 5.9          # ideal P-P bond (Angstrom) -> 0.59 nm
SPRING_K_INIT = 5000.0  # pair spring initial
ANCHOR_K = 1000.0       # anchor for satisfied pairs

print("=" * 60)
print("POST-HOC PAIR REPAIR (PPR) v3 - backbone-aware")
print("=" * 60)

# Load PDB
print("\n[1] Loading PDB...")
pdb = PDBFile(PDB_PATH)
n_atoms = sum(1 for _ in pdb.topology.atoms())
print(f"  Atoms: {n_atoms}")

# Build residue -> P atom map + sequence
res_p = {}
seq = []
seen = set()
for atom in pdb.topology.atoms():
    if atom.name.strip() == 'P':
        res_p[atom.residue.index] = atom.index
    rid = (atom.residue.chain.id, atom.residue.index)
    if rid not in seen:
        seen.add(rid)
        rn = atom.residue.name.strip()
        m = {'A':'A','U':'U','G':'G','C':'C','RA':'A','RU':'U','RG':'G','RC':'C',
             'ADE':'A','URA':'U','GUA':'G','CYT':'C'}
        seq.append(m.get(rn, 'N'))

sequence = ''.join(seq)
n_res = len(seq)
print(f"  Residues: {n_res}")

# ViennaRNA
print("\n[2] ViennaRNA (circ)...")
md = RNA.md(); md.circ = 1
fc = RNA.fold_compound(sequence, md)
(ss, mfe) = fc.mfe()
print(f"  MFE: {mfe:.2f}")

pairs = []
stack = []
for i, c in enumerate(ss):
    if c == '(':
        stack.append(i)
    elif c == ')':
        if stack:
            pairs.append((stack.pop(), i))
print(f"  Pairs: {len(pairs)}")

# Initial distances
pos_list = pdb.positions.value_in_unit(u.nanometers)
pos_A = np.array([[p.x, p.y, p.z] for p in pos_list]) * 10.0

def compute_pair_dists(positions_A):
    dists = []
    for (i, j) in pairs:
        if i in res_p and j in res_p:
            d = np.linalg.norm(positions_A[res_p[i]] - positions_A[res_p[j]])
            dists.append((i, j, d))
    return dists

def compute_bond_stats(positions_A):
    bonds = []
    for ri in range(n_res - 1):
        if ri in res_p and ri+1 in res_p:
            d = np.linalg.norm(positions_A[res_p[ri]] - positions_A[res_p[ri+1]])
            bonds.append(d)
    # BSJ bond
    if 0 in res_p and n_res-1 in res_p:
        bonds.append(np.linalg.norm(positions_A[res_p[n_res-1]] - positions_A[res_p[0]]))
    bonds = np.array(bonds)
    return bonds

dists = compute_pair_dists(pos_A)
bonds = compute_bond_stats(pos_A)

sat_init = sum(1 for _, _, d in dists if d < THRESHOLD)
print(f"\n  Initial pair sat: {sat_init}/{len(dists)} ({sat_init/len(dists)*100:.1f}%)")
print(f"  Initial bond RMSD: {np.sqrt(np.mean((bonds - BB_R0)**2)):.4f} A")
print(f"  Initial BSJ: {np.linalg.norm(pos_A[res_p[n_res-1]] - pos_A[res_p[0]]):.3f} A")

# === Build system ===
print("\n[3] Building constrained system...")
current_positions = pdb.positions
spring_k = SPRING_K_INIT

for rnd in range(MAX_ROUNDS):
    unsat = [(i, j, d) for i, j, d in dists if d >= THRESHOLD]

    print(f"\n  --- Round {rnd+1}/{MAX_ROUNDS} ---")
    print(f"  Unsatisfied pairs: {len(unsat)}, spring_k={spring_k:.0f}")

    if len(unsat) == 0:
        print("  All satisfied!")
        break

    # Build system from scratch each round
    system = mm.System()
    for _ in range(n_atoms):
        system.addParticle(12.0)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(1000, 0, 0), mm.Vec3(0, 1000, 0), mm.Vec3(0, 0, 1000)
    )

    # Force 1: Backbone bonds (P-P consecutive)
    bb_force = mm.HarmonicBondForce()
    for ri in range(n_res - 1):
        if ri in res_p and ri+1 in res_p:
            bb_force.addBond(res_p[ri], res_p[ri+1], BB_R0/10.0, BB_K)
    # BSJ bond
    if 0 in res_p and n_res-1 in res_p:
        bb_force.addBond(res_p[n_res-1], res_p[0], BB_R0/10.0, BB_K)
    system.addForce(bb_force)

    # Force 2: Pair springs (pull unsatisfied pairs)
    pair_force = mm.HarmonicBondForce()
    for (i, j, d) in unsat:
        if i in res_p and j in res_p:
            pair_force.addBond(res_p[i], res_p[j], TARGET/10.0, spring_k)

    # Force 3: Anchor springs (preserve satisfied pairs at current distance)
    for (i, j, d) in dists:
        if d < THRESHOLD and i in res_p and j in res_p:
            pair_force.addBond(res_p[i], res_p[j], d/10.0, ANCHOR_K)
    system.addForce(pair_force)

    n_pull = sum(1 for _ in unsat if _[0] in res_p and _[1] in res_p)
    n_anchor = sum(1 for i,j,d in dists if d < THRESHOLD and i in res_p and j in res_p)
    print(f"  Forces: {n_res} backbone bonds + {n_pull} pull + {n_anchor} anchor")

    # Simulate
    integrator = mm.LangevinIntegrator(300*u.kelvin, 1.0/u.picosecond, 0.002*u.picoseconds)
    platform = mm.Platform.getPlatformByName('CPU')
    sim = Simulation(pdb.topology, system, integrator, platform)
    sim.context.setPositions(current_positions)

    t0 = time.time()
    sim.minimizeEnergy(maxIterations=MIN_ITERS)
    elapsed = time.time() - t0
    print(f"  Minimization: {elapsed:.1f}s")

    # Get positions
    state = sim.context.getState(getPositions=True)
    current_positions = state.getPositions()
    pos_new_A = np.array([[p.x, p.y, p.z] for p in current_positions.value_in_unit(u.nanometers)]) * 10.0

    # Recompute
    dists = compute_pair_dists(pos_new_A)
    bonds = compute_bond_stats(pos_new_A)

    sat = sum(1 for _, _, d in dists if d < THRESHOLD)
    bond_rmsd = np.sqrt(np.mean((bonds - BB_R0)**2))
    bsj = np.linalg.norm(pos_new_A[res_p[n_res-1]] - pos_new_A[res_p[0]])
    global_rmsd = np.sqrt(np.mean(np.sum((pos_A - pos_new_A)**2, axis=1)))

    print(f"  Pair sat: {sat}/{len(dists)} ({sat/len(dists)*100:.1f}%)")
    print(f"  Bond RMSD: {bond_rmsd:.4f} A")
    print(f"  BSJ closure: {bsj:.3f} A")
    print(f"  Global RMSD: {global_rmsd:.2f} A")

    spring_k *= 1.5

# Write output
print(f"\n[4] Writing...")
with open(OUT_PATH, 'w') as f:
    PDBFile.writeFile(pdb.topology, current_positions, f)
print(f"  -> {OUT_PATH}")

# Final
pos_final = np.array([[p.x, p.y, p.z] for p in current_positions.value_in_unit(u.nanometers)]) * 10.0
sat_f = sum(1 for _, _, d in dists if d < 15.0)
p15 = sum(1 for _, _, d in dists if 15.0 <= d < 30.0)
p30 = sum(1 for _, _, d in dists if d >= 30.0)
bonds_f = compute_bond_stats(pos_final)
bond_rmsd_f = np.sqrt(np.mean((bonds_f - BB_R0)**2))
bsj_f = np.linalg.norm(pos_final[res_p[n_res-1]] - pos_final[res_p[0]])
rmsd_f = np.sqrt(np.mean(np.sum((pos_A - pos_final)**2, axis=1)))

print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
print(f"  {'Metric':<25} {'Before':>10} {'After':>10}")
print(f"  {'-'*45}")
print(f"  {'Pair sat (<15A)':<25} {sat_init/len(dists)*100:>9.1f}% {sat_f/len(dists)*100:>9.1f}%")
print(f"  {'Bond RMSD (A)':<25} {np.sqrt(np.mean((compute_bond_stats(pos_A)-BB_R0)**2)):>10.4f} {bond_rmsd_f:>10.4f}")
print(f"  {'BSJ closure (A)':<25} {np.linalg.norm(pos_A[res_p[n_res-1]]-pos_A[res_p[0]]):>10.3f} {bsj_f:>10.3f}")
print(f"  {'Global RMSD (A)':<25} {'--':>10} {rmsd_f:>10.2f}")
print(f"  {'Satisfied':<25} {sat_init:>10} {sat_f:>10}")
print(f"  {'Partial (15-30A)':<25} {'--':>10} {p15:>10}")
print(f"  {'Unsatisfied (>30A)':<25} {'--':>10} {p30:>10}")

results = {
    'total_pairs': len(dists),
    'initial_satisfaction': sat_init / len(dists),
    'final_satisfaction': sat_f / len(dists),
    'bond_rmsd_before': float(np.sqrt(np.mean((compute_bond_stats(pos_A)-BB_R0)**2))),
    'bond_rmsd_after': float(bond_rmsd_f),
    'bsj_before': float(np.linalg.norm(pos_A[res_p[n_res-1]]-pos_A[res_p[0]])),
    'bsj_after': float(bsj_f),
    'global_rmsd': float(rmsd_f),
    'satisfied': sat_f,
    'partial_15_30': p15,
    'unsatisfied_gt30': p30,
}
rp = OUT_PATH.replace('.pdb', '_results.json')
with open(rp, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results: {rp}")
