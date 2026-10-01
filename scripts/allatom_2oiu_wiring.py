"""2OIU through the SHIPPED pipeline order: CG refine -> reconstruct -> all-atom -> marginals.

The first attempts fed the CRYSTAL P trace straight into the reconstruction and then into OpenMM, which
starts at 4.1e25 kJ/mol of atomic overlap and NaNs during annealing. The shipped path does not do that:
scripts/benchmark_2oiu.py runs the CG refiner (openmm_gpu_refine) FIRST, and only then is the structure
reconstructed and refined. So this follows it:

  1. 2OIU chain P -> sequence + crystal P coords -> ViennaRNA secondary structure and bpp pairs
  2. CG refinement (openmm_gpu_refine, 10000 steps + 4-replica REMD)
  3. reconstruction from the REFINED P trace (aform_from_template, base pairs anchored) minus the
     third phosphate oxygen the template adds to every residue
  4. the repo's own circular builder (internal templates, manual H) -> amber14-OL3 + OBC1
  5. short anneal + minimisation, then implicit-solvent MD at 300 K
  6. P-trace marginals against the tables the loop inverts
"""
import sys
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "output_2oiu_allatom"
OUT.mkdir(exist_ok=True)


def read_p_chain(path):
    from openmm.app import PDBFile
    pdb = PDBFile(str(path))
    seq_chars, p_coords = [], []
    for res in pdb.topology.residues():
        if res.chain.id != "P":
            continue
        base = {"RA": "A", "RU": "U", "RG": "G", "RC": "C"}.get(res.name.strip(), res.name.strip()[:1])
        if base not in "ACGU":
            continue
        seq_chars.append(base)
        for atom in res.atoms():
            if atom.name.strip() == "P":
                p = pdb.positions[atom.index]
                p_coords.append([p.x, p.y, p.z])
    return "".join(seq_chars), np.asarray(p_coords, dtype=float)


def main():
    import torch
    import ViennaRNA
    from openmm import unit, Platform, LangevinMiddleIntegrator
    from openmm.app import Simulation, NoCutoff, HBonds
    import boltzmann_bonded as B
    import ibi_bonded as I
    from torusfold.scheme2.aform_from_template import reconstruct_all_atom
    from torusfold.scheme2.amber_refine import _build_topology_and_modeller

    ps = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0

    seq, p_crystal_nm = read_p_chain(REPO / "artifacts" / "2oiu" / "2OIU.pdb")
    L = len(seq)
    print(f"2OIU chain P: {L} nt; crystal BSJ |P0 - P_last| = "
          f"{np.linalg.norm(p_crystal_nm[0] - p_crystal_nm[-1]):.2f} nm", flush=True)

    fc = ViennaRNA.fold_compound(seq)
    ss = fc.mfe()[0]
    fc.pf()
    bpp = fc.bpp()
    pairs = [(i - 1, j - 1, float(bpp[i][j])) for i in range(1, L + 1) for j in range(i + 1, L + 1)
             if bpp[i][j] > 0.1]
    print(f"ss {ss}\npairs {len(pairs)}", flush=True)

    # ---- 1. CG refinement, the step the shipped benchmark does first ----
    crystal_pdb = OUT / "crystal_p.pdb"
    with crystal_pdb.open("w", encoding="utf-8") as fh:
        fh.write("HEADER 2OIU crystal P coords (nm -> A)\n")
        for i in range(L):
            x, y, z = p_crystal_nm[i] * 10.0
            fh.write(f"ATOM  {i + 1:5d}  P   RA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}\n")
        fh.write("END\n")
    from torusfold.scheme2.openmm_gpu_refiner import openmm_gpu_refine
    print("CG refinement ...", flush=True)
    out_pdb, e_cg = openmm_gpu_refine(str(crystal_pdb), str(OUT), seq, ss, nstep=10000,
                                      remd_n_steps=5000, use_remd=True, remd_n_replicas=4,
                                      verbose=False)
    print(f"  CG refine -> {out_pdb}  E = {e_cg}", flush=True)
    from openmm.app import PDBFile
    ref = PDBFile(str(out_pdb))
    p_refined = []
    for res in ref.topology.residues():
        for atom in res.atoms():
            if atom.name.strip() == "P":
                q = ref.positions[atom.index]
                p_refined.append([q.x, q.y, q.z])
    p_refined = np.asarray(p_refined, dtype=float)[:L]
    print(f"  refined P trace: {p_refined.shape}, BSJ = "
          f"{np.linalg.norm(p_refined[0] - p_refined[-1]):.2f} nm", flush=True)

    # ---- 2. reconstruct from the REFINED trace ----
    st = reconstruct_all_atom(p_refined * 10.0, seq, pairs=pairs)
    keep, spans, idx = [], [], []
    for i, (a, b) in enumerate(st.residue_atom_spans):
        start = len(keep)
        local = {}
        for k in range(a, b):
            if st.atoms[k].atom_name == "OP3":      # the third non-bridging O the template adds
                continue
            local[st.atoms[k].atom_name] = len(keep)
            keep.append(st.atoms[k])
        spans.append((start, len(keep)))
        idx.append(local)
    st.atoms, st.residue_atom_spans, st.residue_atom_index = keep, spans, idx
    print(f"reconstructed {len(st.atoms)} atoms (OP3 dropped on every residue)", flush=True)

    modeller, ff, _ = _build_topology_and_modeller(st)
    topo = modeller.topology
    n_h = sum(1 for a in topo.atoms() if a.element is not None and a.element.symbol == "H")
    print(f"builder: {topo.getNumAtoms()} atoms, {n_h} H", flush=True)
    system = ff.createSystem(topo, nonbondedMethod=NoCutoff, constraints=HBonds,
                             ignoreExternalBonds=True)
    integ = LangevinMiddleIntegrator(300 * unit.kelvin, 5 / unit.picosecond,
                                     0.0005 * unit.picosecond)   # soft start: tiny timestep
    try:
        plat = Platform.getPlatformByName("OpenCL")
    except Exception:
        plat = Platform.getPlatformByName("CPU")
    sim = Simulation(topo, system, integ, plat)
    sim.context.setPositions(modeller.positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    print(f"start energy {e0:.3g} kJ/mol on {plat.getName()}", flush=True)
    # staged warm-up: 0.5 fs for 2000 steps, then 1 fs, then minimise
    sim.step(2000)
    print("  after 1 ps at 0.5 fs: E = "
          f"{sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.3g}",
          flush=True)
    sim.integrator.setStepSize(0.001 * unit.picosecond)
    sim.step(2000)
    sim.integrator.setStepSize(0.002 * unit.picosecond)
    sim.minimizeEnergy(maxIterations=2000)
    e = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    print(f"warmed + minimised: E = {e:.0f} kJ/mol", flush=True)

    p_idx = [a.index for a in topo.atoms() if a.name == "P"]
    print(f"P atoms in the topology: {len(p_idx)} (expected {L})", flush=True)
    every = int(1.0 / 0.002)
    frames = []
    for k in range(int(ps)):
        sim.step(every)
        frames.append(np.array(sim.context.getState(getPositions=True)
                               .getPositions().value_in_unit(unit.nanometer))[p_idx])
        if (k + 1) % 20 == 0:
            print(f"    ... {k + 1} ps", flush=True)
    print(f"MD {ps} ps, {len(frames)} frames", flush=True)

    ref_tabs = I.load_clean_tables(str(REPO / "results" / "refit_smooth5.npz"))
    print(f"  {'coord':9s} {'mean':>9} {'sd_MD':>8} {'sd_target':>10} {'ratio':>7}", flush=True)
    for name in ("bb_bond", "angle", "dihedral", "stack"):
        vals = []
        for f in frames:
            t = torch.tensor(np.repeat(f, 3, axis=0), dtype=torch.float64).reshape(1, 3 * f.shape[0], 3)
            vals.append(B.coords_of(t, name).reshape(-1).numpy())
        v = np.concatenate(vals)
        print(f"  {name:9s} {v.mean():9.4f} {v.std():8.4f} {float(ref_tabs[name]['sigma']):10.4f} "
              f"{v.std() / float(ref_tabs[name]['sigma']):7.3f}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
