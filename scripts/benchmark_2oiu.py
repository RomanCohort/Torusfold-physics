"""Benchmark: test CG refinement + scoring on 2OIU crystal structure."""
import sys, os

def main():
    sys.path.insert(0, "C:/Users/颜子壹/TorusFold-scheme2-rl/src")

    import numpy as np
    from openmm.app import PDBFile

    OUT = "C:/Users/颜子壹/TorusFold-scheme2-rl/benchmark_2oiu"
    os.makedirs(OUT, exist_ok=True)

    # ── 1. Parse 2OIU ──
    pdb = PDBFile(os.path.join(OUT, "2oiu.pdb"))
    topo = pdb.topology
    positions = pdb.positions

    residues_p = [r for r in topo.residues() if r.chain.id == "P"]
    seq_chars = []
    p_coords = []
    for res in residues_p:
        rn = res.name.strip()
        base_map = {"A": "A", "U": "U", "G": "G", "C": "C", "RA": "A", "RU": "U", "RG": "G", "RC": "C"}
        base = base_map.get(rn, rn[0])
        if base not in "ACGU":
            continue
        seq_chars.append(base)
        for atom in res.atoms():
            if atom.name.strip() == "P":
                pos = positions[atom.index]
                p_coords.append([pos.x, pos.y, pos.z])

    seq = "".join(seq_chars)
    p_coords = np.array(p_coords)
    L = len(seq)
    print(f"2OIU: {L} nt")
    print(f"Sequence: {seq}")
    ref_bsj = np.linalg.norm(p_coords[0] - p_coords[-1])
    print(f"Crystal BSJ: {ref_bsj:.2f} A")

    # ── 2. ViennaRNA pairs ──
    import ViennaRNA
    fc = ViennaRNA.fold_compound(seq)
    ss_result = fc.mfe()
    ss = ss_result[0]
    print(f"ViennaRNA SS: {ss}")

    # Get bpp (need pf() first)
    fc.pf()
    bpp = fc.bpp()
    pairs = []
    for i in range(1, L+1):
        for j in range(i+1, L+1):
            if bpp[i][j] > 0.1:
                pairs.append((i-1, j-1, float(bpp[i][j])))
    print(f"Pairs: {len(pairs)} (bpp>0.1)")

    # ── 3. Run CG refinement on crystal P coords ──
    from torusfold.scheme2.openmm_gpu_refiner import openmm_gpu_refine
    import tempfile

    # Write crystal P coords as PDB for the refiner
    tmp_pdb = os.path.join(OUT, "crystal_for_refine.pdb")
    with open(tmp_pdb, 'w') as f:
        f.write("HEADER 2OIU crystal P coords\n")
        for i in range(L):
            x, y, z = p_coords[i]
            f.write(f"ATOM  {i+1:5d}  P   RA A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}\n")

    print(f"\nRunning CG refinement on crystal coords...")
    output_pdb, e_refined = openmm_gpu_refine(
        tmp_pdb, OUT, seq, ss,
        nstep=10000,  # Short for benchmark
        remd_n_steps=5000,
        use_remd=True,
        remd_n_replicas=4,
        verbose=True,
    )
    print(f"  Output PDB: {output_pdb}")

    # ── 4. Compute RMSD ──
    pred_pdb_path = output_pdb
    if os.path.exists(pred_pdb_path):
        pred_pdb = PDBFile(pred_pdb_path)
        pred_residues = [r for r in pred_pdb.topology.residues()]
        pred_p = np.zeros((L, 3))
        for i, res in enumerate(pred_residues[:L]):
            for atom in res.atoms():
                if atom.name.strip() == "P":
                    pos = pred_pdb.positions[atom.index]
                    pred_p[i] = [pos.x, pos.y, pos.z]

        # Kabsch alignment
        centroid_p = p_coords.mean(axis=0)
        centroid_pred = pred_p.mean(axis=0)
        P_c = p_coords - centroid_p
        Q_c = pred_p - centroid_pred
        H = Q_c.T @ P_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        pred_aligned = Q_c @ R + centroid_p

        rmsd = np.sqrt(np.mean(np.sum((pred_aligned - p_coords)**2, axis=1)))
        pred_bsj = np.linalg.norm(pred_p[0] - pred_p[-1])

        print(f"\n{'='*60}")
        print(f"2OIU Benchmark Results")
        print(f"{'='*60}")
        print(f"Length: {L} nt")
        print(f"Crystal BSJ: {ref_bsj:.2f} A")
        print(f"Predicted BSJ: {pred_bsj:.2f} A")
        print(f"P-only RMSD (after Kabsch): {rmsd:.2f} A")
        print(f"Refined energy: {e_refined:.0f} kJ/mol")
    else:
        print(f"\nNo output PDB found!")

if __name__ == '__main__':
    main()
