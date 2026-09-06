"""
TriRNASP Python scorer — 三体统计势打分.

 reimplements TriRNASP (Tan-group, Wuhan University) scoring in pure Python/numpy.
 Reference: Tongwei Yuan et al. Biophysical Journal 125(11), 2526-2540 (2026).

两种模式:
  1. full_scoring(): O(n³) 完整三体打分, 用于 post-hoc 分析
  2. window_scoring(): O(n × w²) 滑窗近似, 用于 MC 采样 (w=窗口大小)

用法:
    scorer = TriRNASPScorer("Energy/")
    energy = scorer.full_scoring(sequence, coords_3bead)
    # coords_3bead: (N, 3, 3) — 每个nt的P/C4'/N坐标 (Å)
"""
import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from itertools import combinations


# ── 常量 (与 TriRNASP.c 一致) ──
R0 = 8.0           # 距离截断 (Å)
BIN_WIDTH = 2.0    # Rough bin width
BIN_WIDTH1 = 0.6   # Fine bin width
INTERVALS = 4      # Rough bins: floor(8.0 / 2.0)
INTERVALS1 = 13    # Fine bins: floor(8.0 / 0.6)

# Bead type codes (与 compute_code() 一致)
# C4': A=0, U=1, C=2, G=3
# N:   A=4, U=5, C=6, G=7  (purine→N9, pyrimidine→N1)
# P:   A=8, U=9, C=10, G=11
BASE_IDX = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
PURINE = {'A', 'G'}
PYRIMIDINE = {'U', 'C'}


def _compute_type_code(base: str, atom: str) -> int:
    """Compute 12-type code from base + atom name."""
    b = BASE_IDX.get(base.upper())
    if b is None:
        return -1
    if atom == 'C4':
        return b          # 0-3
    elif atom == 'N':
        if base.upper() in PURINE:
            return 4 + b   # 4-7 (N9 for purines)
        elif base.upper() in PYRIMIDINE:
            return 4 + b   # 4-7 (N1 for pyrimidines)
        return -1
    elif atom == 'P':
        return 8 + b       # 8-11
    return -1


class TriRNASPScorer:
    """TriRNASP energy table loader and scorer."""

    def __init__(self, energy_dir: Optional[str] = None):
        """Load energy tables from Energy/ folder.

        Args:
            energy_dir: path to Energy/ folder. Auto-detect if None.
        """
        if energy_dir is None:
            base = Path(__file__).parent.parent / "external" / "TriRNASP" / "Energy"
            energy_dir = str(base)

        self.energy_dir = Path(energy_dir)
        if not self.energy_dir.exists():
            raise FileNotFoundError(f"Energy dir not found: {energy_dir}")

        # Load tables
        self.rough = self._load_table("Rough.energy", INTERVALS, is_fine=False)
        self.fine = self._load_table("Fine.energy", INTERVALS1, is_fine=True)

        print(f"    [TriRNASP] Loaded: Rough={self.rough.nbytes/1e6:.1f}MB, "
              f"Fine={self.fine.nbytes/1e6:.1f}MB")

    def _load_table(self, filename: str, intervals: int, is_fine: bool) -> np.ndarray:
        """Load energy table from file into numpy array.

        Indexing: IDX(n1,n2,n3,b12,b13,b23)
        = ((((n1*12 + n2)*12 + n3)*I + b12)*I + b13)*(2*I) + b23
        """
        filepath = self.energy_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Energy table not found: {filepath}")

        # Determine I (for 2*I range of b23)
        I = intervals
        shape = (12, 12, 12, I, I, 2 * I)
        table = np.zeros(shape, dtype=np.float64)

        with open(filepath) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 7:
                    continue
                n1, n2, n3, b1, b2, b3 = int(parts[0]), int(parts[1]), int(parts[2]), \
                                           int(parts[3]), int(parts[4]), int(parts[5])
                val = float(parts[6])
                if (0 <= n1 < 12 and 0 <= n2 < 12 and 0 <= n3 < 12 and
                    0 <= b1 < I and 0 <= b2 < I and 0 <= b3 < 2 * I):
                    table[n1, n2, n3, b1, b2, b3] = val

        return table

    def _get_bead_types(self, sequence: str) -> np.ndarray:
        """Get type codes for 3-bead model: (L, 3) array.

        Returns:
            (L, 3) int array: [C4'_type, N_type, P_type] per residue
        """
        L = len(sequence)
        types = np.zeros((L, 3), dtype=np.int32)
        for i, base in enumerate(sequence):
            types[i, 0] = _compute_type_code(base, 'C4')  # C4'
            types[i, 1] = _compute_type_code(base, 'N')   # N
            types[i, 2] = _compute_type_code(base, 'P')   # P
        return types

    def _coords_to_atoms(self, coords_3bead: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert (L, 3, 3) to flat atom arrays.

        Args:
            coords_3bead: (L, 3, 3) — [residue][C4'/N/P][xyz] in Å

        Returns:
            xyz: (N_atoms, 3) in Å
            types: (N_atoms,) int type codes
        """
        L = coords_3bead.shape[0]
        xyz = coords_3bead.reshape(-1, 3)  # (3L, 3)
        types = np.zeros(3 * L, dtype=np.int32)
        for i in range(L):
            types[3 * i] = _compute_type_code('', 'C4')      # placeholder
            types[3 * i + 1] = _compute_type_code('', 'N')
            types[3 * i + 2] = _compute_type_code('', 'P')
        return xyz, types

    def full_scoring(self, sequence: str, coords_3bead: np.ndarray,
                     verbose: bool = False) -> float:
        """Full O(n³) TriRNASP scoring.

        Args:
            sequence: RNA sequence (ACGU)
            coords_3bead: (L, 3, 3) — [residue][C4'/N/P][xyz] in Å
            verbose: print progress

        Returns:
            energy in kBT (lower = better)
        """
        import time
        L = len(sequence)
        assert coords_3bead.shape == (L, 3, 3), f"Expected ({L},3,3), got {coords_3bead.shape}"

        t0 = time.time()

        # Get bead types for all atoms
        atom_types = []
        atom_coords = []
        res_ids = []  # which residue each atom belongs to
        for i, base in enumerate(sequence):
            for atom_name in ['C4', 'N', 'P']:
                tc = _compute_type_code(base, atom_name)
                if tc >= 0:
                    atom_types.append(tc)
                    atom_coords.append(coords_3bead[i, ['C4', 'N', 'P'].index(atom_name)])
                    res_ids.append(i)

        atom_types = np.array(atom_types, dtype=np.int32)
        atom_coords = np.array(atom_coords, dtype=np.float64)
        res_ids = np.array(res_ids, dtype=np.int32)
    def _get_residue_types(self, sequence: str) -> np.ndarray:
        """Get type codes for each residue's 3 beads.

        Returns:
            (L, 3) int array: [C4'_type, N_type, P_type] per residue
        """
        L = len(sequence)
        types = np.zeros((L, 3), dtype=np.int32)
        for i, base in enumerate(sequence.upper()):
            types[i, 0] = _compute_type_code(base, 'C4')
            types[i, 1] = _compute_type_code(base, 'N')
            types[i, 2] = _compute_type_code(base, 'P')
        return types

    def score_from_pdb(self, pdb_path: str) -> float:
        """Score a PDB file (3-bead format).

        Args:
            pdb_path: path to PDB file with P/C4'/N atoms

        Returns:
            energy in kBT
        """
        atoms = []
        res_ids = []
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                atom_name = line[12:16].strip()
                res_id = int(line[22:26].strip()) - 1  # 0-indexed
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                if atom_name == "P":
                    tc = _compute_type_code('A', 'P')  # placeholder base
                elif atom_name == "C4'":
                    tc = _compute_type_code('A', 'C4')
                elif atom_name in ("N9", "N1"):
                    tc = _compute_type_code('A', 'N')
                else:
                    continue

                atoms.append((tc, res_id, x, y, z))

        if not atoms:
            return 0.0

        n = len(atoms)
        total_energy = 0.0

        R0_sq = (R0 - 0.3) ** 2
        inv_bw = 1.0 / BIN_WIDTH
        inv_bw1 = 1.0 / BIN_WIDTH1

        for i in range(n):
            ti, ri, xi, yi, zi = atoms[i]
            for j in range(i + 1, n):
                tj, rj, xj, yj, zj = atoms[j]
                d12_sq = (xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2
                if d12_sq >= R0_sq:
                    continue

                for k in range(j + 1, n):
                    tk, rk, xk, yk, zk = atoms[k]
                    d13_sq = (xi-xk)**2 + (yi-yk)**2 + (zi-zk)**2
                    if d13_sq >= R0_sq:
                        continue
                    d23_sq = (xj-xk)**2 + (yj-yk)**2 + (zj-zk)**2

                    # Clash/contact penalties
                    if d12_sq <= 1.21 or d13_sq <= 1.21 or d23_sq <= 1.21:
                        total_energy += 0.5
                        continue
                    if ((d12_sq <= 2.89 and d12_sq > 1.21 and abs(ri - rj) > 1) or
                        (d13_sq <= 2.89 and d13_sq > 1.21 and abs(ri - rk) > 1) or
                        (d23_sq <= 2.89 and d23_sq > 1.21 and abs(rj - rk) > 1)):
                        total_energy += 0.5
                        continue
                    if ((d12_sq > 2.89 and d12_sq <= 4.0) or
                        (d13_sq > 2.89 and d13_sq <= 4.0) or
                        (d23_sq > 2.89 and d23_sq <= 4.0)):
                        total_energy += 0.2
                        continue

                    # Bin distances
                    d12 = np.sqrt(d12_sq)
                    d13 = np.sqrt(d13_sq)
                    d23 = np.sqrt(d23_sq)

                    b12 = int(d12 * inv_bw)
                    b13 = int(d13 * inv_bw)
                    b23 = int(d23 * inv_bw)

                    D12 = int(d12 * inv_bw1)
                    D13 = int(d13 * inv_bw1)
                    D23 = int(d23 * inv_bw1)

                    if b12 > INTERVALS - 1 or b13 > INTERVALS - 1 or b23 > 2 * INTERVALS - 1:
                        continue

                    if ti >= 0 and tj >= 0 and tk >= 0:
                        # Rough
                        total_energy += self.rough[ti, tj, tk, b12, b13, b23]
                        # Fine
                        if (D12 < INTERVALS1 and D13 < INTERVALS1 and D23 < 2 * INTERVALS1):
                            total_energy += self.fine[ti, tj, tk, D12, D13, D23]

        n_nt = len(res_ids) // 3
        mu = sum(1 for t in atom_types if 0 <= t <= 3) / max(1, n_nt)

        if verbose:
            print(f"    [TriRNASP] {n_nt} nt, {n_atoms} atoms, "
                  f"R0={R0:.1f}Å, μ={mu:.2f}")

        R0_sq = (R0 - 0.3) ** 2  # 与C代码一致的截断
        inv_bw = 1.0 / BIN_WIDTH
        inv_bw1 = 1.0 / BIN_WIDTH1

        total_energy = 0.0

        # O(n³) triplet enumeration with distance cutoff
        for i in range(n_atoms):
            ti = atom_types[i]
            ri = res_ids[i]
            xi, yi, zi = atom_coords[i]
            for j in range(i + 1, n_atoms):
                dx = xi - atom_coords[j, 0]
                dy = yi - atom_coords[j, 1]
                dz = zi - atom_coords[j, 2]
                d12_sq = dx*dx + dy*dy + dz*dz
                if d12_sq >= R0_sq:
                    continue

                tj = atom_types[j]
                rj = res_ids[j]

                for k in range(j + 1, n_atoms):
                    dx2 = xi - atom_coords[k, 0]
                    dy2 = yi - atom_coords[k, 1]
                    dz2 = zi - atom_coords[k, 2]
                    d13_sq = dx2*dx2 + dy2*dy2 + dz2*dz2
                    if d13_sq >= R0_sq:
                        continue

                    dx3 = atom_coords[j, 0] - atom_coords[k, 0]
                    dy3 = atom_coords[j, 1] - atom_coords[k, 1]
                    dz3 = atom_coords[j, 2] - atom_coords[k, 2]
                    d23_sq = dx3*dx3 + dy3*dy3 + dz3*dz3

                    tk = atom_types[k]
                    rk = res_ids[k]

                    # 惩罚规则 (与 C 代码一致)
                    if d12_sq <= 1.21 or d13_sq <= 1.21 or d23_sq <= 1.21:
                        total_energy += 0.5
                        continue
                    if ((d12_sq <= 2.89 and d12_sq > 1.21 and abs(ri - rj) > 1) or
                        (d13_sq <= 2.89 and d13_sq > 1.21 and abs(ri - rk) > 1) or
                        (d23_sq <= 2.89 and d23_sq > 1.21 and abs(rj - rk) > 1)):
                        total_energy += 0.5
                        continue
                    if ((d12_sq > 2.89 and d12_sq <= 4.0) or
                        (d13_sq > 2.89 and d13_sq <= 4.0) or
                        (d23_sq > 2.89 and d23_sq <= 4.0)):
                        total_energy += 0.2
                        continue

                    d12 = np.sqrt(d12_sq)
                    d13 = np.sqrt(d13_sq)
                    d23 = np.sqrt(d23_sq)

                    b12 = min(int(d12 * inv_bw), INTERVALS - 1)
                    b13 = min(int(d13 * inv_bw), INTERVALS - 1)
                    b23 = min(int(d23 * inv_bw), 2 * INTERVALS - 1)

                    D12 = min(int(d12 * inv_bw1), INTERVALS1 - 1)
                    D13 = min(int(d13 * inv_bw1), INTERVALS1 - 1)
                    D23 = min(int(d23 * inv_bw1), 2 * INTERVALS1 - 1)

                    if ti >= 0 and tj >= 0 and tk >= 0:
                        total_energy += self.rough[ti, tj, tk, b12, b13, b23]
                        total_energy += self.fine[ti, tj, tk, D12, D13, D23]

        elapsed = time.time() - t0
        if verbose:
            print(f"    [TriRNASP] E={total_energy:.3f} kBT, {elapsed:.1f}s")

        return total_energy

    def window_scoring(self, sequence: str, coords_3bead: np.ndarray,
                       window: int = 30, verbose: bool = False) -> float:
        """Fast O(n × w²) approximate scoring using sequence window.

        Only considers atom triplets where all 3 atoms are within
        ±window residues of each other. Fast enough for MC (~10ms).

        Args:
            sequence: RNA sequence (ACGU)
            coords_3bead: (L, 3, 3) — [residue][C4'/N/P][xyz] in Å
            window: sequence window size (± residues)
            verbose: print progress

        Returns:
            approximate energy in kBT
        """
        L = len(sequence)
        assert coords_3bead.shape == (L, 3, 3)

        residue_types = self._get_residue_types(sequence)
        total_energy = 0.0

        R0_sq = (R0 - 0.3) ** 2
        inv_bw = 1.0 / BIN_WIDTH
        inv_bw1 = 1.0 / BIN_WIDTH1

        for i in range(L):
            # 3 beads for residue i
            for bi in range(3):
                ti = residue_types[i, bi]
                xi, yi, zi = coords_3bead[i, bi]

                j_start = max(i - window, 0)
                j_end = min(i + window + 1, L)

                for j in range(max(i + 1, j_start), j_end):
                    for bj in range(3):
                        tj = residue_types[j, bj]
                        xj, yj, zj = coords_3bead[j, bj]
                        d12_sq = (xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2
                        if d12_sq >= R0_sq:
                            continue

                        k_start = max(j, j_start)
                        k_end = min(j_end, L)

                        for k in range(max(j + 1, k_start), k_end):
                            for bk in range(3):
                                tk = residue_types[k, bk]
                                xk, yk, zk = coords_3bead[k, bk]
                                d13_sq = (xi-xk)**2 + (yi-yk)**2 + (zi-zk)**2
                                if d13_sq >= R0_sq:
                                    continue
                                d23_sq = (xj-xk)**2 + (yj-yk)**2 + (zj-zk)**2

                                # 惩罚规则
                                if d12_sq <= 1.21 or d13_sq <= 1.21 or d23_sq <= 1.21:
                                    total_energy += 0.5
                                    continue
                                if ((d12_sq <= 2.89 and abs(i - j) > 1) or
                                    (d13_sq <= 2.89 and abs(i - k) > 1) or
                                    (d23_sq <= 2.89 and abs(j - k) > 1)):
                                    total_energy += 0.5
                                    continue
                                if ((2.89 < d12_sq <= 4.0) or
                                    (2.89 < d13_sq <= 4.0) or
                                    (2.89 < d23_sq <= 4.0)):
                                    total_energy += 0.2
                                    continue

                                d12 = np.sqrt(d12_sq)
                                d13 = np.sqrt(d13_sq)
                                d23 = np.sqrt(d23_sq)

                                b12 = min(int(d12 * inv_bw), INTERVALS - 1)
                                b13 = min(int(d13 * inv_bw), INTERVALS - 1)
                                b23 = min(int(d23 * inv_bw), 2 * INTERVALS - 1)

                                D12 = min(int(d12 * inv_bw1), INTERVALS1 - 1)
                                D13 = min(int(d13 * inv_bw1), INTERVALS1 - 1)
                                D23 = min(int(d23 * inv_bw1), 2 * INTERVALS1 - 1)

                                if ti >= 0 and tj >= 0 and tk >= 0:
                                    total_energy += self.rough[ti, tj, tk, b12, b13, b23]
                                    total_energy += self.fine[ti, tj, tk, D12, D13, D23]

        return total_energy

    def score_pdb(self, pdb_path: str, verbose: bool = False) -> float:
        """Score a PDB file.

        Parses PDB to extract atom types and coordinates, then scores.

        Args:
            pdb_path: path to 3-bead PDB file
            verbose: print progress

        Returns:
            energy in kBT
        """
        atoms = []  # [(type_code, res_id, x, y, z)]

        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue

                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21]
                res_id = int(line[22:26].strip()) - 1  # 0-indexed
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                # 确定 bead type
                if atom_name == "P":
                    tc = _compute_type_code(res_name[0] if res_name else 'A', 'P')
                elif atom_name in ("C4'", "C4"):
                    tc = _compute_type_code(res_name[0] if res_name else 'A', 'C4')
                elif atom_name in ("N9", "N1", "N"):
                    tc = _compute_type_code(res_name[0] if res_name else 'A', 'N')
                else:
                    continue

                if tc >= 0:
                    atoms.append((tc, res_id, x, y, z))

        n = len(atoms)
        if n < 3:
            return 0.0

        # 提取为数组
        atom_types = np.array([a[0] for a in atoms], dtype=np.int32)
        res_ids = np.array([a[1] for a in atoms], dtype=np.int32)
        coords = np.array([[a[2], a[3], a[4]] for a in atoms], dtype=np.float64)

        # 用 full_scoring 的核心逻辑
        R0_sq = (R0 - 0.3) ** 2
        inv_bw = 1.0 / BIN_WIDTH
        inv_bw1 = 1.0 / BIN_WIDTH1
        total_energy = 0.0

        for i in range(n):
            ti = atom_types[i]
            ri = res_ids[i]
            xi, yi, zi = coords[i]
            for j in range(i + 1, n):
                dx = xi - coords[j, 0]
                dy = yi - coords[j, 1]
                dz = zi - coords[j, 2]
                d12_sq = dx*dx + dy*dy + dz*dz
                if d12_sq >= R0_sq:
                    continue

                tj = atom_types[j]
                rj = res_ids[j]

                for k in range(j + 1, n):
                    dx2 = xi - coords[k, 0]
                    dy2 = yi - coords[k, 1]
                    dz2 = zi - coords[k, 2]
                    d13_sq = dx2*dx2 + dy2*dy2 + dz2*dz2
                    if d13_sq >= R0_sq:
                        continue

                    dx3 = coords[j, 0] - coords[k, 0]
                    dy3 = coords[j, 1] - coords[k, 1]
                    dz3 = coords[j, 2] - coords[k, 2]
                    d23_sq = dx3*dx3 + dy3*dy3 + dz3*dz3

                    tk = atom_types[k]
                    rk = res_ids[k]

                    if d12_sq <= 1.21 or d13_sq <= 1.21 or d23_sq <= 1.21:
                        total_energy += 0.5
                        continue
                    if ((d12_sq <= 2.89 and abs(ri - rj) > 1) or
                        (d13_sq <= 2.89 and abs(ri - rk) > 1) or
                        (d23_sq <= 2.89 and abs(rj - rk) > 1)):
                        total_energy += 0.5
                        continue
                    if ((2.89 < d12_sq <= 4.0) or
                        (2.89 < d13_sq <= 4.0) or
                        (2.89 < d23_sq <= 4.0)):
                        total_energy += 0.2
                        continue

                    d12 = np.sqrt(d12_sq)
                    d13 = np.sqrt(d13_sq)
                    d23 = np.sqrt(d23_sq)

                    b12 = min(int(d12 * inv_bw), INTERVALS - 1)
                    b13 = min(int(d13 * inv_bw), INTERVALS - 1)
                    b23 = min(int(d23 * inv_bw), 2 * INTERVALS - 1)

                    D12 = min(int(d12 * inv_bw1), INTERVALS1 - 1)
                    D13 = min(int(d13 * inv_bw1), INTERVALS1 - 1)
                    D23 = min(int(d23 * inv_bw1), 2 * INTERVALS1 - 1)

                    if ti >= 0 and tj >= 0 and tk >= 0:
                        total_energy += self.rough[ti, tj, tk, b12, b13, b23]
                        total_energy += self.fine[ti, tj, tk, D12, D13, D23]

        if verbose:
            print(f"    [TriRNASP] PDB {pdb_path}: {n} atoms, E={total_energy:.3f} kBT")

        return total_energy


# ────────────────────────────────────────────────────────
# Quick test
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    energy_dir = sys.argv[1] if len(sys.argv) > 1 else None
    pdb_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        scorer = TriRNASPScorer(energy_dir)

        if pdb_path:
            e = scorer.score_pdb(pdb_path, verbose=True)
            print(f"Energy: {e:.3f} kBT")
        else:
            # Test with random coords
            seq = "AUGCAUGCAU"
            coords = np.random.rand(10, 3, 3) * 10.0
            e = scorer.full_scoring(seq, coords, verbose=True)
            print(f"Random test: E = {e:.3f} kBT")
    except Exception as e:
        print(f"Error: {e}")
