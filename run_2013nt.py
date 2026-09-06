# -*- coding: utf-8 -*-
"""run_2013nt.py - end-to-end isRNAcircLong test on a 2013-nt circRNA.

Usage: python run_2013nt.py
"""
import os
# Disable OpenCL (Windows LLVM JIT reports "Can't get available size")
os.environ["OPENMM_CPU_THREADS"] = os.environ.get("OPENMM_CPU_THREADS", "32")

import sys
import time
from pathlib import Path

# Monkey-patch before importing openmm to force OpenCL off;
# fixes "LLVM ERROR: Can't get available size" on the Windows OpenCL JIT.
def _patch_openmm_no_opencl():
    """Make Platform.getPlatformByName('OpenCL') raise -> CPU fallback."""
    try:
        import openmm as _mm
        _orig = _mm.Platform.getPlatformByName
        def _safe_get(name):
            if name in ("OpenCL", "CUDA"):
                raise RuntimeError(f"Disabled: {name}")
            return _orig(name)
        _mm.Platform.getPlatformByName = staticmethod(_safe_get)
    except ImportError:
        pass

_patch_openmm_no_opencl()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Read the sequence from a file (not hard-coded, to protect IP)
_seq_file = ROOT / "sequence.txt"
if _seq_file.exists():
    SEQUENCE = _seq_file.read_text().strip().replace("\n", "")
else:
    print(f"Error: sequence file not found: {_seq_file}")
    print("Create sequence.txt and write the RNA sequence (T->U conversion supported)")
    sys.exit(1)


def main():
    # MUSES multi-source secondary-structure consensus (structRFM-inspired)
    ss_path = ROOT / "test_2013nt_ss.txt"
    if ss_path.exists():
        ss = ss_path.read_text().strip()
        # Validate the SS string (must contain at least one base pair)
        if ss.count('(') == 0 and ss.count(')') == 0:
            print(f"  SS is all dots ({len(ss)}nt), re-predicting...")
            ss = None
        else:
            print(f"  Existing SS: {ss.count('(')}bp stem")
    else:
        ss = None

    if ss is None:
        print("MUSES multi-source SS consensus prediction...")
        try:
            from torusfold.scheme2.multisource_ss import multisource_consensus_ss
            import numpy as np
            ss, _ = multisource_consensus_ss(SEQUENCE)
            print(f"  MUSES consensus: {ss[:50]}...")
        except Exception as e:
            print(f"  MUSES unavailable: {e}, falling back to ViennaRNA...")
        if ss is None or ss.count('(') == 0:
            import RNA
            md = RNA.md()
            md.circ = 1  # circular mode
            fc = RNA.fold_compound(SEQUENCE, md)
            ss, mfe = fc.mfe()
            print(f"  ViennaRNA MFE={mfe:.1f}, pairs={ss.count('(')}")
        ss_path.write_text(ss)

    assert len(SEQUENCE) == len(ss), \
        f"sequence length {len(SEQUENCE)} != structure length {len(ss)}"

    print("=" * 70)
    print("isRNAcircLong: 2013nt circRNA")
    print("=" * 70)
    print(f"  Sequence: {len(SEQUENCE)}nt")
    print(f"  Pairs: {ss.count('(')}bp stem")
    print(f"  Output: {ROOT / 'output_2013nt'}")

    from torusfold.scheme2.isrnaclong import isrnaclong_pipeline

    t0 = time.time()
    result = isrnaclong_pipeline(
        sequence=SEQUENCE,
        secondary_structure=ss,
        output_dir=str(ROOT / "output_2013nt"),
        max_seg_len=200,
        overlap=30,              # larger overlap region -> more accurate Kabsch alignment
        n_relax_rounds=20,       # Level 2 iteration count (early stopping truncates automatically)
        use_rl_relax=True,
        use_rl_mcts=True,
        rl_n_simulations=50,     # MCTS search iterations
        n_rest2_replicas=16,     # REST2 replica count (saturates 32 CPU cores)
        rest2_nsteps=100000,     # REST2 steps: full sampling
        md_step_scale=0.5,       # Level 2 MD steps: 1M x 0.5 = 500K/round
        nrep=16,                 # Level 2 REMD concurrent replica count
        platform="auto",
        use_rhofold=True,
        use_pyrosetta=True,  # Level 2.6: conditional PyRosetta refinement (WSL)
        use_ppr=True,       # Level 5.5: base-pair hydrogen-bond repair
        ppr_max_rounds=5,
        n_candidates=1,
        use_msa=True,        # pseudo-MSA fallback (auto-degrades when no Rfam file)
        rfam_dir=str(ROOT / "msa_work"),
        rfam_cm=os.environ.get("RFAM_CM"),  # optional: Rfam.cm path for cmsearch (env RFAM_CM)
        msa_blocks=None,     # no MSA file present -> no blocks (auto pseudo-MSA)
        resume=True,  # resume from checkpoint (keeps level-1 checkpoints)
        verbose=True,
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"  Segments: {result.n_segments}")
    print(f"  Pair rate: {result.pair_rate:.2%}")
    print(f"  Cross-segment: {result.cross_segment_ok_rate:.2%}")
    print(f"  Energy: {result.energy_cg:.0f}")
    print(f"  PDB: output_2013nt/isrnaclong_final.pdb")

    # Per-residue confidence statistics
    if hasattr(result, 'per_res_confidence') and result.per_res_confidence is not None:
        import numpy as np
        conf = result.per_res_confidence
        print(f"\n  Per-residue confidence:")
        print(f"    Mean: {conf.mean():.3f}")
        print(f"    Median: {np.median(conf):.3f}")
        print(f"    Low (<0.3): {(conf < 0.3).sum()} residues ({(conf < 0.3).mean():.1%})")
        print(f"    High (>0.7): {(conf > 0.7).sum()} residues ({(conf > 0.7).mean():.1%})")


if __name__ == "__main__":
    main()
