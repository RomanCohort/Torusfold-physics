# -*- coding: utf-8 -*-
"""run_2013nt.py — 2013nt circRNA isRNAcircLong 端到端测试

用法: C:/ana/envs/comfyui/python.exe run_2013nt.py
"""
import os
# 禁用 OpenCL (Windows 上 LLVM JIT 报 "Can't get available size")
os.environ["OPENMM_CPU_THREADS"] = os.environ.get("OPENMM_CPU_THREADS", "32")

import sys
import time
from pathlib import Path

# 在 import openmm 之前 monkey-patch, 强制跳过 OpenCL
# 解决 Windows 上 OpenCL JIT "LLVM ERROR: Can't get available size"
def _patch_openmm_no_opencl():
    """让 Platform.getPlatformByName('OpenCL') 抛异常, 走 CPU fallback."""
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

SEQUENCE = (  # 注: 含T和小写, 管线入口自动 .upper().replace("T","U")
    "CTACCGTTTAATATTGCGTCATATtcggcgaccatttgtgtggtaaaaaaaaaaaaccaaaaaaaaaaaac"
    "aaaaaaaaaaaataattgactaaGATATCTTAAAACAGCGGATGGGTACCCCACCATCCGACCCACTGGG"
    "TGTAGTACTCTGGTACTTCGTACCTTTGTACGCCTGTTCTTCCCATTGTACCCTTCCTGAACTTCCAACC"
    "CAAGTAACGTTAGAAGCTCAACATTTAGTACAACAGGAAGCACCACATCCAGTGGTGTTTAGTACAAGCA"
    "CTTCTGTTTCCCCGGAGCGAGGTATAGGCTGTACCCACTGCCAAAAACCTTTAACCGTTATCCGCCAACC"
    "AACTACGTAAAAGCTAGTAGTATTATGTTTTTAACTAGGCGTTCGATCAGGTGGATTTCCCCTCCACTAG"
    "TTTGGTCGATGAGGCTAGGAATTCCCCACGGGTGACCGTGTCCTAGCCTGCGTGGCGGCCAACCCAGCCC"
    "ACTCACTATTTGTTTTCGCGCCCAGTTGCAAAAAGTGTCGGGGCTGGGACGCCTTTTTATAGACATGGTGT"
    "GAAGACTCGCATGTGCTTGGTTGTGATTCCTCCGGCCCCTGAATGCGGCTAACCTTAACCCTGGAGCCTT"
    "GTGTCACAAACCAGTGATGATAAGGTCGTAATGAGCAATTCCGGGACGGGACCGACTACTTTGGGTGTCCG"
    "TGTTTCTTATTTTTCTTATTATTGTCTTATGGTCACAGCATATATATAACATATACTGTGATCATGgctag"
    "cGCCACCATGgatgcaatgaagagagggctctgctgtgtgctgctgctgtgtggagcagtcttcgtttcgcc"
    "cagccaggaaatccatgcccgattcagaagaGGATCCAGTGTGTGTTGGTGCGTTAACTCAGTTGGCAGCG"
    "GCGGAAGTAGATTGTTCCGCGAGAGATATCGGCTGGGCAGCGGTGGCAGTTGGCTGAAAGAGGGTGTGCTC"
    "GGACTCGGTAGTGGTGGTAGCGCCGTGTTTGCCGACCAGGTGATCGTGGGAAGTGGAGGTAGCTTTCAGGC"
    "GAGGCTTCGCTTGCGCGTACTCGTTCCACCCCTGGGATCTGGCGGCTCTGCTGTGACTTCCGAGTTCCACC"
    "TGGTTGGCAGCGGTGGCAGCGGTGTGGCCACTCTTGCCTGGATGGTGGGCTCCGGCGGCAGTGGCCTCCAT"
    "AACTTCTCAGACGGTCTGGGCAGCGGAGGCAGCCTCGAGaagtttctgaacacagccaaagatcggaaccg"
    "ctgggaggagcctgaccagcagctctacaacgtagaggccacatcctacgccctcctgGGCTCCGGCGGTA"
    "GCaagtttctgaacacagccaaagatcggaaccgctgggaggagcctgaccagcagctctacaacgtagagg"
    "ccacatcctacgccctcctgGGCGGAGGTGGCAGCGGCaagtttctgaacacagccaaagatcggaaccgc"
    "tgggaggagcctgaccagcagctctacaacgtagaggccacatcctacgccctcctgGGCTCCGGCGGTAG"
    "Caagtttctgaacacagccaaagatcggaaccgctgggaggagcctgaccagcagctctacaacgtagaggc"
    "cacatcctacgccctcctgGGATCTGGCGGCAGCatcgtgggcattgttgctggcctggctgtcctagcag"
    "ttgtggtcatcggagctgtggtcgctactgtgatgtgtaggaggaagagctcaggtggaaaaggagggagcta"
    "ctctcaggctgcgtccagcgacagtgcccagggctctgatgtgtctctcacagctGGTGGCTCCGATTATAA"
    "GGATGATGACGACAAGTGAatcgatGCTGGAGCCTCGGTGGCCATGCTTCTTGCCCCTTGGGCCTCCCCC"
    "CAGCCCCTCCTCCCCTTCCTGCACCCGTACCCCCGTGGTCTTTGAATAAAGTCTGAaccacacaaatggtc"
    "gccgaCTCAGTAGATGTTTTCTTGGGT"
)


def main():
    # MUSES 多源二级结构共识 (structRFM 启发)
    ss_path = ROOT / "test_2013nt_ss.txt"
    if ss_path.exists():
        ss = ss_path.read_text().strip()
    else:
        print("MUSES 多源 SS 共识预测...")
        try:
            from torusfold.scheme2.multisource_ss import multisource_consensus_ss
            import numpy as np
            ss, _ = multisource_consensus_ss(SEQUENCE)
            print(f"  MUSES 共识: {ss[:50]}...")
        except Exception:
            print("  MUSES 不可用, 回退 ViennaRNA...")
            from viennaRNA import RNA
            ss, mfe = RNA.fold(SEQUENCE)
            print(f"  ViennaRNA MFE={mfe:.1f}")
        ss_path.write_text(ss)

    assert len(SEQUENCE) == len(ss), \
        f"序列长度 {len(SEQUENCE)} != 结构长度 {len(ss)}"

    print("=" * 70)
    print("isRNAcircLong: 2013nt circRNA")
    print("=" * 70)
    print(f"  序列: {len(SEQUENCE)}nt")
    print(f"  配对: {ss.count('(')}bp stem")
    print(f"  输出: {ROOT / 'output_2013nt'}")

    from torusfold.scheme2.isrnaclong import isrnaclong_pipeline

    t0 = time.time()
    result = isrnaclong_pipeline(
        sequence=SEQUENCE,
        secondary_structure=ss,
        output_dir=str(ROOT / "output_2013nt"),
        max_seg_len=200,
        overlap=20,
        n_relax_rounds=10,
        use_rl_relax=True,
        use_rl_mcts=True,
        rl_n_simulations=20,
        n_rest2_replicas=4,
        rest2_nsteps=30000,
        md_step_scale=0.1,  # Level 2 每轮 MD 步数减到 1/10 (1M→100K), 控制总耗时
        nrep=4,  # Level 2 REMD 并发副本数
        platform="auto",
        use_rhofold=True,
        n_candidates=1,
        use_msa=True,
        rfam_dir=str(ROOT / "msa_work"),
        rfam_cm="",  # Rfam.cm 在 WSL, 当前走已知家族复用 + 伪MSA兜底
        msa_blocks=[
            # Rfam 家族深同源: A3M 已按 chunk 精确对齐 (列数 == chunk 长度,
            # 参考行 == chunk 序列). cmalign --mapali 家族坐标 -> 重投影到 chunk.
            {"start": 101, "end": 185,
             "msa_path": str(ROOT / "msa_work/CRE_chunk.a3m.fa"),
             "source": "Rfam_CRE"},
            {"start": 535, "end": 786,
             "msa_path": str(ROOT / "msa_work/IRES_chunk.a3m.fa"),
             "source": "Rfam_IRES"},
            # RNAcentral blast 命中 (3' 端区域), 已裁剪到 chunk 窗口
            {"start": 1663, "end": 1971,
             "msa_path": str(ROOT / "msa_work/rnacentral_out/rnacentral_chunk.a3m.fa"),
             "source": "RNAcentral"},
        ],
        resume=True,  # 断点续跑: 每个 Level 完成后保存 checkpoint, 中断后下次自动从最近 Level 恢复
        verbose=True,
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("结果")
    print("=" * 70)
    print(f"  时间: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"  分段: {result.n_segments}")
    print(f"  配对率: {result.pair_rate:.2%}")
    print(f"  跨段: {result.cross_segment_ok_rate:.2%}")
    print(f"  能量: {result.energy_cg:.0f}")
    print(f"  PDB: output_2013nt/isrnaclong_final.pdb")


if __name__ == "__main__":
    main()
