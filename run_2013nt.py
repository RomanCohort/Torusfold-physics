# -*- coding: utf-8 -*-
"""run_2013nt.py — 2013nt circRNA isRNAcircLong 端到端测试

用法: python run_2013nt.py
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

# 序列从文件读取 (不硬编码, 保护知识产权)
_seq_file = ROOT / "sequence.txt"
if _seq_file.exists():
    SEQUENCE = _seq_file.read_text().strip().replace("\n", "")
else:
    print(f"错误: 未找到序列文件 {_seq_file}")
    print("请创建 sequence.txt, 将 RNA 序列写入 (支持 T→U 自动转换)")
    sys.exit(1)


def main():
    # MUSES 多源二级结构共识 (structRFM 启发)
    ss_path = ROOT / "test_2013nt_ss.txt"
    if ss_path.exists():
        ss = ss_path.read_text().strip()
        # 验证 SS 有效性 (不能全是点)
        if ss.count('(') == 0 and ss.count(')') == 0:
            print(f"  SS 全是点 ({len(ss)}nt), 重新预测...")
            ss = None
        else:
            print(f"  已有 SS: {ss.count('(')}bp stem")
    else:
        ss = None

    if ss is None:
        print("MUSES 多源 SS 共识预测...")
        try:
            from torusfold.scheme2.multisource_ss import multisource_consensus_ss
            import numpy as np
            ss, _ = multisource_consensus_ss(SEQUENCE)
            print(f"  MUSES 共识: {ss[:50]}...")
        except Exception as e:
            print(f"  MUSES 不可用: {e}, 回退 ViennaRNA...")
        if ss is None or ss.count('(') == 0:
            import RNA
            md = RNA.md()
            md.circ = 1  # 环化模式
            fc = RNA.fold_compound(SEQUENCE, md)
            ss, mfe = fc.mfe()
            print(f"  ViennaRNA MFE={mfe:.1f}, pairs={ss.count('(')}")
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
        overlap=30,              # 加大重叠区, Kabsch 对齐更准
        n_relax_rounds=20,       # Level 2 迭代轮数 (早停会自动截断)
        use_rl_relax=True,
        use_rl_mcts=True,
        rl_n_simulations=50,     # MCTS 搜索次数
        n_rest2_replicas=16,     # REST2 副本数 (CPU 并行打满 32 核)
        rest2_nsteps=100000,     # REST2 步数: 充分采样
        md_step_scale=0.5,       # Level 2 MD 步数: 1M×0.5=500K/轮
        nrep=16,                 # Level 2 REMD 并发副本数
        platform="auto",
        use_rhofold=True,
        use_pyrosetta=True,  # Level 2.6: PyRosetta 条件式精修 (WSL)
        use_ppr=True,       # Level 5.5: 碱基对氢键修复
        ppr_max_rounds=5,
        n_candidates=1,
        use_msa=True,        # 伪 MSA 兜底 (无 Rfam 文件时自动降级)
        rfam_dir=str(ROOT / "msa_work"),
        rfam_cm=os.environ.get("RFAM_CM"),  # 可选: cmsearch 的 Rfam.cm 路径 (env RFAM_CM)
        msa_blocks=None,     # MSA 文件不存在, 不指定块 (自动伪 MSA)
        resume=True,  # 断点续跑 (level 1 checkpoint 保留)
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

    # 每残基置信度统计
    if hasattr(result, 'per_res_confidence') and result.per_res_confidence is not None:
        import numpy as np
        conf = result.per_res_confidence
        print(f"\n  每残基置信度:")
        print(f"    均值: {conf.mean():.3f}")
        print(f"    中位: {np.median(conf):.3f}")
        print(f"    低置信 (<0.3): {(conf < 0.3).sum()} 残基 ({(conf < 0.3).mean():.1%})")
        print(f"    高置信 (>0.7): {(conf > 0.7).sum()} 残基 ({(conf > 0.7).mean():.1%})")


if __name__ == "__main__":
    main()
