# 开发机交接

## 一、代码

本机在 `D:\torusfold-hybrid`，**12.3 MB**。远端已是最新（`3d1df83`）：

```
git clone https://gitlab.igem.org/2026/software/jlu-fbh/torusfold-hybrid.git
```

分支 `main`。GitHub 镜像是 `github.com/RomanCohort/Torusfold-physics`，分支 `master`。

`results/boltzmann_tables_clean.npz`（22.7 KB，参考表）**在仓库里**，跟着走，不用单独传。

## 二、数据 —— 这一步最容易漏

`scripts/boltzmann_bonded.py` 要一份**在仓库外面**的数据库：
**191 个 PDB，686.7 MB**。本机在 `D:\torusfold-cgdata\rsRNASP\Training_set`。

现在可以指：

```
export TORUSFOLD_RSRNASP=/path/to/rsRNASP/Training_set
```

不设就回落到 Windows 的默认值（本机行为不变，已验证两条路径都解析正确）。

**注意：另外 15 个脚本各自抄了一份同样的字面量**，它们**不读这个环境变量**——
`cg_occupancy_reference_scheme.py`、`check_bonded_distributions.py`、`check_bonded_modes.py`、
`check_signed_pseudotorsion.py`、`decompose_database_variance.py`、
`fold_signal_under_real_weights.py`、`measure_cpu_constants.py`、
`measure_targets_on_training_set.py`、`measure_rcm_weight_range.py`、
`measure_pair_targets_on_training_set.py`、`measure_pair_weight_quality.py`、
`measure_pair_clash_bsj_constants.py`、`recalibrate_ff_targets.py`、`smoke_cg_stage.py`、
`test_sequence_dependence.py`。要用得逐个改。

**IBI 那条链只经过 `boltzmann_bonded.py`，所以只设这一个就够。**
（另有两个脚本读的是**另一份**数据 `D:\torusfold-cgdata\cgRNASP\cgRNASP\data`：`cg_rnasp_evaluator.py`、`inspect_potential_table.py`。）

## 三、IBI 那条链要什么

| 脚本 | 作用 | 依赖 |
| :-- | :-- | :-- |
| `scripts/boltzmann_bonded.py` | 数据库加载、表、`mixed_energy`（autograd 取力） | numpy, torch |
| `scripts/ibi_round0.py` | 测残差（**现在就能跑**） | torch |
| `scripts/ibi_bonded.py` | 更新式 + 失败处理（675 行，25 个测试） | numpy |
| `scripts/sample_bonded_chain.py` | 旧采样器（**只跑键合项，会解开**，见下） | numpy, torch |

**ViennaRNA 和 OpenMM 不在这条链上。** grep 确认 `import RNA` 全仓库只有 3 个脚本
（`measure_pair_weight_quality.py`、`measure_dividefold_tier_room.py`、`ppr_repair_v3.py`），都不在 IBI 链里。
不过还是照 `requirements.txt` 装一遍稳妥。

## 四、**还没写的东西 —— 这是关键**

**IBI 现在只能测残差，不能迭代。** 缺的是一个**跑整场、但把三个角坐标（angle / dihedral / stack）
换成表格势**的采样环。

`sample_bonded_chain.py` 不能直接改成那个：它刻意只跑 P 原子的四个键合项
（docstring 第 10 行「no nonbonded terms」，第 13–19 行解释为什么排除两个 intra 项），
而**参考边缘分布是以折叠为条件的**，所以一条没有非键合项的链会解开，跟参考比的是两件事。

## 五、到开发机后的第一件事

**量直方图收敛时间，不是先写采样环。** 因为每轮采样多久决定了 IBI 在那台机器上是否可行。

本机速率 **23.1 steps/s**（8 副本批处理，1L2X，81 珠，纯 CPU）：

```
1 ps   =   500 步 = 22 s
16 ps  =  8000 步 = 5.8 min     ← ibi_round0.py 8 8000 就是这一档
100 ps = 50000 步 = 36 min
200 ps =100000 步 = 72 min
```

**有 GPU 的话这个数会变，先在那台机器上重量一次**，再决定每轮跑多长。

反面的数据点：第 0 轮跑到 8000 步时 `bb_bond` 的比值还在往上走（800 步 1.42 → 8000 步 1.58），
**那是没收敛的统计量**。拿没收敛的直方图更新表，更新的是噪声。

## 六、GPU

`torch_cgsim` 自己挑设备（`torch.device("cuda" if torch.cuda.is_available() else "cpu")`），
`determine_k_bb.py`、`check_field_after_fix.py` 等照此。ROCm 机器上 torch 的 `cuda` 命名空间同样可用。

## 七、不要带过去的

- `old_torch_cgsim_ref.py`（115 KB，`8eaaf7f` 的旧场快照；**未跟踪**，含旧常数和被禁的
  `ANGLE_K = K_ANGLE` 别名。它可以从 `git show 8eaaf7f:src/torusfold/scheme2/torch_cgsim.py` 随时重建）
- `short_traj_cmp.py`（根目录那份是 `scripts/` 里**已跟踪**版本的字节副本）

## 八、那台机器上最先跑的三条

```
python -m pytest tests/ -q                      # 131 个测试，确认环境对
python scripts/audit_field_state.py             # 打印全部在场常数 + 哪条路径是活的
python scripts/ibi_round0.py 8 8000             # 第 0 轮残差，同时也是速率基准
```
