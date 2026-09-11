# 开发机交接

## 一、怎么传

**整个文件夹拷走就行。** 数据已经在里面了，不需要环境变量。

```
D:\torusfold-hybrid\                12.3 MB   代码（+.git 10.9 MB）
  └ _cgdata\                       335.1 MB   数据，gitignore 掉的
      ├ rsRNASP\Training_set\       13.8 MB   192 个文件，191 个 PDB（沉积结构数据库）
      └ cgRNASP\                   321.3 MB   2841 个文件（cgRNASP 参考实现的数据）
```

合计 **约 360 MB**，压缩后更小（PDB 是文本）。

**_cgdata/ 已加进 .gitignore**，所以 git status 永远干净，git add -A 也绝不可能把它带进历史。
这条规矩存在的原因：曾经有 4.9 GB 进过历史，清了三轮 git filter-repo。

**如果嫌大：** _cgdata/cgRNASP 那 321 MB 只被两个脚本用
（cg_rnasp_evaluator.py、inspect_potential_table.py）。删掉它，其余全部照常，总量降到 ~26 MB。

**根目录已经干净了。** 之前有两个未跟踪文件会跟着整夹拷贝走，已移出仓库，现在在
D:\torusfold-physics\_repo_strays\ —— 没删，要就自己拿：

- old_torch_cgsim_ref.py —— 115 KB，8eaaf7f 的旧场快照，含旧常数和被禁的 ANGLE_K = K_ANGLE。
  能从 git show 8eaaf7f:src/torusfold/scheme2/torch_cgsim.py 随时重建
- short_traj_cmp.py —— 与 scripts/ 里**已跟踪**版本逐字节相同

## 二、远端

仓库已推到最新，所以也可以直接 clone：

```
git clone https://gitlab.igem.org/2026/software/jlu-fbh/torusfold-hybrid.git
```

分支 main。GitHub 镜像 github.com/RomanCohort/Torusfold-physics，分支 master。
**但 clone 拿不到 _cgdata/**（它在 gitignore 里），所以走网盘就整夹拷贝，走 clone 就得单独把数据放回去。

## 三、路径怎么解析

scripts/_cgdata.py 是唯一一处解析，顺序：

1. 环境变量 TORUSFOLD_RSRNASP / TORUSFOLD_CGRNASP（若设了）
2. 仓库内的 _cgdata/...（若存在）
3. 原来的绝对路径 D:\torusfold-cgdata\...（兜底）

第 3 条**故意放最后、故意保留**：docs/statistical_potentials_as_forces.md 里每个数字都是读第 3 条产生的，
一个会静默优先别处的解析器，会把那些测量和引用它们的代码放到不同的数据上。

三条分支都验过：不设 → 仓库内；设了 → 覆盖生效；cgRNASP → 存在。
**17 个脚本全部改成走它了**（之前是 17 份同样的字面量 —— 那是 bug，不是配置）。

## 四、IBI 那条链要什么

| 脚本 | 作用 | 依赖 |
| :-- | :-- | :-- |
| scripts/_cgdata.py | 数据路径解析 | 标准库 |
| scripts/boltzmann_bonded.py | 数据库加载、表、mixed_energy（autograd 取力）| numpy, torch |
| scripts/ibi_round0.py | 测残差（**现在就能跑**）| torch |
| scripts/ibi_bonded.py | 更新式 + 失败处理（675 行，25 个测试）| numpy |
| scripts/sample_bonded_chain.py | 旧采样器（**只跑键合项，会解开**，见下）| numpy, torch |

**ViennaRNA 和 OpenMM 不在这条链上。** grep 确认 import RNA 全仓库只有 3 个脚本
（measure_pair_weight_quality.py、measure_dividefold_tier_room.py、ppr_repair_v3.py），都不在 IBI 链里。
不过还是照 requirements.txt 装一遍稳妥。

## 五、**还没写的东西 —— 这是关键**

**IBI 现在只能测残差，不能迭代。** 缺一个**跑整场、但把三个角坐标（angle / dihedral / stack）
换成表格势**的采样环。

sample_bonded_chain.py 不能直接改成那个：它刻意只跑 P 原子的四个键合项
（docstring 第 10 行「no nonbonded terms」，第 13–19 行解释为什么排除两个 intra 项），
而**参考边缘分布以折叠为条件**，所以一条没有非键合项的链会解开，跟参考比的是两件事。

## 六、到那台机器上的顺序

**先把 commit 钉住。** 下面每个数字都是 `b2d0641` 上的：

```bash
git checkout b2d0641                 # 或者 clone 之后 git pull
python -m pytest tests/ -q           # 135 个测试，确认环境对
python scripts/audit_field_state.py  # 全部在场常数 + 哪条路径是活的
```

**第 0 步：量速率。** `ibi_round0.py 8 8000` 既是第 0 轮残差，也是速率基准。
本机（纯 CPU，8 副本批处理，1L2X 81 珠）是 **22.3-23.1 steps/s**：

```
1 ps   =   500 步 = 22 s
16 ps  =  8000 步 = 5.8 min     ← 第 0 轮就是这一档
100 ps = 50000 步 = 36 min
200 ps =100000 步 = 72 min
```

有 GPU 就先重量一次，别拿这个数排计划。

**第 1 步：把基线复现出来，对不上就别往下走。** `ibi_round0.py 8 8000` 在 `b2d0641` 上应给出：

| 坐标 | sim/ref |  | 坐标 | sim/ref |
| :-- | --: | :-- | :-- | --: |
| bb_bond | 1.703 | | angle | 1.149 |
| intra_pc | 1.624 | | dihedral | 0.969 |
| intra_cn | 1.606 | | stack | 1.345 |

联合 `mean|ln(sim/ref)|` = **0.3263**。这条链是随机的，1.5% 以内算同一档；差得多说明数据路径或
常数不对，先回去查 `_cgdata.py` 的三条分支。

**第 2 步：K_PAIR 扫描。这是 §3ay 留下的唯一未决项，而且它便宜，并且能证伪我的判断。**
§3ay 测到向导改正号之后联合残差从 0.2937 走到 0.3263。成因我写成了读法（已标注无独立测量）：
两种形状在井内同力、曲率反号（-k/(4w²) → +k/(4w²)），把碱基对的等效弹簧从 600-130 抬到 600+130。
**若这个读法对，降 K_PAIR 就应该把联合残差拉回来。** 所以扫它：

```bash
python scripts/ibi_round0.py 8 8000      # K_PAIR=600 基线，应复现 0.3263
# 再把 K_PAIR 改到 470 / 400 / 340，各重跑一次，比联合 mean|ln(sim/ref)|
```

判据用 README 那一条：**改善不到约 10% 就不算改善。** 600→470 若只动 2%，我的读法就是错的，
该去别处找，而不是继续降 K_PAIR。区间是 [204.8, 2173.9]（`scripts/select_k_pair_by_ranking.py`
的配准位移诱饵给上界），区间内没有别的测量——**这一扫本身就是在给它造一个。**

**第 3 步：查最小化为什么卡。** 这是 README 验收表里那行 `currently fails`。
`check_field_after_fix.py:102-116` 是最速下降 + 步长减半，6 次重启、max|F| 690.19。
先换 LBFGS 看能不能降到 236.3 以下：

- 能 → 是方法的问题，那条读数以后按 LBFGS 记，验收表改一行；
- 不能 → 场里真有一条没配平的力，那才是物理问题，得定位到哪一对珠子。

**第 4 步：写那个采样环。这是 IBI 真正的阻塞点，也是唯一挡着下一步的东西。**
见第五节。补一句它必须长什么样：从**天然结构**起步、跑**整场**（含非键合项），
只把 angle / dihedral / stack 三个调和项换成表格势，表格势的力用
`boltzmann_bonded.mixed_energy` 的 autograd 梯度。少一样都会丢掉「折叠」这个条件，
而参考边缘分布**是以折叠为条件的**。

**第 5 步：写更新环之前，先量直方图收敛。**
反面数据点：第 0 轮跑到 8000 步时 bb_bond 的比值还在往上走（800 步 1.42 → 8000 步 1.58），
**那是没收敛的统计量**。拿没收敛的直方图更新表，更新的是噪声。先定「跑到多少步直方图不再动」。

**不要顺手动的两样：**

- `openmm_gpu_refiner.py`（CPU / OpenMM 那条路）——**它是另一个模型、另一套靶**，不在这一轮里。
- `K_BSJ` / `K_BSJ_GUIDE` / `K_BSJ_CONTACT`——这份数据库里没有一条共价闭合的链，
  在那台机器上也定不了标。

## 七、GPU

torch_cgsim 自己挑设备（torch.device("cuda" if torch.cuda.is_available() else "cpu")），
determine_k_bb.py、check_field_after_fix.py 等照此。ROCm 机器上 torch 的 cuda 命名空间同样可用。
