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

**GitHub 能不能推取决于加速器开没开，不是远程坏了。** `127.0.0.1 github.com` 是**加速器自己的机制**，
不是广告拦截表：那台机器上 `127.0.0.1:443` 的监听进程就是 `Steam++.Accelerator`，
旁边的 `www.github.com 140.82.112.4` 和那一堆 steam / googleapis 域名是同一套配置。

**所以不要删 hosts 里那一行**——删了会把加速器弄坏。（我先前把它当成拦截表，说错了，已改。）

| 加速器 | `Resolve-DnsName github.com` | `git push github` |
| :-- | :-- | :-- |
| **开** | 127.0.0.1（本机代理在听） | **通**，原生命令就行 |
| **关** | 127.0.0.1（没人听） | `fatal: unable to connect to server` |

关的时候，不碰 hosts 的一次性绕法：

```
git -c http.sslBackend=schannel -c http.curloptResolve=github.com:443:<IP> push github main:master
```

**`http.sslBackend=schannel` 这一半不能省**，原因在本仓库自己的配置里：`.git/config` 有
`http.sslBackend = openssl`，而系统级 `C:/Program Files/Git/etc/gitconfig` 是 `schannel`。
仓库级覆盖系统级，而这个 Windows 构建**没有编 openssl**，所以走 libcurl 的那条路会报
`fatal: Unsupported SSL backend 'openssl'`。**普通推送不受这行影响**，只有 `curloptResolve` 那条路撞它。

**IP 要探，不能抄。** 关加速器时 GitHub 的接入点很不稳——`20.205.243.166` 上午 0.88 s、下午就超时。

```
foreach ($ip in '20.205.243.166','140.82.116.3','140.82.113.4','140.82.112.3') {
  $c = curl.exe -sS -m 8 --resolve github.com:443:$ip -o NUL -w '%{http_code} %{time_total}' https://github.com 2>&1
  "$ip -> " + ($c -join ' ')
}
```

**GitLab（origin）是另一回事。** 它不需要加速器，但**凭据助手里的 token 是坏的**：原生的
`git push origin main` 会报 `fatal: Authentication failed`，而且**没有 TTY 时会挂住等输入**
（一次 600 s 超时就是这么来的）。用带 token 的 URL 推：

```
git push https://oauth2:<token>@gitlab.igem.org/2026/software/jlu-fbh/torusfold-hybrid.git main
```

推完记得撤销 token。

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

## 六、到那台机器上的顺序（v2，第一轮实测之后重写）

**第一轮报了六件事，三件推翻了我原计划的数字。** 下面的顺序按实测重排；完整记录在
docs/statistical_potentials_as_forces.md §3az。

**第 0 步：先确认这份代码是哪个版本。目录快照不等于版本。**
第一轮那份没有 .git，内容是 `424e1d7`，而原计划里每个数都是 `6e7a44b` 的。
**两边对不上时先查版本，别先怀疑数据。** 没有 .git 时用内容认：

**最硬的一条是跑出来的数**：`ibi_round0.py 8 8000` 在修好的场上给 bb_bond **1.703**、联合 **0.3263**；
给 1.588 / 0.2937 就是旧场。种子固定（20260218），所以这条逐位可复现，比数测试文件硬。

```bash
python -c "import io,re;print(re.search(r'is (\d+) tests', io.open('README.md',encoding='utf-8').read()).group(1))"
#   131 -> 424e1d7（旧场：向导符号未改，无 §3ay，无 docs/silent_defects.md）
#   135 -> 6e7a44b 及以后（当前；HEAD 是 fd862e6，其后只动文档）
python -c "import io;s=io.open('src/torusfold/scheme2/torch_cgsim.py',encoding='utf-8').read();print('old guide shape present:', '(r0 - dist)' in s or 'r0-dist' in s)"
```

**第 1 步：环境。pytest 装不上不阻塞。** `tests/` 只是回归网，不在这条链上。
要确认的只有两样：torch 能 import，和 `_cgdata` 解析到仓库内那份：

```bash
python -c "import sys;sys.path.insert(0,'scripts');import _cgdata;print(_cgdata.rsrnasp())"
```

（`audit_field_state.py` 崩在五路径对比是真 bug：`6e7a44b` 上还在，本轮的修复提交之后才没有。）

**第 2 步：速率已经有了，别重量。** 那台机器 **60.2-62.1 steps/s**（8 副本），本机 22.3-23.1。
16 ps = **130 s**，不是 5.8 min。**本机排不下的一次扫描，在那边是几分钟的事。**

```
8000 步  =  16 ps = 2.2 min
40000 步 =  80 ps = 11 min
100000 步= 200 ps = 28 min
```

**第 3 步：分块 J。这是现在唯一的闸门，也是采样环能不能开工的判据。** —— **40-200 ps 已跑完，见 §3ba。**

新场、40-200 ps：**J = 0.0968**，六项里三项几乎精确（intra_cn 1.000、stack 0.992、intra_pc 1.010），
**但累计 J 在 100 ps 触底后升了 6%**（0.0911 → 0.0968），说明 100-200 ps 那段更差。
累计量分不清「早期运气好」和「真平衡」，所以判据换成**块间散布**：

```bash
# 采样窗口 40-200 ps，切成 8 个不相交的块，每块各自算 J，并印出每块的每个坐标 sim/ref
python scripts/ibi_round0.py 8 100000 0 0.1 25 20000 --blocks=8
```

**看块间散布，不看累计线的走向。** 几个 percent 以内算平；几十 percent 说明还在漂，
任何残差都是混合量。`--blocks=N` 会指出**是哪个坐标在动**——已知还在动的是 dihedral
（40→200 ps 之间 0.720 → 0.708，持续收窄）。

块不平就加长：NSTEPS 翻倍、burn 保持 20000 不动。按 60 steps/s，200000 步 ≈ 56 min。
（`ibi_round0.py` 的 provenance 现在还会印 `guide shape: E(0.5 nm)=... E(3.0 nm)=...`，
用来分辨向导是长程形状还是改前的短程奖励——常数指纹认不出形状。）

**块平之前，不要写采样环，也不要再比任何残差。**

**第 4 步：K_PAIR 扫过了，结论是保持 600；要不要重扫，等第 3 步的平台。**
四档 600 / 470 / 400 / 340，最好一档 +3.31%，在 10% 门槛之下。但**那次扫描跑在旧场（`424e1d7`）上**，
检验的不是 §3ay 那个读法（新形状把等效弹簧从 600-130 抬到 600+130）——旧形状在井内的曲率是负的，
降 K_PAIR 是继续往下压。**两个问题不一样。** 而且残差是瞬态时，这一扫本身也该在收敛窗口上重做。

**第 5 步：最小化。** 旧场上最速下降 1500 次到 max|F| 25.68、6000 次 69.80，都在 236.3 之下、0 次重启；
新场上是 4427 次 / 6 重启 / 690.19。**换 LBFGS 再测一次，但手写的线搜索不算数**——
第一轮那个手写版落到 5000.00 = force_cap，进了帽里，不能作为 L-BFGS 的结论。

**第 6 步：采样环。前提是第 3 步给了收敛窗口。**
**只换 angle 和 dihedral，不换 stack。** K_STACK = 0，而 §3ac 已证明 stack 是
`|b_i|^2 + |b_(i+1)|^2 - 2|b_i||b_(i+1)|cos_a` 的精确恒等式（R² = 1.000000，残差 6.7e-16）；
给一个已消融的精确冗余项换表格势会把冗余原样引回来，而且模型根本没有 base 平面。
其余同前一版：从**天然结构**起步、跑**整场**（含非键合项）、表格势的力用
`boltzmann_bonded.mixed_energy` 的 autograd 梯度。

**仍然不要动的两样：**

- `openmm_gpu_refiner.py`（CPU / OpenMM 那条路）——**另一个模型、另一套靶**。
- `K_BSJ` / `K_BSJ_GUIDE` / `K_BSJ_CONTACT`——这份数据库里没有一条共价闭合的链。
  在那台机器上也定不了标。

## 七、GPU

torch_cgsim 自己挑设备（torch.device("cuda" if torch.cuda.is_available() else "cpu")），
determine_k_bb.py、check_field_after_fix.py 等照此。ROCm 机器上 torch 的 cuda 命名空间同样可用。
