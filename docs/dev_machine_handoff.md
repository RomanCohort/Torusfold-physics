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

## 七、还要做的实验（按顺序）

**E1 不平，后面全是白跑。** 顺序不是偏好，是依赖：E2/E3/E4 全都要跟 E1 给的窗口比。

| | 实验 | 命令 | 成本 | 判据 |
| :-- | :-- | :-- | --: | :-- |
| **E1** | 平稳性 | `ibi_round0.py 8 100000 0 0.1 25 20000 --blocks=8` | 28 min | 块间散布几个 percent |
| E1b | 不平就加长 | `8 200000 0 0.1 25 40000 --blocks=8` | 56 min | 同上；burn 翻倍用来分辨「还没够」和「窗口滑走」 |
| **E2** | 多结构 | `8 100000 <IDX> 0.1 25 20000 --blocks=8`，IDX = 1..4 | 4 x 28 min | 把 0.0968 变成分布 |
| **E3** | 结清 §3ay | 见下 | 28 min | 旧场收敛 J 更大 → 符号改对了 |
| **E4** | K_PAIR 正经版 | 2 档 x 3 种子，同一收敛窗口 | 6 x 28 min | 差值 > 块间散布才算差 |
| **E5** | 采样环 | 见第六节第 6 步 | 一天 | **E1 通过才开工** |

**E2 是现在最缺的一条。** 整条 §3ax / §3ay / §3ba 的线只用了一条链（1L2X），而参考是 126 条
池化的。第三个参数就是结构序号（池子是 `len(pairs) >= 8 and 24 <= L <= 34` 筛出来的）。
**「一条链的残差」不叫残差，叫轶事。**

**E3 是唯一能翻掉一条已写下结论的实验**，而且它现在才做得起——§3ay 那个「代价 11%」是在瞬态
窗口上量的。旧现场的 `_sigmoid_f` 会让 `guide shape:` 那行印出 `SHORT-RANGE REWARD`，
所以顺带把那个探针也验了：

```bash
git checkout 424e1d7 -- src/torusfold/scheme2/torch_cgsim.py
python scripts/ibi_round0.py 8 100000 0 0.1 25 20000 --blocks=8    # 先看 guide shape 那行
git checkout HEAD -- src/torusfold/scheme2/torch_cgsim.py          # 用完立刻还原
```

**别再做的两件：**

- **不要用「单种子两档」比 K_PAIR。** 漂移的尺子已经有了：累计 J 在 100 ps 后升了 6%。
  效应若比它小，那是噪声。差值必须大于块间散布。
- **不要在 E1 之前碰采样环。** 残差还在漂的时候，更新式更新的是漂移。

## 八、GPU

torch_cgsim 自己挑设备（torch.device("cuda" if torch.cuda.is_available() else "cpu")），
determine_k_bb.py、check_field_after_fix.py 等照此。ROCm 机器上 torch 的 cuda 命名空间同样可用。

## 九、**不要按镜像名杀进程** —— 这台机器上跑着多个互不相干的活

写这条是因为它今天造成了实际损失：`armA` 第一次启动就死了，A/B 六次夭折。`results/plan_c/run_armA.cmd:14-30`
已经记了现场（`exit code -1`，正是 `Stop-Process -Force` 留下的；随后 120 个孤儿 spawn worker，
子进程在 `reduction.duplicate` 里报 `PermissionError [WinError 5]` —— 父进程没了，句柄复制不了）。

**禁止**：

```
taskkill /F /IM python.exe          # 杀掉这台机器上每一个 python
Stop-Process -Name python -Force    # 同上
Get-Process python | Stop-Process   # 同上，没有 CommandLine 过滤
```

**要这样** —— 按 CommandLine 认领，再按 PID 杀：

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*serve.py*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

同一目录下的三个 ps1（`extend_when_round3_done` / `reload_when_round0_done` /
`switch_when_round4_done`）已经是这个写法：`Where-Object { $_.CommandLine -like '*ibi_loop.py*' }`
之后 `taskkill /PID $p.ProcessId /T /F`。**照抄它们的形状**，不要简化成按名字。

### 已经存在的防御，不要拆掉

`C:\ana\envs\circrna3d\python_ibiA.exe` 和 `C:\ana\envs\comfyui\python_ab.exe`
是**解释器的副本，只是换了个名字**。名字里带下划线不是笔误：

- `multiprocessing.spawn` 用 `sys.executable`，所以整个 worker 树都继承这个私有名字
- 一次针对 `python.exe` 的宽杀**碰不到它们**

所以看到 `python_ibiA.exe` / `python_ab.exe` 在跑，**那就是正在进行的实验，别当成陌生进程清掉**。
`run_armA.cmd:42` 的 `set PY=...python_ibiA.exe` 就是这套防御的接入点。

### 顺带一条：不要用会等待的长命令去轮询别人的实验

跑在同一个工作区的会话之间不共享状态。一个用 `Get-Process python` 或
`taskkill /IM` 做"清理"的会话，等于把别人跑了几小时的东西一起清掉，而它自己看不到这一点。

## 十、**锁** —— 2026-10-02 那十九小时就是这么丢的，别再重新发现一次

这一节是整份交接里最难靠读代码补回来的部分，所以写全。

### 症状长什么样

`run_armA.cmd` 的 retry 循环**每次 attempt 都在一秒内返回 `exit=0`、写入 0 字节、
arm 从不启动**。日志看起来像成功：`exit=0` 是写上去的，`echo` 行也正常打印。
八个 attempt 一轮，二十秒一次，整下午都是这样。

### 真正发生了什么

`scripts/ibi_loop.py:958` 用上下文管理器开 worker 池：

```python
with ctx.Pool(processes=min(_N_WORKERS, len(remaining))) as pool_procs:
```

**上下文管理器只在正常退出时回收 worker。** driver 一旦被强杀、或在 with 块之外异常退出，
`__exit__` 不执行，32 个 worker 就成了孤儿 —— 而它们**继承了重定向的 stdout 句柄**，
于是继续持有 `results/plan_c/armA.out`。

而 **cmd 的 `>>` 打不开被锁的目标时，它跳过那条命令** —— 不是报错，是跳过。
ERRORLEVEL 保持 0，循环体照常跑完。所以：

```
echo 行正常打印      ← 块在执行
exit=0 被记录        ← 命令从未运行，ERRORLEVEL 没被动过
python 从未启动      ← 没让它做任何事
目标文件 mtime 冻结   ← >> 从未打开它
```

实测两次：`armA_smoke.out` 被 4 个 10-01 的孤儿持有 **19 小时**（文件 mtime 冻在
`19:11:32`，正是它们启动后 11 秒）；随后 `armA.out` 又被上一次 run 的孤儿 driver 持有。

### 怎么判断是不是它

**先查锁，不要先查引号。** 引号不是原因，这一点我用一整天和三次错误结论换来的：

```powershell
$f = 'results\plan_c\armA.out'
try { [IO.File]::Open($f,'Append','Write','None').Close(); 'openable' }
catch { 'LOCKED: ' + $_.Exception.Message }
```

`Move-Item` 同样的路径也能测：锁住的文件重命名也会失败。

### 怎么解

按 **PID** 停掉持有者。已经写成脚本，`run_armA.cmd` 每次启动前会调用它：

```
"%PY_BOOT%" results\plan_c\stop_stale_workers.py
```

它按 CommandLine 匹配 `ibi_loop` 找 driver，按父 PID 找 worker，再逐个 `taskkill /F /PID`。
`PY_BOOT` 是**另一个解释器**（`comfyui\python.exe`），故意不用 `%PY%`。

### 两个坑，都踩过

1. **别用 PowerShell 的一行模糊匹配。** 最自然的写法
   `Get-CimInstance Win32_Process | Where-Object CommandLine -like '*ibi_loop*'`
   **会匹配到它自己的命令行**，于是杀掉自己的进程树。这个错误在 2026-10-02 犯了**两次**。
   所以清理脚本用 Python 写：它跑在 `python.exe` 下，只杀 `python_ibiA.exe`，两者不可能混淆。

2. **前置检查不够。** 我加过一个"进循环前试写目标"的检查，实测**它通过而运行照样失败** ——
   检查采样的是一个瞬间，而重定向需要的是整段跨度。**先杀遗留进程才是真修复**，
   检查只是把静默 no-op 变成具名 ABORT 的放大器。两个都要，但别把后者当前者。

### 兄弟任务的免疫是结构性的

`plan_c_ab2oiu` 跑 `python_ab.exe`，armA 跑 `python_ibiA.exe`，清理脚本的过滤器只认后者，
所以它碰不到兄弟任务。**这不是巧合，是那两个改名解释器的用处之一** —— 别把它们改回 `python.exe`。

