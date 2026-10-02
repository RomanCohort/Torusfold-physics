# 工作区残留物 —— 五件东西,从仓库根目录移到 `_strays/`(2026-10-02)

**`_strays/` 不进 git**(`.gitignore:136` 有规则和理由)。它是个**停放处**,不是删掉的垃圾桶:
里面每一样都还在,想拿就拿。删掉它们的判断依据写在下面每一项里。

## 为什么会有这个目录

这五件东西原来在**仓库根目录**,和 `README.md`、`serve.py` 平级。它们跟着**整夹拷贝**走 ——
所以每换一台机器就多一份,而且 `git status` 永远显示五个未跟踪文件。

`docs/dev_machine_handoff.md` 第一节其实已经交代过这件事:

> **根目录已经干净了。** 之前有两个未跟踪文件会跟着整夹拷贝走,已移出仓库,
> 现在在 `D:\torusfold-physics\_repo_strays\` —— 没删,要就自己拿

问题是**那台机器上的 `_repo_strays\` 在这台机器上不存在**(已验证:`D:\torusfold-physics\_repo_strays`、
`D:\torusfold-hybrid`、`..\_repo_strays` 三个路径全都不存在)。整夹拷贝把它们又带回来了,
于是文档说「干净」、实际不干净。

所以这一次不靠「另一台机器上有个文件夹」,而是**在仓库里放一个被 ignore 的停放处**,
把每一样是什么写下来。

## 逐项

### `old_torch_cgsim_ref.py` —— 115,277 字节,2026-09-11

`8eaaf7f` 那个提交的 `src/torusfold/scheme2/torch_cgsim.py` 快照。**它是唯一副本。**

为什么在:它保存着**修正前的旧力场**,而修正的内容是「旧的手性引导项把远距离碱基对**推开**」
(病态势),新场在 ~r0+3w ≈ 1.6 nm 内**吸引**。要有东西能证明这个修正是真的,
就必须能同时加载旧场和新场跑同一条轨迹。

**谁引用它**(已跟踪的代码指着它,这四处在文件没提交时是断的):
- `scripts/short_traj_cmp.py:21`
- `scripts/verify_dihedral_force_fd.py`
- `docs/statistical_potentials_as_forces.md`
- `docs/dev_machine_handoff.md:25`

**重建路径(已实测可行)**:

```bash
git cat-file blob 8eaaf7f:src/torusfold/scheme2/torch_cgsim.py > old_torch_cgsim_ref.py
```

**行尾的坑。** 重建出的 blob 是 **LF**,112,733 字节;盘上这份是 **CRLF**,115,277 字节,
差值 2,544 = 恰好 2,544 行。**内容规范化为 LF 后两者逐字节相同**(已核对
`CONTENT IDENTICAL: True`)。所以重建是可行的,但别用 `ConvertTo-Json` 之类的工具过一遍。

**另有一个陷阱**:PowerShell 5.1 的 `>` 重定向写的是 **UTF-16LE**,不是 UTF-8。
用 `git show ... > file` 会得到一个双倍大小、无法当 Python 加载的文件
(实测 227,444 字节 = 2 × 113,722)。要字节正确,用 `git cat-file` + 重定向到文件后
`[System.IO.File]::WriteAllBytes`,或直接用 bash 的重定向。

**结论:可重建,所以删掉不会永久丢失 —— 但这是本机唯一一份,留在 `_strays/` 里。**

### `input_relax_1l2x` —— 1,423 字节,2026-09-13

`1L2X` 那条链的 **oxDNA / oxRNA 弛豫输入**。它是 oxDNA `RELAX_INITIAL_CONFIGURATION`
配方在 RNA 上的对应物,头部注释说明了它为什么必须存在:

> 晶体构象的成键骨架距离落在 FENE 势阱之外,所以常规 RNA MD 拒绝加载它
> ("Distance between bonded neighbors ... exceeds acceptable values")。
> 这个相互作用用**恒力(线性势)替代 FENE 键**,线性势没有最小值可供落在外边,
> 于是把结构推向 FENE 最小距离。

**`1L2X` 出现在很多已跟踪的文件里** —— `README.md`、`docs/cg_allatom_interface.md`、
`docs/dev_machine_handoff.md`、`docs/ibi_loop_and_oxrna_findings.md`、
`docs/pipeline_audit_2026-09-13.md`、`docs/statistical_potentials_as_forces.md`、
`docs/timeline.md`,以及 9 份 `results/ibi_*/manifest.json` 和 15 个 `scripts/`。

**但没有任何已跟踪文件引用这个输入文件本身**(`git grep input_relax` 无结果),
而且**没有重建路径** —— 它不是从某个提交生成的,是手写的。

**结论:唯一副本 + 无出处 + 被大量文档依赖的那条链的弛豫参数只在这里。留。**

### `_partial_round0.py` —— 2,513 字节,2026-09-18

回合 0 的聚合草稿:把 `results/ibi_relax/tasks_r0` 下每个 `.npz` 读出来,汇总
out-of-support 比例、`joint_J` 分布、入场能量/最大力、以及 1500 步入场弛豫之后
有多少条链**离开了 5000 的上限**。

路径是硬编码的 `C:\baidunetdiskdownload\torusfold-hybrid\results\ibi_relax\tasks_r0`,
但它读的数据**仍然在**(2,601 项),所以脚本还能跑。

**结论:草稿性质的分析工具,数据还在。留,但不必提交。**

### `_progress.py` —— 779 字节,2026-09-17

14 行,从 `results/ibi_full/run.log` 里把每链耗时正则解析出来,算总 core-seconds、
最长链、中位耗时。

**`results/ibi_full/` 这个目录不存在**(已验证),所以这个脚本**已经跑不起来了**。
它分析的是一轮「没有弛豫」的旧回合,那一轮的记录不在本机。

**结论:已废。这里唯一一个可以放心永久删除的文件。**

### `short_traj_cmp.py` —— 5,426 字节,2026-09-11

**和 `scripts/short_traj_cmp.py` 逐字节相同**(两者 SHA1 均为 `6D2CEE2AFAC8`,
140 行)。`docs/dev_machine_handoff.md:27` 也是这么说的。

它是一次整夹拷贝留下的重复件。注意**已跟踪的那一份同样有硬编码的 `D:/` 路径**
(`sys.path.insert(0, 'D:/torusfold-hybrid/src')`、`load_old('D:/torusfold-hybrid/old_torch_cgsim_ref.py')`),
所以两份在本机都跑不起来 —— 运行记录是
`docs/traj_comparison_2026-09-10.txt`,在 D 盘那台机器上产生的。

**结论:纯重复件。可以永久删除;想留也无害,因为它已有一份在 `scripts/` 里。**

## 如果以后要清理

```powershell
# 确认它是重复件再删
(Get-FileHash '_strays\short_traj_cmp.py').Hash -eq (Get-FileHash 'scripts\short_traj_cmp.py').Hash
# 确认它已废再删
Test-Path 'results\ibi_full\run.log'   # False 就意味着 _progress.py 没有输入
```

**这个目录不该变大。** 往里加东西之前先问:它是源,还是某次运行的残留?
如果是源,它属于 `scripts/` 或 `docs/`;如果是残留,它属于 `.gitignore` 里的某个
`output_*`/`results/` 规则,不属于这里。
