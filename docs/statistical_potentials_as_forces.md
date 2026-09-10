# 把统计势注入力：这不是新想法，而且我们的失败有名字

> 2026-09-10。为队列项 3（"如何把统计势注入力"）定位。
> 结论先行：**这件事文献里做过，其中一件就是我们在用的那套模型的作者做的。**
> 而 `docs/NOTES.md` 里 TriRNASP 的失败，在文献里有明确的原因和解法。

## 1. 文献

| 文献 | 做了什么 | 对我们 |
| :-- | :-- | :-- |
| **IsRNA / IsRNA1 / IsRNA2**（PMC9731381） | CG 力场的参数化方法叫 **iterative simulated reference state approach**。`E_total = E_bond(b) + E_angle(θ) + E_torsion(φ) + E_bp(r,θ,φ) + E_pair(r)`，简谐 + 高斯 + LJ 型，参数由统计方法确定 | **我们的 CG 力场本身就是统计参数化的**（同一血统）|
| **cgRNASP**（NAR Genom Bioinform 2023;5(1):lqad016, doi:10.1093/nargab/lqad016, PMC9985339） | "a series of residue-separation-based CG statistical potentials at **different CG levels**"，短程/长程按残基间距分开。与全原子 rsRNASP 性能相当，RNA-Puzzles 上略好，且"strikingly more efficient" | TriRNASP 失败的解法 |
| **cgRNASP-CN**（Commun. Theor. Phys. 2022, doi:10.1088/1572-9494/ac7042；GitHub `Tan-group/cgRNASP-CN`） | 最小 CG 表示上的统计势 | 同上 |
| **IsRNA2+**（JCTC, doi:10.1021/acs.jctc.6c01116） | 标题即方法："Developing Explicit Base Stacking Potentials for the IsRNA2+ Coarse-Grained RNA Force Field **Using Iterative Reweighting**" —— 给现成 CG 力场加显式项 | 这就是"注入"本身 |
| **综述** Building RNA coarse-grained force fields: Design principles and training strategies（Biophys J 2026, doi:10.1016/j.bpj.2026.03.041） | CG 力场的设计考量、训练策略；序列依赖、二级结构、三级模体等结构信息如何并入；ML 方向的展望 | 该先读的入门 |
| **rsRNASP**（Biophys J 2022, PMC8758408） | 全原子统计势 | 我们那份元验证计划的对象 |
| Bernauer et al.，**可微**统计势（全原子 + CG 两个层级） | 从 cgRNASP 的引用里看到 | 【未核实原文】要"进力"就得有解析梯度 |
| SimRNA / iFoldRNA / NAST / Vfold / HiRe-RNA / oxRNA / RACER | 统计势 + MC / 离散 MD / CG MD 的范例群 | 见下条引文 |

## 2. 四条要点

### 2.1 "注入"这个说法要改

我们的 CG 力场（IsRNAcirc 血统）本来就是统计参数化的。所以不是"把外来的统计势注射进物理力场"，
而是"**在一个统计势场上再加一层统计项**"。这不是措辞问题 —— 它决定会不会**双重计数**：
知识型势本质上是自由能（PMF）的一类，里面已经隐含着物理相互作用。

### 2.2 失败的原因是分辨率错配，文献里有原话

cgRNASP 原文：

> 现有的传统统计势（RAPDF、KB、DFIRE-RNA 等）**全都基于全原子表示**。因此，**至今各个不同
> CG 层级上仍然急需可靠的 CG 统计势**。

同文另一句（点名了整条路线）：

> 几乎所有现有的物理模型都基于不同层级的 CG 表示，而不是全原子 —— 包括 SimRNA、iFold、NAST、
> IsRNA、Vfold、HiRe-RNA、oxRNA、RACER 和我们带盐效应的 CG 模型。

所以 TriRNASP 失败的病根不是"统计势不能进力场"，而是**拿全原子的统计去配 CG 的场**。
我们记的"量级差 ~50 倍、极小值不一致"正是这个错配的症状。

### 2.3 解法有两条，我们上次两条都没走

| 路线 | 代表 | 做法 |
| :-- | :-- | :-- |
| **同层级计数** | cgRNASP / cgRNASP-CN | 直接在 CG 表示上统计，不改力场 |
| **迭代再加项** | IsRNA2+（iterative reweighting）；IsRNA（iterative simulated reference state）| 反复调整势，直到模拟出的统计量复现目标统计量 |

我们上次是**一次性硬塞一个固定系数**（`trirnasp_scale = 0.002`）。
那既不是同层级计数，也不是迭代标定 —— 所以受挫是预期的。

### 2.4 进力就需要梯度

打分函数只要能量值；进力场要解析梯度。可微的统计势是有的（Bernauer et al.），
但我们没有核过原文，也没有确认它在 CG 层级上的形式是否可直接用。

## 3. 重写后的三个问题

1. **在哪一级计数？** 我们的珠子是 P / C4' / N（三颗），比 IsRNA2 的九类还粗。
   在 P/C4'/N 上直接统计的势，本轮检索**没找到先例**。可能是机会（更省、副本更多），
   也可能是死路（太粗、无判别力）。可测。
2. **怎么加才不打架？** 走迭代（IsRNA2+ / IsRNA 路线），而不是固定系数。
3. **加成，还是重参数化？** 底层场已经是统计的，所以"再加一层"和"重新拟合"是两条不同的路。

## 4. 证据空缺（诚实标注）

- **IsRNA2+ 正文未读**：ACS 付费，EuropePMC 未收录（按标题检索无结果）。只拿到标题里的方法名。
- **cgRNASP 只到摘要级**：具体在哪几个 CG 层级上做、效率数字是多少，尚未核到正文。
- **2026 综述正文未读**：只拿到摘要。
- **Bernauer 可微统计势未核**：仅从 cgRNASP 的引用列表中看到。

## 5. 下一步

1. 读 cgRNASP 全文 —— 尤其 **CG 层级的定义**与**效率数字**。这两项直接决定我们能不能在
   P/C4'/N 上做。
2. 读 2026 综述 —— 训练策略那一节。
3. 然后才谈要不要动 `torch_cgsim.py`。