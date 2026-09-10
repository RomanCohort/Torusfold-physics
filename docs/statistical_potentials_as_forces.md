# 把统计势注入力：这不是新想法，而且我们的失败有名字

> 2026-09-10。为队列项 3（"如何把统计势注入力"）定位。
> 结论先行：**这件事文献里做过，其中一件就是我们在用的那套模型的作者做的。**
> 而 `docs/NOTES.md` 里 TriRNASP 的失败，在文献里有明确的原因和解法。

## 1. 文献

| 文献 | 做了什么 | 对我们 |
| :-- | :-- | :-- |
| **IsRNA / IsRNA1 / IsRNA2**（PMC9731381） | CG 力场的参数化方法叫 **iterative simulated reference state approach**。`E_total = E_bond(b) + E_angle(θ) + E_torsion(φ) + E_bp(r,θ,φ) + E_pair(r)`，简谐 + 高斯 + LJ 型，参数由统计方法确定 | **我们的 CG 力场本身就是统计参数化的**（同一血统）|
| **cgRNASP**（NAR Genom Bioinform 2023;5(1):lqad016, doi:10.1093/nargab/lqad016, PMC9985339） | 在**三个 CG 层级**上做的统计势：① **3 珠 = P / C4\' / N9(嘌呤) 或 N1(嘧啶)** —— 代表作；② cgRNASP-PC，2 珠 = P / C4\'；③ cgRNASP-C，1 珠 = C4\'。短程/长程按残基间距分开。与全原子 rsRNASP 性能相当，RNA-Puzzles 上略好，"strikingly more efficient" | **①的珠子定义与我们的 P/C4\'/N 是同一约定** —— 分辨率错配这一条被它填掉了 |
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

1. **在哪一级计数？—— 开关已经查清：我们的 3 珠是名义上的，不是真的。**

   cgRNASP 的 3 珠是 **P / C4' / N9(嘌呤) 或 N1(嘧啶)**，都是真实重原子。我们的不是：

   | 位置 | 珠子怎么来的 |
   |---|---|
   | `torch_gpu_refine.py:132-133` | `pos_c4 = pos_p + bb_dir*0.034`；`pos_n = pos_p + bb_dir*(-0.015)` —— **沿骨架方向的偏移** |
   | `openmm_gpu_refiner.py:224-225` | `p + rng.normal(0, 0.3, 3)` —— **随机扰动**，注释写 “corrected later by minimize” |
   | `openmm_gpu_refiner.py:233-235` | 索引定义 `P(i)=3i, C4(i)=3i+1, N(i)=3i+2` |
   | `aform_from_template.py:52-53` | 全原子重构用的是**另一组**三锚点：**P / C1' / C4'** |

   两条后果：

   - **cgRNASP 的统计不能直接搬。** 珠子数一样（3）、名字一样（P/C4'/N），
     但 N 不是糖苷氮，C4' 也不一定是真 C4'。同名不同物。
   - 更要紧的一条：`torch_gpu_refine.py` 里 C4' 和 N **都与 P–P 方向共线**，
     所以这两颗珠子只编码局部主链切向，不携带任何垂直于主链的信息，**也就没有序列信息**。
     在它们上面加统计势，等于给“P 坐标 + 切向”再加一层统计 —— 增量可能很小。

   **出路**：`aform_from_template.py` 的 1EHZ 重构本来就会给出每个残基的**真实**全原子几何。
   从那里读出真正的 C4' 与 N9/N1，就得到一个货真价实的 CG 表示，可以在上面按 cgRNASP 的方式统计。
   **这件事必须先做**，否则后面所有关于统计势的讨论都是在装饰珠子上做文章。
2. **怎么加才不打架？** 走迭代（IsRNA2+ / IsRNA 路线），而不是固定系数。剩下两件没人替我们做：
   把 cgRNASP 的统计转成**可微**形式（它原本是打分用的距离依赖势，不是力），以及定标。
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