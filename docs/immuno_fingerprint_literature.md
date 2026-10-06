# 免疫指纹:文献校准笔记

这份文件只做一件事:把「结构 → 免疫指纹」的每个特征钉到**具体文献和具体数字**上,
并标出哪些是机制明确的、哪些是代理量、哪些文献直接否掉了。

结论先说:**原设计的五特征里有两条站不住** —— TLR7 对 circRNA 基本不响应,
NLRP3 的"dsRNA 长度"锚点是错的机制。替换方案在最后一节。

写这份文件的起因是 NLRP3 那条走 (b)(查文献校准)而不是 (a)(直接编个代理量)。

---

## 一、RIG-I 是 circRNA 的主要受体,TLR7 不是

Chen et al., *Mol Cell* 2019(PMC6778039)的原文结论:

> RIG-I is necessary and sufficient for innate immunity to foreign circRNA
> (Chen et al., 2017) while toll-like receptors are not responsive to circRNAs
> (Wesselhoeft et al., 2019).

**这条直接改掉了原设计里的 TLR7 项。** 我原来写「TLR7 认单链 GU-rich + 可及性」在
**线性 RNA / mRNA 疫苗**语境下是对的,但 circRNA 语境下 TLR 不参与。所以:

- 「可及单链 GU 含量」这个量**可以算,但它预测的不是 TLR7 对 circRNA 的响应**
- 要留在指纹里的话,必须改名成「内体 TLR 通路的外推风险」并标注 `mechanism: "extrapolated"`,
  或者干脆删掉

## 二、m6A 是「自己人」标记,而且位置是 +40~100 nt —— 这是唯一带剂量数据的校准点

### 位置

m6A-irCLIP 在**环化接头 3' 侧 50–100 nt 内**检出 m6A 富集;转录组范围内,
内源 circRNA 的 m6A 富集在 **BSJ 之后 +40–100 nt 窗口**(Chen 2019, Fig. 2D/2E)。

### 剂量-反应(这是能拿来校准的硬数字)

| 构建体 | 改动 | 抗病毒基因诱导 |
|---|---|---|
| circFOREIGN (wt) | — | 基线 |
| ΔRRACH | 12 个 RRACH 位点 → RRUCH | ↑ **约 2 倍** |
| A-less | 接头后**前 200 碱基内全部 A → U**(37 个位点) | ↑ **约 100 倍** |

### 机制细节(决定指纹该怎么算)

- m6A 的安装**由内含子身份决定,与 circRNA 外显子序列无关** —— circSELF 和 circFOREIGN
  **外显子序列完全相同**,只因为环化用的内含子不同(人 ZKSCAN1 vs 噬菌体 td),
  一个带 m6A 不免疫,一个不带 m6A 强免疫
- **未修饰和 m6A 修饰的 circRNA 都能结合 RIG-I,但只有未修饰的能激活它**
  (导致 MAVS 纤维化)
- YTHDF2 是必须的读者蛋白:1% 或 10% m6A 在 `YTHDF2−/−` 细胞里**不再抑制**免疫
- **1% m6A 就足以钝化**体内佐剂活性(CD8 T 细胞反应和抗体滴度都下降)

### 这对 TorusFold 意味着什么(重要)

**BSJ 是构建设计的选择,不是结构的性质。** 同一条外显子序列,换个内含子环化,
免疫原性可以完全不同 —— 而 TorusFold 拿到的输入是**成熟 circRNA 序列**,
接头在哪必须由外部提供。

好消息是**这个信息在仓库里已经有位置了**:

- `src/torusfold/circrna_library/schema.py:522` —— `class BSJSpec`,`bsj_index: Optional[int]`,
  并且校验 `bsj_index < sequence_length`、必须是非负整数
- `src/torusfold/circrna_library/circular_qc.py:186-189` —— 文档字符串明确写:
  「``bsj_index`` is a boundary index: index 0 means residue ``L-1`` to residue 0.
  The function **never infers a BSJ from file order**.」
- 同一文件还有 `bsj_closure` 检查:circularity 或 BSJ index 没显式给出时,
  状态是 `NOT_ASSESSED` 而不是猜一个

这是**正确**的设计 —— 不猜接头。

**但指纹拿不到它。** `artifacts/2013nt/isrnaclong_final.pdb` 全文只有 1 行 REMARK,
没有 `bsj_index`;`provenance.json` 也没记。所以:

| 场景 | `bsj_index` 有吗 |
|---|---|
| 走 CircuForge 构建体记录 → 有(`BSJSpec`) |
| 直接丢一个 PDB 给指纹 → **没有**,必须由调用方传 |

**结论:m6A 项的 API 必须把 `bsj_index` 做成必填参数**,不能有"从文件顺序猜"的默认行为 ——
和 `circular_qc.py` 保持一致。若未提供,该项输出 `NOT_ASSESSED`,
**不要**退化成"整条序列扫 DRACH"(那样算出来的数没有校准依据,而且会让人误以为算过)。


## 三、PKR 认长度,而且 circRNA 可以是 PKR 的**抑制剂**而不是激活剂

机制数据(PMC2570377, *J Mol Biol*):

> PKR activation requires a minimum dsRNA length of ~30 bp and reaches a maximum by 85 bp.

更细的绑定数据:

- dsRBD 可以和短到 **15–16 bp** 的序列结合
- 短于 30 bp 时,结合化学计量符合**重叠配体结合模型**(结构域在螺旋轴上重叠排布)
- 后来的研究把激活下限推到 **19–21 bp**(带 2 nt 3' 突出),甚至平末端 19 bp

另一条更关键(Liu et al., *Mol Cell* 2021, S1097276521010091):

> Exon back-splicing-generated circular RNAs, as a group, can **suppress**
> double-stranded RNA-activated protein kinase R (PKR) in cells.
> ... directly ligated circular RNAs that form **short dsRNA regions** efficiently
> suppress PKR activation **10³- to 10⁶-fold** higher than C16 and 2-AP.

也就是说:**长 dsRNA → 激活 PKR;短 dsRNA 区段的 circRNA → 抑制 PKR**。
同一份指纹里这两者要分开报,不能合成一个"PKR 分数"。

而且同一篇指出:**带多余片段的 RNA 环(比如 group I 内含子产生的)才有免疫原性;
T4 RNA 连接酶直接连接的环免疫原性最小。** 又是「接头工艺决定免疫」,不是「结构决定免疫」。

## 四、NLRP3:原设计的代理量选错了

三条独立的证据都指向「dsRNA 长度不是 NLRP3 的驱动量」:

| 来源 | 结论 |
|---|---|
| PMC4234566, *Human NLRP3 inflammasome senses multiple types of bacterial RNAs* | 明确写「the double-stranded structure of bacterial RNA **is not required**」 |
| FEBS Lett, *Activation of the NLRP3 inflammasome by intracellular poly I:C* | 「**double stranded structure is required** for IL-1β secretion in response to [poly I:C]」 |
| Science, *Human NLRP1 is a sensor for double-stranded RNA* | **NLRP1**(不是 NLRP3)才认 dsRNA,且要求 **>500 bp**,活性随长度上升 |

⚠️ 上面第二、三条**互相冲突**(一条说 poly I:C 需要双链,一条说细菌 RNA 不需要双链)。
这个冲突我没解决 —— 它可能来自 poly I:C 与天然 RNA 的差别、或 NLRP3 vs NLRP1 的混淆。
**在解决之前,任何"dsRNA 负荷 → NLRP3"的代理量都是没有依据的。**

所以原设计的 (b) 路线「用 dsRNA 负荷 + ssRNA 可及性做主成分」**基础假设就不成立**。

## 五、(b) 路线的真实可行性:没有可校准的数据集

(b) 的可执行含义是「拿公开文献里已有的(结构特征, 免疫读出)对做校准」。
查下来的情况:

| 想要的数据 | 实际有没有 |
|---|---|
| circRNA 的**实验结构** + 免疫读出 | **没有**。Chen 2019 全文是 m6A-irCLIP + qRT-PCR,没有结构解析 |
| circRNA 的**预测结构** + 免疫读出 | **没有**。没人把二级结构预测和免疫读出配对发表过 |
| 一组**序列** + 免疫读出 | **有**,就是 Chen 2019 的 wt / ΔRRACH / A-less 三档 |
| PKR 的 **dsRNA 长度 → 激活** 曲线 | **有**,体外纯化体系,曲线干净 |

**结论:能校准的只有 m6A 那一项,而且是用序列命中数校准,不是用结构校准。**
PKR 那一项可以用体外的长度-激活曲线做**参数标定**(不是校准,因为对象不是同一种分子)。

---

## 六、修正后的特征表

| 项 | 原设计 | 修正后 | 机制依据 | 等级 |
|---|---|---|---|---|
| **m6A / RIG-I** | DRACH 命中 + 可及性 | **+40~100 nt(相对接头)窗口内的 DRACH 密度**,再乘每个位点的可及性权重 | Chen 2019 Fig 2D/2E + Fig 3B/3C 剂量数据 | **A**(有校准数据) |
| **PKR** | dsRNA ≥30 bp 段计数 | **拆成两个反向的量**:① ≥33 bp 连续 A-form 螺旋(激活)② 短 dsRNA 区段总量(抑制) | PMC2570377 长度阈值;Liu 2021 抑制活性 | **A**(体外曲线可标定) |
| **接头工艺** | 未纳入 | **新增**:接头类型/来源作为元数据输入,直接调制 m6A 项 | Chen 2019(circSELF vs circFOREIGN 同序列不同免疫);Liu 2021(group I vs T4 连接) | **A** |
| **miRNA sponge** | seed 互补 + 可及性 | 不变 | AGO2 沿单链滑动(PMC4768356),可及性决定能否接触 | B |
| **TLR7** | 单链 GU-rich + 可及性 | **降级**为 `extrapolated`,或删 | Chen 2019 直接否掉 circRNA 的 TLR 响应 | **C** |
| **NLRP3** | dsRNA 负荷 + ssRNA 可及性 | **暂缓**,先解决第四节那个冲突 | 机制未定 | **D** |

## 七、下一步要先定的事

1. **NLRP3 那两个冲突怎么处理?** 我要再查一轮原文(poly I:C 那篇和 NLRP1 那篇的
   实验体系差异),还是直接降级为 `extrapolated`?
2. **TLR7 删还是降级?** 删掉更诚实,但下游如果已经假设有这一项,接口会变。

---

## 八、第一层已实现:`pair_graph_from_coords` 的实测结果

本节记录代码跑出来的数,不是计划。

### 交付的 2013 nt 模型**没有碱基配对几何**

这是最重要的发现,而且它改变了对交付件的描述:

| 量 | 2013 nt 模型 | 2OIU(晶体) | 1QC0(晶体) |
|---|---|---|---|
| 相邻 C1'–C1'(中位) | **6.51 Å** | 5.52 Å | 5.45 Å |
| 全局最近 C1'–C1'(中位) | **5.84 Å** | 5.29 Å | 5.32 Å |
| 全局最近 C1'–C1'(**最大**) | **9.43 Å** | 6.81 Å | 5.74 Å |
| 每残基原子数 | 21.3 | 21.5 | 31.0 |

**骨架不压缩** —— 相邻距离和晶体结构同量级,所以不是解析问题、也不是坐标被缩放。
异常的是第三行:任何残基与它最近邻居的距离**最大只有 9.43 Å**,而 Watson-Crick 配对
需要约 10.4 Å。也就是说,**这份模型里没有任何残基有处于配对距离的伙伴**。

配套证据:全局最近邻中位数 5.84 Å,说明它不是把 2013 个核苷酸摊开成一个环,而是
**折成一团、大量残基挤在一起**(最小 C1'–C1' 只有 2.48 Å,物理上不可能)。

**结论:2013 nt 交付模型上,针对 A-form 几何的配对检测器返回 0 对配对是正确结果,
不是 bug。** 检测器 0.3 秒跑完 2013 残基。这意味着:**不能用这个模型做 PKR 那类
"长 A-form 螺旋"特征** —— 不是检测器不够好,是输入里没有那个信息。

要在 TorusFold 上做免疫指纹,得用一个**真的带配对几何的预测输出**(pipeline 的
中间产物),而不是这个交付用的查看器模型。**这是一个必须先解决的输入问题。**

### 1QC0 上的表现(唯一有独立真值的验证)

19 bp A-form 双链、1.55 Å。检测器返回 **18 对、0 假阳**:

- 全部链间配对,全部 Watson-Crick,全部 C1'–C1' 在 **10.24–10.76 Å**
- 氢键偏差最大 **0.455 Å**(理想 2.95 Å)
- 寄存器全对:C101–D138、C102–D137 …… C119–D120,连续反向 1:−1
- 同文件的 A+B 碎片双链也是完整的 9/9

**判据不是距离。** 1QC0 是 19 层堆叠,它的 C1'–C1' 矩阵里满是相邻碱基 7.4–8.0 Å 的接触,
全部落在任何距离阈值内。把它们挡掉的是**碱基特异的氢键边判据**(G-C:
O6–N4 / N1–N3 / N2–O2;A-U:N1–N3 / N6–O4),实测 34 个堆叠候选被它拒掉。

### 建设中修掉的四个真 bug(都是我自己写错的)

| bug | 症状 | 根因 |
|---|---|---|
| 距离判据单独用 | 1QC0 出 24 对,含 19 个链内伪配对 | A-form 里同链相隔 2–4 残基的 C1' 也是 10–11 Å 且平面平行 |
| 糖苷键方位判据取反 | **拒掉全部 19 对真配对** | 真实 A-form 里两个糖苷键**平行**(实测 3–7°),我按"平行=堆叠"排除,反了 |
| 平面法向未定号 | 11 对真配对被判"平面相距 170°" | SVD 法向符号任意;共面必须测 `min(θ,180−θ)`,且 dihedral 会随之翻号 |
| 下界缺失 | 出现 5.28 Å 的"配对" | 没有 C1' 下界,骨架邻居偶然满足氢键边 |

### 未解决,已记录而非隐藏

- **1QC0 漏 1 对**:C110–D129。C1'–C1' 10.67 Å、糖苷角 12.0° 都在范围内,但三条 G-C
  接触里只有 N2–O2 合格(2.72 Å),O6–N4 是 4.8 Å、N1–N3 是 6.1 Å。这个模型里它不是
  Watson-Crick 边。这一对大概是真实的局部扭曲,我**没有解决,检测器按"不存在"处理**
  而不是猜。
- **不给 cis/trans 标签**。早先版本给过,18 对全部报成 `trans` —— 对 WC 螺旋是错的。
  字段已删除,而不是修成一个我无法验证的值。
- **非标准配对完全不报**。`PAIR_EDGES` 只有 WC 和 G•U wobble,所以真正的 Hoogsteen、
  碱基三联体、mismatch 一律报"没有"。拒绝列表让这可见,但默认输出是欠报的。

### 合成 fixture 不能用作验证锚,原因已查明

`ideal_aform_helix` 生成的"理想 A-form"**检测不到任何配对**,这是**正确结果**:

单参数螺旋把两条链的 C1' 放在同一圆柱上相隔 180°,于是配对距离由半径决定、堆叠距离
由半径加升程决定。真实 A-form 需要 ~10.4 Å 和 ~6.2 Å,**没有任何一个半径能同时给出**:
r=5.2 得到 10.4 配对但 3.5 堆叠;r=6.5 得到 6.2 堆叠但 13.0 配对。真实双链能同时满足,
是因为糖苷键不是径向的 —— 那是这个生成器没有的第二个自由度。

测试里**断言它检测不到配对**,防止有人为了让它"通过"而放松判据。验证锚是 1QC0。

## 九、管线实跑:pairing 在哪、以及一个必须先说清的陷阱

跑法:`tools/run_pipeline_with_checkpoint.py --length 200`(缩小参数,但每个阶段都开着)。
输出 `results/immuno_run/`。

### 配对的三个载体都是 0-based,坐标层是 1-based

这是**最容易静默出错**的一点。实测(200 nt 跑,序列互补性检验):

| 读法 | 互补配对命中 |
|---|---|
| 按 **0-based** 读 `pairs` | **82 / 82** |
| 按 1-based 读 `pairs` | 37 / 82 |

错的读法**看上去不像错的**,只是差一些。而且 `pairs[i][2] == bpp[i, j]` 逐项吻合,
所以 `pairs`、`bpp`、`bpp_high`、`bpp_mid`、`far_pairs`、`stem_blocks` **用的是同一套 0-based 索引**。

`src/torusfold/immuno/checkpoint_pairs.py` 把转换收在一处,并且**用序列互补性自动校验索引基数** ——
将来格式变了会直接抛错,而不是静默偏移一位。

### 四个列表不能互换

| 载体 | 内容 | 200 nt 实测 |
|---|---|---|
| `pairs` | 所有约束,含伪结派生的补充项和非标准接触;第三个数是**约束权重,不是概率** | 82 项 |
| `far_pairs` | 长程子集 | 21 项 |
| `stem_blocks` | **连续螺旋段**,一段一个列表 —— PKR 特征要的是这个 | 6 段,长度 5/5/5/5/4/4 |
| `bpp` | ViennaRNA 配分函数概率矩阵,唯一经过标定的概率 | 200×200,6644 非零,峰值 0.9229 |

`pairs` 的第三个数**不等于** `bpp[i,j]`:82 项里只有 21 项相等(与 `bpp_high` 的 3 项一致)。
`stem_blocks` 合计 28 对,`ss_consensus` 解出 33 对,`pairs` 82 项 —— **三个不同的数,别混用**。

### `ss_consensus` 这个字符串不能直接解析

实测括号**不平衡**:`(` 26 个但 `)` 28 个,`{` 13 个但 `}` 只有 7 个。
也就是说它混合了多种括号且不闭合。**要用配对信息就读 `pairs` / `stem_blocks`,不要解 `ss_consensus`。**

### 交付模型的 21.3 原子/残基来自 CG_to_allatom,而它不在仓库里

本次跑出来的**所有** PDB 都是 **1.0 原子/残基 —— 每残基一个 P**,包括文件名叫 `.pdb` 的那些。
原因:`src/torusfold/scheme2/isrnacirc_wrapper.py` 需要 `CG_to_allatom.exe`,而**仓库里一个 `.exe` 都没有**。

二进制在本机别处:`C:\Users\<user>\deploy\IGEM集成方案\tools\IsRNAcirc\IsRNAcirc_standalone\bin\`
—— 但路径**含非 ASCII 字符**,而 wrapper 注释明确说该 exe(GBK 编码原生 Windows 程序)不支持非 ASCII 路径。

**已解决**:把 exe + 4 个 DLL + 5 个系数文件复制到 ASCII 路径 `_isrnacirc_ascii/`(55.7 MB,已 gitignore),
设 `ISRNACIRC_BIN_DIR` / `ISRNACIRC_ROOT` / `CG_TO_ALLATOM_COEFF`。实测转换成功:

```
输入 seg_0_cg.pdb   200 atoms / 200 res / 1.0 atoms/res
输出 _test_aa.pdb  2400 atoms / 200 res / 12.0 atoms/res   ← 有 C1'
```

得到的是 **12 原子/残基**(骨架 + 每碱基),不是交付模型的 21.3 —— 后者走的是更后面的路径。

### 交叉验证:checkpoint 的配对在 3D 里一个都没形成

几何检测器跑这份粗粒化转全原子的结构,对 checkpoint 的 82 对配对:

```
in both                     : 0
checkpoint only (no contact): 82
coordinates only (unlisted) : 0
```

不在"both"里的那些,实测 C1'–C1' 距离是 **18–55 Å**(配对需要约 10.4 Å):

```
  1-109  44.73 A     5- 18  25.81 A
  1-146  36.67 A     5- 30  18.44 A
  2-108  55.23 A
  4-143  51.62 A
```

**读法:这次跑被我截断在 Level 1.5 的 CG 弛豫,还没到把碱基对拉起来的阶段**,
所以结构根本没折叠 —— 不是检测器错了,也不是 checkpoint 错了,是**运行没跑到**。

**但它顺带证明了检测器不会撒谎**:它没有把 checkpoint 声称的 82 对配对编出来。0 就是 0。

### 交付的 2013 nt 模型:骨架是理想化的,而且有物理上不可能的接触

用**不依赖任何本模块判据**的量核对过,所以这不是解析问题:

| 量 | 交付模型 | 参照 |
|---|---|---|
| 同残基 `C1'–O4'` | 1.40–1.42 Å | 标准 1.41 ✓ |
| 同残基 `C1'–C2'` | 1.53–1.54 Å | 标准 1.52–1.54 ✓ |
| 同残基 `C1'–N9/N1` | 1.60–1.70 Å | 标准 1.6 ✓ |
| 骨架 `P–P`(2012 对) | **min 5.87 / 中位 5.90 / max 6.05** | 2OIU 晶体 4.89–7.13,均值 5.86 |
| 骨架缺口 > 8 Å | **0 处** | —— |
| 最近 `C1'–C1'` | **2.48 Å**(A781–A782,**相邻残基**) | 相邻应 5–6 Å |
| 每残基原子数 | 20 / 22 / 23 | —— |

两条结论:

1. **骨架是理想化的。** 2012 对 P–P 全落在 5.87–6.05 Å(宽度 0.18 Å),而真实晶体结构是
   4.89–7.13 Å。零缺口、方差极小 —— 这不像动力学弛豫出来的东西,更像按固定骨架步长铺出来的。
   对比:本次缩水跑的 CG 输出(200 残基,1 原子/残基)才是会动的结构。

2. **有物理上不可能的接触。** A781 与 A782 的 C1' 相距 **2.48 Å**。相邻核苷酸的 C1' 不可能
   近于 5 Å,2.48 Å 比一个 C–C 键还短。同样问题的还有 A1924–A1925(2.61)、A382–A383(2.85)。
   **这是真实的几何冲突,不是判据问题。**

**这解释了为什么检测器在它上面返回 0 对配对,以及对描述这个产物意味着什么:**

- 不是"这条 RNA 恰好没有碱基对" —— 2013 nt 的 RNA 几乎必然有螺旋
- 是**这份文件的几何不支持配对**:任何残基的最近 C1' 邻居最远只有 9.43 Å,
  而 Watson-Crick 需要约 10.4 Å。**没有任何残基有一个处于配对距离的伙伴。**
- 结合骨架理想化 + 2.48 Å 冲突,**这个交付模型不应该被当作预测出来的结构看待**,
  它的主要用途是给 3D viewer 渲染(它的来源就是 viewer 的 payload)

**对免疫指纹的影响:所有依赖几何的特征(螺旋长度、可及性、m6A 可及性)都不能在这个文件上算。**
不是检测器不够好,是输入里没有那个信息。

### 结论:免疫指纹需要**两样东西同时存在**,目前没有任何单一产物同时具备

| 产物 | 配对 | 全原子坐标 | 几何可用? |
|---|---|---|---|
| `_checkpoint.json`(本次跑) | **有** | 无 | —— |
| 流水线的 PDB(1.0 原子/残基) | 无 | 无 | 是,但只有 P |
| `_test_aa.pdb`(CG→全原子,12 原子/残基) | 无 | **有**(无碱基原子) | 是 |
| `artifacts/2013nt/`(交付件) | 无 | 有,21.3 原子/残基 | **否** —— 骨架理想化 + 2.48 Å 冲突 |

要真正算指纹,需要**一次跑到 Level 2 之后的运行的全原子输出,配上它自己的 checkpoint**。
`run_2013nt.py` 的参数(20 轮、16 replicas、100k 步、RhoFold、PyRosetta)是那个量级。
本次缩水跑只证明了:**管线能跑通、checkpoint 会写、CG→全原子转换可用(修好路径之后)**。

## 十、完整跑一次的结果:机制全通,但采样没工作

`tools/run_pipeline_with_checkpoint.py --length 200 --full --out results/immuno_full`
(参数同 `run_2013nt.py`,两处改动写在该文件的注释里:`metad_n_steps` 200000→2000、
`use_rhofold` True→False)。

### 通了的部分

| 环节 | 结果 |
|---|---|
| 流水线跑通、每个阶段都到达 | Level 1 → 1.5 → 2(GPU REMD)→ … |
| checkpoint 落盘 | `level=1.5`,21 键,`pairs` 82 / `stem_blocks` 6 / `far_pairs` 21 |
| **CG→全原子**(修 ASCII 路径之后) | `cg2aa/merged_aa.pdb` **12.0 原子/残基,2400 原子** |
| 环状二级结构 | ViennaRNA `md.circ=1`:45 对(线性只有 35 对) |
| 冲突修复 | Level 1 `clash=19503` → Level 1.5 `clash=0, bond_q=0.99` |
| 结构约束进模型 | Level 1 `pair_rate=1.00` |

**修 ASCII 路径这件事在完整管线里也被确认了**:修之前 `cg2aa/*.pdb` 全是 P-only,
修之后管线自己写出 12.0 原子/残基的文件。

### 没通的部分:两个采样器都没在工作

```
[GPU-2D] ⚠️ Tri energy share too low (0.000%), TriRNASP is nearly ineffective!
[Torch GPU] REMD complete: final E=683, T-acc=0%
```

- **`T-acc=0%`** —— 副本交换接受率为零,一次交换都没发生。REMD 的意义就在于跨温度交换,
  接受率为 0 时它退化成 16 条互不相通的独立轨迹。**猜测**:16 副本 / 300 K 起、8 个温度点的
  阶梯,是从 2013 nt 那套参数继承来的,对 200 nt 这个链长可能太窄。**这一条我没有验证,只是记录。**
- **`Tri/CG=0.000`** —— TriRNASP 能量贡献为零,知识引导项完全没起作用。

### 直接后果:预测的配对在结构里一个都没形成

```
checkpoint pairs            : 82
coordinate pairs            : 0
in both                     : 0
predicted-but-not-formed    : 82 pairs, C1'--C1' 4.6 - 55.9 A
geometric helix runs        : 0
```

4.6 Å 和 55.9 Å 两端都出现了 —— **不是"没折起来",是有的地方挤得过近、有的地方拉得过开**,
一个典型的没有充分采样的构象。配对需要约 10.4 Å。

**结论:管线机制全部可用(配对能写、全原子能出、报告能跑),但用这套继承参数在 200 nt 上
采不到正确结构。** 所以在拿到一个采样有效的运行之前,**免疫指纹仍然算不出来** ——
这次卡在采样,不是卡在借口。

### 下一步的前置条件(按顺序)

1. **先让采样有效**:确认那 8 个温度点是什么、`T-acc` 为什么是 0。在采样无效的运行上继续做
   任何指纹工作都是白费。
2. **再谈序列长度**:200 nt 是 2013 nt 序列的前缀,可能不是一个有意义的折叠单元。
3. **然后才是指纹**:需要一个 `T-acc` 明显非零、且几何检查显示配对真的形成的运行。

### 追查 `T-acc=0%`:判据正确,**阶梯在工作**,但结构本身不能用

#### 先纠正我自己的一个错误结论

我一开始根据日志里的一行 `T-acc=['0/8',...]` 判断"温度阶梯太粗,接受率恒为零"。**这个结论是错的**,
而且错得很有教育意义 —— 我看到的是日志的**中段**。读完 294 行完整日志后:

| 阶段 | 温度边数 | 接受率 |
|---|---|---|
| `block n/10`(共 40 行) | **1** | 全部 `0/8`,即 0% |
| `block n/100`(共 3 行) | **7** | `['2/8','4/8','1/8','1/8','2/8','2/8','2/8']` → 12.5%–50% |

**`['0/8']` 只有一个元素,而 `n_t-1=7` 时应该有 7 个。** 所以那 40 行是 `n_t=2`
(只有一对相邻温度)的阶段;后来变成 `n_t=8`,`attT[0]` 每次 +8,接受率**非零**。

结论改成:**这套 8×8 网格交换是能工作的**(每边 12.5%–50%,`2/8`~`5/8`),
早期那个 0% 是起始构象还是 19503 个冲突时的暂态,不是阶梯的固有性质。
**不要在只看头几块的情况下改 `t_lo/t_hi`。**

#### 代码层面确实成立的几条

| 事实 | 位置 |
|---|---|
| 接受判据是标准 Metropolis:`log_alpha = (β_a−β_b)(E_a−E_b)` | `torch_cgsim.py:3391-3396` |
| `attT[k] += 1` 无条件,`accT[k] += 1` 只在接受时 | `torch_cgsim.py:3390/3396` |
| `β = 1/(KB_KJ·T)`,`KB_KJ = 0.008314462618` —— **单位正确** | `torch_cgsim.py:2866`、`:233` |
| 阶梯是几何的 `np.geomspace(t_lo, t_hi, n_t)` | `torch_cgsim.py:2728` |
| `t_lo=300.0, t_hi=1000.0` **硬编码** | `torch_gpu_refine.py:782` |
| `n_t = ceil(预算 / n_lam)`,默认 lambda 元组 8 项 | `torch_gpu_refine.py:762-763` |

#### 第二条独立发现:`TriRNASP` 的能量目录**全仓库没有任何地方赋值**

`trirnasp_energy_dir` 在 `openmm_gpu_refiner.py`、`rest2_remd_2d.py`、`torch_cgsim.py`、
`torch_gpu_refine.py` 里一路作为参数传递,**没有任何调用点给它赋值**。
所以 `Tri/CG=0.000` 不是"权重调小了",是**这一项根本没接上线**。

#### 接受率我测不出来,原因记下来免得重走

两个能量场各缺一半:

| 能量场 | 需要 | 手上 | 结果 |
|---|---|---|---|
| `cg_energy`(1 珠子) | 每残基 1 个珠子 | P-only ✓ | 能算,但**不是生产场**,且实测对温度几乎无响应 |
| `cg_energy_forces`(3 珠子,**生产**) | C4、NN 坐标 | **没有** | `IndexError: index 338 out of bounds for dimension 0 with size 200` |

流水线的 PDB 每残基只有 1 个 P,而它**不保存每副本能量**。两边都断了。走错的三条路:

1. **手工重建 3 珠子状态** → `E(start)=132588 kJ/mol`(流水线自己是 683),
   `C_v=283835 kJ/(mol·K)`。重建错误让 82 个配对的 NN 珠子全错位,配对项爆炸。
2. **对 1 珠子坐标调 `cg_energy_3bead`** → `atoms_per_residue = 1`,**没有 C4/NN 槽位**。
3. **零力数组 + `force_fn=None`** → BAOAB 会**原样重用传入的力**,所以漂移项无力,
   系统只受恒温器和噪声支配自由扩散(300 K 时 E=36030,1000 K 时 36155),
   并报出**不可能**的接受率 1.0000。

### 运行完整跑完了,以及一个比"配对没形成"更根本的问题

运行一路到 Level 4(2.3 → 2.5 → 2.6 PyRosetta → 3 RL → 3.5 Metadynamics → 4 REST2),
checkpoint 停在 `level=3.5`,`final_allatom.pdb` 存在。但:

**CG→全原子的输出里没有碱基环原子,所以配对检测器连几何都算不出来。**

```
final_allatom.pdb : 200 残基,0 配对,0 螺旋
   809 个候选全部被拒,理由全是 "missing"    ← 不是几何不满足
   candidates considered 809   rejection reasons {'missing': 809}
```

这**不是**"配对没形成"的证据 —— 是**测量不了**。同时:

| 文件 | 最近 C1'–C1' min | median | max |
|---|---|---|---|
| `cg2aa/merged_aa.pdb` | **1.07** | 4.82 | 10.16 |
| `remd_r0/remd_r0.pdb` | **1.46** | 4.86 | 9.89 |
| `final_allatom.pdb` | **1.96** | 5.40 | 9.08 |

两件事都指出来了:

1. **`max` 全部低于 10.4 Å** —— 和交付的 2013 nt 模型**同一个签名**:没有残基有处于配对距离的伙伴。
2. **`min` 是 1.07–1.96 Å** —— C1'–C1' 比一个 C–C 键还短,**物理上不可能**。

第 2 点与流水线自己的 `clash_count=0` 矛盾,也和 `07_validation.json` 里的
`bond_quality=0.992` 矛盾。那个矛盾我没有解决,只记录:**流水线的 clash 判据和
C1'–C1' 的最小距离说的不是同一件事。**

### 更新后的前置条件

1. **让全原子输出带上碱基原子** —— 否则配对检测器无法工作(现在全是 `missing`)。
   `CG_to_allatom.exe` 给的是 12 原子/残基(骨架 + 每碱基一个代表原子);
   交付模型是 21.3 原子/残基。**需要弄清 21.3 那条路径。**
2. **给 `trirnasp_energy_dir` 赋值** —— 接线问题,不是调参问题。
3. **澄清 1.07 Å 的 C1'–C1' 与 `clash_count=0` 为什么不冲突。**
4. **不要动温度阶梯** —— 实测每边 12.5%–50%,是工作的。

## 十一、闭环:21.3 原子/残基从哪来,以及为什么配对仍然算不出来

这是第十节留下的问题的答案,而且是一条完整可复现的链条。

### 21.3 那条路径找到了:`allatom_reconstruct.py`

交付的 2013 nt 模型是 **21.3 原子/残基**并带碱基环原子。本次跑写出的三个全原子文件
(`cg2aa/merged_aa.pdb`、`final_allatom.pdb`、`remd_r0/remd_r0.pdb`)都是 **12.0 原子/残基**
且**没有环原子** —— 所以配对检测器在它们上面 809/809、980/980 全部因 `missing` 被拒。
那是**测量墙,不是结果**。

`src/torusfold/scheme2/allatom_reconstruct.py` 的 `reconstruct_all_atom()` 就是缺的那一步:

> The scheme2 CG solver outputs (L, 3) coordinates with one P atom per nucleotide.
> This module expands each P point into a full all-atom residue using standard
> A-form RNA geometry **… Bases: A/G purines (start at N9, 9-10 atoms),
> C/U pyrimidines (start at N1, 8 atoms)**

在本次跑自己的 CG 轨迹上跑它:

| 文件 | 原子/残基 | 环原子 |
|---|---|---|
| `final_allatom.pdb` | 12.0 | 否 |
| `cg2aa/merged_aa.pdb` | 12.0 | 否 |
| `_reconstructed_allatom.pdb`(本次重建) | **21.2** | **是** |
| `artifacts/2013nt/isrnaclong_final.pdb`(交付件) | **21.3** | **是** |

**21.2 vs 21.3 —— 这就是那条路径。** 脚本:`tools/reconstruct_then_crosscheck.py`。

### 配对检测这次真的跑起来了,结论从"测不了"变成"确实是零"

重建之后 868 个候选的拒绝理由**全是几何**,**没有一个是 `missing`**:

```
rejection reasons: {"C1'--C1'": 434, 'glycosidic': 335, 'no': 90, 'base': 9}
checkpoint pairs 82   coordinate pairs 0   in both 0
predicted-but-not-formed: 82 pairs, C1'--C1' 3.3 - 45.9 A   (配对需要 ~10.4 Å)
helix runs: 0
```

所以:**这次跑没有形成任何它自己预测的碱基对。** 这是测量结果,不是测量失败。

### 而根因是 `reconstruct_all_atom` 的模板框架本身

它的每残基局部坐标系是:

```
b = normalize(P[i+1] - P[i])      骨架方向
r = normalize(P[i] - centroid)    径向
u = cross(b, r)                   法向
```

**只有两个向量,没有二面角自由度。** 所以它**无法表达** A-form 双链那种"两个碱基面对面
朝同一侧伸出糖苷键"的排布 —— 配对被骨架排布**结构性地排除了**。这不是参数没调好,
是这个重建方案的设计范围里没有"碱基与碱基之间的相对取向"。

配套证据:实测骨架 `P–P` = **5.90 Å 恒定**(交付模型与 CG 模型一致),而真实 A-form
是 **5.9–7.1 Å**。骨架锁在中间值上,两段链就落不到配对距离。

### 于是免疫指纹的阻塞点是两个,不是一个

| # | 阻塞 | 性质 | 归属 |
|---|---|---|---|
| 1 | 全原子产物缺碱基环原子 | 工具链:本机 `CG_to_allatom.exe` 只有 P-only 模式(实测它自己打印 `Detected 1-bead P-only CG`),3 珠子输入 exit 1 | 本仓库可用 `allatom_reconstruct` 绕过 |
| 2 | 重建的碱基取向不含二面角自由度 | **方法**:模板框架排除了配对几何 | 需要上游改变碱基放置方案 |
| 3 | `trirnasp_energy_dir` 全仓库无赋值 | 接线 bug | 一行环境变量或一个参数 |
| 4 | Tri 警告与 `use_trirnasp=False` 并存 | 尚未查明 | —— |

**#2 才是真正的墙。** #1 我已经绕过去了,#3 是接线,#4 待查。
在 #2 解决之前,任何跑出来的结构都不会形成配对,因而依赖几何的免疫特征
(螺旋长度、可及性)都算不出有意义的值。

## 十二、第四个阻塞:Tri 警告是误报,已修(诊断层,无行为改动)

### 日志里的矛盾

```
[GPU-2D] ⚠️ Tri energy share too low (0.000%), TriRNASP is nearly ineffective!
         Consider raising trirnasp_scale
```

而 `isrnaclong.py:1833` 传的是 `use_trirnasp=False`。查下来是**警告本身有问题**。

### 根因

`torch_cgsim.py:2872-2884` —— `tri_pot` 只在 `use_trirnasp and sequence` 时构造:

```python
tri_pot = None
if self.use_trirnasp and self.sequence:
    try:
        tri_pot = TriRNASPTorch(self.sequence, energy_dir=self.trirnasp_energy_dir, ...)
        if verbose: print("TriRNASP statistical potential loaded ...")
    except Exception:
        print("TriRNASP failed to load ...")
        tri_pot = None
```

`:3452` 的分子:

```python
e_tri_min = tri_total[i_min] if tri_pot is not None else 0.0
```

**`tri_pot is None` → 分子恒为 0 → `tri_cg_ratio == 0`。** 而警告的 `elif tri_cg_ratio < 0.001:`
**没有任何前提判断 TriRNASP 是否被启用**,所以必然触发。

日志里也**从来没有出现过** "TriRNASP statistical potential loaded" —— 证实它从未被构造。

**这条警告的害处**:它建议读日志的人去调 `trirnasp_scale`,而那是一个
**按设计关闭、并且即使打开也会加载失败**(`trirnasp_energy_dir` 全仓库无赋值)的参数。
照做的人会白费一次运行。

### 修法

把 `tri_pot` 提成显式前提,并让"按设计关闭"在日志里可读:

```python
if tri_pot is None:
    if rep == 0:
        reason = ("use_trirnasp=False" if not self.use_trirnasp
                  else "TriRNASPTorch failed to load; see the 'failed to load' line above")
        print(f"    [GPU-2D] TriRNASP: off ({reason}). "
              f"Tri/CG=0 is expected, not a diagnostic.")
elif tri_cg_ratio > 0.1:
    ...  # 原来的 too high
elif tri_cg_ratio < 0.001:
    ...  # 原来的 too low
```

**只改打印语句。** 没有触碰物理、常量,或任何传入模拟/交换判据的量。
`pytest -k "cgsim or trirnasp or remd or tri"` → 39 passed。

### 并发编辑提示

改这个文件时另一个会话正在同时改它 —— 同一个文件里有他们对 `PAIR_NN` 坐标的修复
(配对引导项从读 P 珠子改成读 N 珠子,并把配对 P–P 从 18.16 Å 拉向 10 Å 的那个 bug)。
两个改动在**不同函数**(`cg_energy_forces` vs `BatchedREMD2D`),已确认互不冲突。
**提交时注意别替别人提交。**

### 顺带:`isrnaclong.py:1835` 那个注释说的最优值永远到不了

```
trirnasp_scale=0.002,  # optimal: Tri/CG ≈ 11%, the balance point
```

`use_trirnasp=False` 加上 `trirnasp_energy_dir` 无赋值,**9–11% 这个"平衡点"目前不可达**。
这不是建议改回去 —— 是记下来:有人标过一个最优值,而当前配置到不了它。

## 十三、评估"给重建加一个自由度":**已经实现过了,而且它证明了墙在更上游**

我提议的做法 —— 让 `allatom_reconstruct` 的碱基能相对彼此定向 —— 在动手前查了一下,
**仓库里已经有三个东西做这件事**:

| 文件 | 做什么 |
|---|---|
| `src/torusfold/scheme2/allatom_reconstruct.py` | 手工模板,**固定朝向**。docstring 自己说这套几何"deviates from the amber14 OL3 force-field equilibrium, so after minimization amber_field stayed positive (+70,000 kJ/mol)" |
| `src/torusfold/scheme2/aform_from_template.py` | **取代**上面的手工模板。用 1EHZ tRNA 真实晶体残基 + Kabsch 三点对齐(P/C1'/C4'),并且 **`pairs=` 参数会让配对残基的锚轴指向伙伴** |
| `scripts/place_bases_hbond.py` | 绕糖苷键旋转碱基(chi),目标函数含"配对接触 + 保持原有碱基平面" |
| `torch_gpu_refine.py:951` | 已经有一个 "Watson-Crick edge repair (chi)" 步骤 |

`aform_from_template.reconstruct_all_atom(p_coords, sequence, pairs=None)` 的 docstring:

> When given, a paired residue's perpendicular anchor axis points at its partner,
> so the base faces the base it pairs with instead of radiating from the centroid.

**正是我提议的自由度。** 所以不该重写,该测它够不够。

### 测下来:不够,而且原因写在它自己的 docstring 里

用本次跑的 CG 轨迹 + checkpoint 的 82 个配对,对比开/关 `pairs`:

| | 原子/残基 | 配对召回 | 未形成配对的 C1'–C1' 中位 |
|---|---|---|---|
| `pairs=None`(径向回退) | 21.3 | **0.0%** | 17.3 Å |
| `pairs=<82 配对>` | 21.3 | **0.0%** | 14.8 Å |

**顺带确认了 21.3 原子/残基的来源:`aform_from_template` 给 4266 原子 / 200 残基 = 21.3。**

`aform_from_template` 的 docstring 已经预言了这个结果:

> What it cannot do is close a POSITIONAL gap: a base whose partner sits eight
> Angstroms away cannot be paired by any twist, and the remaining 9 of 12 are
> exactly that.

实测吻合:82 对里 **25–31 对在 12 Å 以内(扭转或可救),51–57 对在 12 Å 以外
(位置缺口,任何扭转都够不到)**。

### 墙在更上游:CG 阶段自己就没把配对珠子放在配对距离上

直接在 CG 轨迹上测三种距离:

| 集合 | n | min | **median** | max |
|---|---|---|---|---|
| 相邻骨架珠子 | 199 | 5.33 | **5.86** | 6.33 |
| **checkpoint 配对** | 82 | 3.63 | **15.43** | 49.45 |
| 非配对、非相邻 | 39238 | 1.82 | 35.05 | 88.22 |

相邻骨架 5.86 Å 落在 A-form 的 5.9–7.1 区间下沿(略短)。但**配对珠子的中位距离是
15.43 Å**,而 A-form 配对 P–P 应该约 **18 Å**;82 对里 **34 对超过 18 Å**。

配对项是对这个距离的谐振子,所以这些配对一直在被拉,而结构的其余部分在抵抗。
**这就是那面墙**:不是碱基朝向(已实现,且在伙伴足够近的产物上有效 —— 2OIU 上
关键接触 1/12 → 3/12、堆叠保持 58.3%),而是 **CG 结构本身不把配对珠子保持在配对距离上**。

### 附带发现:同一份运行里有**两个不同的"配对率"**

```
_plots/07_validation.json :  level1_5 pair_rate = 0.549,  level2 = 0.561
我自己测 checkpoint 的 82 对 : P-P < 12 A 只有 28.0%
```

`0.549` 不可能来自那 82 个配对。所以流水线内部的 `pair_rate` 和 checkpoint 的 `pairs`
指的是**不同的东西** —— 大概率是 `pairs`(硬约束)与 `scan_pairs`/chirality 候选之分,
或者阈值不同。

**这个我没有解决,只记录。** 但它意味着:**看到 `pair_rate=0.55` 就以为"一半碱基对形成了"
是错的** —— 用 checkpoint 的配对列表自己量,只有 28%。

### 修正后的结论

阻塞点从"缺一个自由度"变成:

| # | 阻塞 | 性质 |
|---|---|---|
| 1 | 全原子产物缺环原子 | 已绕过(`aform_from_template` 给 21.3/残基) |
| 2 | ~~碱基朝向缺自由度~~ | **已实现,且不是墙** |
| 3 | **CG 阶段没把配对珠子放到配对距离** | **真正的墙** |
| 4 | `trirnasp_energy_dir` 无赋值 | 接线 |
| 5 | Tri 警告误报 | 已修(诊断层) |
| 6 | 两个不同的 `pair_rate` 定义 | 未查明 |

**#3 意味着:改重建没有用。** 要么让 CG 力场真正抓住配对(它现在只有一个 N–N 谐振项,
没有朝向项,注释里写着 "The CG force field has no term that targets pairing geometry at all"),
要么接受当前产物不折叠,在下游把配对称作"意图"而不是"事实"。

## 参考文献

- Chen YG, Chen R, Ahmad S, et al. **N6-methyladenosine modification controls circular RNA immunity.** *Mol Cell* 2019;76(1):96-109. https://pmc.ncbi.nlm.nih.gov/articles/PMC6778039/
- Paramasivam A, Vijayashree Priyadharsini J. **Novel insights into m6A modification in circular RNA and implications for immunity.** *Cell Mol Immunol* 2020;17(6):668-669. https://pmc.ncbi.nlm.nih.gov/articles/PMC7264242/
- **Mechanism of PKR Activation by dsRNA.** *J Mol Biol* 2008. https://pmc.ncbi.nlm.nih.gov/articles/PMC2570377/
- Liu CX, et al. **RNA circles with minimized immunogenicity as potent PKR inhibitors.** *Mol Cell* 2021. https://www.sciencedirect.com/science/article/pii/S1097276521010091
- **Human NLRP3 inflammasome senses multiple types of bacterial RNAs.** https://pmc.ncbi.nlm.nih.gov/articles/PMC4234566/
- **Activation of the NLRP3 inflammasome by intracellular poly I:C.** *FEBS Lett* 2010. https://www.sciencedirect.com/science/article/pii/S0014579310008525
- **Human NLRP1 is a sensor for double-stranded RNA.** *Science* 2021. https://www.science.org/doi/10.1126/science.abd0811
- Wesselhoeft RA, et al. 2019(circRNA 免疫原性,经 Chen 2019 引用)
- **Differences in the immunogenicity of engineered circular RNAs.** https://pmc.ncbi.nlm.nih.gov/articles/PMC10257478/
- **A Dynamic Search Process Underlies MicroRNA Targeting.** https://pmc.ncbi.nlm.nih.gov/articles/PMC4768356/
