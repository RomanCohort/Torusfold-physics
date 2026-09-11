# Attribution — 草稿

> 类别与格式取自你给的 `Attribution 填写要求.docx`。该文件里类别是**图片**，
> 我用 Windows 自带 OCR（`Windows.Media.Ocr`，引擎语言 zh-Hans-CN）读出，
> 并对照同文件的两份范例（Laura Sun / Malak）确认了填写结构。
>
> 类别表共 17 项：Analysis、Conceptualization、Background Research、Investigation、
> Data Curation、Notebook/record keeping、Project Administration、Fundraising、
> Public Engagement、Entrepreneurship、Hardware、**Software**、Wiki Coding、Visualization、
> Writing、Safety、Other。
>
> 范例结构：**Name / Role / Tasks（类别列表）/ Specific Tasks（第三人称短文，带引用编号）**。

---

## 卡片

| 字段 | 内容 |
| :-- | :-- |
| **Name** | `<填入姓名>` |
| **Role** | Student |
| **Tasks** | Software, Analysis, Data Curation |

## Specific Tasks（约 144 词）

`<Name>` is a methodical, exacting computational biologist who rebuilt the calibration of
TorusFold's coarse-grained RNA force field. Across Software, Analysis and Data Curation, they
re-derived the stiffnesses of four bonded coordinates from the statistical distributions of
deposited structures (`k = kBT/sigma^2`), zeroed a fifth that was exactly redundant, added four
backbone distances the field had never covered, and re-cut the excluded-volume range to the
database minimum with its stiffness by Boltzmann inversion. In the same field they found and
repaired nine silent defects — a thermostat running at 0.4 T, an effective mass 100x too large, a
non-symplectic integrator, pair-summed forces that dropped duplicate-index contributions, and a
solvation force that vanished under `torch.no_grad`. They established the first measured upper
bound on the base-pairing spring by scoring register-shift decoys, and reported the null results as
plainly as the others.

---

## 填写时要注意

- **第三人称。** 全文用 `they`/`<Name>`，不出现 "I"。（要求第 1 条）
- **形容词两个。** `methodical`（有条理）、`exacting`（严苛）。
  想换的话这两个位置最好留给**工作态度**，不要留给能力。（要求第 2 条）
- **引用编号。** 范例里每条具体工作是 `(1, 2)` / `(3-5)` 这样标到 Wiki 或
  GitLab 上的证据。现在一个都没有 —— 需要你填：
  - 力场常数与四个新项的来源 → 分析脚本 / 数据库说明
  - 九个缺陷 → 各自对应的 commit
  - 配准位移诱饵的上界 → 那个脚本
- **词数。** 现在 144 词，落在那句「150 词左右」上。

## 两个必须你定的

1. **这是谁的 attribution。** 我按「一名队员（Student）」写。姓名、Role 我不知道，留了占位。
2. **AI 是否参与要如实处理。** 这一条我不替你决定，但要说清楚：
   这一系列改动**是 AI 助手执行、由你指挥的**。iGEM 对 AI 使用有单独的披露要求，
   而这份草稿现在写成 `they re-derived` / `they found and repaired`，读起来像全是手做的。
   如果实际是「设计判据、审阅、拍板」由你完成、实现由 AI 辅助，那更准确的说法是
   `directed` / `specified and verified`，并且另附一份 AI 使用说明。
   你说一声要哪个版本，我改。

## 可以加、但我没证据的类别

- **Background Research** —— 如果你确实读了 cgRNASP / IsRNA2 / rsRNASP 那批文献
  （`docs/statistical_potentials_as_forces.md` 第 1 节有那批引用），这项成立。
- **Visualization** —— 漏斗图 / 能量图。这次会话里我没做过图，所以没选。
- **Writing** —— 那份 2380 行的 `docs/statistical_potentials_as_forces.md`。
