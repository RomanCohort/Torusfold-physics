# What the pipeline can and cannot predict: sequence length

This is a property of the molecule, not a defect in the code. It is written down
separately from `docs/silent_defects.md` on purpose: those sixteen rows are mistakes
in this repository, each pinned by a test. This one is a limit of the method, and
filing it beside them would make it look like something to be fixed.

## The result that prompted it

A 10 nt circle was run end to end (job `20d1b54d`, sequence `GGAAACGCGA`, secondary
structure `((((....))`, the shipped defaults other than length). All twelve stages
executed and every one produced output. The structure is not a ring.

Measured on the output, all stages agreeing:

| structure | P–P step | contour | end-to-end | Rg |
| :-- | --: | --: | --: | --: |
| `vfold3d/assembled.pdb` (Level 1) | 5.31 A | 47.8 A | **30.86 A** | — |
| `level1_5_relaxed.pdb` (Level 1.5) | 5.93 A | 53.4 A | **5.40 A** | — |
| `latest_cg.pdb` (Level 2) | 5.94 A | 53.4 A | **6.12 A** | 5.53 A |
| `_final_cg_for_aa.pdb` (Level 2.5) | 5.86 A | 52.7 A | **8.74 A** | — |

For comparison, the delivered 2,013 nt model: Rg **72.45 A** against **3,427 A** for
a straight rod of the same contour, principal moments `[2094.4, 1789.4, 1364.6]`,
axis ratio 1.5. It closes. The 10 nt one does not, and its moments
`[20.8, 6.2, 3.5]` describe an elongated blob, not a ring.

The per-residue backbone geometry is correct throughout — the P–P step sits at
5.86–5.94 A against 5.9 A for A-form RNA at every stage. Nothing is malformed. The
chain simply does not come back to its own start.

## Why, in closed form

A ring of `n` steps of length `s` has radius `R = s / (2 sin(pi/n))`, so the
backbone must bend by `180 - 360/n` degrees at every residue. With `s = 5.93 A`,
measured above:

| n | ring radius | bend per residue | |
| --: | --: | --: | :-- |
| 10 | 9.6 A | **36.0 deg** | |
| 12 | 11.5 A | 30.0 deg | |
| 16 | 15.2 A | 22.5 deg | |
| 20 | 19.0 A | 18.0 deg | |
| 24 | 22.7 A | 15.0 deg | |
| 30 | 28.4 A | **12.0 deg** | approximately the ceiling |
| 40 | 37.8 A | 9.0 deg | |

A phosphodiester backbone tolerates only a few degrees of effective bend per
residue before the stacking and backbone terms resist. Below roughly **30 nt** a
circle is not merely unlikely, it is geometrically unavailable: closing it would
require per-residue curvature the chain will not adopt, so the minimiser leaves a
gap instead. The measured gaps above (5.4 to 8.7 A of a 53 A contour) are that
refusal.

The practical statement: **this pipeline predicts circRNA structure for sequences
long enough to close. A 10 nt input runs every stage and returns a structure that is
not circular, and no stage fails.**

## What is not the cause

- Not the coarse-grained resolution. Level 1.5 and Level 2 agree to 0.02 A on the
  step length; both are describing the same molecule.
- Not the assembly or the segmentation. This input is a single segment, so
  `segmented_vfold3d_pipeline` does no Kabsch stitching and there is no boundary to
  get wrong.
- Not a failure of the predictors. The run reports `methods=['rhofold', 'rnabpflow']`
  with `conf=0.821` on the segment. RhoFold+ did produce an output; it is simply an
  output for a chain that cannot circularise.

One distinction worth keeping, because the two look alike in a log:

    abnormal P-P bond length (avg=0.00A), generating compact coordinates

is what `torch_gpu_refine` prints when the input it was handed has collapsed
phosphates — with `use_rhofold=False` on this same 10 nt sequence, RhoFold+ returns
a near-degenerate trace and the pipeline substitutes generated coordinates. **That
line does not appear in the `use_rhofold=True` run above.** So the bent-rod result
is the model's own output, not a fallback wearing its clothes.

## What would make this visible instead of surprising

The pipeline currently returns a non-circular structure for a non-circularisable
sequence and says nothing. A reader reasonably reads the output as "the model failed
on short input" when the accurate reading is "this input has no circular solution".

The check is arithmetic and cheap, and it can be made from a length alone before any
stage runs:

    bend_per_residue = 180 - 360/n
    warn when bend_per_residue > 12 degrees   # n below about 30

Nothing in this document has been wired into the pipeline. It is recorded here
because it was measured, and because the next person to run a short test sequence
will otherwise spend the same time working out why the answer has no ring in it.
