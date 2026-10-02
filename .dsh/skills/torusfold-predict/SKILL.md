---
name: torusfold-predict
description: >-
  Predict the 3D structure of a circular RNA with TorusFold, for a user who works
  in a wet lab and has not run structural software before. Covers starting the
  tool, checking that the input sequence can physically form a circle, choosing
  the run settings that control how long it takes, and reading the output without
  mistaking a self-consistency number for an accuracy measurement. Use when
  someone asks to predict, fold, or model a circRNA structure, asks why a
  prediction failed or produced something that does not look circular, asks what
  a TorusFold score means, or asks how to run TorusFold at all.
whenToUse: >-
  A user wants a circRNA 3D structure predicted, wants help preparing a sequence
  or dot-bracket input for it, is looking at TorusFold output and asking what a
  number or a picture means, or is deciding whether their sequence is even a
  sensible thing to predict.
---

# Predicting a circRNA structure with TorusFold

You are helping someone who understands RNA but not this software. Assume they do
not know what a dot-bracket string is, what MD or REMD means, or why a "closure"
number matters. Explain in their vocabulary and never let a number stand without
saying whether it is a measurement, a self-consistency check, or an estimate.

## Before anything: is this sequence a candidate?

**Check the length first, and say so out loud.** This is the single most useful
thing you can do for a biologist here, because the failure it prevents is
invisible in the output.

A circular RNA has to bend back until its two ends meet. Closing a ring of `n`
residues requires each residue to turn by `360/n` degrees — the n steps share one
full turn — and the phosphodiester backbone only tolerates a few degrees before
stacking and backbone terms resist:

| length | bend needed per residue | |
| --: | --: | :-- |
| 10 nt | 36° | not achievable |
| 20 nt | 18° | not achievable |
| 30 nt | 12° | approximately the ceiling |
| 100 nt | 3.6° | fine |

**Below roughly 30 nt the pipeline still runs, every stage succeeds, and the
structure it returns is not a ring.** Nothing in the output says this. If the
sequence is short, tell the user before they spend two hours on it.

Typical circRNAs are hundreds to thousands of nucleotides, so this mostly matters
for test sequences and for very short constructs. Full detail is in
`docs/sequence_length_limits.md`.

## Getting it running

The web interface is the intended route for someone who is not going to type
commands:

```
start.bat                # Windows: finds Python, checks the external tools, serves,
                         # and opens http://127.0.0.1:8877
start.bat --check        # report the environment and exit, start nothing
```

Then paste the sequence in the left panel and press **Predict**. There is a
**Demo structure** button that loads a finished 2,013 nt model if they want to see
a good result before committing to a run.

For someone who would rather not use a browser:

```
activate_deps.bat                  # sets the external-tool paths; starts nothing
python run_2013nt.py               # a full end-to-end prediction
python scripts/verify_headline.py  # recompute the published numbers, numpy only
```

`run_2013nt.py` reads `sequence.txt` at the repository root if present, otherwise
the committed `artifacts/2013nt/sequence.txt`. It is the wrong entry point for a
new sequence — for that, use the web interface, or call
`torusfold.scheme2.isrnaclong.isrnaclong_pipeline(sequence=..., secondary_structure=..., output_dir=...)`
directly.

If the external predictors are missing, `start.bat --check` names each one and
whether it resolved. A missing predictor does not stop the run: it is skipped and
the ensemble is weaker. That is why the check exists.

## The input

**Sequence.** Plain text, `ACGU` or `ACGT` (T is converted to U), FASTA headers
starting with `>` are ignored. The web panel also accepts a dropped `.fa`,
`.fasta` or `.txt` file.

**Secondary structure.** Dot-bracket, same length as the sequence: `(` and `)`
are paired, `.` is unpaired. For `GGAAACGCGAAACG` that is `((((....))))..`.

Two rules that matter:

- **The lengths must match exactly.** A mismatch stops the run at Level 1 with a
  length error. Count the characters.
- **Do not pair the two ends of the sequence to each other.** The ends have to
  come together to close the circle, so a base pair holding them apart works
  against the thing being predicted. If a structure-folding tool handed the user a
  dot-bracket that pairs position 1 with position n, remove that pair.

If they do not have a dot-bracket, the pipeline can build restraints from a
predicted one; ask what they have rather than guessing.

## Choosing settings

Defaults are sensible. Only four settings are worth changing, and all four trade
time for thoroughness. The panel groups them under **Sampling** and **Enhanced
sampling**.

| setting | default | what changing it does |
| :-- | --: | :-- |
| `md_step_scale` | 0.1 | Multiplies the per-round Level 2 step count. **The dominant cost.** 0.05 for a quick look, 0.3–0.5 when a structure is not converging |
| `n_rest2_replicas` | 16 | Replica count. One CPU core each. 4 for a quick look |
| `metad_n_steps` | 200000 | Total Level 3.5 steps. 20000 for a quick look |
| `n_relax_rounds` | 6 | Level 2 iterations. Early stopping usually cuts this short anyway |

Two more worth knowing about:

- `use_rhofold` (default **off**) — turns on the RhoFold+/RNAbpFlow/trRosettaRNA2
  ensemble for the first-stage prediction. Better, and slower, and needs the
  external tools installed.
- `use_pyrosetta` (default on) — full-atom refinement. Requires WSL with
  PyRosetta. **Skips in under a second when absent**, so leaving it on is safe.

**Set expectations on time.** This is not a fold-prediction web server that answers
in seconds; it runs molecular dynamics. A short test sequence is tens of minutes to
a couple of hours. The committed 2,013 nt demo wants **30–60 GB of memory and hours
to days**. Say this before they start, not after. The interface shows a stage
counter and an estimate that sharpens as measured stages accumulate — early on it
reads "projected" and is a guess; later it reads "estimated" with a range.

**Runs are resumable.** Every stage writes a checkpoint, and `resume` is on by
default. An interrupted run continues from the last finished stage. The
checkpoint is only reused when the sequence *and* every content-bearing setting
match; otherwise it is set aside and the run starts over rather than mixing two
configurations. The **Resume** tab in the right panel reports which will happen
and can roll back to an earlier level. If a run restarts from scratch, that tab is
where the reason is written.

## Reading the output

### The picture

The 3D panel shows the newest structure the run has produced. A coarse-grained
trace early on is a single phosphate per residue, drawn as sticks; the finished
all-atom model is drawn as a ribbon. **A model that is still running looks worse
than it will** — that is expected, not a failure.

What "correct" looks like at the end: a closed loop, no long straight run of
backbone, no chain passing through itself.

### The numbers, and what each one is

This is where a biologist is most likely to be misled, so be explicit about which
kind of number each one is.

| shown as | what it is | how to read it |
| :-- | :-- | :-- |
| **BSJ closure** | distance between the first and last phosphate | **A measurement.** For a circle it should be small — the delivered 2,013 nt model reads 5.898 Å. A large value means it did not close |
| **Bond RMSD** | deviation of adjacent-phosphate distances from A-form geometry | **A measurement.** Should be well under 1 Å; the delivered model reads 0.0082 Å |
| **Shape / Rg** | radius of gyration and whether the shape is compact or elongated | **A measurement.** A ring-like fold has an Rg far below a straight rod of the same length |
| **rsRNASP1** | statistical potential energy of the structure | **A real score, but read it carefully — see below** |
| **pair satisfaction** | fraction of base pairs *found in the structure* that are at hydrogen-bond distance | **Self-consistency, not accuracy.** See the warning below |
| **DFIRE**, **3dRNAscore** | — | Shown as `n/a`. **This pipeline does not compute them.** If they read `n/a`, that is not a failed run |

**rsRNASP1 has two traps, both written into the source comments.** First, the
PASS/FAIL threshold at -2000 has no recorded provenance — it has never been
calibrated against a reference set, so do not present a PASS or FAIL as meaningful.
Second, **the value is only comparable between sequences of the same length.** The
measured reference points are: crystal 1a9nR (27 nt) -3146.6, crystal 1h4sT
(61 nt) -7757.6, and this pipeline's own 139 nt prediction **+2289.3**, which is
positive. Comparing across lengths is meaningless.

**"Pair satisfaction" is not an accuracy score, and part of the documentation once
said it was.** An earlier version attributed the panel's figure to
`pdb_analyzer.compute_pair_satisfaction`; run on the delivered structure that
function returns **47.2%**, not the 100.0% shown elsewhere. Three different
quantities in this codebase are called "pair rate" and only one is inferred from
the structure — the other two score a predicted pair list against the geometry
that same list was used to build, which is close to tautological. The correction is
recorded in `docs/REPRODUCTION_RESOURCES.md` section 5.

### What has actually been validated

Exactly one experimental cross-check exists in this repository, and it is worth
naming so nobody claims more: **PDB 2OIU**, the only experimentally resolved
circRNA structure. Starting from the crystal structure, the Level-2 relaxation
takes 17 minutes on CPU and ends at **1.83 Å RMSD** from the crystal. That is
evidence the force field does not distort a known structure — it is not evidence
that a prediction from sequence alone is accurate.

Everything else is internal self-consistency. **No accuracy claim against
experimental ground truth is available for a de-novo prediction**, and you should
say so rather than let a good-looking structure imply otherwise.

## Reporting back

When you hand results to the user, include:

1. The sequence length, and whether it can form a circle at all.
2. Which settings were used, if not the defaults.
3. BSJ closure and bond RMSD, named as measurements.
4. That pair satisfaction and the shape numbers are self-consistency checks.
5. The 2OIU result as the only experimental validation, if validation comes up.
6. Any stage that was skipped, and why — a missing external tool or an absent
   checkpoint both cause skipping, and `start.bat --check` and the **Resume** tab
   are where that is visible.

## Where the details are

- `docs/sequence_length_limits.md` — why short sequences cannot close, measured
- `docs/REPRODUCTION_RESOURCES.md` — hardware, runtimes, the access path for a
  reader who will not run anything, and the pair-satisfaction correction
- `docs/DEPLOY_EXTERNAL.md` — installing the external predictors
- `docs/silent_defects.md` — the force-field defects that were found and fixed,
  each with the measurement that found it
- `README.md` — install, usage, and the AI/model disclosure table
- `artifacts/2013nt/quality.json` — every number the shipped viewer displays, with
  a note on whether a third party can recompute it
