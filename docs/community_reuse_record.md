# A reuse record, executed: 2OIU through the shipped CG stage

**This page answers one question a community-reuse page has to answer: has this pipeline ever been run on a
real circular RNA, and can somebody else check the result?** Until now the answer was "yes, on 2OIU — and
you will have to take our word for the 2,013 nt example, whose artifact is decoded rather than re-run".
`artifacts/reuse_demo/2oiu/` is a record a reader can execute and check, produced by
`scripts/record_2oiu_reuse.py` on 2026-10-06 in about 21 seconds of GPU time. The paste-ready wiki wording is in
`docs/wiki_community_reuse_section.md`.

## What has actually been run here

| System | Length | Status | What a reader can do with it |
|---|---:|---|---|
| **2OIU** (`artifacts/2oiu/2OIU.pdb`) | 71 nt | **Run through the CG stage and recorded** — this page | Verify the record without a GPU, or repeat the run and compare hashes |
| 2,013 nt example (`artifacts/2013nt/`) | 2,013 nt | **Decoded, not re-run** — `provenance.json` says `"is_a_rerun": false` | Verify its hashes against the viewer payload it came from; the run that produced it is not in this repository |
| Any other circRNA | — | **Never run.** The one candidate in the tree, `ct1.fa` = `mmu_circ_0011663`, a 706 nt mouse circRNA (49 % GC), was prepared and then dropped in favour of 2OIU | Nothing yet |

2OIU is the one system here whose output can be compared against something the pipeline did not compute.
That is the entire reason the record is built on it: a reuse record whose only evidence is self-consistency
is not evidence.

## What the record contains

| File | What it is | SHA-256 |
|---|---|---|
| `input_p.pdb` | The deposited P trace, P-only, the start the refiner was given | `a79fc7c4cad3173b…` |
| `spec.json` | The mature sequence, its dot-bracket, and the base-pair probabilities above 0.1 — the input the run consumed, committed because the source copy lives in `results/`, which is not in the repository | see `record.json` |
| `2oiu_reuse.pdb` | The all-atom product: 1,550 atoms | `8619bb95849012e5…` |
| `2oiu_reuse_cg.pdb` | The CG P trace the run ended on | `2dfe598427bed0b8…` |
| `run_output.txt` | The run's own stdout, including the lines that say what the environment changed (named `.txt` because the repository ignores `*.log`) | — |
| `record.json` | Every number below, plus the sampled bead frame, both pair lists, the host, the environment and every hash | — |
| `production_tables.npz` | The force field the run used, copied here because `results/` is git-ignored | see `record.json` |
| `refit_smooth5_with_base.npz` | The reference binning grids the refine mode reads **from the directory of the tables file** | see `record.json` |
| `artifacts/2oiu/2OIU.pdb` | The deposit itself, the reference row, hashed by the record | `fee585a743a98272…` |

`artifacts/reuse_demo/2oiu_repair/` holds the same thing with `TORUSFOLD_HBOND_REPAIR=1`.

## The run

From the repository root, with the six environment settings the calibrated protocol needs (findings
Part 29 added the last of them so that no call site has to be edited):

```
TORUSFOLD_CG_TABLES=results/production_tables.npz
TORUSFOLD_BASE_STACK=16.6:0,1,0.19
TORUSFOLD_REFINE_MODE=refine
TORUSFOLD_REFINE_STEPS=1000
TORUSFOLD_BEAD_SOURCE=artifacts/2oiu/2OIU.pdb
TORUSFOLD_SEED=20261005

python scripts/record_2oiu_reuse.py --spec artifacts/reuse_demo/2oiu/spec.json --repeat 2
```

The recorded command names the source spec in `results/`, which is not in the repository; the copy
committed beside the product was checked by re-running with `--spec artifacts/reuse_demo/2oiu/spec.json`,
which produces the same product hash, `8619bb95849012e5…`.

Host: Windows 10 (build 26200), Python 3.11.15, torch 2.12.0a0+rocm7.13.0a20260313 on an AMD Radeon
8060S iGPU. Wall time 21.3 s, final energy 1499.4 kJ/mol, 10 `on_report` callbacks, 1,550 atoms in the product.
The record also names the commit its run was made from (`c7990a5`, the commit before the one that adds the
record) and the timestamp to the second. `--repeat 2` re-runs the identical protocol twice into a scratch directory: all three products
are byte-identical, so the recorded seed reproduces the run, not merely the protocol.

| Quantity (Angstrom unless stated) | 2OIU deposit | Product | Product, with the repair |
|---|---:|---:|---:|
| P-trace deviation from the deposit (Kabsch) | 0 (reference) | **1.386** | 1.386 |
| Base-to-base distance, through the bead map the field scores on | 5.306 | **5.201** | 5.201 |
| Base-to-base rise, same map | +3.186 | **+3.329** | +3.329 |
| Cosine between neighbouring base normals, same map | 0.901 | **0.899** | 0.899 |
| Base-to-base rise, measured on the ring planes of the file | +3.598 | +3.140 | +2.819 |
| Helical steps judged stacked (12 steps) | 100 % | **58.3 %** | 50.0 % |
| Rise over the stacked steps | 3.397 | 3.292 | 3.222 |
| Rise over all 12 helical steps (mean ± sd) | 3.397 ± 0.134 | 3.427 ± 0.519 | 2.556 ± 0.905 |
| Mean angle between the planes of a helical step | 7.8° | 17.8° | 17.4° |
| Watson–Crick key contacts within 3.6 Å | 12 / 12 | **1 / 12** | **3 / 12** |
| Bases rotated by the repair | — | — | 39 |
| Back-splice-junction closure of the product (\|P0 − P70\|) | 5.915 | 6.248 | 6.248 |
| Radius of gyration of the product | — | 24.14 | 24.14 |

Both base-level columns are reported because they answer different questions and disagree by a measured
amount: the first is the fixed linear map the field itself scores on, the second measures the ring planes
of the reconstructed atoms. On the deposit they differ by 0.41 Å of rise; findings Part 25 has the
explanation.

## The tree the run came from, and what a clean checkout gives

A record is only as reproducible as the source it came from, so `record.json` carries a fingerprint: the
SHA-256 of the nine files whose bytes decide what the run produces, and a flag for each saying whether it
was in a commit. Three of them were **not** in a commit when this record was made — uncommitted edits in a
shared working tree — and the record names them.

This was measured rather than assumed, and then narrowed down to one file. The same command, run from a
clean worktree of the commit the record names, produces a *different* product: energy 2638.3 against 1499.4
kJ/mol, trace deviation 3.15 Å against 1.386 Å, 25.0 % of helical steps stacked against 58.3 %, 0 of 12 key
contacts against 1 — and putting back **only** `src/torusfold/scheme2/torch_cgsim.py` reproduces the
recorded product `8619bb95…` byte for byte, while putting back `torch_gpu_refine.py` alone does not. The
causal file is the force-field one; the refiner's uncommitted changes move bookkeeping, not the product. So
one file stands between this record and a clean checkout that reproduces its headline numbers exactly, and
the record says which one instead of implying the question does not exist.

The file is worth naming precisely, because it is the reason the two rows above look the way they do. The
uncommitted change in `src/torusfold/scheme2/torch_cgsim.py` moves the **pair guide** from the P beads to
the N beads. `PAIR_NN = 1.00 nm` is documented in that module as a target on N beads (native N–N 9.9 Å
against P–P 18.2 Å), so reading the P beads instead pulled the phosphate of every paired residue toward
10 Å — the term was strongest exactly on native geometry. Re-measured at the current commit with the same
command and the same record: energy 2638.3 against 1499.4 kJ/mol, 3.15 Å against 1.386 Å, 25.0 % against
58.3 % stacked, 0/12 against 1/12 contacts, product `cf2405c1…` against `8619bb95…`. Landing that one
file is all it takes for a clone to reproduce this record; until then the fingerprint is what tells a
reader which of the two fields they are looking at. Verification is
unaffected — it re-derives every number from the committed files and passes there too — but "repeat the run
and compare hashes" is a claim about the tree the fingerprint describes, not about the commit alone. That is
the honest form of the claim, and the fingerprint is what makes it checkable rather than a footnote. It is
also the reason the copies of the force field above are in this directory: without them, a checkout has no
field at all, because `results/` is git-ignored.

## Where the gap comes from, measured

Six numbers in that table look like failures — 58.3 % stacked, 1 of 12 contacts. Two deterministic reference
rows, both re-derived by `--verify-only`, say which step each belongs to. Both are reconstructions the
shipped code performs on the **deposit itself**: one handed its own bead frame, one handed its own P trace
and therefore carrying zero sampling error.

| Quantity (Å unless stated) | Deposit | Reconstruction from its own beads | Reconstruction from its own P trace | Recorded product |
|---|---:|---:|---:|---:|
| Helical steps stacked (of 12) | 100 % | 91.7 % | 83.3 % | **58.3 %** |
| Watson–Crick key contacts within 3.6 Å | 12/12 | 5/12 | **2/12** | **1/12** |
| base_dist, bead map | 5.306 | 5.289 | 4.615 | 5.201 |
| base_rise, bead map | +3.186 | +3.216 | +2.444 | +3.329 |
| base_cos, bead map | 0.901 | 0.905 | 0.884 | 0.899 |
| Rise over all 12 steps (mean ± sd) | 3.397 ± 0.134 | 3.190 ± 0.352 | 2.862 ± 0.412 | 3.427 ± 0.519 |
| Mean angle between the planes of a step | 7.8° | 14.6° | 15.3° | 17.8° |

Read the contacts row down: the **reconstruction alone**, handed the deposit's exact trace and with no
sampling error at all, already keeps only 2 of the 12 contacts. The CG model pairs bases with a harmonic on
the N–N distance and has no orientation term, so nothing in the chain of tools is even attempting that
criterion; 2/12 is the ceiling of that step rather than a failure of the sampler, and what the sampler
costs on top of it is one contact. The stacking row decomposes the same way, and this residual really is
the sampler's: the template reconstruction of the deposit reaches 83.3 %, and 1.39 Å of drift takes it to
58.3 %. Both statements are cheap to make and would have been impossible to make from the product column
alone, which is the whole reason the rows are in the record.

## How to check it: no GPU, no torch import, no ViennaRNA

```
python scripts/record_2oiu_reuse.py --verify-only
python scripts/record_2oiu_reuse.py --verify-only --out artifacts/reuse_demo/2oiu_repair --repair
```

The two print 66 checks and 67 checks respectively, with no failures — the extra one re-runs the repair
and confirms it rotates the 39 bases the record says it did. The checks are not a re-print of the record: they re-hash all four
files, re-derive every geometric number from the committed product, and **rebuild the all-atom product from
the recorded bead frame with the same code and compare it byte for byte**. Only numpy, the repository's own
templates and the committed PDBs are needed — the verify path imports neither torch nor OpenMM — and it has
been run both from the ROCm environment and from a second interpreter (Python 3.14, no OpenMM) with the same
result.

One caveat, because it is exactly the kind of thing a record exists to expose: the rebuild is byte-exact,
but the product's own P/C4'/N atoms do **not** sit exactly on the recorded beads — they are up to 0.76 Å
away, because the reconstruction least-squares-fits a rigid template onto three points. An earlier version
of this verification asserted they should coincide and failed on a correct product; the byte-level rebuild
is the check that is both strong and true.

## What this record does not claim

* **It is a refinement, not a prediction from sequence.** The start is the deposited P trace, so this
  measures whether the field and the reconstruction preserve a structure that is already right. The
  sequence-only path exists (`--spec` omitted, ViennaRNA folding the sequence) but is not what is recorded
  here.
* **One circular RNA is not a benchmark.** 2OIU is the only circRNA here with a structure; nothing on this
  page says the pipeline behaves this way on a 700 nt or 2,000 nt circle.
* **The base pairing is still the weak point.** 1 of 12 key contacts survive the production path, 3 of 12
  with the repair. The remaining pairs are 4–13 Å apart: a positional gap that no rotation of a base can
  close (findings Part 30).
* **The protocol is 21 seconds long.** It reproduces a structure; it is not a converged ensemble.
* **The run needs the tree, not just the commit.** Three source files were uncommitted when the record
  was made; their hashes are in `record.json`, and a run from the commit alone gives a visibly worse
  product (3.15 Å, 25.0 % stacked, 0/12 contacts — measured, above). Verification does not care: it is a
  function of the committed record and passes on a clean checkout.
* **The start format has a ceiling.** The compact circular start the pipeline generates for a new sequence
  cannot be written past about 1,060 nt at the shipped 5.9 Å per step, because the PDB coordinate columns
  (31–54, `8.3f`) overflow and the loader's whitespace fallback cannot recover merged fields. Longer chains
  need a real starting structure.

## Two corrections this record produced

* **1,550 atoms, not 1,551.** The findings' Part 29 table says 1,551; the record counts the product file's
  own ATOM records, of which there are 1,550. The `--verify-only` rebuild settles it.
* **"Rise" meant two different things.** Part 30's repair row (3.42 → 2.54 Å) is the mean over all helical
  steps; this record's `rise_stacked` (3.292 → 3.222 Å) is the mean over the steps that pass the stacking
  test. Both are now fields in `record.json`; the record's all-step numbers, 3.427 → 2.556 Å, reproduce
  Part 30's within 0.02 Å.

## Cross-checks behind the record

* The pair list the run consumed is ViennaRNA's own output on the deposit's sequence, to the last
  probability: `python scripts/record_2oiu_reuse.py --check-spec --spec <spec.json>` re-folds it
  (ViennaRNA 2.7.2, identical dot-bracket, identical 27 pair indices and probabilities, max |Δ bpp| = 0.0).
* The deposit is hashed in the record, so a reader can confirm the reference row is computed from the same
  bytes the RCSB deposit's chain P gives.
* The record is not the first run of this protocol (findings Part 29 measured 1.41 Å of trace deviation,
  5.200 Å / +3.335 Å / 0.897 on the base map) and it does not reproduce that artifact byte for byte, because
  that run fixed no seed. The geometry agrees to three decimals, which is what "reproduced" means without a
  seed.
