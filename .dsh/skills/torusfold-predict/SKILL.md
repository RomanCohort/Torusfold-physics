---
name: torusfold-predict
description: "Help a wet-lab biologist get a circular RNA structure predicted with TorusFold, by doing the work for them in natural language. They should never have to open a terminal, edit a file, choose a parameter, or know what a dot-bracket is. Use this whenever someone mentions a circRNA, circular RNA, a circle or ring structure, or a BSJ, and wants it modelled, folded, predicted, or simulated; whenever they paste an RNA sequence and ask what it looks like in 3D; whenever they ask whether a TorusFold result is any good, or what a number like closure, bond RMSD, rsRNASP1, pair satisfaction, 3dRNAscore or DFIRE means; whenever they ask why a prediction is taking so long, produced something that does not look like a ring, or stopped; and whenever they ask how to run, start, or set up TorusFold at all."
whenToUse: >-
  A biologist with a circRNA sequence wants a 3D structure, asks whether a
  TorusFold result is any good, asks what a score or a closure number means, or
  cannot get the tool to run. Also when a sequence is too short for the chemistry
  to close a circle, which is the most common way a biologist gets a confidently
  wrong answer here.
---

# Getting a biologist a circRNA structure without making them operate anything

The person you are helping knows RNA and knows their bench. They do not know this
software, and they should not have to. Your job is to ask for the one thing only they
can supply — the sequence — and to handle everything else: deciding whether the
question is answerable, starting the tool, choosing settings, watching the run, and
explaining what came back in their vocabulary.

Rules for the whole conversation:

- **Do not hand them a command to run.** Run it yourself.
- **Do not hand them a choice of parameters.** Pick, and say why in one line.
- **Never mention a filename, a path, or a port** unless they ask for it.
- **Never let a number stand alone.** Say whether it is a measurement, a
  self-consistency check, or an estimate.
- If their sequence cannot answer their question, say so **before** spending an hour
  of their time on it.

## Step 1 — ask for the sequence, and nothing else

Ask for the sequence in one sentence. They may paste it, attach a file, or name a
construct. They do **not** need to supply a secondary structure — the tool derives
one itself. Do not ask for one, and if they offer one, accept it and move on.

While you have it, check three things without asking:

- **Characters.** `A C G U`, or `T` which is converted. Anything else is a typing
  error or a different molecule — ask.
- **Length.** This decides everything; see the next step.
- **Is it actually circular?** A circRNA has no 5′ or 3′ end. If what they pasted has
  ends that clearly cannot join, or they describe a linear construct, ask.

## Step 2 — decide whether to run, before running

**This is the most valuable thing you do in the whole task.** A circle has to bend
back until its two ends meet, and closing a ring of `n` residues needs each residue
to turn by `360/n` degrees:

| length | turn per residue | outcome |
| --: | --: | :-- |
| 10 nt | 36° | impossible |
| 20 nt | 18° | impossible |
| 30 nt | 12° | roughly the limit |
| 100 nt | 3.6° | fine |
| 1000 nt | 0.36° | fine |

**Below about 30 nt the software runs successfully, every stage completes, and the
structure it returns is not a ring. Nothing in the output warns about this.** Left
alone, you would hand a biologist a confident non-answer.

So, by length:

- **Under ~30 nt** — say plainly that a circle of that length cannot close, so a
  structure prediction is the wrong tool, and ask what they are actually trying to
  learn. There may be a better question — about their BSJ junction, or their
  construct's design — that needs no 3D at all. If they still want to see the
  geometry, run it, but tell them first that the result will not be circular.
- **30–150 nt** — fine, and quick enough to run without ceremony.
- **Over ~150 nt** — fine, but **say how long it will take and ask before starting.**
  Minutes for the early stages, then the molecular dynamics dominates. Get a yes
  first. Cost here is set by the configuration, not the length, so never quote a
  round number: at the checked-in settings of `run_2013nt.py` the 2,013 nt demo is
  the full configuration — ≈60 GB and ≈14 days on the GPU path, a figure that was
  estimated and never measured — against ≈30 GB and ≈7 h for the CPU path in the
  low configuration, and that 7 h was measured on an earlier build, not on this
  checkout. `docs/REPRODUCTION_RESOURCES.md` §2 is the table; quote it.

If they are against a deadline, offer the faster settings rather than the full run:
fewer sampling steps and fewer replicas give a rougher structure in a fraction of the
time. Those are `md_step_scale`, `n_rest2_replicas` and `metad_n_steps` — describe
them as "a quick look" versus "thorough", not by name.

## Step 3 — start it if it is not already running

Check whether the tool is answering:

    curl -s http://127.0.0.1:8877/api/health

If nothing answers, start it from the repository root and wait for it to come up. On
Windows that is `start.bat`. Two flags are worth knowing: `--check` reports the
environment and exits having started nothing, and `--setup` re-searches for the
external tools. A missing external predictor does not stop a run — it makes the
ensemble weaker — so it is worth knowing about but not worth blocking on.

## Step 4 — submit it, then keep them informed

Submit the sequence and let the server own the run. **Do not run the pipeline in the
foreground**: it takes hours, and you would be unable to report anything until it
finished.

    POST http://127.0.0.1:8877/api/predict
    {"sequence": "<their sequence>", "params": {}}

That returns a `job_id` immediately. Then poll `GET /api/current` and translate what
you see into their language:

| what you see | what to tell them |
| :-- | :-- |
| `stage_index` of `stage_total` | "Stage 4 of 12 — building the coarse-grained fold" |
| `eta.state = "projected"` | "Still calibrating the time estimate" — at this point it is a guess |
| `eta.state = "estimated"` | Give the range as a range. It narrows as stages complete |
| a long, silent stage | Normal. The heavy dynamics stages print little. Do not report it as stuck |

Only one prediction runs at a time and a second request is refused. If one is already
running, say so rather than trying to start another.

**The run is resumable.** If it is interrupted, restarting continues from the last
finished stage rather than from the beginning — but only when the sequence and every
setting match. If it ever starts over, the reason is recorded in the tool's Resume
panel; check there before telling them it "just restarted".

## Step 5 — explain what came back

Lead with the shape, not the numbers: did it close into a ring, or not. Then the
numbers, each labelled for what it is.

| what to report | what it actually is |
| :-- | :-- |
| **Closure** — the distance between the two ends | **A measurement.** Near zero for a circle; the finished 2,013 nt model reads 5.898 Å. A large value means it did not close |
| **Bond geometry** | **A measurement.** Small is healthy; under 1 Å is good |
| **Compactness / shape** | **A measurement.** A ring-like fold is far more compact than a straight chain of the same length |
| **rsRNASP1** | A real statistical score — but see the two traps below |
| **Pair satisfaction** | **A self-consistency check, not accuracy.** See below |
| **DFIRE, 3dRNAscore** | Show `n/a` because this software does not compute them. Not a failure |

**rsRNASP1 has two traps.** The PASS/FAIL threshold shown beside it has never been
calibrated against a reference set, so a PASS or FAIL there means nothing — do not
repeat it as a verdict. And the value is only comparable between sequences of the
same length: the measured references are crystal 1a9nR (27 nt) -3146.6, crystal 1h4sT
(61 nt) -7757.6, and this pipeline's own 139 nt prediction **+2289.3**, which is
positive. Comparing across lengths is meaningless.

**"Pair satisfaction" is not an accuracy score, and the documentation once said it
was.** It reports the fraction of base pairs *found in the structure* that sit at
hydrogen-bond distance, which partly restates the geometry it was measured from. The
figure the documentation once attributed to `compute_pair_satisfaction` is 47.2% on
the delivered structure, not the 100.0% shown elsewhere. Only one quantity here is
inferred from the structure; two others share the name "pair rate" and score a
predicted pair list against the geometry that same list was used to build.

**What has actually been validated: one thing.** PDB 2OIU, the only experimentally
resolved circRNA structure — starting from the crystal, this software's relaxation
ends 1.83 Å from it. That says the physics does not distort a known structure. It is
**not** evidence that a prediction from sequence alone is accurate. No accuracy figure
against experimental ground truth exists for a de-novo prediction, and you must not
imply one. If they ask how accurate this is, the honest answer is: the geometry is
physically consistent and the closure is real, and there is no experimental structure
to score it against.

## How to answer what they will actually ask

**"Is it good?"** Report closure and bond geometry as facts, say the rest is
self-consistency, and do not invent a quality grade.

**"Why doesn't it look like a ring?"** Check the length first — under ~30 nt it cannot
be, by arithmetic. Otherwise the run may still be early: during the dynamics stages
the panel shows a working trace, not the answer.

**"Why is it taking so long?"** It runs molecular dynamics, not a lookup. Give them
the stage count and the estimate, and offer the faster settings only if they want
them.

**"Can I stop it and change something?"** Yes, and it resumes from the last finished
stage. Changing the sequence or a setting starts it over deliberately, because mixing
two configurations would produce a structure belonging to neither.

**"Can I look at the structure myself?"** Tell them the page at
`http://127.0.0.1:8877/` shows it in 3D, and that it can be exported as a PDB for
PyMOL or ChimeraX. This is the one place a URL is worth giving.

## Where the detail is, if you need it

- `docs/sequence_length_limits.md` — the closing arithmetic, measured
- `docs/REPRODUCTION_RESOURCES.md` — runtimes, memory, and the pair-satisfaction correction
- `docs/DEPLOY_EXTERNAL.md` — installing the external predictors
- `artifacts/2013nt/quality.json` — every number the interface shows, with whether a third party can recompute it
