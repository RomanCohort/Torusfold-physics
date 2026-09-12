# Timeline: the repair and improvement arc

**128 commits in six days** (2026-09-06 to 2026-09-11), on top of six annual release commits that
are the repository's skeleton rather than the project. Every entry below is a commit in this
repository. The reasoning and the numbers behind them are in
docs/statistical_potentials_as_forces.md, whose section numbers are cited where they exist.

Where a measurement was later superseded, the entry says so rather than quoting the older figure
silently -- that happened three times in this arc, and the corrections are part of the record.

## At a glance

| date | commits | phase |
| :-- | --: | :-- |
| to 2026-06-09 | 6 | annual release commits, 2022-2026: repository skeleton |
| 2026-09-06 | 9 | the repository as a deliverable |
| 2026-09-07 | 1 | the pipeline documented |
| 2026-09-09 | 13 | judge-facing alignment, and the first force-field restoration |
| 2026-09-10 | 32 | geometry, truth sets, and the statistical potentials |
| 2026-09-11 | 73 | the force field itself |
| **total** | **134** | |

Three of those 134 commits are the ones that each overturned something written earlier:
`0daaac3` (the dihedral mechanism I gave was wrong), `f3e5731` (retract the PAIR_NN claim -- read
off one structure), and `a52648a` (the converged window reverses section 3av).

## What the force field was before

Each change below names its own before and after in the commit subject, which is why this table
can be built at all:

| constant | before | after | commit |
| :-- | --: | --: | :-- |
| `K_BPP` | 600 | 13.4 | `99ab850` |
| `K_STACK` | 500 | 0 | `d357a5b` |
| `K_ANGLE` | 600 | 28.1 | `9dc314b` |
| `K_DIH` | 500 | 7.2 | `9dc314b` |
| `K_BB` | 500 | 1122.4 | `920fbc4` |
| `force_cap` | 200 | 5000 | `90bcac8` |
| `K_INTRA_*` | one spring | split into two measured ones | `6b214ac` |

The pattern is the point, and it is not one-directional. Three stiffnesses were 21 to 69 times
stiffer than the distribution of deposited structures allows -- `K_ANGLE` 600 against 28.1,
`K_DIH` 500 against 7.2, `K_BPP` 600 against 13.4. One was too SOFT: `K_BB` 500 against 1122.4.
One was zeroed because it is exactly redundant with two terms that do the same work. Four
short-range backbone pairs had no term of any kind, and four constants had no criterion at all.

## Phase 1 -- 2026-09-06, the repository as a deliverable (9 commits)

- `c05c22d` the pipeline itself arrives: isRNAcircLong circRNA 3D prediction.
- `33e408a` whole codebase localised to English; iGEM checklist scaffolding.
- `f0bf853` the mission statement, written at the top of the README.
- `10ab73b`, `82657a1`, `649e5ce` the demo viewer: pre-built output, the standalone Mol* viewer
  removed because it was obsolete, and later a single self-contained offline file.
- `9489f02`, `00b751c` deployment guides for the external predictors, pointed at their papers.
- `5c52e6b` 32 stray zero-byte files removed from the repository root.
- `aa046e5` repository identity aligned with the official registration (Oncology Village /
  CirCure), and a licence classifier corrected.

## Phase 2 -- 2026-09-07, the pipeline documented (1 commit)

- `51559f8` pipeline architecture and coarse-grained force-field figures added to the README.

## Phase 3 -- 2026-09-09, judge-facing alignment and the first field restoration (13 commits)

- `6365504`, `fc58c18`, `f542e0e`, `ebf6f82`, `7f88ae1` architecture, resources and reproduction
  references brought in line with the real pipeline: dual GPU/CPU backend, the Level-2 REMD stage
  that dominates the seven-hour wall time, neutral phrasing of the GPU platform rationale.
- `9215333`, `3bc8ad9` DivideFold configured by environment variable, deployed and cited.
- `cb4d308` viewer statistics panel filled from the repaired structure.
- `e92f5a4`, `1f2d849`, `8eaaf7f` internal notes; acceptance measured at 30-50 percent; and a
  finite-difference gradcheck for the explicit force paths.
- `450605d` **the first field repair.** `F = -dE/dx` restored, the bounded near-attraction guides
  rewritten in all three implementations, and a dihedral term restored. What follows shows how
  much was still wrong after this.

## Phase 4 -- 2026-09-10, geometry, truth sets, and the statistical potentials (32 commits)

**4a. The all-atom reconstruction was building on degenerate anchors.**
`34d98f5` the anchor audit found that `aform_template.npz` had never been committed and that the
four Kabsch targets were degenerate. `4f55c16`, `ea42389`, `7d74059`, `c0ed20f`, `67dcdd6` fixed
the anchors, directed the roll by each base's partner, rebuilt the local frame per anchor, and
passed the pairing list through the reconstruction call sites.

**4b. The truth set was wrong.** `f9fb229` -- HETATM modified residues were being dropped from the
1EHZ reference, so every comparison against it was against an incomplete structure. Corrected and
re-measured.

**4c. The three-bead model is nominal, and that blocks a whole approach.** `54dc12d`, `950c4da`,
`1b7df21`, `b04ba24` -- the C4-prime and base beads are grown from P, by offset or at random,
rather than taken from geometry, so cgRNASP's statistics (which count exactly at our three-bead
level) cannot be injected into our coordinates. `70daa97` fixed the fabrication where it could be.

**4d. The statistical-potential route, built and tested.** `f6d1e57` the literature map -- these
are not new, and TriRNASP's published failure is a warning. `8343ea9` the Boltzmann-inversion
recipe under its two established names. `7eb6d61` Boltzmann-inverted bonded potentials, which
pass the funnel test the harmonic terms fail. `38e0858`, `b8e3cf2` the cgRNASP tables are
piecewise constant, so injection is blocked on differentiability rather than on data.

**4e. Three target constants did not match native geometry.** `9a7d041`, `69de2dc`, `249acd3`,
`74e3640` -- recalibrated to the native modes rather than the mean, which changes which structure
the field prefers. `c4a430a` adds a shuffled-type control and reports honestly that the gain is
not significant (p = 0.077).

**4f. A loader bug and a wrong mechanism.** `0daaac3` the residue loader was joining across gaps,
and the dihedral mechanism stated in the previous commit was wrong. `568bc3e` the angle and
dihedral forces did not match their own energies, and the batched path was dead code.

## Phase 5 -- 2026-09-11, the force field itself (73 commits)

**5a. Recalibration by criterion.** `9ceac47` tables refit on cleaned data, local springs set to
`kBT/sigma^2`, with a retraction of my own variance test. `6ea4bc8`, `6dfcf5e` what actually carries
the fold signal -- and that the loud terms do no fold work. `9dc314b` `K_ANGLE` 600 to 28.1 and
`K_DIH` 500 to 7.2. `99ab850` `K_BPP` 600 to 13.4, which drops the share of the field sitting on
the force cap from 53.77 to 6.20 percent. `d357a5b` `K_STACK` 500 to 0, because the term is
exactly redundant with the bond and angle terms.

**5b. The excluded volume, in three defects and then one law.** `373dad4` three silent clash
defects, and the discovery that the gradcheck could never have passed as written. `0e2a38d` one
excluded-volume potential for four callers, derived from the database. `da35752` the criterion it
should match. `2498d7b`, `0fe923b` the range does not need to change; `K_PAIR` has a lower bound
and nothing else.

**5c. The integrator and the thermostat -- four separate defects.** `c8b1ffb` the Langevin
thermostat ran at 0.4 T, tested against an exact Ornstein-Uhlenbeck moment. `5c755d6` the noise
amplitude. `876d894` the force kick assumed 100 times the stated mass, so every GPU timescale was
ten times long. `bad550a`, `dfd0308` the deterministic part made symplectic, and the last caller
of the integrator moved off the non-symplectic fallback.

**5d. Forces that were silently short.** `9be2d4c` every pair-summed force was dropping all but one
contribution per bead. `c9bb145` the force depended on the caller's autograd mode, so the shipped
GPU path integrated a field with solvation energy and no solvation force. `558f8ff` four short
backbone pairs had no term at all, and `e_intra` was computed then dropped.

**5e. Two channels that were effectively switched off.** `0b8b89e` the RCM complement table was
DNA, so half of every A-U match was invisible. `7da785a` the pair weights were being overwritten
by a quantity that scores at chance and is exactly zero for 88.84 percent of true pairs.

**5f. The IBI workflow.** `950b3ea` the iterative half of the Boltzmann workflow, proved on a
known answer. `d34d198` CPU-path constants measured and a salt parameter nothing could read
deleted. `3f69234` IBI round 0. `c035b00`, `c876ce1` the harness fingerprint, and then the hole in
it that the script's own comment warns about. `1756c72` the equilibration plateau is at about
20 ps. `ed3e5a5` a static scan for terms computed and never read.

**5g. The constants with no criterion, settled where they could be.** `2807830` a nine-item
measurement batch. `920fbc4` `K_BB` to 1122.4. `1ce64c3` `K_BSJ` by transferability, `K_BSJ_GUIDE`
by the criterion. `424e1d7` the verification after those four changes: the best dynamics of the
session and a wash on the distributions. `3d1df83` the force cap measured as a guard, not a law.

**5h. The dead entry points stopped being a trap.** `5cef567` the audit -- one live path and four
dead ones. `459301b` the four now raise unless an explicit flag is set, so a caller cannot reach a
different potential under the same constant names by accident.

**5i. Documentation and the deliverable.** `cb65795` the attribution draft. `3e8f143` the
promotional description. `b66a75a`, `0ac8f38` the database path resolvable one way, and the data
bundled inside the folder. `b2d0641` the README update log.

**5j. The guide terms were pointing the wrong way.** `6e7a44b` `_sigmoid_f` was a short-range
reward under a long-range name. Corrected at five sites. The two shapes give the same force at the
well and opposite curvature there, so the correction is not free; the measured cost is in section
3ay.

**5k. The dev machine found that the numbers were transients.** `f9be972` -- a version mismatch, a
real bug in `audit_field_state.py`, and (the one that matters) the discovery that IBI round 0 was
sampling 3.2-16 ps of a window that had not converged. `e020882` the burn separated from the run
length, and `sim/ref` printed live. `a52648a` the converged window (40-200 ps): joint residual
**0.0968** instead of 0.3263, and **section 3av's direction reversed** -- angle and dihedral are
too narrow, not too soft. `f4b6c6e` the single-chain ceiling measured, which rules out the obvious
excuse: the largest sim/ref one chain can reach is 0.961 to 0.998, so the dihedral's 28 percent
deficit is real headroom.

## What is still open

The current state, stated as limits rather than as plans:

- **The field is not stationary at 200 ps.** The block spread and a dihedral that is still
  narrowing between 40 and 200 ps both say so. The blocked-J instrument (`--blocks=N`) is the gate.
- **The residual is one chain's.** Every dynamics result in this arc used 1L2X. The reference is
  pooled over 126 chains, so the honest next experiment is four more structures.
- **The dihedral's 29 percent deficit owns 59 percent of the joint residual, and no constant can
  fix it** -- its one-dimensional prediction is correct by construction. It needs a potential whose
  shape is not `kBT/sigma^2`, which is what IBI is for.
- **The sampler the IBI loop needs does not exist yet.** It has to run the full field with the
  angle and dihedral terms replaced by table potentials, starting from a native structure.
- **Section 3ay's 11 percent cost has never been measured on a converged window.** One run against
  the pre-correction field settles it.
- **`K_BSJ`, `K_BSJ_GUIDE` and `K_BSJ_CONTACT` remain uncalibrated**, because no deposited chain is
  covalently closed.

The ordered experiment list is in `docs/dev_machine_handoff.md`, section 7.