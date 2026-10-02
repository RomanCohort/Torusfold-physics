# Plan C — change the target, not the operator (the IsRNA2 shape)

*The second of two independent tracks opened on 2026-09-21. Plan B
(`plan_b_coupled_update.md`) keeps the target and changes the operator; this one changes the
target and keeps the operator.*

## The point of departure

`docs/force_field_comparison.md` records what IsRNA2 / isRNAcirc actually do:

> **iteration: yes, by construction** — each round relaxes the native structures and draws 35,000
> snapshots to deduce the parameters

and the same document's conclusion:

> **The iteration is the method.** If we want what IsRNA2 has, the IBI loop is not polish -- it is
> [the method].

Note what the target is there: **not** a marginal pooled from a heterogeneous structure database,
but *the ensemble the field itself produces when it is started from native structures*. That is a
fixed-point equation, and a field can satisfy it by construction. Ours is different: we invert the
pooled marginal of 867 deposited chains, of which the loader's own comment says the histogram is
dominated by the longest (protein-held cryo-EM) chains, and we ask a decoupled 1-D inversion to hit
it. Four rounds of evidence says it cannot: bb_bond's correction went to zero with the ratio
frozen at 0.856 -- a self-consistent point of the iteration, not the reference.

## What the repository already has for this loop

| piece | where | note |
| :-- | :-- | :-- |
| relaxation of a deposited start under the injected Hamiltonian | `ibi_core.relax_positions` | built this session; 65 percent of chains start with clipped forces, 1500 steps clears about two thirds of them |
| full-field sampling with tables injected | `ibi_core.run_round` | the sampler the loop already uses |
| entry diagnostics | `entry{energy_0, max_force_0, at_cap_0}`, `relax` | say whether a start was ever physical — the acceptance test needs exactly these |
| marginal inversion and tables | `boltzmann_bonded.fit`, `_table_from_values` | usable as the fitting step, or replaceable by a parametric fit |
| per-task checkpoints + dead-worker retry | `ibi_loop` | a self-consistent loop is many short rounds, so this matters less here, but it is free |

## The loop, concretely

```
reference_0 = the deposited-pool marginals                      (as today, one round's worth)
for r in 0..R:
    sample every chain from ITS DEPOSITED GEOMETRY under field_r (relax + run_round)
    ensemble_r  = the pooled distributions of that run          <- this replaces the database marginal
    theta_{r+1} = fit(field form, ensemble_r)                   <- parameters, not 1000-bin tables
    field_{r+1} = field_r with the three terms replaced by theta_{r+1}
converged when ensemble_r reproduces itself (the distributions stop moving between rounds)
```

Two deliberate differences from the current loop:

1. **The reference is generated, not deposited.** Round 0 uses the database marginals (a starting
   guess); every round after that compares the field against *the ensemble the field produced from
   native starts*.
2. **The fit is parametric.** With a fixed functional form the loop is fitting tens of numbers
   against an ensemble, not inverting thousands of bins; that is what makes "the iteration is the
   method" affordable, and it is also what makes the result transferable.

## Acceptance, and why it is a better test than J

- **Native retention**: a held-out deposited chain, relaxed and sampled under the final field,
   stays near its deposited geometry (RMSD), i.e. the field does not melt chains it was started
  from. Today's entry diagnostics measure the first half of that (65 percent of starts are at the
  force cap; 24 percent do not leave it in 1500 steps).
- **Stationarity**: the distributions stop moving between rounds (the block-spread instrument
  already exists: `--blocks`).
- **Then, and only then, J**: with a self-consistent target J is expected to go to zero by
  construction, so a J that does NOT fall is a bug in the loop rather than a statement about the
  field.

## The cheapest decisive experiment

Same seven-chain pool, three arms, CPU-cheap because the rounds are short:

| arm | reference | fit |
| :-- | :-- | :-- |
| C0 (control) | deposited pooled marginal | table (today's loop) |
| C1 | ensemble generated from native starts | table |
| C2 | ensemble generated from native starts | Fourier/Chebyshev K=4-8 |

**The decisive number is not J.** It is: after R rounds, does a held-out native chain relax to its
own geometry (RMSD) better than it does under today's field? If C1/C2 do not beat C0 on native
retention, the target change bought nothing on this system and the honest answer is that IsRNA2's
success comes from its parametrisation plus its target, not from the target alone.

**Cost**: the seven-chain pool makes each round minutes; the whole matrix is a few hours. The
867-chain version of whatever wins is 18 h per round.

## Risks

- **Self-consistency can be vacuous.** A field that melts everything has a perfectly
  self-consistent target (the melted ensemble). Native retention is therefore part of the
  acceptance, not an optional extra — and the chain that does not melt is the one that says the
  loop is doing something.
- **Reference drift.** If the fit is bad, the next reference is further from anything physical and
  the loop walks away. The block-spread and native-retention instruments are the guard rails.
- **Parametrisation burden.** Choosing the functional form (harmonic vs Fourier vs table) is the
  real work in this plan; it is also the thing every published field has spent years on.

## Relationship to Plan B

Independent, and deliberately so: B asks whether a better operator closes the gap on the *current*
target; C asks whether the current target is the wrong thing to chase. B's result is meaningful even
if C is right (a coupled update is still what one wants on any target), and C's is meaningful even
if B works (a self-consistent target plus a better operator is the combination both plans are
ultimately aiming at).

## First free check — done 2026-09-21, and it says where the acceptance test has to come from

The plan's acceptance is native retention, so the first question is whether the *existing* artifacts
can already tell which chains fail to stay: 867 round-0 tasks carry `joint_J` and the entry state
(`energy_0`, `max_force_0`, `at_cap_0`, and the relaxation's end state).

| comparison | Spearman with J |
| :-- | --: |
| chain length L | +0.144 |
| deposited energy E0 | +0.043 |
| deposited max\|F\| F0 | +0.039 |
| relaxed energy E1 | +0.093 |
| relaxed max\|F\| F1 | +0.044 |
| accepted relaxation steps | -0.093 |

and by group:

| group | n | median J | p90 | max |
| :-- | --: | --: | --: | --: |
| started AT the 5000 cap | 566 | 0.1746 | 0.2434 | 1.636 |
| started below the cap | 301 | 0.1735 | 0.2432 | 0.558 |
| cap start that LEFT the cap | 368 | 0.1640 | 0.2500 | 0.421 |
| still at the cap after 1500 steps | 211 | 0.1900 | 0.2332 | 1.636 |

**The entry state does not predict the residual.** Starting with clipped forces is the norm (566 of
867) and it costs nothing measurable on average; clearing the cap buys a little (0.1640 against
0.1746). The only chain whose entry state is spectacular is the worst one (J=1.636, L=595,
E0 = 321686 kJ/mol, still capped after 1500 steps at E1 = 191890) — and even that one is a single
data point, not a group.

The tail is **chain-specific, not start-specific**, and the next seven worst are L=21-27 chains with
modest entry energies (E0 = 2.5k-26k): short chains are the ones a length-pooled reference fits
worst. Two consequences for this plan:

- **The acceptance test cannot be read off the entry diagnostics.** Native retention needs
  coordinates, and the sampler does not store positions in its task results — so the first real step
  of Plan C is a small run that RECORDS the geometry (deposited vs relaxed vs sampled mean, per
  chain), on ~20 chains, minutes of CPU. That is the missing instrument, and it is cheap.
- **Do not expect the tail to fall out of a better start.** 1500 -> 5000 relaxation steps will
  shrink the "still capped" group, and this table says that group's median J is 0.1900 against
  0.1735 for everyone else — a real but small effect. The residual lives in the field and the
  reference, which is what this plan is about.

## Baseline measured — 2026-09-21 (the number C1 and C2 have to beat)

`scripts/measure_native_retention.py`, 24 length-stratified chains (L 21-662), 12 ps each under
`tables_r3.npz` with 5000 relaxation steps, Kabsch-aligned RMSD against the deposited geometry
(`results/native_retention/retention_tables_r3.json`):

| quantity | value |
| :-- | --: |
| median deposited -> sampled-mean RMSD | **0.83 A** |
| median ensemble spread (frames against their own mean) | 0.22 A |
| chains that moved more than 10 A | **0 of 24** |
| worst two | 9AXT_1 (L=89) 1.92 A, 7QVP_7 (L=662) 1.29 A |
| closest bead approach | 0.243-0.322 nm, no interpenetration |
| per-chain J | 0.028-0.251 (median about 0.12) |

**The acceptance baseline is a high bar, and it is not where the four rounds' J said the problem
was.** The field already holds deposited geometry to under an Angstrom over 12 ps, with an ensemble
spread five times smaller than its own displacement from the deposit: the relaxation moves a chain
by about 0.8 A and then it sits still. So C1 and C2 do not have to *fix* retention — they have to
not lose it while changing what the loop targets, and a J-based criterion would have hidden that
the field is already usable for the architecture claim (refinement from a coarse start), which is
the same conclusion `force_field_comparison.md` reaches from the other side: the published fields
are judged by melting thermodynamics or native discrimination, not by a pooled marginal.

**Launch note for the same instrument on 24 chains:** the pool worker reserves 950 MB of commit if
the launcher does not pin `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_BLOCKTIME=0` (measured; the
production launcher has carried that pin since 2026-09-18, this one did not).

## Three arms, one pool — measured 2026-09-22 (`scripts/plan_c_loop.py`)

The implementation of the section at the top of this file. Pool 7 chains (L=26-34), holdout 4
(L=25-33), 4 rounds per arm, 8 replicas x 5000 steps (10 ps), burn 1000 (2 ps), stride 5, relax
5000, retention measured over 6000 steps (12 ps) from the deposited geometry, 6 worker processes x
1 torch thread, seed 20260922, K=8. Record: `results/plan_c/plan_c_run1.json`, `plan_c_run1.log`,
fields under `results/plan_c/fields/run1/`.

| field | pool median / spread / moved>10 A | holdout |
| :-- | --: | --: |
| the shared start field every arm begins from | 0.40 / 0.21 / 0 | 0.50 / 0.20 / 0 |
| `tables_r3.npz` — the field the 0.83 A number was measured under | 0.41 / 0.20 / 0 | 0.49 / 0.20 / 0 |
| C0: deposited target, `plan_update` (what rounds 0-4 did) | 0.42 / 0.20 / 0 | 0.49-0.51 / 0.21 / 0 |
| C1: the field's own ensemble as target, smoothed-table fit | 0.39-0.41 / 0.19-0.20 / 0 | 0.48-0.50 / 0.20-0.21 / 0 |
| C2: same target, Chebyshev T_1..T_8 | no number — round 1 diverged | |

**Changing the target did not move retention.** C1's 0.02-0.03 A is smaller than the 0.19-0.21 A
spread of a single arm on a single round, and every arm — the control included — sits in the same
0.39-0.51 A band as the two baselines it has to beat. On this pool the score is already at its
ceiling: four rounds of *any* of these updates neither gain nor lose geometry, which is the same
statement `measure_native_retention.py` makes at 24 chains. Note that the absolute numbers are not
comparable to the 0.83 A baseline of the section above: that pool is 24 chains spanning L=21-662 and
this one is seven chains of L=26-34, which retain better. Only the comparison inside this table
carries.

**C1's deposited residual rises by construction, not by failure**: J_dep 0.1371 -> 0.1775 -> 0.1820
-> 0.1971 against C0's 0.1371 -> 0.0901 -> 0.0974 -> 0.1070. Once the reference is the ensemble the
field itself produced, the deposited pooled marginal is no longer the quantity being minimised, so
its residual is free to grow; reading that as a regression would be reading the wrong number. It is
the reason this plan fixed retention as the verdict before running anything.

**The parametric fit is what breaks.** C2's first round produced max|dU| = 52.6 / 44.6 / 45.7 kJ/mol
(bb_bond / angle / dihedral) against C0's 8.0 / 2.9 / 5.6 at the same seed, pool and round. A
Chebyshev projection of -kBT ln p onto T_1..T_8 is unbounded where the empirical density is thin, and
the loop injects that correction into the sampler before anything checks it. The stabilisation rerun
(`plan_c_run1_c2`) produced no data — its ensemble file is empty — so the K-and-damping question is
open, and it is now the only live question this plan has.

Two things this run does not settle: the pool is 7 short chains, and a 12 ps window at n=7 cannot see
a 0.03 A effect; and no arm was ever compared against the 24-chain, L=21-662 instrument the 0.83 A
number lives in. Both are cheap and both use the same recorder.


