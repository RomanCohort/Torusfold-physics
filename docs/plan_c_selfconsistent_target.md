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
