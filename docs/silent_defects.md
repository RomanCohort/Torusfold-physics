# The silent defects in the coarse-grained force field

The word has one meaning here: the code ran, returned a number, and nothing raised. Every entry
was found by building a measurement that could see it, not by reading the source.

**The count is 16.** It counts root causes, not edits and not sites: one mistake (the
excluded-volume neighbour table) has three faces and lives in two classes and is counted once.
The count matters because two documents cite it - `docs/attribution.md` and `docs/description.md` -
and both said "nine" with no list behind it. A count with no list cannot be checked, and this
repository does not keep those. Rows 1 to 13 are the field and its production path; 14 to 16 are
the modules around it, found by the same sweep.

| # | what was wrong | silent because | pinned by |
| --: | :-- | :-- | :-- |
| 1 | The Langevin O-step noise carried a spurious `sqrt(dt)`, so the effective temperature was 0.4 T at the shipped gamma and dt | every energy, force and gradient was correct; only the ensemble was wrong, and a wrong ensemble looks healthy | `tests/test_integrator_thermostat.py` (three tests); `scripts/check_kinetic_temperature.py` |
| 2 | Both force kicks were divided by `unit_conv = 100`, an effective mass of 100x and a clock 10x slow | a stationary distribution does not depend on the mass, so every equilibrium average stayed correct | `tests/test_integrator_thermostat.py::test_force_kick_uses_the_stated_mass_not_a_hundredth_of_it`; `scripts/integrator_mass_probe.py` |
| 3 | The second B half-kick reused `f(x_n)` instead of `f(x_{n+1})`, so the map was not symplectic | the trajectory still looked like dynamics; only the amplitude grew, by 1.8e-5 per step | `tests/test_integrator_thermostat.py::test_free_oscillation_survives_when_the_tail_kick_uses_the_new_positions` |
| 4 | The angle force's leading term had its sign inverted and its cosine term carried an extra `n2` | a force was returned, and it correlated -0.93 with its own energy gradient, so it read as a force | `scripts/diagnose_angle_gradient.py` (1.927 relative error, against 6.2e-08 for the fix) |
| 5 | The dihedral force was a hand-derived approximation with arbitrary 0.25 coefficients | the comment admitted the full formula was too complex, which reads as a decision rather than a bug | `scripts/gradcheck_per_term.py` (1.836 relative error) |
| 6 | Pair-summed forces accumulated with an indexed in-place add, which keeps one write per duplicate index | a residue sits in several pairs, so the shortfall looked like a plausible weaker force | `tests/test_force_gradcheck.py` (header carries the 1ET4 measurement) |
| 7 | The GB/SA pair energy had a hard pair-list cutoff and no matching switching function | the step is 0.074 kJ/mol, small enough to be read as noise | `scripts/measure_gb_jump_per_step.py`, `scripts/measure_gb_discontinuity.py` |
| 8 | The GB/SA and Manning force block was guarded by `torch.is_grad_enabled()`, so the shipped GPU path ran without a solvation force | its energy was still added, so the field looked complete; 7.77 of a 2447.92 maximum is 0.3 percent | `tests/test_ff_bonded_targets.py::test_the_force_does_not_depend_on_whether_the_caller_wants_a_graph` |
| 9 | The excluded-volume neighbour table was built from `pos_nm[0]` alone, within a single cell, and masked on `dist[0]` | 63 of 64 replicas had no excluded volume at all and nothing raised | `tests/test_clash_single_potential.py`; the three faces are recorded on `GPUCellList.build` and `_ClashNeighborList` |
| 10 | Four backbone pairs with `abs(i-j) <= 2` had no term of any kind | the excluded volume skips that range and no bonded term covered them; a hand-written audit table of six rows missed the fourth | `tests/test_backbone_13_terms.py`; found by enumerating the classes, not by the table |
| 11 | `cg_energy_3bead` computed `e_intra` and left it out of its own return | its only caller differences two runs of the same function, and a term missing from both cancels out of the difference | `tests/test_no_unreturned_energies.py`; `scripts/scan_unreturned_energies.py` |
| 12 | The excluded-volume law existed in four independent copies | the four agreed at the time, so nothing was wrong yet; the next edit to one of them would have left three stale | `tests/test_clash_single_potential.py` fails if a fifth copy appears or if the four stop agreeing |
| 13 | `_sigmoid_f` was inverted: it pulled hardest when a pair was too close and vanished at long range | it computed a real force and its numbers were plausible; the module's own docstring described the other sign | `tests/test_pair_clash_bsj_criterion.py`, `scripts/measure_pair_equilibrium.py`, `scripts/measure_guide_shape_fix.py` |
| 14 | `screeningLength` was added to a plain `mm.NonbondedForce`, which has no user expression | a global parameter on that force can never be read, and the parameter list read as a claim of screening | `tests/test_nonbonded_block_invariants.py`; `scripts/characterize_nonbonded_block.py`, section B |
| 15 | `_COMPLEMENT` in the RCM prefilter was the DNA alphabet | the prefilter accepted A-U and `_validate_rcm` then rejected it in one of the two orientations | `tests/test_pair_weight_source.py::test_the_complement_table_is_rna_not_dna` |
| 16 | `isrnaclong_pipeline` overwrote every method-agreement weight with the RCM confidence | the confidence scores at chance and is exactly 0.0 for 88.84 percent of true pairs, so the run completed with the base-pairing spring off for about 89 percent of pairs | `tests/test_pair_weight_source.py::test_pair_weights_do_not_default_to_the_rcm_confidence` |

Fifteen of the sixteen produced a wrong number while raising nothing. Row 12 is structural: the
four copies agreed at the time, and the defect was the next edit.

## Two omissions that look like row 11 and are not

`scripts/scan_unreturned_energies.py` reports rather than fails, because some energies are dropped on
purpose. `tests/test_no_unreturned_energies.py` records both survivors and they are the same deliberate
omission in `physical_relaxation.py`: `e_stack` is built and not returned, which is why
`K_STACK = 0.0` is dead twice over. Keeping the omission visible matters, because switching
`K_STACK` back on would otherwise look like it re-enables a restraint while changing no energy.

## This file is an index, not the record

Each row points at the place where the defect and its measurement are written down. Those comments
carry the arithmetic, the parameter point it was measured at, and in two cases a correction to an
earlier claim. Read them. If a defect is added or a row is found to be wrong, change the count
here and in the two documents that cite it.
