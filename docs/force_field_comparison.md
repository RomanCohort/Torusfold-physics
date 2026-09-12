# IsRNA2 / IsRNAcirc's force field against ours

Two fields, compared term by term. Theirs is the one isRNAcirc runs; ours is the production field in
`src/torusfold/scheme2/torch_cgsim.py`, which is under recalibration. Sources:

- theirs -- IsRNA2, PMC9731381 (`E_total = E_bond + E_angle + E_torsion + E_bp + E_pair`) and its
  circular extension isRNAcirc, PLoS Comput. Biol. 2024, e1012293 (PMC11542809).
- ours -- read from the source and from the constants table in the README.

Everything below about their field is from the two papers. Everything about ours is from the code.

## The headline

They are **different kinds of field**, not two versions of one.

Their field is **knowledge-based**: each term takes a functional form chosen to fit distributions,
and the parameters come from an iterative procedure that compares simulated statistics against
target statistics. The nonbonded term is a typed pair potential **with a well** -- a Morse term plus
two Gaussians pinning the local minima -- and it carries base-base stacking, noncanonical pairing,
base-backbone and backbone-backbone in one function.

Our field is an **elastic network with an implicit-solvent tail**: every bonded term is harmonic
(in a distance, or in a cosine), the excluded volume is one soft divergence, and the long-range
part is Generalised-Born / solvent-accessible-surface plus a Manning term plus Mg2+ -- which their
field does not have at all. **There is no generic attractive nonbonded term.** The only attractions
are the ones we place by hand: the Watson-Crick spring, the closure terms, a soft BPP constraint,
and the ion and solvation terms.

## Beads

| | IsRNA2 / IsRNAcirc | ours |
| :-- | :-- | :-- |
| beads per nucleotide | **5** | **3** |
| bead types | **11** | 3 |
| definition | P + S (sugar) + three base beads | P + C4' + N9/N1 |
| base beads | typed by base identity and by edge (Watson-Crick / sugar / Hoogsteen) | three beads, no base identity |
| how positions are obtained | mapped from the all-atom structure | from the 1EHZ template reconstruction since `70daa97`; **before that, grown from P** |
| per-bead diameter | yes, 2.5 to 3.7 Angstrom (Table 1) | one `CLASH_SIGMA = 0.3975 nm` for every pair |

The last row of that table is the one that limits us most. Their excluded volume is per bead type;
our whole field has a single range, and we measured per-type ranges and rejected them because the
criterion could not supply a per-type stiffness -- but we never had a typed attraction to begin with
either.

## Terms, side by side

| term | IsRNA2 / IsRNAcirc | ours |
| :-- | :-- | :-- |
| bond | harmonic **+ a Gaussian** | harmonic, `0.5 K (d - d0)^2` |
| angle | harmonic **+ a Gaussian** | harmonic **in the cosine** |
| torsion | **quadruple Fourier** | harmonic in the cosine, one target |
| base pair | `E_bond(r1) + E_bond(r2) + E_angle(theta) + sum of 5 E_torsion(phi)` -- **eight restraints per pair**, tabulated for GC/AU/GU | **one harmonic N-N spring** |
| stacking | inside the typed `E_pair` | harmonic on P(i)-P(i+2), **`K_STACK = 0`** |
| excluded volume | `eps (sigma/r)^9`, per-type sigma | one soft `~ k sigma^4 / r^2`, one sigma |
| nonbonded well | **Morse + two Gaussians**, cutoff 6.9 to 13.5 Angstrom | **none** |
| solvation | none | **GB/SA + Manning + Mg2+** |
| circular closure | a **protocol**: harmonic restraint with a gradually raised constant plus simulated annealing, then the standard bonded terms for the new 5'-3' link | three terms with constants (`bsj closure`, `bsj guide`, `bsj contact`) |

Two things follow immediately. First, their field restrains a base pair with eight internal
coordinates; ours restrains it with one distance. That is not refinement -- it is a different
amount of the geometry being held. Second, ours has no attractive well and theirs is built around
one, which is why so much of our behaviour depends on the springs we place by hand and why `K_PAIR`
has a floor and a ceiling and nothing inside: it is doing a job a statistical pair potential would
do.

## How the parameters were obtained

| | IsRNA2 / IsRNAcirc | ours |
| :-- | :-- | :-- |
| method | **iterative simulated reference state** | `k = kBT / sigma^2` per coordinate, from the database |
| data | 70 simulated structures, 26 to 188 nt, with many noncanonical pairs | 191 deposited PDB entries, 126 gap-free chains |
| what is matched | the distribution a simulation produces, against a target distribution | one-dimensional reference width per coordinate |
| iteration | **yes, by construction** -- each round relaxes the native structures and draws 35,000 snapshots to deduce the parameters | **not yet.** IBI is designed, partially built, and not iterated |

This is the sharpest difference and the one that matters for the work in progress. Their field is
iterative; ours is a one-shot criterion plus an iteration that does not run yet. Theirs is not an
iteration *added to* a fixed field -- the iteration is how the field exists.

## Sampling

| | IsRNA2 / IsRNAcirc | ours (the calibration run) |
| :-- | :-- | :-- |
| integrator | LAMMPS Langevin, NVT, dt = 1 fs | torch Langevin, dt = 0.002 ps |
| replicas | **REMD, 10 replicas, 200 to 425 K** | 8 replicas, all at 300 K |
| length | 50 ns per replica, three duplicate runs -- **1.5 microseconds total** | 16 to 200 ps per replica |
| analysis | last 25 ns at 50 ps intervals, 5,000 snapshots, top 10 percent by energy, clustered by pairwise RMSD | window-averaged histograms |

Three orders of magnitude, and it is not incidental. Their parameterisation draws 35,000 snapshots
*per iteration*; the production run then samples 1.5 microseconds. Our current uncertainty is
exactly here: the field is not stationary at 200 ps, on one chain, with no replica exchange.

## What is the same

- Harmonic bonds and angles, of some form, in both.
- Both restrain a predefined secondary structure.
- Excluded volume and stacking are present in both.
- Both use replica-exchange MD in the production pipeline, and both reconstruct to all-atom and
  refine at the end.

## What this means for us

1. **The iteration is the method.** If we want what IsRNA2 has, the IBI loop is not polish -- it is
   the field. That is already the plan; this is the independent confirmation of why.
2. **A typed, attractive nonbonded term is the thing we lack.** Their Morse plus two Gaussians is
   where stacking and noncanonical pairing live. Ours turns stacking off (`K_STACK = 0`, because it
   is exactly redundant) and has no replacement.
3. **Eight restraints per base pair against our one** is measurable, not philosophical -- and it is
   the kind of thing a funnel test would see.
4. **Sampling.** Any claim from a 200 ps single-chain window should be labelled as such, and the
   plan to run four more structures is the minimum, not a luxury.
5. **The bead objection is smaller than it was, and what is left is a different one.** `70daa97`
   made the production path take C4' and N9/N1 from the 1EHZ reconstruction instead of growing them
   from P (measured: the old offsets put |C4'-N| at 4.900 Angstrom against the field's own 3.35
   target, and put both beads collinear with the backbone; the real ones land at 3.885 and 3.359 with
   P-to-C4' 48.9 degrees off axis). So "same name, different object" is gone for the production path.
   What remains is a **resolution** difference -- 3 untyped beads against their 11 typed ones -- and
   that is a separate question from whether our beads are real. The `openmm_gpu_refiner.py` path
   still random-perturbs them, and the reconstructed base bead still sits 2.010 +/- 2.328 Angstrom
   from the crystal N9/N1, which affects scoring our models rather than counting a potential.

## A note on scope

Our field is under recalibration, and several entries in the tables above are known gaps rather
than claims. The comparison is between their published field and our current one, taken from the
source at this commit.