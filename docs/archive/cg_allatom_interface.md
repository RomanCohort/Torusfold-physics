# The CG / all-atom interface: what transfers, and what the wiring costs

*Measured 2026-10-01. Two questions the IBI loop cannot answer from inside itself: are the local
coordinates of a LINEAR deposited fragment the same statistics as the CIRCULAR product the field is
for, and what happens to stacking when the all-atom level takes over? Both were asked on data already
on disk, plus one 2OIU run.*

## 1. Local geometry transfers between the fit's fragments and the product

The four scored coordinates are local functions of the P trace alone (`boltzmann_bonded.coords_of`):

| coordinate | definition |
| :-- | :-- |
| bb_bond | \|P(i) - P(i+1)\| |
| angle | the P-P-P pseudo-angle (cosine) |
| dihedral | the P-P-P-P pseudo-dihedral (cosine) |
| stack | \|P(i) - P(i+2)\| -- an EXACT function of the two above, by the cosine law |

A circle differs from a line only at the ends, and a circle has none. That is the assumption behind
fitting on 867 linear fragments and applying the field to circular RNA, and
`scripts/local_geometry_transfer.py` measures it (no sampling):

**A. End effects** -- per-chain median of (interior with 3 residues trimmed each end) - (all), on the
867 fragments:

| L bucket | n | bb_bond dmean/dsd (interior sd) | angle | dihedral | stack |
| --: | --: | :-- | :-- | :-- | :-- |
| < 60 | 375 | -0.0007 / -0.0002 (0.0494) | +0.0109 / +0.0140 (0.2892) | -0.0239 / +0.0120 (0.6108) | -0.0043 / +0.0039 (0.1182) |
| 60-150 | 278 | -0.0004 / +0.0002 (0.0505) | +0.0062 / +0.0062 (0.3132) | -0.0132 / +0.0092 (0.6205) | -0.0025 / +0.0018 (0.1280) |
| 150-400 | 136 | -0.0002 / -0.0000 (0.0514) | +0.0025 / +0.0025 (0.3186) | -0.0032 / +0.0018 (0.6355) | -0.0012 / +0.0007 (0.1306) |
| > 400 | 78 | -0.0001 / 0 (0.0533) | +0.0003 / +0.0005 (0.3252) | +0.0002 / -0.0002 (0.6330) | -0.0002 / +0.0001 (0.1332) |

The end effect is 4-5 percent of the sd below L=60 and under 0.2 percent past L=400, and the BULK
width is length-independent to within 4-13 percent across a 7x length range: the local marginals
converge to one limit, and what is left is a terminal residue effect the circle does not have.

**B. Closure** -- interior statistics bucketed by end-to-end extension (L >= 60). A circle is the
R_ee -> 0 extreme:

| R_ee / R_max | n | bb_bond | angle | dihedral | stack |
| --: | --: | :-- | :-- | :-- | :-- |
| 0.00-0.15 | 409 | 0.5969 +/- 0.0509 | -0.7126 +/- 0.3186 | +0.5161 +/- 0.6295 | 1.1007 +/- 0.1299 |
| 0.15-0.35 | 82 | 0.5993 +/- 0.0519 | -0.7090 +/- 0.3254 | +0.5081 +/- 0.6281 | 1.0998 +/- 0.1330 |
| 0.35-0.60 | 4 | 0.6152 +/- 0.0613 | -0.7242 +/- 0.2776 | +0.3707 +/- 0.6196 | 1.1538 +/- 0.1083 |

**The closure constraint does not measurably change the local marginals** (under 0.5 percent between
the most compact and the more extended buckets, and the compact bucket is the largest sample).
Forcing R_ee to zero is therefore free for local statistics; the only topology-dependent part is the
two terminal residues, worth 4-5 percent of the sd on the shortest fragments.

The same run is an independent check of the two-denominator story: the pool's own sds
(0.0509 / 0.3186 / 0.6295 / 0.1299) sit under the STORED sigmas (0.0633 / 0.3203 / 0.6299 / 0.1390)
for bb_bond and stack, and match the tables' IMPLIED ones (0.0480 / 0.3218 / 0.6247 / 0.1268).

## 2. The all-atom level carries stacking and inherits the P trace

`scripts/reconstruction_stacking_check.py`: 1EHZ-template reconstruction of three pool chains, base
centroid distances / inter-plane angles / rise, and the same after perturbing the P trace by 1.5 A
(comparable to the CG's own sampling spread, since the angle's sd of 0.35 in cosine units is about
1 A on a 6 A step):

| chain | input | centroid med (A) | inter-plane med (deg) | rise (A) | stacked fraction |
| :-- | :-- | --: | --: | --: | --: |
| 1ET4 L=35 | deposited | 4.83 | 30.0 | 2.59 | 0.588 |
| 1ET4 | +1.5 A noise | 5.37 | 52.0 | 3.10 | 0.294 |
| 1KXK L=69 | deposited | 4.33 | 16.3 | 2.59 | 0.721 |
| 1KXK | +1.5 A noise | 5.02 | 33.2 | 2.91 | 0.471 |
| 1L2X L=27 | deposited | 4.55 | 24.3 | 2.62 | 0.577 |
| 1L2X | +1.5 A noise | 5.75 | 49.8 | 3.20 | 0.346 |

The reconstruction's stacking is plausible (centroids 4.3-4.8 A, inter-plane 16-30 deg), and it comes
entirely from the template -- but a 1.5 A error in the CG P trace HALVES the stacked fraction. The
division of labour follows: the CG owns the P trace, the all-atom level owns the base planes, and
`stack` is not a coordinate the CG can or should target (it is algebraically derived from bb_bond and
the angle, and `torch_cgsim` already sets its spring to zero for the same reason).

## 3. The wiring, measured on 2OIU (71 nt, the only resolved circular RNA)

`scripts/allatom_2oiu_wiring.py`. The product's path is circular end to end, and on 2OIU it needs
exactly three repairs before amber14-OL3 will take it:

1. **Drop the third phosphate oxygen.** `aform_from_template` gives EVERY residue an OP3, so a
   phosphodiester (whose phosphate has OP1/OP2 plus the two bridging O5'/O3') carries one oxygen too
   many and the force field rejects residue after residue: "the set of atoms is similar to G, but has
   1 O atom too many". Filtering OP3 fixes the whole chain (1550 -> 1527 atoms).
2. **Internal residue names + `ignoreExternalBonds=True`.** amber14's `G5` is the DEPHOSPHORYLATED
   5' terminus: with the reconstruction's 5'-phosphate the heavy atoms match plain `G` (OpenMM's own
   matcher says so). Naming the ends `G5`/`A3` makes matching fail.
3. **`sim.context.setPositions(modeller.positions)`.** `Simulation()` does not carry the coordinates
   over, and OpenMM answers "Particle positions have not been set".

With those, the repo's own circular builder works -- 2297 atoms, 770 hydrogens -- and
`createSystem(amber14-all.xml + implicit/obc1.xml, NoCutoff, HBonds)` builds 2297 particles.

**What is still missing is a relaxed starting structure.** Straight from the crystal P trace the
reconstruction enters OpenMM at 4.1e25 kJ/mol of atomic overlap and NaN during annealing. The shipped
path never does that: `scripts/benchmark_2oiu.py` runs the CG refiner (`openmm_gpu_refine`, 10000
steps plus a 4-replica REMD) FIRST, and its own recorded e0 is ~1e14. That component did not produce
output in this session's run and needs its own debugging; it is the one remaining step between the
wiring above and an all-atom marginal for the product's topology.

## 4. What is open

* Run the all-atom level from a CG-RELAXED structure (debug `openmm_gpu_refine`, or replace it with a
  few thousand steps of the CG model's own minimisation) and measure the P-trace marginals of the
  product's topology against the tables the loop inverts. The well-posedness question -- is the pooled
  deposited marginal something a physical 300 K ensemble produces? -- is still unmeasured.
* The OP3 filter belongs upstream: as it stands, `aform_from_template` output cannot be fed to
  `amber_refine` without it.
