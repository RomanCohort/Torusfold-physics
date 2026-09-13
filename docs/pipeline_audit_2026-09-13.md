# Pipeline audit — 2026-09-13

Scope: the whole TorusFold pipeline — Level 0/1 structure prediction, the CG physics core,
all-atom refinement, the enhanced-sampling layer, the IBI closed loop, and the analysis/scoring
layer.

Method: six independent auditors, one per slice. Every finding below is one of two things, and
the marker says which:

| Marker | Means |
| --- | --- |
| **[已验] verified** | I read the cited line myself, or I executed the reproduction command and saw the stated result. |
| **[未验] unverified** | Reported by an auditor. I did not independently confirm it. Treat as a lead, not a fact. |

Two ground rules the auditors were held to, and that this document follows:

1. Every finding quotes the code it is based on, at `file:line`.
2. A decision that a comment or a test already records is a decision, not a bug.

**20 of the 52 findings below are verified. 32 are not.** The unverified ones are not withdrawn —
they are leads with a stated reproduction, kept because the alternative was losing them.
(Counts are of the itemised findings in groups A-E; group F records a change made during the
audit rather than a finding.)

## Fixes applied

| Finding | Status |
| --- | --- |
| **D1** | fixed — all seven sites; verified by compiling every touched script and confirming no bare `C.K_INTRA` remains |
| **D3** | fixed — `amber_refine.py:625` now writes `1.06` |
| **D5** | fixed — `reduction="none"` on both losses; verified that the old expression reproduced the unweighted mean exactly (1.327507 == 1.327507) and the new one does not |
| **B4** | fixed — the relaxation moved into the `else` branch; verified by AST that `relax_structure` is now inside `orelse` |
| **B1** | fixed — segments now genuinely overlap; verified end-to-end (see the entry) |
| **B2** | fixed — the ViennaRNA source works for the first time; verified across L = 40…2013 |
| **D21** (new, found while fixing B2) | fixed — same defect as D18, in `multisource_ss.py` |
| **B3** | **part-verified** — changed, mechanism established, magnitude not measurable (see the entry) |
| **Level 0 vote** | semantics corrected after B2 changed the premise (see F1) |
| **C1** | fixed — one `_remd_report_count` shared by worker and both masters; verified at four parameter sets |
| **C2** | fixed — `KB_KJ` replaces scipy's per-molecule `k`; verified the old/new ratio is exactly N_A |
| **C3** | fixed — cell list unioned over replicas; verified the union is a strict superset that the old list under-covered by half in a toy case |
| **C4** | fixed — clip removed per the in-source analysis; `GB_FORCE_CAP` kept but now has no consumer |
| **C5** | fixed — ladder built after the clamps; verified by position in the source |
| **C6** | fixed — `rng_seed` threaded to the builder; static verification only at the time, and OpenMM **is** available in `C:\ana\envs\circrna3d`, so this can be run for real |
| **C7** | fixed — three sites, including a third one in `torch_cgsim.py:2335` that neither audit found |
| **C8** | fixed — the `minimal` branch now binds `_sf/_bjf/_bjg` to None; the NameError was being swallowed as "anneal skipped" |
| **C9** | fixed — `set_pair_k` scales the build-time per-bond k instead of overwriting it with `scale*K_PAIR` |
| **C11** | **reverted.** The 34 `arange` replacements were undone; `torch_cgsim.py` is back to the 12 `_arange_dev` sites the project itself chose. The premise does not reproduce — see the entry |
| **A3** | **fixed.** All 9 `cg_energy_forces` calls in `BatchedREMD2D` now pass the cell list. Measured effect of the omission: 753.7 kJ/mol/nm, **21.2 % of the maximum force** |
| **D2** | fixed and verified end to end — re-running `ibi_update.py` on `results/ibi_chainA_r1` now emits `_source` with the seed, the original cmdline and the 17-constant fingerprint. The run also reproduced the recorded `max|dU| 1.8272 kJ/mol = 0.7326 kBT` exactly, so the change is purely additive |
| **C10** | **BLOCKED — do not "fix" without a decision.** See the entry |
| **C12** | **BLOCKED** — a modelling choice with no recorded answer. See the entry |

### C10 is not a wiring bug — it is an open question the repository already refused to settle

`BatchedREMD2D.__init__` (`torch_cgsim.py:2534-2537`) accepts `relax_bond_k=500.0`,
`relax_angle_k=200.0`, `relax_pair_k=500.0`, `restraint_k=500.0`, stores them, and never passes
them to `cg_energy_forces` — so on this path they are inert and the module constants
(`K_ANGLE=28.1` and friends) are what actually run.

But `metadynamics_gpu.py` **does** wire the same four through (`:200-201`, `:214-215`,
`:244-245`, `:306-307`), and `isrnaclong.py` passes them (`:2037-2040`, `:2174-2177`). So the two
production paths are silently running **different angle constants**: 200.0 in metadynamics,
28.1 in REMD.

That discrepancy is already recorded in the repository, and the record declines to resolve it.
`scripts/cg_potentials.py:16-20`:

> The module constant is K_ANGLE=28.1, but every production caller passes relax_angle_k=200.0
> (isrnaclong.py:1950, :2087) -- a factor of 7 -- so a sampler that inherited "the shipped angle"
> could mean either one and would print neither. force_reference._v_fn refuses that spec for the
> angle rather than choosing.

`scripts/force_reference.py:123` repeats it. Wiring `BatchedREMD2D` through would make the two
paths agree at 200.0, which is a 7x change to the relaxation stiffness on the primary path, and
choosing 200.0 is exactly what the repository said it would not do without a measurement. So
this is left alone, deliberately, with the divergence made visible here.

### C12 is a modelling choice with no recorded answer

`cg_energy_3bead:951` excludes `|residue_i - residue_j| <= 1` from the excluded-volume sum.
`cg_energy_forces:1420` excludes `|bead_i - bead_j| <= 2`, which for P beads (one per residue) is
`|residue_i - residue_j| <= 2`. The two fields therefore disagree by one residue of
neighbourhood. Neither site states an intent, and both are defensible — excluding 1-3 pairs is
standard when an explicit angle term carries that interaction, which both fields have. Changing
either shifts an energy that has been calibrated. Flagged rather than guessed.

Nothing else has been changed. The remaining findings are still open, and the 32 unverified ones
should be confirmed before they are fixed — the four fixes above are all in the verified set.

---

## Group A — known to affect results that already exist

### A1. Checkpoint resume replays a completed run as a fresh measurement — **[已验] verified**

`scripts/ibi_remd_residual.py:219-241`

```python
if str(z["config_key"]) != _config_key():
    return None
...
step = int(z["step"])
return step, pos, vel
```

`_load_ckpt` compares `config_key` and reads `step`, but never rejects a checkpoint whose
`step` already equals `NSTEPS`. The docstring claims a checkpoint is resumed only when "its
step is below NSTEPS". The shipped `results/remd_ckpt_idx0.npz` has `step = 330000`, and
`NSTEPS` defaults to `330000`, so re-running the same command resumes a finished run: the
`while` loop does not execute once and the stored accumulators are printed as if they were a
new measurement.

Reproduction (run; output shown):

```bash
python - <<'EOF'
import numpy as np
z = np.load("results/remd_ckpt_idx0.npz", allow_pickle=True)
print("ckpt step =", z["step"])          # -> 330000
EOF
grep -n "^NSTEPS" scripts/ibi_remd_residual.py   # -> 100:NSTEPS = int(_positional[0]) if _positional else 330000
```

Related, same function: `_config_key()` (`:176-180`) folds `NSTEPS` and the raw `_set_str`
into the key, so (a) a run killed early cannot be continued with a larger `NSTEPS`, and (b) the
same field written with a different `--set` ordering is rejected as a different field.

### A2. Injected table potentials are a flat clamp; the deployed path has a wall — **[未验] unverified**

`scripts/cg_potentials.py` → `force_reference.py:99-104`, against `boltzmann_bonded.py:489-508`

```python
u = (q - lo) / binw - 0.5
i0 = floor(u).clamp(0, len(U) - 2)
f  = (u - i0).clamp(0, 1)
return U[i0] + f * (U[i0 + 1] - U[i0])
```

The sampler treats the potential as flat outside the tabulated support; the deployment path
(`boltzmann_bonded.energy` / `mixed_energy`) applies a quadratic wall outside it. So the IBI
fixed point is found on a potential that differs from the one that gets deployed. The angle
table (`hi = 0.7998`) is the most exposed, since the top of the cosine range falls outside.

Not closed: whether deployment actually goes through `mixed_energy`. The auditor marked this
explicitly. **Do not act on this one before closing that question.**

### A3. Excluded volume is missing from the REMD exchange criterion — **[已验] verified**

`src/torusfold/scheme2/torch_cgsim.py:2878, 2879, 2884, 2897, 2900` against the gate at `:1343`

```python
# :1343
if cell_list is not None:
    e_c, f_c = _clash_f(pos_nm, cell_list, _clash_k, CLASH_SIGMA)
    total_E += e_c; total_F += f_c
```

```python
# :2878-2884  (BatchedREMD2D._energy_split)
en_ref, _  = cg_energy_forces(p, pairs_t, pw, lam=1.0)
en_base, _ = cg_energy_forces(p, pairs_t, pw, lam=0.0)
...
en_own, _  = cg_energy_forces(p, pairs_t, pw, lams=own_lams)

# :2897, :2900  (the 500-step Langevin relaxation)
_e, _f = cg_energy_forces(pos, pairs_t, pw, lams=lams_t)
```

None of these pass `cell_list`, so the excluded-volume term is absent from the exchange
criterion and from the relaxation forces, while the main MD loop (`:2983`) does pass it. The
dynamics run on a field the exchange decision does not see. Different replicas hold different
geometry, so this is not a constant offset that cancels.

**Still open:** which already-produced artifacts came out of `BatchedREMD2D`. The REMD runs in
`results/remd_*.log` did not — see the note under C7.

**Magnitude, measured — and this is the largest of the audit's findings.** On the real 24-replica
configuration in `results/remd_ckpt_idx0.npz`, the same function with and without the cell list
differs by:

| | with − without |
| --- | ---: |
| max \|dE\| per replica | 19.430 kJ/mol |
| max \|dF\| | 753.690 kJ/mol/nm |
| relative to the maximum force | **21.12 %** |

For comparison the C3 + C4 changes, measured identically, move the force by 0.28 %. The
excluded volume is an entire term, not a neighbour-list correction.

**Fixed** by building the cell list before `_energy_split` is defined (it used to be created
further down, in the MD loop, so the exchange criterion and the relaxation ran before it
existed) and passing it at all nine call sites in the class. `GPUCellList` rebuilds itself
lazily in `get_pair_info`, so one instance serves every call site. Verified: the class
constructs on `cuda` in `C:\ana\envs\comfyui`, and the full suite still passes 145 tests.

Reproduction: read `:1343` for the gate, then confirm `cell_list` is absent from the five call
sites above.

---

## Group B — live on the main path, but nothing has been produced from them yet

### B1. The Kabsch assembly never runs — **[已验] verified**

`src/torusfold/scheme2/segmented_vfold3d.py:196-214` and `:786-816`

```python
# :197  (per segment, seg_idx > 0)
overlap_start = max(pos, best_boundary - effective_overlap)
...
"overlap_end": best_boundary if overlap_start >= 0 else -1,
```

```python
# :802
target = full_coords[ol_start:ol_end]
...
# :806
if len(moving) > 0 and np.any(target) and np.any(moving):
    aligned, R, rmsd = kabsch_align(moving, target)
```

`overlap_start` is `max(start, end - overlap)`, i.e. the **tail** of the current segment, and
`overlap_end` is that segment's own end. Assembly fills `full_coords` left to right and, when
it reaches this segment, has only written up to `start`. The region `[ol_start, ol_end)` is
therefore still all zeros, `np.any(target)` is `False`, and the alignment is skipped. The
segments are concatenated unrotated.

Compounding it: `:216` sets `pos = best_boundary`, so consecutive segments are exactly adjacent
in sequence coordinates and share no residues at all. There is no overlap region to align
against even if the ordering were fixed.

Consequence: `run_2013nt.py`'s comment "larger overlap -> more accurate Kabsch alignment"
describes a mechanism that does not execute.

Reproduction: read `:196-214` for how the overlap is defined and `:787-806` for the order in
which `full_coords` is filled.

**Fixed** by starting each segment `effective_overlap` residues before its predecessor's end
(`split_sequence`), so the overlap is the segment's head and covers the predecessor's tail, and
`overlap_end` is the predecessor's end rather than the segment's own. The segments become
`effective_overlap` residues longer than `max_seg_len`; nothing enforces that bound, and the
MSA-anchored path already produces chunks up to 1.5x `max_seg_len`.

Verified end to end: chunks are predicted in independent random frames (random rotation plus a
40 A translation), then assembled and compared against the known ground truth after an optimal
global superposition.

| case | unaligned concat | after `confidence_weighted_assemble` |
| --- | --- | --- |
| 2 segments, no noise | 294.883 A | **0.000 A** |
| 5 segments, 0.5 A noise/chunk | 1015.064 A | **5.870 A** |

The residual 5.87 A is inherent to rigid-chunk assembly — each chunk is fitted on a 20-residue
overlap, and the small angular error is amplified over the remaining ~180 nt — not a defect.
The overlap invariant (`overlap_start == segment start`, inside the predecessor's span) was
asserted separately on both cases.

### B2. `multisource_ss`'s ViennaRNA source has never worked — **[已验] verified**

`src/torusfold/scheme2/multisource_ss.py:30-38`

```python
for i in range(1, L + 1):
    if bp[i] > 0 and int(bp[i]) > i:
        ...
except (ImportError, Exception):
    return None, None
```

`fc.bpp()` returns a 2-D `(L+1, L+1)` tuple in ViennaRNA 2.x; the code indexes it as if it were
a vector. That raises `TypeError`, which the bare `except` at `:37` converts to `(None, None)`.
With the Nussinov source gated behind `L <= 500` (`:161`), `multisource_consensus_ss` returns
all dots for any sequence longer than 500 nt.

Reproduction (run; output shown):

```bash
python - <<'EOF'
import RNA, numpy as np
seq = "GGGAAACCC" * 8
fc = RNA.fold_compound(seq); fc.pf()
print("np.array(bp).shape =", np.array(fc.bpp()).shape, " seq len =", len(seq))
try:
    bp = fc.bpp()
    for i in range(1, len(seq) + 1):
        if bp[i] > 0 and int(bp[i]) > i: pass
    print("loop OK")
except Exception as e:
    print("RAISES:", type(e).__name__, e)
EOF
# np.array(bp).shape = (73, 73)  seq len = 72
# RAISES: TypeError '>' not supported between instances of 'tuple' and 'int'
```

Knock-on effect: `run_2013nt.py:67` sees `ss.count('(') == 0` and falls back to an
`md.circ = 1` ViennaRNA MFE fold. That fallback is the **same call** Level 0's own `mfe_set`
makes on the same `fold_compound` (`isrnaclong.py:643-645`), so the secondary structure handed
to Level 0 is not an independent prediction — it is `mfe_set` again.

**Fixed.** Two things were needed, and the second only showed up after the first: `fc.bpp()`
returns an empty tuple if `fc.pf()` has not been called, silently rather than by raising, so the
partition function is now computed first; and the `(L+1, L+1)` matrix is indexed properly. The
`except` was narrowed to `ImportError`, because the bare `except (ImportError, Exception)` is
precisely what kept this invisible: a broken source looked identical to an absent one.

| L | before | after |
| --- | --- | --- |
| 600 | 0 pairs (all dots) | 180 |
| 1200 | 0 pairs (all dots) | 372 |
| 2013 | 0 pairs (all dots) | 616 |

This changes what Level 0 receives on the production path. `run_2013nt.py:67` no longer fires
its circular fallback, so the input is now a **linear** fold and genuinely differs from the
circular `mfe_set` — which is exactly the case the Level 0 pair vote was written for, and it is
now exercised rather than degenerate. The comment in `_fuse_pair_sources` has been updated to
describe both routes.

### B3. Pre-fold Adam takes autograd over an energy whose solvation part is detached — **[已验] verified**

`src/torusfold/scheme2/torch_gpu_refine.py:144-147`

```python
e, _ = cg_energy_forces(pos_3bead, pairs_t, pw)
...
e.sum().backward()
```

The returned forces are discarded and the gradient is taken by autograd. But
`torch_cgsim.py:1442` and `:1468` add the GB/SA and Manning energies with `.detach()`:

```python
total_E += _gb_e_tot.detach()
total_E += e_mi.detach()
```

Detached terms contribute nothing to `backward()`. So this minimisation sees the solvation
**energy** in `e` while its **force** is absent from the gradient — the field is not
`-dE/dx`. This is the same F != -grad E defect that `torch_cgsim.py:1444-1461` documents and
fixes for callers that use the returned `total_F`; a caller using autograd bypasses that fix.

Reproduction: read `torch_gpu_refine.py:144-147` alongside `torch_cgsim.py:1442/1468`.

**Changed, but NOT fully verified — treat this entry as part-verified.** The pre-fold loop now
takes its gradient from the returned force (`pos_3bead.grad = (-f).clone()`) instead of
`e.sum().backward()`.

What is established: detached tensors contribute nothing to `backward()`, and the source's own
comment at `torch_cgsim.py:1444-1461` states the consequence in the same terms — the energy kept
`_gb_e_tot.detach()` while "the force lost the whole GB/SA + Manning contribution".

What is NOT established: the magnitude. Three configurations were built to measure it (random,
hand-built helix, and one at exact bond equilibrium) and **all of them saturated the global force
cap** (`max|F|` between 4899 and 4999 against a cap of 5000), so the returned force is clipped
while the autograd gradient is not, and the difference between them is dominated by the clip
rather than by the solvation term. No trustworthy number was obtained.

There is also an unresolved design point: `total_F` is the *capped* force, so switching to it
changes the descent direction wherever the cap is active. Using the field the dynamics integrate
is self-consistent; using the true gradient is the other defensible choice. This is flagged, not
settled.

### B4. Level 1.5 restores from checkpoint and then re-runs anyway — **[已验] verified**

`src/torusfold/scheme2/isrnaclong.py:1159-1168`

```python
_l15_from_ckpt = (_l1_from_ckpt and "coords_relaxed" in ckpt)
if _l15_from_ckpt:
    coords_vfold = ckpt["coords_relaxed"]
    if verbose:
        print(f"\n[Level 1.5] restored globally relaxed coordinates from checkpoint")
else:
    if verbose:
        print(f"\n[Level 1.5] CG global restraint relaxation...")
_l15_ok = False
try:
    ...relax_structure(...)      # runs unconditionally
```

The `else:` holds only the print. `_l15_ok = False` and the `try:` sit at the same indentation
as the `if`, so `relax_structure` — 5000 steps — runs whether or not the checkpoint was
restored. The "restored from checkpoint" message is true and the restoration is then discarded.

Reproduction: check the indentation of `:1167-1168` against `:1160`.

---

## Group C — fallback paths (not currently reached)

`isrnaclong.py:1496-1501` prefers `torch_gpu_refine` and falls back to `openmm_gpu_refiner`
only if that import fails; `:2135` vs `:2166` does the same for REMD. The findings in this
group sit on the fallback side of those branches.

### C1. REMD report count mismatches the master's expectation — **[已验] verified**

`openmm_gpu_refiner.py:1105/1130/1154` (worker) against `:1316-1322` (single round) and
`:1582-1588` (multistage)

```python
# worker
n_anneal_steps = max(200, int(n_steps * 0.4))
for step_i in range(max(n_steps - n_anneal_steps, exchange_interval)):
    if (step_i + 1) % exchange_interval == 0:
        conn.send(("report", worker_idx, total_energy, pos))
...
conn.send(("done", worker_idx, ...))     # :1174
```

```python
# multistage master
n_ex = n_steps_per_round // max(10, n_steps_per_round // 10)
...
if msg[0] == "report":                   # :1588 — "done" leaves energies[ri] = None
```

With the defaults (`use_multistage_remd=True`, `remd_n_steps=100000` ->
`n_steps_per_round = 33333`, `exchange_interval = 3333`): the worker sends **6** reports, the
master waits for **10**. On the 7th receive the master gets `"done"`, `energies[ri]` stays
`None`, and `(ui - uj)` at `:1597` raises `TypeError`. There is no try/except at the call site
(`:2142`).

The two counts only coincide when `exchange_interval > 0.6 * n_steps`, i.e. when there is a
single exchange point. The default `n_steps_per_round // 10` does not satisfy that.

Reproduction: evaluate both expressions at the defaults, or read the four cited lines.

### C2. Metropolis criterion is off by a factor of N_A — **[已验] verified**

`openmm_gpu_refiner.py:1259` and `:1333-1335`; the same construction at `:1598-1600`

```python
from scipy.constants import k as kB       # 1.380649e-23 J/K, per molecule
beta_i = 1.0 / (kB * temperatures[ri] / 1000.0)
beta_j = 1.0 / (kB * temperatures[ri + 1] / 1000.0)
exponent = np.clip((beta_i - beta_j) * (ui - uj), -30, 30)
```

`ui` is an OpenMM potential energy in kJ/mol (`:1156` says so in a comment). The matching
beta is `1/(R*T)` with `R = 0.008314` kJ/(mol*K), giving 0.401 at 300 K. The code computes
`1000/(1.380649e-23 * 300) ~ 2.4e23` — larger by exactly Avogadro's number. `exponent` is
therefore always clipped to +/-30; since beta decreases with T, the sign is decided by
`ui - uj` alone and the swap becomes a deterministic rule rather than a Metropolis test.

Reproduction: compute `1000/(1.380649e-23*300)` against `1/(0.008314462618*300)`.

### C3. The GB neighbour list is built from replica 0 only — **[已验] verified**

`openmm_gpu_refiner.py:1415` against `:1429`

```python
p_coords = gb_pos[0]                                    # :1415
p_cell = torch.floor(p_coords / GB_CUTOFF).long()
...
d_ij = gb_pos[:, pi_g, :] - gb_pos[:, pj_g, :]          # :1429 — all B replicas
```

The cell list and the pair list are derived from replica 0's geometry, then applied across the
replica batch. A pair that is a neighbour in replica 3 but not in replica 0 is dropped from
replica 3's GB interaction.

Reproduction: read `:1415` and `:1429`.

### C4. GB force is clipped at 50 while its energy is added in full — **[已验] verified**

`src/torusfold/scheme2/torch_cgsim.py:1495-1498` against `:1442` and `:1468`

```python
gb_f = -gb_pos.grad
_gb_f_mag = gb_f.norm(dim=-1, keepdim=True).clamp(min=1e-12)
gb_f = gb_f * torch.clamp(GB_FORCE_CAP / _gb_f_mag, max=1.0)   # :1497
total_F[:, P(torch.arange(L, device=dev))] += gb_f
```

**This one is already documented in the source.** `:1474-1494` is a long comment that names it
("It is a heater, by construction"), explains why the clip fires exactly where the energy is
largest, reports that raising the outer cap from 200 to 5000 moved the mean kinetic temperature
from 440.8 K to 606.6 K, and states that the clip "can now be removed". The clip is still
present. So this is a recorded decision that was never executed, not an undiscovered defect.

Reproduction: read `:1474-1498`.

### C5. Temperature ladder is built before the memory clamp — **[已验] verified**

`openmm_gpu_refiner.py:1263` against `:1268`

```python
temperatures = [300.0 * (1.10 ** i) for i in range(n_replicas)]     # :1263
...
n_replicas = _clamp_replicas_by_memory(n_replicas, mem_per_proc_gb=1.5)   # :1268
```

If the clamp lowers `n_replicas`, the ladder is truncated and the hottest replica never reaches
its design temperature. The multistage path does this in the correct order (`:1545` before
`:1546`) with a comment saying why.

### C6. Per-worker seeding has no effect — **[已验] verified**

`openmm_gpu_refiner.py:273` against `:815`

```python
rng = np.random.default_rng(42)      # :273, inside _build_3bead_system_gpu
...
_np.random.seed(42 + worker_idx)     # :815, in the worker
```

`default_rng(42)` is an independent `Generator`; seeding the legacy global state does not touch
it. Every worker therefore produces the same C4'/N perturbation, contradicting the docstring at
`:809`.

**Correction to the auditor:** it reported that trajectory diversity is lost entirely. It is
not — `:819` varies `bsj_k_scale=0.1 + 0.05 * worker_idx` per worker. Only the initial
perturbation is identical.

### C11 outcome — reverted, and why

The premise was testable after all, and it did not hold up. `_arange_dev`'s docstring names
torch 2.12a0+rocm7.13 on Radeon 8060S / gfx1151 — and that environment exists on this machine:
`C:\ana\envs\comfyui` has `torch 2.12.0a0+rocm7.13.0a20260313`, hip 7.2.0, an
**AMD Radeon(TM) 8060S Graphics** at **gfx1151**. Version, device and arch all match.

Bare `torch.arange(..., device='cuda')` was then run **in that environment**, in the usage pattern
`cg_energy_forces` actually has (as an index into a batched tensor), at L = 27 / 200 / 2013,
inside `no_grad`, over 200 iterations. **It never aborted.** One branch remains untested:
`torch.compile`, because that environment has no working Triton.

The 34 replacements were therefore reverted. `torch_cgsim.py` is back to the 12 `_arange_dev`
sites the project chose. The reason to revert is not that the docstring is certainly wrong — it
is that the cost is certain (a host→device copy per call in the hottest function) and the
benefit could not be demonstrated. **If the abort ever reappears, the fix is one line at the
site where it happens.**

Method note, recorded because it caused two wrong answers in this session: the pipeline's
environments are the conda envs. `C:\ana\envs\circrna3d` is the CPU one (torch 2.13.0+cpu,
pytest 9.1.1, OpenMM 8.5.2) and `C:\ana\envs\comfyui` is the GPU one. Checking only the
interpreters listed by `py -0p` (3.14 / 3.13 / 3.12) produced both the false claim that pytest
was unavailable and the false claim that OpenMM was. **Use `circrna3d` for tests and `comfyui`
for anything touching the GPU.**

### C7. The temperature-axis exchange criterion has an inverted sign — **[已验] verified**

`src/torusfold/scheme2/rest2_remd_2d.py:123-125`, and the same shape at `:163-166` and
`src/torusfold/scheme2/rest2_sampler.py:160-163`

```python
d_beta = _beta(temps[ri]) - _beta(temps[ri + 1])                     # beta_cold - beta_hot > 0
expo   = np.clip(d_beta * (energies[a] - energies[b]), -30, 30)      # = Delta
if expo <= 0 or np.random.rand() < np.exp(-expo):                    # min(1, exp(-Delta))
```

Detailed balance gives `accept = min(1, exp(Delta))` with
`Delta = (beta_i - beta_j)(E_i - E_j)`. The code computes exactly that `Delta` and then accepts
with `exp(-Delta)` — the reciprocal. It accepts unconditionally in the cases that should
usually be rejected.

The internal control is decisive: the lambda axis in the same function (`:140-142`) uses
`d_lam = lambdas[cj + 1] - lambdas[cj]`, i.e. the "target minus source" ordering, which makes
`expo = -Delta` and `exp(-expo) = exp(Delta)` **correct**. Two axes in one file, one right and
one flipped.

**This did not produce the results in `results/`.** Those logs come from
`scripts/ibi_remd_residual.py` — its prints at `:147`, `:150` and `:434` match the log verbatim —
and that script uses `torch_cgsim.py:2355` `exchange_log_alpha_temperature` with `:2369`
`metropolis_accept_log_alpha`, which is the correct rule (`log_alpha >= 0` -> accept, else
`exp(log_alpha)`). The `note:` line in the log that mentions `rest2_remd_2d.py` is comparing
defaults; it is not a statement of which sampler ran.

Reproduction:

```bash
grep -n "T-acc\|REMD grid:\|sampling the cold rung" scripts/ibi_remd_residual.py
grep -n "def exchange_log_alpha_temperature" -A 5 src/torusfold/scheme2/torch_cgsim.py
grep -n "def metropolis_accept_log_alpha" -A 8 src/torusfold/scheme2/torch_cgsim.py
```

### C8. `_run_remd_worker` references undefined names on the `minimal` branch — **[未验] unverified**

`openmm_gpu_refiner.py:1024-1025` unpack only `_pf` when `minimal=True`; `:1107-1109` then
passes `_bjf`/`_bjg` to `_run_annealing`, raising `NameError` that `:1114-1115` catches and
prints as "anneal skipped". The auditor notes both current call sites pass `minimal=False`, so
this is latent.

### C9. `_run_annealing`'s `set_pair_k` discards per-pair weights — **[未验] unverified**

`openmm_gpu_refiner.py:733-740` rewrites every pair bond's `k` to `scale * K_PAIR`, dropping
the per-pair `w` written at build time (`:441-443`).

### C10. `relax_*_k` parameters are accepted and never used — **[未验] unverified**

`torch_cgsim.py:2543-2546` stores `relax_bond_k` / `relax_angle_k` / `relax_pair_k` /
`restraint_k`, which are never passed to `cg_energy_forces`; the defaults also disagree with
the current constants.

### C11. Production path uses bare `arange`, the fallback uses the safe wrapper — **[未验] unverified**

`cg_energy_forces` uses `torch.arange(..., device=dev)` at eight sites while `_arange_dev`
(`:467-474`) exists to work around a device-specific abort on Radeon 8060S, and the fallback
field `cg_energy_3bead` uses it. Whether this aborts on the target device was not tested.

### C12. `cg_energy_3bead` excluded-volume mask disagrees with the cell list — **[未验] unverified**

`:948-952` excludes on residue distance <= 1 (bead distance <= 5) where the production cell
list uses bead distance <= 2 (`:1812`).

---

## Group D — tooling, provenance, training and metrics

### D1. `K_INTRA` was split, and seven scripts still reference the removed name — **[已验] verified**

Reported as one site; it is seven. `torch_cgsim` replaced `K_INTRA` with `K_INTRA_PC`
(P-C4', 20752.7) and `K_INTRA_CN` (C4'-N, 36399.2). Four scripts crash, one silently skips a
term, one prints `nan`, one carries a stale label:

| Site | Failure |
| --- | --- |
| `scripts/refit_tables_clean.py:35` | `AttributeError` — the baseline table generator |
| `scripts/calibrate_boltzmann.py:41` | `AttributeError` |
| `scripts/decompose_database_variance.py:80` | `AttributeError` |
| `scripts/funnel_vs_stiffness.py:39` (`getattr` with no default, `:48`) | `AttributeError` |
| `scripts/attribute_gradcheck_to_term.py:51` | silently prints "(no such constant)" and skips the term |
| `scripts/describe_current_force_field.py:80` | `getattr(..., nan)` — prints `nan` |
| `scripts/check_ff_targets_vs_native.py:62-63` | label string `"K_INTRA 400"`, stale |

The first site, as originally reported:

`scripts/refit_tables_clean.py:35`

```python
K_HARM = {"bb_bond": C.K_BB, "intra_pc": C.K_INTRA, "intra_cn": C.K_INTRA, ...}
```

`torch_cgsim` no longer defines `K_INTRA`; it was split into `K_INTRA_PC` / `K_INTRA_CN` /
`K_INTRA_PN` (`:140-141`, `:181`). The script that regenerates `boltzmann_tables_clean.npz` —
the baseline input to the entire IBI loop — raises `AttributeError`. Provenance of the shipped
table cannot be reproduced on a new machine.

Reproduction (run; output shown):

```bash
python - <<'EOF'
import sys; sys.path.insert(0, 'src')
import torusfold.scheme2.torch_cgsim as C
print("K_INTRA    =", hasattr(C, "K_INTRA"))
print("K_INTRA_PC =", hasattr(C, "K_INTRA_PC"), getattr(C, "K_INTRA_PC", None))
try:
    K_HARM = {"bb_bond": C.K_BB, "intra_pc": C.K_INTRA}
except AttributeError as e:
    print("RAISES:", e)
EOF
# K_INTRA    = False
# K_INTRA_PC = True 20752.7
# RAISES: module 'torusfold.scheme2.torch_cgsim' has no attribute 'K_INTRA'
```

### D2. The update step discards the sampler's fingerprint — **[已验] verified**

`scripts/ibi_update.py:91-97`

```python
manifest = hist_dir / "manifest.json"
...
_pots = json.loads(manifest.read_text(encoding="utf-8")).get("potentials", {})
```

The manifest carries a `fingerprint` written by the sampler; `ibi_update.py` reads only
`potentials`. The word does not occur in the file at all. So a table produced from a patched
field carries no record of which field, seed, or round produced it, and a mismatched read-back
cannot be detected. `ibi_round0.py:271-284` takes care to write that provenance.

Reproduction (run; output shown):

```bash
grep -c "fingerprint" scripts/ibi_update.py     # -> 0
```

### D3. `amber_refine.py` pairing restraint target is 10x too small — **[已验] verified**

`src/torusfold/scheme2/amber_refine.py:621`

```python
pair_force.addBond(ci, cj, [0.106])
```

`CustomBondForce` lengths are nm, and the file header at `:16` states `r0=1.06 nm
(10.6 A C1'-C1')`; the last annealing stage at `:721` is `(100.0, 1.06)`. The literal `0.106`
is 1.06 A. The default path overwrites it during annealing, so it is masked — except when
`pair_anneal_stages=[]` is passed, which takes the branch at `:722-725` and skips annealing
entirely, leaving `0.106` live. The docstring at `:402-403` says that path should "use k=100
r0=1.06 directly".

Reproduction: read `:16`, `:621`, `:719-725`, `:402-403`.

### D4. The repository has two incompatible `WC_TARGET_DIST` conventions under one comment — **[已验] verified**

| Value | Site | Comment |
| --- | --- | --- |
| 10.6 | `pdb_analyzer.py:22`, `constraint_solver.py:31`, `immune_heuristic.py:275` | C1'-C1' |
| 10.6 | `refine.py:26` | C1'-C1' **(approximated by P here)** |
| 10.5 | `rl_optimizer.py:52` | C1'-C1' |
| 20.0 | `segmented_vfold3d.py:37`, `trrna2_calibrator.py:27` | C1'-C1' |

An auditor reported `trrna2_calibrator.py:27` as an isolated typo. It is not isolated — the
same value and comment appear in `segmented_vfold3d.py:37`, and `refine.py:26` shows the
authors know C1'-C1' is being used as a proxy for P. Which value is correct depends on the atom
type at each call site, and that is not uniform. **Resolve the convention before changing any
of these numbers.**

Reproduction (run):

```bash
grep -rn "WC_TARGET_DIST\|IDEAL_WC_DIST\|pair_dist_target" src/ | grep -v molstar
```

### D5. `weight_mask` has no effect on the loss — **[已验] verified**

`src/torusfold/scheme2/multitask_loss.py:87-93` and `:124-127`

```python
loss_ss = F.cross_entropy(ss_logits[known_mask], ss_labels[known_mask].long())
if weight_mask is not None:
    w = weight_mask[known_mask]
    loss_ss = (loss_ss * w).sum() / w.sum().clamp(min=1.0)
```

`F.cross_entropy` defaults to `reduction='mean'` and returns a **scalar**. `(scalar * w).sum()
/ w.sum()` is identically that scalar. The same holds for the `binary_cross_entropy` case at
`:124-127`. Overlap-chunk down-weighting does not reach the gradient, and
`build_overlap_weight_mask` (`multitask_heads.py:511`) is computed for nothing.

Reproduction: read `:87-93`; note the default `reduction` is not overridden.

### D6. `pair_satisfaction` judges P-P distances against a C1'-C1' target — **[已验] verified**

`src/torusfold/scheme2/immune_heuristic.py:275` and `:311-322`

```python
pair_dist_target = 10.6
...
# pair_satisfaction: fraction of pairs meeting the P-P target distance (10.6 +/- 1.5)
d = float(np.linalg.norm(cg_coords[i] - cg_coords[j]))
if abs(d - pair_dist_target) < 1.5:
```

The docstring at `:268` says `cg_coords` are CG **P** coordinates, and `:274`'s
`bond_length = 5.9` (the P-P step) is used for `bond_rmsd` at `:311-312`, confirming it. 10.6 A
is the C1'-C1' distance. If the input really is P coordinates, the indicator is near zero for
real structures. The comment at `:314` labels the target "P-P", so the code agrees with the
comment and both disagree with the value.

**Not measured.** The repository's own standard is to measure before asserting; the actual P-P
distribution across a WC pair should be measured on a reference structure before this number is
changed.

### D7. `ibi_round0.py` displays a correction computed with a different pseudocount — **[未验] unverified**

`:314` uses `(counts[c] + 1.0) / (counts[c].sum() + len(counts[c]))` where the real update
(`ibi_bonded.py:104`, `DEFAULT_PSEUDO = 0.5`) uses 0.5. The logged correction size therefore
does not match the table actually written. Magnitude is small.

### D8. `ibi_round0_circular.py`'s fingerprint omits `K_BSJ_CONTACT` — **[未验] unverified**

`:109-112` lists `K_BSJ` and `K_BSJ_GUIDE` but not `K_BSJ_CONTACT`, the split that
`ibi_round0.py:152-158` records as the cause of a run reporting the fingerprint of an
unpatched field.

### D9. `two_start_ergodicity.py`'s annealing checkpoint does not fingerprint the field — **[未验] unverified**

`:107-111` validates `L`, `anneal_steps`, `anneal_temp` and `seed` but nothing about the force
field, so a changed constant would silently reuse a start B annealed under the old field.

### D10. `ncm_detector.global_pair_assembly`'s `wc_hard` / `wc_soft` are dead — **[未验] unverified**

`:334-340` declares them; the body (`:363-415`) uses only `min_prob=0.01`, so every WC pair
above 1% enters the result regardless of the documented thresholds.

### D11. `trrna2_calibrator` divides by requested samples, not successful ones — **[未验] unverified**

`:294-317` accumulates only when `tr_result.dist is not None` but divides by
`n_trrna2_samples`.

### D12. `multitask_heads._build_own_features` has colliding feature indices — **[未验] unverified**

`:284-287` maps `offset in 1..5` to `9 + min(offset, 4)`, giving `{10,11,12,13,13}`, and `:293`
then overwrites index 13 with `is_stem`.

### D13. `ncm_detector` tandem rule sums both flanks — **[未验] unverified**

`:52` documents "minimum WC pairs on each side"; `:217-221` tests
`(flank_up + flank_dn) >= _TANDEM_MIN_FLANK`, which admits a one-sided flank.

### D14. `rcm._validate_rcm` returns after the first mismatch — **[未验] unverified**

`:65-73` can only return 0 or 1, so the `max_mismatch` parameter exposed by
`compute_rcm_score` cannot distinguish 1 from 2.

### D15. `build_overlap_weight_mask`'s `linear` branch ramps instead of dipping — **[未验] unverified**

`multitask_heads.py:511-527`: `cosine` is lowest at the centre of the overlap, `linear` rises
monotonically from `min_weight` to 1.0 across it. The default is `linear`.

### D16. `pair_graph` counts WC pairs without wobble in one place and with it in another — **[未验] unverified**

`:78-88` uses `_is_wc_pair`; the quality gate at `:288-291` uses `_is_complementary`, which
includes G-U.

### D17. `pdb_analyzer._get_vdw_radius` cannot match two-letter elements — **[未验] unverified**

`:114-126` takes `name[:1].upper()`, so `FE`/`MG`/`CA`/`NA` fall through to the 1.5 A default.

### D18. `overlap_confidence._confidence_weighted_consensus` uses a zero diagonal as confidence — **[未验] unverified**

`:82-89`: the diagonal of a pairing matrix is zero, so every predictor weight collapses to
`max(conf, 0.01)` — equal weighting, not confidence weighting.

### D19. `ensemble_predictor` fallback branches are unreachable / truncated — **[未验] unverified**

`:408-424` — the geometric-circle fallback cannot be reached because `np.zeros((L,3))` is not
empty; `:528-534` — the `len(coords) > expected_L` branch is dead behind the `>=` return above
it, so a short P-atom list is silently truncated and returned.

### D21. `multisource_ss` derives per-predictor confidence from a diagonal that is always zero — **[已验] verified, and fixed**

Found while fixing B2; it is the same defect D18 reports, at a second site.

`src/torusfold/scheme2/multisource_ss.py:107-112`

```python
# per-predictor confidence: average of the diagonal elements
diag = np.diag(bpp[:L, :L])
conf = float(np.mean(diag[diag > 0.05])) if np.any(diag > 0.05) else 0.0
confidences.append(max(conf, 0.01))
```

The diagonal of a base-pair matrix is identically zero — a base does not pair with itself — so
`np.any(diag > 0.05)` is never true, every predictor takes the `0.01` floor, and the
"confidence-weighted consensus" below is equal weighting.

Reproduction (run; output shown):

```bash
python - <<'EOF'
import sys; sys.path.insert(0,'src')
import numpy as np
from torusfold.scheme2.multisource_ss import _vienna_fold_consensus
seq = "ACGU" * 30
ss, bpp = _vienna_fold_consensus(seq)
print("diag max (what the old code averaged):", float(np.max(np.diag(bpp))))   # -> 0.0
iu = np.triu_indices(len(seq), k=1)
print("offdiag mean where >0.05           :", float(np.mean(bpp[iu][bpp[iu] > 0.05])))  # -> 1.0
EOF
```

**Fixed** by taking the mean over the upper triangle instead. This is deliberately recorded as a
*new* finding rather than folded into D18: D18 is unverified and sits in
`overlap_confidence.py`; this one is verified, sits in `multisource_ss.py`, and was found
independently. The two should be checked against each other.

**Residual, not fixed:** for L <= 500 the majority-vote step further down
(`multisource_ss.py:124+`) still discards most of the structure — with the ViennaRNA source
producing 88 pairs at L=300, the consensus returns 3. That is a separate defect in the vote, not
in the source, and it is not covered by any finding above.

### D20. Level 1 resume restores default confidences — **[未验] unverified**

`isrnaclong.py:1016` reconstructs `[0.5] * n_segments` because the fresh branch never saves
`chunk_confidences`; `:1135` then overwrites `_chunk_unc` with `[1.0]`.

---

## Group E — enhanced sampling and metadynamics (all **[未验] unverified**)

None of the following were independently confirmed. They are stated as reported.

### E1. `rest2_remd_2d` acceptance statistics count only accepted swaps — **[已验] verified**

`_record_axis` (`:356-368`) is called from exactly one site, `:297`, inside the loop over
**accepted** swaps, and increments `att_T`/`att_L` unconditionally. Attempts therefore equal
acceptances and the reported acceptance is always 1.0. The `pass` at `:308` is where
rejected attempts would have been counted.

Note this is `rest2_remd_2d.py` only. The acceptance figures in `results/remd_*.log`
(0.26-0.33 on the temperature axis) come from `ibi_remd_residual.py`, which increments
`attT[k]` at `:338` before the decision and is therefore correct.

### E2. `rest2_sampler`'s exchange criterion does not match its own ensemble — **[未验] unverified**

`:124-125` sets `lambda_i = T_low / T_i` and `:160-163` applies the plain T-REMD form to
lambda-scaled total energies; with a per-replica Hamiltonian the correct acceptance contains
cross terms. The module docstring's own formula does not match the code either.

### E3. Two-dimensional exchange can select one replica on both axes — **[未验] unverified**

`rest2_remd_2d.py:292-299` sends swap commands for the temperature and lambda axes in the same
round, while the worker (`:490-508`) consumes exactly one command per round, so a replica
selected on both axes applies a stale command on the next round.

### E4. `metadynamics_gpu` bias force has an inverted sign and an extra division by Rg — **[未验] unverified**

`:370-376`: `f_on_p = (dV_drg / (rg + 1e-8)) * drg_dx` where `drg_dx` already carries `1/rg`.

### E5. `metadynamics_sampler`'s BSJ bias term returns a gradient where the Rg term returns a force — **[未验] unverified**

`:132-137` versus `:142-144`, both feeding the same `CustomExternalForce` at `:355`.

### E6. `metadynamics_sampler` bias force is written in kJ/mol/A into a nm-coordinate force — **[未验] unverified**

`:71` documents kJ/mol/A; `:396` converts positions to A before the call; `:384-389` writes the
result into `CustomExternalForce`, whose coordinates are nm. A factor of 10.

### E7. The "well-tempered" hill height is not the well-tempered relation — **[未验] unverified**

`metadynamics_sampler.py:400-404` and `metadynamics_gpu.py:270-274` use
`height = h0 / (1 + n_visits / bias_factor)`, which drops the `k_B T (gamma - 1)` energy
denominator; the visit count also ignores the Rg coordinate.

### E8. `metadynamics_sampler`'s best-energy tracker is contaminated by the linear bias term — **[未验] unverified**

`:393-395` reads the total potential energy including `fx*x + fy*y + fz*z`, a term that depends
on the coordinate origin, then uses it at `:413-415` to select the best frame.

### E9. `rest2_sampler` swaps every neighbouring pair each round instead of alternating parity — **[未验] unverified**

`:156-176` decides all adjacent pairs at once and then greedily de-duplicates, so whether
(1,2) is accepted depends on whether (0,1) was.

### E10. `fivebead_folding`'s B1-B1 stacking attraction is unbounded — **[未验] unverified**

`fivebead_folding.py:143-144`: a linear `-eps*(r - sigma)/sigma` attractive arm grows without
bound as `r` increases; it is not a well.

### E11. `spline_smooth_dihedral` computes a spline and discards it — **[未验] unverified**

`segmented_vfold3d.py:618-635`: `smoothed_dihedrals` is computed at `:619-620` and never used;
`:633` substitutes `np.random.randn(n, 3) * 0.1` with no seed.

### E13. `_bpp_pairs_fallback` is off by one on the ViennaRNA bpp matrix — **[未验] unverified**

`segmented_vfold3d.py:1231-1242` reads `M[i, j]` from the 1-based `(L+1, L+1)` matrix without
stripping row/column 0. This is the same shape confusion as B2, in a different function.

### E14. `_refine_with_context`'s `next` branch fades in the wrong direction — **[未验] unverified**

`segmented_vfold3d.py:1670-1681`: `fade = 1.0 - i / ctx_len` peaks at the inner end, while the
boundary is at `refined[-1]`.

---

## Corrections this audit made to its own inputs

These are places where checking the claim changed it. They matter as much as the findings.

1. **The REMD results in `results/` were not produced by the broken sampler.** The logs come
   from `scripts/ibi_remd_residual.py` (its prints at `:147`, `:150`, `:434` match verbatim),
   which uses the correct exchange helpers in `torch_cgsim.py:2355`/`:2369`. An earlier reading
   of this audit attributed them to `rest2_remd_2d.py` on the strength of a `note:` line that
   only compares default grids.

2. **The pipeline does not use the OpenMM REMD path by default.** `isrnaclong.py:1496-1501`
   prefers `torch_gpu_refine` and `:2135`/`:2166` prefers `BatchedREMD2D`. C1, C2, C6 and C7 sit
   on branches that are only reached when the torch import fails.

3. **C4 is already documented in the source**, including its measured kinetic-temperature
   effect. It is an unexecuted decision, not an oversight.

4. **C6's framing was too strong.** Trajectory diversity is reduced, not eliminated —
   `bsj_k_scale` still varies per worker.

5. **D4 is a repository-wide convention split, not a typo.**

6. **E1 is false for the code path that produced the published acceptance numbers.**

## Errors made in the course of this audit

Recorded because the audit is only worth as much as its error rate.

1. A first pass told the user that existing REMD results would have to be redone. That was
   stated before checking which code path produced them. Retracted.

2. The producer of `results/remd_*.log` was misidentified as `rest2_remd_2d.py`, from a
   comparison line in the log rather than from the code that emitted it. Corrected in
   "Corrections" item 1.

3. The Level 0 pair-vote change described in Group F below was made on a premise that B2
   falsifies. It has since been re-specified.

---

## Group F — changes made during this audit

### F1. Level 0 now admits the caller's secondary structure to the pair vote — verified

Before: `secondary_structure` was passed into `isrnaclong_pipeline`, written to
`test_2013nt_ss.txt` upstream, and **never read anywhere in Level 0** — the stage recomputed its
own sources and voted among those.

Now: `_fuse_pair_sources(pf_high_set, mfe_set, divide_set, input_ss_set)` is a module-level
function (`isrnaclong.py`, immediately above `LongPipelineResult`) that takes a fourth source.
Two properties are pinned by `tests/test_level0_pair_vote.py`:

- a pair named only by the input does not become hard (one vote is not enough);
- the input is dropped entirely when it adds nothing beyond `mfe_set`, and otherwise votes on
  all of its pairs.

The second rule went through two versions, and the reason is worth recording. It was first
written as a per-pair subtraction (`input - mfe_set`), chosen at a point when B2 made the input
an exact duplicate of `mfe_set`: subtracting was then equivalent to dropping it. Fixing B2
changed that — the input is now a **linear** fold while `mfe_set` is circular, so a pair both
models find is two independent models agreeing, which is precisely what the `>=2` threshold is
for, and subtracting threw that evidence away.

The rule as it stands now covers both routes: the fallback route hands in the identical fold and
the input contributes nothing; the working route hands in a different model and it votes.

Net behaviour: on the fallback route, unchanged from before. On the working route the input is a
live fourth source for the first time. Diagnostics are written to `_plots/00_level0_diag.json`
as `n_input_ss` (effective), `n_input_ss_raw` (as supplied), `n_input_ss_hard`, and
`n_input_ss_single`.

Verification status: `_fuse_pair_sources` was driven directly through 12 assertions; the
`_parse_dotbracket_strict` parser was exercised on well-formed, unbalanced and all-dot input.
**The full Level 0 path was not executed end to end** — `sequence.txt`, `test_2013nt_ss.txt`
and `output_2013nt/` are not in the repository.

`tests/` runs in the pipeline's own environment, not on the interpreters named above.
`C:\ana\envs\circrna3d` (Python 3.11, torch 2.13.0+cpu, OpenMM 8.5.2, ViennaRNA 2.7.2) has
pytest 9.1.1: **the full suite passes, 145 tests in 24 s**, including the new
`test_level0_pair_vote.py`. An earlier draft of this document said pytest was absent on "all
three interpreters" without ever looking in the conda environments; that was wrong. Note that
`torusfold` is *also* installed in that environment, from a different checkout
(`C:\Users\...\TorusFold-scheme2-rl\src\torusfold`), so a script that imports it without putting
this repository's `src/` first will silently test the other tree. The test files do insert it;
anything else should be checked.

The obsolete claim, kept only so it is not repeated:
`tests/` cannot currently be run: pytest is not installed on any of the three interpreters
present (3.14 / 3.13 / 3.12).

---

## Group G — external quality scoring (lociPARSE)

Not an audit finding; added after the audit, at the maintainer's request, following the
statistical-potential discussion. Recorded here because it changes what the pipeline reports.

### G1. `loci_quality.py` — lociPARSE as a superposition-free quality score

`src/torusfold/scheme2/loci_quality.py` (new) wraps lociPARSE (Bhattacharya Lab, J. Chem.
Inf. Model. 2024, 64(22):8655-8664), which predicts per-nucleotide lDDT (`pNuL`) and its mean
(`pMoL`) **without a reference structure** — the property that makes it usable on the 2013-nt
construct, where no experimental structure exists.

Measured here, not assumed:

| | result |
| --- | --- |
| atoms it uses | P, C4', and the glycosidic N (N9 purines / N1 pyrimidines) — `feature_generation.py:119-123` |
| 2OIU, all 3054 atoms | pMoL **0.80** |
| 2OIU reduced to those 3 atoms/residue (426 atoms) | pMoL **0.80** — unchanged |
| cost, one CPU core | 142 nt in 0.09 s; 568 nt in 0.26 s (linear) |
| 1L2X crystal (27 nt) | 0.64 |
| this pipeline's CG output for the same 27-nt sequence | 0.52 |

The atom result is the important one: deleting every non-P/C4'/N atom leaves the score
untouched, so a **CG structure is in-distribution for this model**, not a degraded input. The
27-nt comparison is a direction, not a validation — n = 1, and the paper's 30-target benchmark
is where the discrimination claim comes from.

**Caveat that decides how it may be used: pMoL drifts with length.** Three crystals score 0.64
(27 nt), 0.75 (69 nt), 0.80 (142 nt). Compare candidates of the same length, or one structure
before and after a change. A cross-chunk ranking compares different lengths and is not
supported.

### G2. Wired at the end, not per chunk — and the reason is a hard blocker

`isrnaclong.py` scores the finished structure and writes the result to the summary key
`final.lociparse_pMoL` (next to the `rsrnasp1` placeholder that was already reserved for
exactly this kind of score), plus `_plots/lociparse_pNuL.json` for the per-residue array. It
degrades to `None` when lociPARSE is not importable, so the pipeline is unaffected by its
absence.

**It does not run per chunk, and cannot without a change elsewhere.** Every coordinate that
reaches the assembly is P-only: `rhofold_wrapper._write_pdb` (`:157-168`) writes one P atom per
residue, and `coords_rh` / `coords_tr_p` at the fusion point (`ensemble_predictor.py:323-379`)
are both `(L, 3)` P arrays. C4' and N are discarded inside the wrappers — exactly the two atoms
lociPARSE needs.

So the intended Level-1 use (ranking candidates within a chunk, where the length caveat cancels)
requires the predictor wrappers to retain C4'/N first. Related: **`n_candidates` is a dead
parameter** — `segmented_vfold3d.py:1429` declares it and `:1467` documents ">1 picks the best",
but the function body never reads it, so there are no candidates to rank either. Same class as
C10.

### G3. Not verified end to end

The wiring is verified statically (definition order, compilation) and the scoring logic is
verified functionally (both entry points, plus graceful degradation with the dependency absent).
**The full pipeline was not executed**: `sequence.txt` and the upstream predictors are not
available here, so the score has never been observed appearing in a real `pipeline_summary.json`.

lociPARSE is GPL-3.0. `loci_quality.py` imports it, it does not vendor it. Settle that before
shipping anything that bundles it. It is not installable on this machine's interpreters — its
`setup.py` pins `numpy==1.22.3` / `torch==1.12.0`, which predate Python 3.11, but the package
imports only torch, numpy and tqdm, so it is loaded from a source checkout via
`TORUSFOLD_LOCIPARSE`.

## Open questions

1. Which produced artifacts came out of `BatchedREMD2D` (A3)?
2. Does deployment actually go through `mixed_energy` (A2)?
3. What should `WC_TARGET_DIST` be, per call site (D4)? This needs a measurement, not a vote.
4. `scripts/ibi_sample.py` does not exist, but the task list records "write ibi_sample.py" as
   complete. `scripts/ibi_core.py` holds the sampling core, added by commit `bbd9989`.
