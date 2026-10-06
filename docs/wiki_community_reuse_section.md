# Community-reuse record — paste-ready section for the Software Wiki

This file exists so the record in `docs/community_reuse_record.md` can be lifted into the team's Software
Wiki without rewriting it. It is written in that page's voice (third person, judge-facing, no code fences
beyond one short command) and is meant to sit after **§7.3 "Getting started as another team"** in
`Software-Wiki-CircuForge-TorusFold-English.md`. Renumber the section to whatever the page needs; nothing
in the surrounding text refers to it by number.

Numbers quoted here are the ones in `artifacts/reuse_demo/2oiu/record.json`, produced on the team's
reference machine on 2026-10-06.

---

### 7.4 A reuse record another team can check

Section 7.3 describes how another team can start. To make that concrete, we published one prediction as a
record that can be checked without our machine, our GPU, or our internal data. The record is the
experimental circular RNA **2OIU** — the only circular RNA in our repository with a determined structure,
which is what makes it the one system whose output can be compared against something we did not compute.

The record holds the deposited P trace used as the starting point, the mature sequence and its secondary
structure, the exact command with the six environment settings the run used, the all-atom product, and a
JSON file carrying every measured number together with a SHA-256 for every input and output. One command
re-derives all of it from the committed files:

`python scripts/record_2oiu_reuse.py --verify-only`

That check rebuilds the all-atom product from the recorded bead frame and compares it byte for byte,
re-measures the geometry, and re-hashes every file. It needs NumPy alone — no GPU, no PyTorch, no
ViennaRNA and no external predictor, and it passes from a clean checkout of the repository. What it
supports is two claims: the prediction reproduces the experimental structure to 1.39 Å along the backbone
and to within 2 % of its base-to-base distance, and every number quoted below is re-derivable from files in
the repository rather than from our machine.

A reader who does have a GPU can also repeat the prediction itself: the record carries the force-field
tables the run used, and a clean checkout of the repository reproduces the recorded product byte for byte.
The recorded run was repeated three times and gave byte-identical products, so its seed reproduces the run
and not merely the protocol. The record additionally fingerprints every source file that decided its output,
so any later divergence can be traced to a different source tree instead of guessed at.

| Quantity | 2OIU deposit | Predicted product |
|---|---|---|
| P-trace deviation from the deposit | — (reference) | 1.39 Å |
| Base-to-base distance (bead map) | 5.31 Å | 5.20 Å |
| Base-to-base rise (bead map) | +3.19 Å | +3.33 Å |
| Cosine between neighbouring base normals | 0.901 | 0.899 |
| Helical steps judged stacked | 100 % | 58.3 % |
| Watson–Crick key contacts within 3.6 Å | 12 / 12 | 1 / 12 |

The last two rows need one more comparison to be read correctly, and the record carries it. Handed the
deposit's own bead frame, the shipped reconstruction keeps 91.7 % of the stacking and 5 of the 12 contacts;
handed the deposit's own P trace — the best any P-trace-based tool can do, with zero sampling error — it
keeps 83.3 % and 2 of 12. The model pairs bases with a distance restraint and has no orientation term, so
the pairing criterion is already down to 2 of 12 before the sampler runs, and the sampler costs one contact
on top of that. Stacking decomposes the other way: 100 % → 83.3 % is the reconstruction, and 83.3 % → 58.3 %
is the 1.39 Å of drift.

The record also states what it is not, because a reuse record that only lists successes is not checkable.
It is a refinement that starts from the deposited trace, not a prediction from sequence alone. The base
pairing stays the weak point: the optional Watson–Crick edge repair (a rotation about the glycosidic bond,
shipped off by default) brings the contacts from 1 to 3 of 12 while leaving the stacking essentially where
it was, and the pairs that remain unsatisfied are a positional gap the current model does not target at all. And
2OIU is the only circular RNA we have run through the force-field stage; the 2,013 nt example is published
as a decoded artifact with hashes, and its provenance file records that it was not re-run, so a reader can
verify it but cannot reproduce it.

Two boundaries are recorded as well. The compact circular starting conformation the pipeline generates for
a new sequence cannot be written for chains longer than about 1,060 nt at the shipped bond length, because
the PDB coordinate columns overflow and the fixed-column loader cannot recover merged fields — longer
chains need a real starting structure. And the recorded run is a refinement inside a 21-second protocol,
so its numbers describe that protocol rather than a converged ensemble.
