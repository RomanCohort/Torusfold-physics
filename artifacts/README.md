# `artifacts/` — the delivered outputs, in the repository

This directory exists because the outputs were not in the repository, and the
audit note at `docs/archive/pipeline_audit_2026-09-13.md:1022` records the consequence in
the team's own words: **the full Level 0 path was never executed end to end**,
because `sequence.txt`, `test_2013nt_ss.txt` and `output_2013nt/` are not in the
repository. A reader could not start the demo, and could not obtain its output
either. Whatever the reason for holding them back, the result was that the only
thing a third party could check was a still image.

## What is here

| Path | What it is | Size |
|---|---|---|
| `2013nt/isrnaclong_final.pdb` | The delivered 2,013 nt model: 42,831 atoms, Level 4.9 PPR-repaired, the same bytes the shipped viewer renders | 3.5 MB |
| `2013nt/sequence.txt` | The demo target, read out of that structure | 2 KB |
| `2013nt/provenance.json` | Hashes and the decode path, so the files above can be checked rather than trusted | <1 KB |
| `2013nt/quality.json` | Every number the viewer's stat panel displays, with whether a third party can re-derive it and, when they cannot, what the blocker is | 7 KB |
| `2oiu/2OIU.pdb` | The one experimentally resolved circRNA structure (chain P, ≈100 nt) — the input to the force-field check | 84 KB |
| `reuse_demo/2oiu/` | An executable reuse record: 2OIU through the shipped CG stage, with the command, the six environment settings, the product, every measured number and a SHA-256 of every file — plus a verifier that needs no GPU | 156 KB |
| `reuse_demo/2oiu_repair/` | The same run with `TORUSFOLD_HBOND_REPAIR=1`: the optional Watson-Crick edge repair, its 39 rotated bases, and what it does and does not fix | 156 KB |

## How these files came to be here, stated plainly

They were **decoded, not re-run**. `docs/circrna_3d_viewer.html` is self-contained,
so the structure it displays is inlined in it as a gzip+base64 payload;
`scripts/extract_viewer_payload.py` inverts that packing, and re-running it must
reproduce these files byte for byte. That is the strongest form of provenance
available for them: the run that produced the structure was an earlier build that
is no longer in this repository, so the viewer's own payload is the only surviving
copy of the delivered model. `artifacts/2013nt/provenance.json` carries the
hashes; `scripts/verify_headline.py` re-checks them on every run.

The sequence in `2013nt/sequence.txt` is **read out of the structure**, because
nothing else in the repository has it. `sequence.txt` at the root is git-ignored
as a "private sequence" — which is worth saying out loud, since the structure
that encodes it has been in the public repository the whole time as the viewer
payload. The exclusion cost the ability to run the demo and bought no secrecy.

## What you can do with it, in the order the effort goes up

```bash
# 1. Nothing to install. Every number the viewer shows you is re-derived from
#    these committed files, with numpy, in seconds. It says which ones do not
#    come back and why.
python scripts/verify_headline.py

# 2. Open the structure. Any viewer; or the shipped one, which needs no network:
#       docs/circrna_3d_viewer.html
#    or point Mol* / PyMOL / ChimeraX at artifacts/2013nt/isrnaclong_final.pdb

# 3. The executed reuse record: check it, then reproduce it. Verifying re-derives
#    every number and hash from the committed files with numpy alone (~10 s, no GPU);
#    reproducing the run itself took 21 s on the GPU named in record.json, and
#    re-running it three times produced byte-identical products.
python scripts/record_2oiu_reuse.py --verify-only
python scripts/record_2oiu_reuse.py --spec results/plan_c/_2oiu_input.json   # the run

# 4. The one experimental cross-check in this repository: start from the crystal
#    structure and relax it, then compare. ~17 min on CPU, and it needs OpenMM and
#    ViennaRNA. The input is committed now, so it no longer dies on a missing file.
python scripts/benchmark_2oiu.py

# 5. The full pipeline. Still expensive (30-60 GB, hours to days) and still needs
#    the external predictors -- see docs/REPRODUCTION_RESOURCES.md.
python run_2013nt.py     # falls back to artifacts/2013nt/sequence.txt
```

## What is deliberately NOT here

Model weights and predictor checkouts (RhoFold+, trRosettaRNA2, RNAbpFlow) stay
external — they are gigabytes of third-party code with their own licences, and
`docs/DEPLOY_EXTERNAL.md` covers them. The deposited-structure database
(`_cgdata/`, 1.09 GB) stays out too: it is reference data for the statistical
potentials, not a deliverable, and it is what `git filter-repo` had to be run
three times to remove from history once already. The commit that adds this
directory is ~3.6 MB, which is worth checking against the repository size before
adding anything else here.
