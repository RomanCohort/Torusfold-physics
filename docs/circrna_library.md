# circRNA structure library

This is an isolated, provenance-first local data layer for RNA structures. It
keeps experimental structures, experimental constraints, and physics-generated
hypotheses distinguishable. It does not alter or automatically feed the
TorusFold prediction pipeline.

## Layout

```text
data/circRNA_library/
  raw/          immutable source PDB/mmCIF bytes, addressed by SHA-256
  normalized/   canonical-ish mmCIF written by Gemmi
  derived/      entry.json, sequence.fasta, residues.jsonl, mapping.json, qc.json
  manifests/    manifest.jsonl and its metadata/rejection logs
```

Every RNA chain and selected model gets an `entry_id`. The source hash and
label/auth chain and residue identifiers are retained. Modified residues keep
their original component name; uncertain base mappings are represented as
`N`, not guessed.

## Install and import

The core schema, QC, and manifest code uses only the Python standard library.
PDB/mmCIF ingestion additionally needs Gemmi:

```text
python -m pip install -r requirements-circrna-library.txt
```

From a source checkout, use `PYTHONPATH=src` (on PowerShell, set `$env:PYTHONPATH='src'`).

```python
from torusfold.circrna_library.ingest_gemmi import ingest_file, write_artifacts

result = ingest_file('example.cif')
write_artifacts(result, 'data/circRNA_library')
```

Importing `schema`, `circular_qc`, or `build_manifest` does not import Torch,
OpenMM, ViennaRNA, or `torusfold.scheme2`. Gemmi is imported only when
`ingest_file` is called.

## Circularity and BSJ semantics

`CircularityEvidence` is intentionally independent from QC. Evidence tiers
are curated rather than inferred from geometry. A BSJ boundary uses a
zero-based index `i`: its endpoints are `(i - 1) % L` and `i`; index `0` thus
means the last-to-first boundary. QC never assumes the first and last records
form a BSJ when no explicit BSJ index is supplied. Missing endpoint coordinates
produce `not_assessed`, not a failed circularity claim.

```python
from torusfold.circrna_library.circular_qc import assess
report = assess(snapshot, mapping, circularity)
```

The report separates internal backbone gaps from BSJ closure and records
representative atom fallback (`P`, `C4'`, then `C1'`) and thresholds.

## Manifest

After ingestion, build a deterministic manifest:

```text
PYTHONPATH=src python -m torusfold.circrna_library.build_manifest \
  data/circRNA_library data/circRNA_library/manifests/manifest.jsonl
```

Existing manifests are not overwritten unless `--replace` is supplied.
Artifact paths are root-relative and hash/size checked. Curation records are
JSONL objects bound to both `entry_id` and `raw_sha256`; conflicting decisions
are rejected instead of resolved by last-write-wins.

The initial experimental tier is conservative: an experimental RNA structure
without verified circularity remains a reference candidate, not a circular
RNA ground truth. Physics-generated structures should use `record_type`
`physics_generated` and remain separate from experimental benchmark entries.
