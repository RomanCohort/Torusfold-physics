# Model bundle

The source repository contains no third-party model checkpoints.  The large
predictor checkouts and weights are distributed separately, for example through
the team's cloud drive.  Extract them into one directory with this layout:

```
torusfold-models/
  RhoFold/
  RNAbpFlow/
  trRNA2/
  structRFM/
  DivideFold/
  isRNAcirc/          # Windows only
```

Set `TORUSFOLD_MODELS` to that directory.  The Docker setup mounts it at
`/models`; native Windows runs can use:

```bat
set TORUSFOLD_MODELS=C:\torusfold-models
set RHOFOLD_ROOT=%TORUSFOLD_MODELS%\RhoFold
set RNABPFLOW_ROOT=%TORUSFOLD_MODELS%\RNAbpFlow
set TRRNA2_ROOT=%TORUSFOLD_MODELS%\trRNA2
set TF_STRUCTRFM_MODEL=%TORUSFOLD_MODELS%\structRFM
set TF_DIVIDEFOLD_ROOT=%TORUSFOLD_MODELS%\DivideFold
```

Then validate the bundle before starting a prediction:

```bash
python tools/check_model_assets.py --root "C:\torusfold-models"
```

## Cloud-drive rules

- Upload directories, not a checkout of this repository.
- Do not upload `.env.local`, `output_web/`, `__pycache__/`, `.git/`, Hugging
  Face caches, or temporary MSA/PDB files.
- Keep each upstream `LICENSE` and `README` beside its code.
- Prefer one archive per asset: `RhoFold.tar.zst`, `RNAbpFlow.tar.zst`,
  `trRNA2.tar.zst`, `structRFM.tar.zst`, `DivideFold.tar.zst`.
- Publish a `SHA256SUMS` file beside the archives.  On Linux:

  ```bash
  sha256sum *.tar.zst > SHA256SUMS
  sha256sum -c SHA256SUMS
  ```

  On Windows:

  ```powershell
  Get-FileHash .\*.tar.zst -Algorithm SHA256
  ```

- A cloud-drive folder cannot be mounted as a local Docker volume.  Sync or
  download it to a local disk first; network filesystems make the MD stages
  slow and can break checkpoint writes.

## Known source locations on the development machine

These are references for preparing the bundle, not hard-coded runtime paths:

| Asset | Source |
|---|---|
| RhoFold+ | `C:\Users\颜子壹\deploy\IGEM集成方案\tools\RhoFold` |
| trRNA2 | `C:\trRNA2` |
| structRFM | `C:\baidunetdiskdownload\TorusFold-tools\structRFM` |
| DivideFold | `C:\Users\颜子壹\DivideFold-main` |
| isRNAcirc | `C:\baidunetdiskdownload\torusfold-hybrid\_isrnacirc_ascii` |

The development machine records RNAbpFlow at
`C:\baidunetdiskdownload\TorusFold-tools\RNAbpFlow`.  Its ACL blocks ordinary
enumeration on this account, so an elevated copy was made to
`C:\torusfold-models\RNAbpFlow`.  The checkpoint is
`RNAbpFlow/checkpoint/RNA3DB.ckpt` (203,025,869 bytes).  If it has to be
fetched again, it is Zenodo record `18305861`.

## Archive checksums

The manifest can record SHA-256 values per weight file.  Run the checker with
`--sha256` after editing those values.  This is intentionally opt-in because
hashing several gigabytes on every startup is wasteful.
