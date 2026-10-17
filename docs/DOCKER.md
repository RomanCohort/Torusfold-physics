# Containerised CPU/Web layer

## Scope

`docker/Dockerfile.cpu` and `compose.yaml` provide the reproducible Linux
CPU/Web layer:

- Python 3.11
- OpenMM 8.5.2
- ViennaRNA 2.7.2
- NumPy/SciPy, PyTorch CPU, transformers and the analysis extras
- `serve.py`, the SSE log stream, the Predict API and the 3D viewer

They intentionally do **not** include:

- `CG_to_allatom.exe` or its Windows DLLs
- the host's AMD ROCm GPU stack
- an unlicensed PyRosetta installation
- model weights baked into an image layer

Those stay on the host for now.  This keeps the current Windows/ROCm path
working while making the CPU path reproducible on another machine.

## Model bundle

Prepare one local bundle:

```
C:\torusfold-models\
  RhoFold\
  RNAbpFlow\
  trRNA2\
  structRFM\
  DivideFold\
  isRNAcirc\       # Windows-only, not used by the Linux image
```

See `models/README.md` and `models/manifest.json`.  Validate it before launch:

```powershell
python tools/check_model_assets.py --root C:\torusfold-models --sha256
```

The cloud-drive copy has to be synced to local storage.  Do not point a bind
mount at a cloud-drive virtual folder.

## Launch

```powershell
Copy-Item .env.docker.example .env.docker
# edit TORUSFOLD_MODELS_HOST if the bundle is not C:/torusfold-models
docker compose --env-file .env.docker up --build
```

Open <http://127.0.0.1:8877/>.

The host port is bound to `127.0.0.1`, not `0.0.0.0`.  Keep it that way:
the server contains unauthenticated setup/upload/feedback endpoints.

## Memory

The full CPU pipeline has a measured peak around 30 GB.  Leave the container
limit at 48 GB on this 64 GB machine:

```dotenv
TORUSFOLD_MEMORY=48g
```

Docker Desktop's WSL2 VM also needs enough memory.  If `.wslconfig` does not
exist, create it at `C:\Users\<user>\.wslconfig`:

```ini
[wsl2]
memory=48GB
swap=8GB
```

Restart Docker Desktop after changing it.  Do not run the full 2,013 nt job
with Docker Desktop's low default memory limit.

## GPU and Windows-only stages

The CPU image is not a replacement for the reference ROCm run.  For now:

- use the native Windows `comfyui` environment for the ROCm/GPU path;
- use the native Windows isRNAcirc binary for CG-to-all-atom;
- keep PyRosetta on a licensed host or WSL installation.

When the project moves to an NVIDIA cloud machine, add a separate CUDA image
and keep the same model-bundle mount.  Do not mix the ROCm and CUDA wheel sets
in one image.

## Secrets

`.env.local` is machine-specific and contains the feedback SMTP credentials.
It is excluded by `.dockerignore`.  Pass secrets at runtime through the
orchestrator's secret mechanism, not through `ENV` or `ARG` in the Dockerfile.
