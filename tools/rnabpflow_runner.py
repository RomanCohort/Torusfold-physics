"""Run RNAbpFlow's upstream inference on this machine's ROCm GPU.

WHY THIS IS A SEPARATE SCRIPT. `ensemble_predictor._run_rnabpflow` invokes
`<RNABPFLOW_ROOT>/inference_rocm.py`. That file is a team-held adaptation and it is not
on this machine and has no upstream URL; what upstream ships is `inference.py`, which
takes no command-line arguments at all and is driven by `configs/inference.yaml` plus
the current directory. This script is the adapter between those two facts, so the
pipeline can use the checkpoint that IS here.

It runs as a subprocess with `cwd` set to RNAbpFlow's scratch tree, which is what
upstream's `run_inference()` expects:

    <cwd>/configs/inference.yaml    written here
    <cwd>/Inputs/list.txt           written by the caller
    <cwd>/Inputs/<name>/<name>.fasta + map1.npy map2.npy map3.npy
    <cwd>/Predictions/<name>/Sample_0.pdb      <- the output the caller reads

Two things have to happen before `inference` is imported, and both are why this cannot
simply be `python inference.py`:

  * `torusfold.torch_compat.install()` -- without it `import pytorch_lightning` fails,
    because pytorch_lightning pulls torchmetrics, which pulls transformers, which drags
    in torch DTensor, which needs a `torch._C._distributed_c10d` this ROCm torch build
    does not have. See that module for the full chain.
  * `RNABPFLOW_ROOT` goes on `sys.path`, because `inference.py` imports `src.models...`.

The shims are installed at IMPORT time, not inside `main()`, because `inference.py` uses
`mp.spawn`: each worker re-imports this file, and a shim installed only under the
`__main__` guard would be missing in the workers.

Exit code 0 means the PDB was written. Anything else is a failure the caller should
report rather than interpret.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback

# Before anything that might reach torchmetrics/transformers. See the module docstring.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep + "src")
try:
    from torusfold import torch_compat
except ImportError:                                          # pragma: no cover
    torch_compat = None

_SHIM_STATE = torch_compat.install() if torch_compat else {}


def _write_config(cwd: str, name: str, checkpoint: str, num_gpus: int,
                  samples: int, timesteps: int) -> str:
    """Write the config upstream reads from a FIXED relative path under `cwd`."""
    import yaml

    cfg_dir = os.path.join(cwd, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg = {
        "defaults": ["_self_"],
        "hydra": {"run": {"dir": "logs"}},
        "inference": {
            "ckpt_path": os.path.abspath(checkpoint),
            "input_dir": os.path.join(cwd, "Inputs"),
            "output_dir": os.path.join(cwd, "Predictions"),
            "name": name,
            "num_gpus": num_gpus,
            "interpolant": {"sampling": {"num_timesteps": timesteps}},
            "samples": {"samples_per_sequence": samples},
            "output_format": "pdb",
        },
    }
    path = os.path.join(cfg_dir, "inference.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True,
                    help="target id; must match Inputs/<name>/ and is the directory "
                         "the samples land in")
    ap.add_argument("--checkpoint", required=True, help="path to RNA3DB.ckpt")
    ap.add_argument("--device", default="cuda",
                    help="upstream's sampler hardcodes torch.device('cuda:N'), so only "
                         "a cuda device is accepted")
    ap.add_argument("--root", default=os.environ.get("RNABPFLOW_ROOT", ""),
                    help="the RNAbpFlow checkout (default: $RNABPFLOW_ROOT)")
    ap.add_argument("--samples", type=int, default=None,
                    help="override samples per sequence (default: keep list.txt's)")
    ap.add_argument("--timesteps", type=int, default=50)
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root or "")
    if not root or not os.path.isdir(root):
        print("RNABPFLOW_ROOT is not set to a directory: %r" % args.root, file=sys.stderr)
        return 2
    if not os.path.isfile(args.checkpoint):
        print("checkpoint not found: %s" % args.checkpoint, file=sys.stderr)
        return 2
    if not args.device.startswith("cuda"):
        print("--device %r is not supported: upstream's Sampler.sample_GPU builds "
              "torch.device('cuda:N') unconditionally, so a CPU request could only "
              "fail later with a worse message." % args.device, file=sys.stderr)
        return 2

    import torch
    if not torch.cuda.is_available():
        print("no CUDA/ROCm device is available to torch", file=sys.stderr)
        return 2

    cwd = os.getcwd()
    if not os.path.isdir(os.path.join(cwd, "Inputs", args.name)):
        print("expected %s to exist -- the caller prepares Inputs/<name>/"
              % os.path.join(cwd, "Inputs", args.name), file=sys.stderr)
        return 2

    samples = args.samples
    if samples is None:
        # Upstream reads per-target counts from Inputs/list.txt; reproduce that here so
        # the override is only applied when explicitly asked for.
        samples = 1
        list_path = os.path.join(cwd, "Inputs", "list.txt")
        if os.path.isfile(list_path):
            for line in open(list_path, encoding="utf-8"):
                tok = line.split()
                if len(tok) == 2 and tok[0] == args.name:
                    samples = int(tok[1])
                    break

    if torch_compat:
        print(torch_compat.describe(_SHIM_STATE), flush=True)
    cfg_path = _write_config(cwd, args.name, args.checkpoint, 1, samples, args.timesteps)
    print("config: %s" % cfg_path, flush=True)
    print("device: %s | samples: %d | timesteps: %d"
          % (torch.cuda.get_device_name(0), samples, args.timesteps), flush=True)

    sys.path.insert(0, root)
    os.chdir(cwd)
    try:
        inference = importlib.import_module("inference")
        inference.run_inference()
    except BaseException:                                    # noqa: BLE001
        traceback.print_exc()
        return 1

    pdb = os.path.join(cwd, "Predictions", args.name, "Sample_0.pdb")
    if not os.path.isfile(pdb):
        print("inference returned without writing %s" % pdb, file=sys.stderr)
        return 1
    print("wrote %s (%d bytes)" % (pdb, os.path.getsize(pdb)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
