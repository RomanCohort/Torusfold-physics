"""Portable trRosettaRNA2 inference runner.

The upstream checkout is external, so no machine path is compiled into this
file.  Point ``TRRNA2_ROOT`` at a directory whose basename is ``trRNA2`` and
``TRRNA2_MODEL_DIR`` at the directory containing ``config/``, ``models/``,
``config_ss/`` and ``models_ss/``.

The wrapper in ``src/torusfold/scheme2/trrna2_wrapper.py`` expects two files:

    <out>_coords.npy   (L, 3) C1' coordinates
    <out>_dist.npy     (L, L) distance expectation

This runner also writes ``<out>_meta.json`` so confidence is the model's mean
pLDDT rather than a heuristic derived from the distance matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _query_sequence(msa_path: Path) -> str:
    """Return the first sequence in a FASTA/A3M file without gap characters."""
    sequence = []
    seen_header = False
    with msa_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seen_header and sequence:
                    break
                seen_header = True
                continue
            if seen_header:
                sequence.append(line)
    if not sequence:
        raise ValueError("no sequence found in %s" % msa_path)
    return "".join(sequence).replace("-", "").replace(".", "").upper().replace("T", "U")


def _device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _distance_expectation(distance_logits, bins, torch):
    """Collapse the C1' distance distribution to an (L, L) matrix."""
    if distance_logits.ndim == 4:
        distance_logits = distance_logits.squeeze(0)
    if distance_logits.shape[-1] == bins.numel():
        return (distance_logits * bins).sum(dim=-1)
    # Some checkpoints emit a scalar distance rather than bin logits.
    if distance_logits.ndim == 2:
        return distance_logits
    raise ValueError("unexpected C1' distance shape: %s" % (tuple(distance_logits.shape),))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msa", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--msa-rows", type=int, default=500)
    parser.add_argument("--cpu-threads", type=int, default=5)
    parser.add_argument(
        "--device",
        default=os.environ.get("TRRNA2_DEVICE", "cpu"),
        help="auto, cpu, cuda, or cuda:N (default: TRRNA2_DEVICE or cpu)",
    )
    args = parser.parse_args(argv)

    root_raw = os.environ.get("TRRNA2_ROOT", "").strip()
    if not root_raw:
        parser.error("TRRNA2_ROOT is not set")
    root = Path(root_raw).expanduser().resolve()
    if root.name.lower() != "trrna2":
        parser.error(
            "TRRNA2_ROOT must point at the directory named trRNA2 (got %s)" % root
        )
    if not (root / "__init__.py").is_file():
        parser.error("TRRNA2_ROOT does not contain __init__.py: %s" % root)

    model_dir_raw = os.environ.get("TRRNA2_MODEL_DIR", "").strip()
    model_dir = Path(model_dir_raw).expanduser().resolve() if model_dir_raw else root / "models" / "params"
    required = [
        model_dir / "config" / "model_1.json",
        model_dir / "models" / "model_1.pth.tar",
        model_dir / "config_ss" / "model_1_finetune.json",
        model_dir / "models_ss" / "model_1_finetune.pth.tar",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("trRNA2 model files missing: %s" % ", ".join(missing))

    # The external checkout is a top-level `trRNA2` package.
    sys.path.insert(0, str(root.parent))

    import numpy as np
    import torch
    import trRNA2.utils as tr_utils
    from trRNA2.model_3d import Folding
    from trRNA2.model_ss import SSpredictor
    from trRNA2.utils import parse_a3m

    torch.set_num_threads(max(1, args.cpu_threads))
    device = _device(torch, args.device)
    config = _load_json(model_dir / "config" / "model_1.json")
    raw_seq = _query_sequence(Path(args.msa))
    length = len(raw_seq)
    if length == 0:
        parser.error("empty query sequence")

    msa_np = parse_a3m(args.msa, limit=args.msa_rows)
    if msa_np.shape[0] == 0 or msa_np.shape[1] != length:
        parser.error(
            "MSA shape %s does not match sequence length %d"
            % (msa_np.shape, length)
        )
    msa = torch.from_numpy(msa_np).to(device)[None]
    res_id = torch.arange(length, device=device).view(1, length)

    ss_models = []
    for index in (1, 2, 3):
        name = "model_%d_finetune" % index
        config_ss = _load_json(model_dir / "config_ss" / (name + ".json"))
        ss_model = SSpredictor(
            dim_2d=config_ss["dim_pair"],
            layers_2d=config_ss["RNAformer"]["n_block"],
            config=config_ss,
            device=str(device),
        ).to(device)
        state = torch.load(
            model_dir / "models_ss" / (name + ".pth.tar"),
            map_location=device,
            weights_only=True,
        )
        ss_model.load_state_dict(state)
        ss_model.eval()
        ss_models.append(ss_model)

    fold_state = torch.load(
        model_dir / "models" / "model_1.pth.tar",
        map_location=device,
        weights_only=True,
    )
    # Upstream's predict.py performs the same compatibility adjustment for the
    # released checkpoint.  The checkpoint has an extra C4' distance head; the
    # in-place update must happen before Folding is constructed.
    if "to_dist.fc_2d.distance.C4'.1.weight" in fold_state:
        tr_utils.obj["inter_labels"]["distance"] = [
            "C3'", "P", "N1", "C4", "C1'", "CiNj", "PiNj", "C4'",
        ]

    fold_model = Folding(
        dim_2d=config["dim_pair"],
        layers_2d=config["RNAformer"]["n_block"],
        config=config,
    ).to(device)
    fold_model.load_state_dict(fold_state)
    fold_model.eval()

    with torch.no_grad():
        ss_lst = [model(msa, msa_cutoff=args.msa_rows).float() for model in ss_models]
        ss = torch.stack(ss_lst, dim=0).mean(dim=0)
        _, outputs = fold_model(
            raw_seq,
            msa,
            ss,
            res_id=res_id,
            num_recycle=args.num_recycles,
            msa_cutoff=args.msa_rows,
            config=config,
        )

    c1 = outputs["cords_c1'"][-1]
    if c1.ndim == 3:
        c1 = c1.squeeze(0)
    coords = c1.detach().cpu().numpy().astype(np.float32)

    distance_logits = outputs["geoms"]["inter_labels"]["distance"]["C1'"]
    distance_bins = torch.arange(3.0, 41.0, device=distance_logits.device)
    distance = _distance_expectation(distance_logits, distance_bins, torch)
    distance = distance.detach().cpu().numpy().astype(np.float32)

    allatom = outputs["cord_tns_pred"][-1]
    if allatom.ndim == 4:
        allatom = allatom.squeeze(0)
    # Upstream's exporter expects (L, atoms, 3).
    allatom = allatom.permute(1, 0, 2).detach().cpu().numpy().astype(np.float32)

    plddt = outputs.get("plddt")
    if plddt is not None:
        plddt_mean = float(plddt.detach().float().mean().cpu())
    else:
        plddt_mean = 0.0

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_base) + "_coords.npy", coords)
    np.save(str(out_base) + "_dist.npy", distance)
    np.save(str(out_base) + "_allatom.npy", allatom)
    with (Path(str(out_base) + "_meta.json")).open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "length": length,
                "device": str(device),
                "plddt_mean": plddt_mean,
                "model_dir": str(model_dir),
                "num_recycles": args.num_recycles,
            },
            handle,
            indent=2,
        )
    print("saved %s_{coords,dist,allatom}.npy" % out_base)
    print("pLDDT mean: %.3f" % plddt_mean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
