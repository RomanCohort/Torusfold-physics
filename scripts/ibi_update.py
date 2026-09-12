"""One IBI update: read a sampler's histograms, run the update, write the next table.

The two halves of IBI existed in this repository without ever touching. ibi_round0.py (now via
ibi_core.run_round) produces histograms on the table's own bins; ibi_bonded.plan_update turns a
histogram plus a reference into the next table, with six named refusals and a full diagnostics
dict -- and it had NO caller anywhere: the only implementation of the loop is
tests/test_ibi_bonded.py:179-193, against a synthetic engine. The sampler printed dU and wrote
nothing, so the two were joined by a human reading numbers out of a log.

This script is that join, on disk:

    python scripts/ibi_round0.py 8 100000 0 0.1 25 20000 --blocks=8 \
        --angle=table --dihedral=table --write=results/ibi/r0
    python scripts/ibi_update.py --hist=results/ibi/r0 --out=results/ibi/r1_tables.npz

The reference is a SEPARATE input from the table being simulated, and deliberately so. p_ref is
the target -- the pooled 126-chain distribution, fixed for every round. What changes round to
round is the table the sampler ran under. ibi_bonded.py:82-84 states the rule: p_ref "must be
kept from round 0 and never recomputed from the running table". Recomputing it from the running
table would make the update a no-op by construction.

Run: python scripts/ibi_update.py --hist=DIR --out=FILE.npz [--ref=FILE.npz]
                                  [--coords=angle,dihedral] [--gain=1.0] [--json=FILE]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B          # noqa: E402
import ibi_bonded as I                # noqa: E402

DEFAULT_REF = REPO / "results" / "boltzmann_tables_clean.npz"


def _opt(name, default=""):
    pre = f"--{name}="
    for a in sys.argv[1:]:
        if a.startswith(pre):
            return a[len(pre):]
    return default


def table_from_npz(z):
    """The table that was simulated, in boltzmann_bonded format, out of a sampler's npz."""
    return {"lo": float(z["lo"]), "hi": float(z["hi"]), "binw": float(z["binw"]),
            "U": z["U"], "centre": z["centre"], "sigma": float(z["sigma"])}


def hist_from_npz(z):
    """plan_update's SimHistogram, with n and n_outside carried rather than guessed.

    counts alone cannot give these: n is every observation offered (in support or not) and
    n_outside is how many fell outside [lo, hi]. plan_update checks them, and a caller that
    passed counts.sum() for n would be claiming the sampler never went out of support.
    """
    return I.SimHistogram(counts=z["counts"], n=int(z["n"]), n_outside=int(z["n_outside"]),
                          lo=float(z["lo"]), hi=float(z["hi"]), nbins=int(z["nbins"]))


def main():
    hist_dir = Path(_opt("hist", ""))
    out_path = Path(_opt("out", ""))
    ref_path = Path(_opt("ref", str(DEFAULT_REF)))
    coords = [c.strip() for c in _opt("coords", "angle,dihedral").split(",") if c.strip()]
    gain = float(_opt("gain", "1.0"))
    json_path = _opt("json", "")

    if not hist_dir or not str(hist_dir):
        raise SystemExit("--hist=DIR is required (what ibi_round0.py --write produced)")
    if not out_path or not str(out_path):
        raise SystemExit("--out=FILE.npz is required (where the next tables go)")

    ref = np.load(ref_path)
    print(f"reference  {ref_path}")
    print(f"histograms {hist_dir}")
    print(f"update     gain={gain}, coordinates {coords}")
    print()

    # A round is an IBI step only if the potential that was SIMULATED is a table on these bins.
    # The update is U_{i+1} = U_i + kBT*ln(P_sim/P_ref) with U_i the potential that was in force;
    # if the sampler ran the shipped harmonic or a Fourier series, there is no U_i here and
    # "U_i + dU" names nothing -- the sum would be a table plus a correction computed against a
    # different Hamiltonian. The sampler records what it ran in its manifest; read it rather
    # than assume, because assuming is how a run of the wrong field becomes a table.
    manifest = hist_dir / "manifest.json"
    if not manifest.exists():
        raise SystemExit(
            f"{manifest} not found. Without it there is no record of which potential produced "
            f"these histograms, and a non-table potential has no U_i for the update to add to. "
            f"Re-run the sampling with --write so the manifest is written.")
    _pots = json.loads(manifest.read_text(encoding="utf-8")).get("potentials", {})
    for c in coords:
        spec = _pots.get(c)
        if spec is None:
            raise SystemExit(
                f"{c} was sampled under the shipped field (no potential recorded for it). The "
                f"shipped term is a harmonic constant, not a table, so there are no bins holding "
                f"its U_i. Re-run with --{c}=table or --{c}=table_jac:<eps>.")
        if not spec.startswith("table"):
            raise SystemExit(
                f"{c} was sampled under {spec}, which is not a table. The update adds dU to the "
                f"table that was simulated, so a non-table potential has nothing to add to. This "
                f"is a real distinction, not a formality: a Fourier round would produce a table "
                f"plus a correction measured against a different Hamiltonian. Re-run with "
                f"--{c}=table.")

    new_tables, report = {}, {}
    for c in coords:
        hp = hist_dir / f"{c}.npz"
        if not hp.exists():
            raise SystemExit(f"{hp} not found; --write must be run for this coordinate first")
        z = np.load(hp)
        table = table_from_npz(z)
        hist = hist_from_npz(z)
        # The target, from the reference file -- NEVER from the table being simulated.
        p_ref = I.bin_probabilities_from_U(
            {"lo": float(ref[f"{c}__lo"]), "hi": float(ref[f"{c}__hi"]),
             "binw": float(ref[f"{c}__binw"]), "U": ref[f"{c}__U"],
             "centre": ref[f"{c}__centre"]})
        res = I.plan_update(table, hist, p_ref, gain=gain)

        # A refused round is a result, not a crash: the diagnostics carry every measurement
        # either way, and printing them is the point of having named refusals.
        print(f"{c}:")
        print(f"  n={hist.n}  n_outside={hist.n_outside}  "
              f"({hist.n_outside / hist.n:.2%} out of support)" if hist.n else "")
        print(f"  status {res.status}" + (f"  ({res.reason})" if res.reason else ""))
        for k in sorted(res.diagnostics):
            print(f"    {k:28s} {res.diagnostics[k]}")
        if res.table is not None:
            print(f"  max|dU| {max(abs(res.dU)):.4f} kJ/mol = {max(abs(res.dU)) / B.KBT:.4f} kBT")
            new_tables[c] = res.table
        print()
        report[c] = {"status": res.status, "reason": res.reason,
                     "n": hist.n, "n_outside": hist.n_outside,
                     "max_abs_dU": (float(max(abs(res.dU))) if res.table is not None else None),
                     "diagnostics": {k: (float(v) if isinstance(v, (int, float, np.floating))
                                         else v) for k, v in res.diagnostics.items()}}

    if not new_tables:
        print("every coordinate was refused; nothing written. That is a measurement: the round "
              "did not produce a usable update, and the diagnostics above say which check fired.")
        if json_path:
            Path(json_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for c, t in new_tables.items():
        for k in I.TABLE_KEYS:
            if k in t:
                payload[f"{c}__{k}"] = np.asarray(t[k])
        # sigma is a property of the REFERENCE, not of the table being updated: it is the width
        # the residual is reported in, and plan_update's TABLE_KEYS deliberately omit it because
        # an update has no opinion about it. Carry it from the reference, or the file cannot be
        # loaded by ibi_core.load_tables for the next round.
        if f"{c}__sigma" in ref.files:
            payload[f"{c}__sigma"] = ref[f"{c}__sigma"]
    # keep the untouched coordinates so the file is a complete, loadable table set
    for c in B.COORDS:
        if c in new_tables:
            continue
        for k in ("lo", "hi", "binw", "U", "centre", "sigma"):
            if f"{c}__{k}" in ref.files:
                payload[f"{c}__{k}"] = ref[f"{c}__{k}"]
    np.savez(out_path, **payload)
    print(f"wrote {out_path}  ({', '.join(new_tables)})")

    if json_path:
        Path(json_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
