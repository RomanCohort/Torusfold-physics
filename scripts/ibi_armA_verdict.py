"""The A verdict: two full-pool rounds under the per-coordinate rules.

Rules, from the basis project: dihedral refit on a B-spline m=16 basis (edge mass it can carry), angle
refitted to the ensemble the field itself produced (self-consistent), bb_bond unchanged. Acceptance
from docs/plan_c_basis_family.md section 5: |edge gap| <= 0.05 for the dihedral, and for the angle the
drift stops -- its table's implied sigma stops falling monotonically while the sampled sigma tracks it.
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import ibi_core as IC          # noqa: E402

ARM = REPO / "results" / "ibi_armA"
# Resolved, not hard-coded: the campaign record was moved out of results/ on 2026-10-01 and only its
# tables and round jsons came back under _strays (see ibi_core.campaign_root).
CAMPAIGN = IC.campaign_root(need_tasks=True)
REF = REPO / "results" / "refit_smooth5.npz"
COORDS = ("bb_bond", "angle", "dihedral", "stack")
KBT = 2.494


def pooled_counts(round_dir, coord):
    tot = None
    for f in sorted(round_dir.glob("*.npz")):
        with np.load(f) as z:
            c = z[f"counts__{coord}"].astype(np.float64)
        tot = c if tot is None else tot + c
    return tot


def sigma_of(p, centre):
    m = float((p * centre).sum())
    return float(np.sqrt(max((p * (centre - m) ** 2).sum(), 0.0)))


def edge_mass(p, frac=0.05):
    k = max(1, int(len(p) * frac))
    return float(p[:k].sum() + p[-k:].sum())


def implied_sigma(rec):
    U = np.asarray(rec["U"], dtype=float)
    p = np.exp(-(U - U.min()) / KBT)
    p = p / p.sum()
    centre = np.asarray(rec["lo"]) + np.asarray(rec["binw"]) * (np.arange(len(U)) + 0.5)
    return sigma_of(p, centre), p


def main():
    ref_tables = IC.load_tables(str(REF))
    print("Target (refit_smooth5) implied sigma:", flush=True)
    ref_imp = {}
    for c in COORDS:
        s, p = implied_sigma(ref_tables[c])
        ref_imp[c] = (s, p)
        print("   %-9s sigma %.5f  edge mass %.4f" % (c, s, edge_mass(p)), flush=True)

    for rnd in (0, 1):
        tabs = IC.load_tables(str(ARM / f"tables_r{rnd}.npz"))
        j = ARM / f"round{rnd}.json"
        d = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {}
        print("\n=== armA round %d ===" % rnd, flush=True)
        for c, u in (d.get("updates") or {}).items():
            print("   update %-9s %-8s max|dU|=%8.4f kJ/mol (%5.3f kBT) rule=%s"
                  % (c, u.get("status"), u.get("max_abs_dU", float("nan")),
                     u.get("max_abs_dU", float("nan")) / KBT, u.get("rule", "-")), flush=True)
        ps = d.get("per_structure") or []
        js = [p["joint_J"] for p in ps if p.get("joint_J") is not None]
        ja = [p["joint_J_all"] for p in ps if p.get("joint_J_all") is not None]
        if js:
            print("   per-chain J median %.4f   J_all median %s"
                  % (st.median(js), ("%.4f" % st.median(ja)) if ja else "n/a"), flush=True)
        counts = pooled_counts(ARM / f"tasks_r{rnd}", None) if False else None
        for c in COORDS:
            cnt = pooled_counts(ARM / f"tasks_r{rnd}", c)
            centre = np.asarray(tabs[c]["centre"], dtype=float)
            p = cnt / cnt.sum()
            s_sim = sigma_of(p, centre)
            s_imp, p_imp = implied_sigma(tabs[c])
            gap = edge_mass(p_imp) - edge_mass(ref_imp[c][1])
            print("   %-9s sampled sigma %.5f | table implied %.5f | sim/implied %.3f | "
                  "edge gap %+.4f" % (c, s_sim, s_imp, s_sim / s_imp, gap), flush=True)


if __name__ == "__main__":
    main()
