"""The verdict on an arm, judged on the SAMPLED marginal -- the criterion the first one got wrong.

WHY THE CRITERION MOVED (2026-10-04). Arm A was judged on table-side quantities: the dihedral's own
implied edge gap had to fall inside |0.05|, and the angle's implied sigma had to stop falling while the
sampled sigma tracked it. It failed both, and the failure was real -- but the criterion was answering a
question the loop does not get to ask. Measured over the production campaign's four recorded rounds, the
dihedral's table-side edge mass moved 0.1699 -> 0.1515 while the sampled edge mass sat at 0.318-0.361
against a 0.3243 reference: the ensemble did not follow the table at all, so a rule that requires the
TABLE to be right is requiring something the physics never promised. The reverse accident is worse and
is also on the record -- the angle's table walked its implied sigma from 0.3218 to 0.1307 (-59 percent)
while the sampler followed -16 percent, and it kept "improving" a table nobody was sampling.

So an arm is judged here on what the pool did, per controlled coordinate:

    WIDTH   |ln(sigma_sampled / sigma_target)| <= 0.10
    EDGE    |edge_sampled - edge_target| <= 0.05

with the table-side numbers reported beside them as DIAGNOSTICS, plus the one number that explains most
of the failures -- the transmission slope of sampled sigma against implied sigma over the arm's rounds,
which is how a coordinate that cannot be moved by its own table is recognised (0.019 for the stack,
which has no table at all; 1.111 for bb_bond, which is the one coordinate the loop finished).

Exclusions, and they are explicit rather than silent:
  * a coordinate with no table of its own is DERIVED, not judged: the stack is an exact algebraic
    function of bb_bond and the angle (K_STACK = 0), so its marginals are a consequence;
  * a coordinate whose implied sigma never moved has no step to attribute and is reported as such --
    that is not a pass.

Any arm directory works, including the ones written before the round json carried marginals: the
numbers are then recomputed from tables_r{r}.npz and tasks_r{r}/counts__*. Exit code 0 iff every judged
coordinate passes at the arm's last round.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import ibi_core as IC                     # noqa: E402
import ibi_transmission_scan as TS        # noqa: E402

WIDTH_TOL = 0.10
EDGE_TOL = 0.05
NO_TABLE = ("stack",)


def arm_rounds(root):
    """Round indices that have both a table and per-chain histograms."""
    out = []
    for tf in sorted(root.glob("tables_r*.npz")):
        r = int(tf.stem.split("_r")[1])
        if (root / f"tasks_r{r}").is_dir():
            out.append(r)
    return out


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "ibi_armA"
    ref_path = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "results" / "refit_smooth5.npz"
    ref_tables = IC.load_tables(str(ref_path))
    coords = list(IC.load_tables(str(sorted(root.glob("tables_r*.npz"))[0]).replace("\\", "\\"))[0].keys()) \
        if False else ["bb_bond", "angle", "dihedral", "stack"]

    target = {}
    for c in coords:
        p, centre = TS.implied(ref_tables[c])
        target[c] = (TS.sigma_of(p, centre), TS.edge_mass(p))

    print("verdict on %s against %s" % (root.name, ref_path.name))
    print("criteria: |ln(sigma_sampled/sigma_target)| <= %.2f and |edge gap sampled| <= %.2f"
          % (WIDTH_TOL, EDGE_TOL))
    print("excluded (derived, no table of its own): %s\n" % ", ".join(NO_TABLE))

    series = {}
    judged = {}
    for r in arm_rounds(root):
        tabs = IC.load_tables(str(root / f"tables_r{r}.npz"))
        rj = root / f"round{r}.json"
        js = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
        print("=== round %d ===" % r)
        for c in coords:
            cnt, _n = TS.pooled_counts(root / f"tasks_r{r}", c)
            if cnt is None:
                continue
            centre = np.asarray(tabs[c]["centre"], dtype=float)
            p_sim = cnt / cnt.sum()
            p_imp, _ = TS.implied(tabs[c])
            s_sim, e_sim = TS.sigma_of(p_sim, centre), TS.edge_mass(p_sim)
            s_imp = TS.sigma_of(p_imp, centre)
            series.setdefault(c, []).append((s_imp, s_sim))
            st, et = target[c]
            lr = abs(np.log(s_sim / st)) if (s_sim > 0 and st > 0) else float("nan")
            eg = e_sim - et
            u = (js.get("updates") or {}).get(c) or {}
            status = u.get("status", "?")
            if c in NO_TABLE:
                verdict = "derived"
            elif abs(s_imp - series[c][0][0]) < 1e-9 and len(series[c]) == 1:
                verdict = "no step yet"
            elif lr <= WIDTH_TOL and abs(eg) <= EDGE_TOL:
                verdict = "PASS"
            else:
                verdict = "FAIL"
            judged[c] = verdict
            print("  %-9s %-8s sampled %.5f / target %.5f  |ln| %.3f   edge %+.4f  "
                  "(table implied %.5f)  -> %s"
                  % (c, status, s_sim, st, lr, eg, s_imp, verdict))

    print("\ntransmission over this arm (sampled against implied sigma):")
    for c in coords:
        pts = series.get(c) or []
        if len(pts) < 2:
            continue
        slope, icept, r2 = TS.ls_line([p[0] for p in pts], [p[1] for p in pts])
        moved = max(p[0] for p in pts) - min(p[0] for p in pts)
        note = ("no table of its own" if c in NO_TABLE else
                ("table never moved, nothing to attribute" if moved < 1e-6 else
                 "floor %.5f vs target %.5f" % (icept, target[c][0])))
        print("  %-9s slope %.3f  intercept %.5f  R2 %.3f  (%s)"
              % (c, slope, icept, r2, note))

    bad = [c for c, v in judged.items() if v == "FAIL"]
    print("\n%s: %s" % ("FAIL" if bad else "PASS",
                         ("no pass at the last round for " + ", ".join(bad)) if bad
                         else "every judged coordinate inside tolerance at its last round"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
