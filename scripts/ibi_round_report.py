"""The two operators side by side on identical histograms, round by round.

WHY THIS EXISTS. The round-4 boundary switch archived the table operator's four rounds
(results/ibi_relax/table_operator_archive/round*.json) and then REPLAYED those same rounds under
the moment operator from the same 867 task files. For rounds 0-4 both operators therefore saw
exactly the same evidence, and the archive records what one of them did with it -- a full-pool
operator comparison that cost no sampling at all, which is the thing the seven-chain A/B could only
approximate. Rounds 5 and later exist under the moment operator only.

Per round and per coordinate it prints the correction the operator asked for (max|dU|, in kJ/mol and
in kBT), how many samples fell outside the table's support, and the per-chain residual: the median
and p90 of each chain's joint J, plus the same ratio against the TABLE's own sigma where the round
carries it (rounds 5+; before that the field does not exist in the task files, see
ibi_core.implied_sigma for why the two denominators differ by a constant factor on bb_bond and
stack).

Usage: python scripts/ibi_round_report.py [round ...]
"""
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "ibi_relax"
ARCHIVE = OUT / "table_operator_archive"
UPDATED = ("bb_bond", "angle", "dihedral")
COORDS = ("bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack")
KBT = 2.494


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rounds_in(d):
    out = []
    for p in Path(d).glob("round*.json"):
        try:
            out.append(int(p.stem[5:]))
        except ValueError:
            continue
    return sorted(out)


def summarise(d, tag):
    u = d.get("updates", {})
    line = [f"  {tag:9s}"]
    for c in UPDATED:
        e = u.get(c)
        if e is None:
            line.append(f"{c}=--")
        else:
            line.append(f"{c}={e['max_abs_dU']:6.2f}({e['max_abs_dU']/KBT:5.2f}kBT,"
                        f"{e.get('status','?')[:4]},out={e.get('n_outside',0)})")
    print(" ".join(line))
    ps = d.get("per_structure") or []
    js = [p["joint_J"] for p in ps if p.get("joint_J") is not None]
    if js:
        js.sort()
        p90 = js[min(len(js) - 1, int(0.9 * len(js)))]
        print(f"             per-chain J median {st.median(js):.4f} p90 {p90:.4f} n={len(js)}")
    srt = [p.get("sim_ref_table") for p in ps if p.get("sim_ref_table")]
    if srt:
        print("             sim_ref_table median  " + " ".join(
            f"{c}={st.median([r[i] for r in srt if r[i] is not None]):.3f}" if any(
                r[i] is not None for r in srt) else f"{c}=--"
            for i, c in enumerate(COORDS) if c in UPDATED or c == "stack"))
    jt = [p["joint_J_table"] for p in ps if p.get("joint_J_table") is not None]
    if jt:
        print(f"             per-chain J_table median {st.median(jt):.4f} n={len(jt)}")


def main():
    want = [int(a) for a in sys.argv[1:]] or None
    live = rounds_in(OUT)
    arch = rounds_in(ARCHIVE) if ARCHIVE.exists() else []
    print(f"live rounds {live}" + (f", archived (table operator) rounds {arch}" if arch else ""))
    for r in (want or live):
        print(f"=== round {r} ===")
        if arch and r in arch and r in live:
            summarise(load(ARCHIVE / f"round{r}.json"), "table")
        if r in live:
            summarise(load(OUT / f"round{r}.json"), "moments")
        elif arch and r in arch:
            summarise(load(ARCHIVE / f"round{r}.json"), "table")
        if not arch or r not in arch:
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
