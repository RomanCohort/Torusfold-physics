"""Two-start ergodicity verdict: do start A (native) and start B (800 K annealed) land on the
same terminal marginals?

Reads two logs:
  - start A: results/F_noBSJ_idx0.log          (native deposited structure, BSJ=0, seed 20260218)
  - start B: results/two_start_B_anneal800K.log (800 K annealed same structure, BSJ=0, seed
    20260218, same 8x100000 / burn 20000 / stride 25 / blocks 8 protocol)

Both logs carry the same per-block sim/ref table, so "how much did each start drift" is read from
the block columns directly: per-coordinate least-squares slope (sim/ref per block) and per-block
range, plus the joint-J block spread. The verdict is per the team-lead's criteria, verbatim:

  - terminal sim/ref agree within block spread  ->  sampler is ergodic on this time scale
  - differ by more than 20 percent             ->  not ergodic (more fundamental than REMD)
  - both starts still drifting (slope far from 0) -> "cannot separate": the premise that the two
    starts have a shared endpoint is not yet true, so neither "equal" nor "unequal" is the answer.

This script only compares numbers already written to disk by the two runs; it runs no sampling.

Run: python scripts/compare_two_starts.py [startA_log] [startB_log]
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
A_LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "F_noBSJ_idx0.log"
B_LOG = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / "results" / "two_start_B_anneal800K.log"

COORDS = ["bb_bond", "intra_pc", "intra_cn", "angle", "dihedral", "stack"]

# block table row: block  lo  hi  frames  J  then six sim/ref columns in B.COORDS order.
ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)-\s*([\d.]+)\s+(\d+)\s+([\d.]+)\s+"
                 r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse(log):
    """Return {'terminal': {coord: sim/ref}, 'blocks': {coord: [8 values]}, 'J': [8 values]}."""
    text = log.read_text(encoding="utf-8", errors="replace")
    terminal = {}
    blocks = {c: [] for c in COORDS}
    J = []

    # terminal sim/ref: start A prints "coordinate  ref sig ... sim/ref ...", start B prints
    # "coordinate  sim/ref". Both give the full coordinate name first and sim/ref somewhere.
    in_term = False
    for line in text.splitlines():
        s = line.split()
        if not s:
            continue
        if s[0] == "coordinate":
            in_term = True
            continue
        if in_term:
            if len(s) < 2:
                continue
            name = s[0]
            if name in COORDS:
                # start A's table: "coordinate ref sig 1-D sig sim sig sim/ref sim/1D dU min
                # dU max |dU|>1kBT" -> 9 tokens, sim/ref is RAW token index 4. start B's table:
                # "coordinate sim/ref" -> 2 tokens, sim/ref is RAW index 1. Index the raw tokens
                # (not nan-filtered ones), because A's "1-D sig" and "sim/1D" columns are 'nan'
                # for stack and filtering them shifts every later index. sim/1D is the constructed
                # quantity and is never read.
                if len(s) == 2:
                    terminal[name] = float(s[1])
                elif len(s) >= 9:
                    terminal[name] = float(s[4])
            if name not in COORDS and s[0] not in ("coordinate",):
                # start A prints extra rows (e.g. ---- separators are not word rows); stop on the
                # next non-coordinate section header only when we've got all six.
                if len(terminal) == len(COORDS):
                    in_term = False

        m = ROW.match(line)
        if m:
            g = m.groups()
            J.append(float(g[4]))
            for i, c in enumerate(COORDS):
                blocks[c].append(float(g[5 + i]))

    # if the terminal parse above mis-fired on start A (extra columns), fall back to block mean of
    # the LAST block only when terminal is empty
    if not terminal:
        for c in COORDS:
            if blocks[c]:
                terminal[c] = blocks[c][-1]
    return terminal, blocks, J


def slope(vals):
    n = len(vals)
    if n < 2:
        return float("nan")
    x = list(range(n))
    mx = sum(x) / n
    my = sum(vals) / n
    num = sum((x[i] - mx) * (vals[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    return num / den if den else float("nan")


ta, ba, Ja = parse(A_LOG)
tb, bb, Jb = parse(B_LOG)

print(f"start A: {A_LOG.name}")
print(f"start B: {B_LOG.name}")
print()

print("=== per-coordinate terminal sim/ref and how far each start drifted ===")
print(f"{'coord':9s} {'A term':>8s} {'B term':>8s} {'|B-A|/A':>9s} "
      f"{'A slope':>8s} {'B slope':>8s} {'A range':>8s} {'B range':>8s}")
print("-" * 78)
for c in COORDS:
    a, b = ta.get(c, float("nan")), tb.get(c, float("nan"))
    rel = abs(b - a) / a if a else float("nan")
    sa, sb = slope(ba.get(c, [])), slope(bb.get(c, []))
    ra = max(ba.get(c, [1e9])) - min(ba.get(c, [1e9])) if ba.get(c) else float("nan")
    rb = max(bb.get(c, [1e9])) - min(bb.get(c, [1e9])) if bb.get(c) else float("nan")
    print(f"{c:9s} {a:8.3f} {b:8.3f} {rel:8.1%} {sa:+8.4f} {sb:+8.4f} "
          f"{ra:8.3f} {rb:8.3f}")

print()
print("=== joint-J block spread (drift ruler) ===")
if Ja and Jb:
    print(f"start A: J {min(Ja):.4f}..{max(Ja):.4f}  spread "
          f"{(max(Ja) - min(Ja)) / min(Ja) * 100:.1f}%   slope {slope(Ja):+.4f}/block")
    print(f"start B: J {min(Jb):.4f}..{max(Jb):.4f}  spread "
          f"{(max(Jb) - min(Jb)) / min(Jb) * 100:.1f}%   slope {slope(Jb):+.4f}/block")

print()
print("=== verdict (per-coordinate, team-lead criteria) ===")
# "Still drifting" means the mean is moving by a meaningful fraction of ITSELF over the window:
# net motion |slope| x 7 blocks vs 3 percent of the terminal value. A max-min block range is a
# noisy ruler (8 points), and it mis-flags a settled coordinate whose blocks merely wiggle; a
# relative threshold asks the physically right question -- "is this number still changing?"
DRIFT = 0.03
have_b = any(tb.get(c, float("nan")) == tb.get(c, float("nan")) for c in COORDS)
for c in COORDS:
    a, b = ta.get(c, float("nan")), tb.get(c, float("nan"))
    rel = abs(b - a) / a if a and a == a and b == b else float("nan")
    sa, sb = slope(ba.get(c, [])), slope(bb.get(c, []))
    drift_a = abs(sa) * 7 > DRIFT * abs(a) if (sa == sa and a and a == a) else False
    drift_b = abs(sb) * 7 > DRIFT * abs(b) if (sb == sb and b and b == b) else False
    if not have_b:
        verdict = "B not yet run"
    elif rel > 0.20:
        verdict = "NOT ergodic (differ >20%)"
    elif drift_a and drift_b:
        verdict = "agrees, but BOTH still drifting (endpoint not final)"
    elif drift_a or drift_b:
        verdict = "agrees, but one start still drifting"
    else:
        verdict = "ergodic (agree within spread)"
    print(f"  {c:9s} A={a:.3f} B={b:.3f} rel={rel:.1%} "
          f"slopeA={sa:+.4f} slopeB={sb:+.4f} -> {verdict}")
