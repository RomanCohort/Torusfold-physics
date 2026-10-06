"""Full REMD acceptance history from the run log, rather than the tail of it.

The tail showed T-acc=['2/40','5/40','3/40',...] while an earlier report said
'0/8'. Both are real; they are different points in time. This reads every
T-acc line in order so the trend is visible instead of whichever block happened
to be last when someone looked.
"""
from __future__ import annotations

import pathlib
import re
import sys

REPO = pathlib.Path(r"C:\baidunetdiskdownload\torusfold-hybrid")
LOG = REPO / "results" / "immuno_full" / "_run.log"

if not LOG.is_file():
    print(f"no log at {LOG}")
    sys.exit(1)

text = LOG.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()
print(f"log: {len(lines)} lines, {LOG.stat().st_size/1024:.1f} KB")

pat = re.compile(r"\[GPU-2D\]\s+(\d+)/(\d+):.*?T-acc=\[(.*?)\]")
rows = []
for l in lines:
    m = pat.search(l)
    if not m:
        continue
    blk, tot = m.group(1), m.group(2)
    entries = re.findall(r"(\d+)/(\d+)", m.group(3))
    rates = [int(a) / int(b) for a, b in entries if int(b)]
    if not rates:
        continue
    rows.append((int(blk), int(tot), rates))

print(f"T-acc rows found: {len(rows)}")
print()
if not rows:
    print("  none")
    sys.exit(0)

print(f"  {'block':>9} {'edges':>6} {'mean acc':>9} {'min':>7} {'max':>7}  per-edge")
for blk, tot, rates in rows:
    per = " ".join(f"{r:.0%}" for r in rates)
    print(f"  {blk:>4}/{tot:<4} {len(rates):>6} {sum(rates)/len(rates):>9.3f} "
          f"{min(rates):>7.3f} {max(rates):>7.3f}  {per}")

# trend: first third vs last third
n = len(rows)
if n >= 6:
    third = n // 3
    first = [r for _, _, rs in rows[:third] for r in rs]
    last = [r for _, _, rs in rows[-third:] for r in rs]
    print()
    print(f"  first {third} rows : mean {sum(first)/len(first):.4f}  "
          f"zeros {sum(1 for r in first if r == 0)}/{len(first)}")
    print(f"  last  {third} rows : mean {sum(last)/len(last):.4f}  "
          f"zeros {sum(1 for r in last if r == 0)}/{len(last)}")
    print()
    if sum(last) / len(last) > sum(first) / len(first) + 0.02:
        print("  TREND: acceptance is IMPROVING as the structure relaxes.")
        print("  => the earlier 0/8 was an initial-condition artefact, not a")
        print("     property of the temperature ladder. Do not retune t_lo/t_hi")
        print("     on the strength of the first few blocks.")
    else:
        print("  TREND: no improvement. The ladder may genuinely be too coarse.")
