"""State of the armA campaign before committing CPU to a long pipeline run.

armA's results directory has not been written to since 2026-10-04 06:58. The
scheduled task is meant to repeat, so either it is still working somewhere else,
or it stopped and nobody said so. This decides whether a multi-hour pipeline run
would compete with it.

Deliberately read-only: nothing here kills, starts or modifies the campaign.
Stale-worker collection, if needed, is a separate decision and uses explicit PIDs
(matching CommandLine, never an image name) per docs/dev_machine_handoff.md.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(r"C:\baidunetdiskdownload\torusfold-hybrid")
ARM = REPO / "results" / "ibi_armA"


def run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                           encoding="utf-8", errors="replace")
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}: {e}>"


print("=" * 74)
print("1. scheduled task")
print("=" * 74)
print(run(["schtasks", "/query", "/tn", "plan_c_armA", "/v", "/fo", "LIST"])[:2000] or
      "  (no output)")

print()
print("=" * 74)
print("2. results/ibi_armA freshness")
print("=" * 74)
if not ARM.is_dir():
    print(f"  {ARM} absent")
else:
    files = [p for p in ARM.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"  {len(files)} files")
    newest = files[0]
    age = dt.datetime.now() - dt.datetime.fromtimestamp(newest.stat().st_mtime)
    print(f"  newest: {newest.relative_to(ARM)}")
    print(f"          {dt.datetime.fromtimestamp(newest.stat().st_mtime):%Y-%m-%d %H:%M:%S}"
          f"  ({age.total_seconds()/3600:.1f} h ago)")
    print()
    print("  newest 8:")
    for p in files[:8]:
        print(f"    {dt.datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M:%S}  "
              f"{p.relative_to(ARM)}")
    for sub in ("tasks_r0", "tasks_r1", "tasks_r2"):
        d = ARM / sub
        if d.is_dir():
            n = len([p for p in d.iterdir() if p.is_file()])
            print(f"  {sub:<12} {n} files")
    for rf in ("round0.json", "round1.json", "round2.json"):
        p = ARM / rf
        if p.is_file():
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                print(f"  {rf:<12} round={j.get('round')} seconds={j.get('seconds')} "
                      f"structures={len(j.get('per_structure', []))}")
            except Exception as e:  # noqa: BLE001
                print(f"  {rf}: unreadable ({e})")

print()
print("=" * 74)
print("3. any process belonging to the campaign (CommandLine match, by PID)")
print("=" * 74)
out = run(["powershell", "-NoProfile", "-Command",
           "Get-CimInstance Win32_Process | "
           "Where-Object { $_.Name -like 'python*' } | "
           "Select-Object ProcessId,Name,CreationDate,"
           "@{n='CmdLine';e={$_.CommandLine}} | Format-List"])
print(out[:4000] if out.strip() else "  (none)")

print()
print("=" * 74)
print("4. disk space")
print("=" * 74)
for drive in ("C:",):
    out = run(["powershell", "-NoProfile", "-Command",
               f"Get-PSDrive -Name {drive[0]} | "
               "Select-Object Used,Free | Format-List"])
    print(f"  {drive}\n{out}")
