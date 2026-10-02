#!/usr/bin/env python3
"""Restore LF to the 24 files a Python write_text() turned into CRLF.

WHAT HAPPENED. Commit 341e267 rewrote 20 files and created 4 using Path.write_text(), and
on Windows text mode translates "\\n" to os.linesep. Every one of those 24 files was LF
before and CRLF after. Measured, blob against blob:

    artifacts/README.md                 LF -> CRLF
    docs/NOTES.md                       LF -> CRLF
    docs/REPRODUCTION_RESOURCES.md      LF -> CRLF
    docs/archive/README.md              new file, written CRLF
    docs/archive/plan_c_basis_family.md          LF -> CRLF   (edited after the move)
    docs/archive/plan_c_c2_stabilization.md      LF -> CRLF   (edited after the move)
    docs/ibi_loop_and_oxrna_findings.md LF -> CRLF
    environment.yml                     LF -> CRLF
    pyproject.toml                      LF -> CRLF
    requirements.lock                   LF -> CRLF
    run_2013nt.py                       LF -> CRLF
    scripts/{extract_viewer_payload,ibi_bonded,ibi_loop,measure_native_retention,
             plan_c_basis,plan_c_instrument,plan_c_loop,plan_c_same_field_floor,
             verify_headline}.py        LF -> CRLF
    tests/{test_headline_artifacts,test_ibi_driver_rules,test_moment_correction,
           test_plan_c_instrument}.py   LF -> CRLF

The seven files that were only RENAMED kept LF, because shutil/Path.rename preserves
bytes. That is the tell that separates damage from a move, and it is why this list is
exactly the files whose CONTENT was rewritten.

CORRECTION TO AN EARLIER CLAIM. I first wrote that "every text blob in this repository is
LF". That was a generalisation from the few files I had checked, and it is false: the
working tree is mixed. src/torusfold/web/app.js, scripts/force_reference.py and
src/torusfold/scheme2/pdb_analyzer.py are CRLF in the committed blob, and
activate_deps.bat is LF in the blob but CRLF in the working tree. So this script does NOT
normalise the repository -- it restores the 24 specific files that had LF and lost it.
Normalising everything would be a different change, with a different blast radius, and
nobody asked for it.

WHY IT MATTERS: a line-ending rewrite buries a reviewable diff inside a whole-file one.
`git diff --ignore-cr-at-eol --numstat HEAD~1 HEAD` puts the real change at 25 lines where
the raw diff says 8,540 deletions.

NO GIT CALL. Whether a file is CRLF is a property of its bytes, so this reads files and
never shells out. (Shelling out from this interpreter is blocked: subprocess with piped
stdio raises NotADirectoryError on CreateProcess.) The file list is therefore written out
explicitly rather than queried.

Run:  python tools/fix_line_endings.py            (dry run)
      python tools/fix_line_endings.py --apply
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The 24 files measured as LF -> CRLF in 341e267. Explicit on purpose: a glob would also
# catch the files that are legitimately CRLF and rewrite them, which is a different change.
RESTORE = [
    "artifacts/README.md",
    "docs/NOTES.md",
    "docs/REPRODUCTION_RESOURCES.md",
    "docs/archive/README.md",
    "docs/archive/plan_c_basis_family.md",
    "docs/archive/plan_c_c2_stabilization.md",
    "docs/ibi_loop_and_oxrna_findings.md",
    "environment.yml",
    "pyproject.toml",
    "requirements.lock",
    "run_2013nt.py",
    "scripts/extract_viewer_payload.py",
    "scripts/ibi_bonded.py",
    "scripts/ibi_loop.py",
    "scripts/measure_native_retention.py",
    "scripts/plan_c_basis.py",
    "scripts/plan_c_instrument.py",
    "scripts/plan_c_loop.py",
    "scripts/plan_c_same_field_floor.py",
    "scripts/verify_headline.py",
    "tests/test_headline_artifacts.py",
    "tests/test_ibi_driver_rules.py",
    "tests/test_moment_correction.py",
    "tests/test_plan_c_instrument.py",
]

apply = "--apply" in sys.argv
print(f"=== restoring LF to {len(RESTORE)} files ===\n")

need = []
for rel in RESTORE:
    p = ROOT / rel
    if not p.is_file():
        print(f"  MISSING  {rel}")
        continue
    raw = p.read_bytes()
    crlf = raw.count(b"\r\n")
    lf_only = raw.count(b"\n") - crlf
    if crlf:
        print(f"  CRLF     {rel}  ({crlf} CRLF, {lf_only} LF)")
        need.append(p)
    else:
        print(f"  already  {rel}  (LF)")

if apply and need:
    print("\n=== rewriting ===")
    for p in need:
        text = p.read_bytes().decode("utf-8")
        # Binary write: the bytes are controlled here, so no text-mode translation.
        p.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
        print(f"  wrote {p.relative_to(ROOT)}")

    still = [p for p in need if b"\r\n" in p.read_bytes()]
    print(f"\nstill CRLF: {len(still)}")
    if still:
        for p in still:
            print(f"  {p.relative_to(ROOT)}")

if not apply:
    print(f"\n{len(need)} file(s) would be rewritten. DRY RUN -- re-run with --apply.")
