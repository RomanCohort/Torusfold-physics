"""Which file is reporting the archived basis-family record as a broken link?

`tools/check_doc_links.py` matches `docs/<path>.<ext>` anywhere in a scanned file
and treats every hit as a link. That means a *mention* in a docstring counts, and
so does a copy of the repository sitting inside the working tree -- `_iso/` is
exactly that, and it is not in SKIP_DIRS.

This reproduces the checker's own scan, restricted to the one path, and prints
every file and line it would blame. Read-only.

Note the phrasing above: this file deliberately does NOT write the target as a
`docs/`-prefixed path, because writing it would make this very script one of the
hits it is meant to find. The first version did exactly that and reported itself.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(r"C:\baidunetdiskdownload\torusfold-hybrid")
SKIP_DIRS = {".git", "__pycache__", "_calib", "_strays", "results", "output_web",
             "output_2013nt", "_cgdata", "_civenv", "node_modules",
             "_empty_strays", "_docs_video", "_park_20261002"}
SCAN_EXT = {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".js", ".cmd", ".bat",
            ".json", ".mermaid", ".cfg", ".lock"}
SELF = {"check_doc_links.py", "retire_docs.py", "finish_docs_retirement.py"}
# PATH_RE is built at run time below, so this script does not match itself.

TARGET = "plan_c_basis_family.md"

print(f"scanning as the checker does; looking for '{TARGET}'\n")


def _self_name() -> str:
    """This file's own name, so the scan does not blame itself.

    It has to build the target string at run time for the same reason: a literal
    `docs/` + filename in this source would be matched by PATH_RE, and the
    diagnostic would report itself as the broken link it is looking for.
    """
    return pathlib.Path(__file__).name


_PAT = re.compile(r"docs/([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)(?::(\d+))?")

hits = []
for f in sorted(ROOT.rglob("*")):
    if not f.is_file() or f.suffix not in SCAN_EXT:
        continue
    if SKIP_DIRS & set(f.parts) or f.name in SELF or f.name == _self_name():
        continue
    try:
        text = f.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    for m in _PAT.finditer(text):
        if m.group(1) != TARGET:
            continue
        line = text[:m.start()].count("\n") + 1
        hits.append((f.relative_to(ROOT), line))

print(f"{len(hits)} hit(s) that the checker would blame:\n")
for rel, line in hits:
    inside = "_iso/" if str(rel).startswith("_iso") else ""
    tag = "INSIDE A NESTED CHECKOUT" if inside else "in this repository"
    print(f"  {rel}:{line}   <- {tag}")

print()
iso = ROOT / "_iso"
if iso.is_dir():
    n = sum(1 for p in iso.rglob("*") if p.is_file())
    print(f"  _iso/ holds {n} files and is a nested copy of the repository.")
    print(f"  _iso is in SKIP_DIRS: {'_iso' in SKIP_DIRS}")
    print(f"  _iso is gitignored  : ", end="")
    import subprocess
    r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "_iso"],
                       capture_output=True, text=True)
    print("yes" if r.returncode == 0 else "NO -- so it is an untracked copy that "
          "travels with the tree and the checker walks into it")
