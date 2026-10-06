#!/usr/bin/env python3
"""Every docs/ path mentioned anywhere in the repo must point at a file that exists.

This is the decisive test, and it is better than searching for particular filenames: it
finds dangling references to files nobody thought to check, including ones this session
never touched.

WHY THE PREVIOUS CHECKS WERE NOT ENOUGH. Two attempts at counting references gave wrong
answers, both by construction:

  1. tools/retire_docs.py rewrote references only in seven file names I had guessed at.
     The real citations are spread over scripts/, src/, tests/, artifacts/ and the three
     dependency files -- 34 of them in 20 files, none of which were in that list.
  2. The audit that classified reconstruction_anchor_audit.md as "0 inbound references"
     scanned docs/, tests/ and the repo root but not src/. A scan that misses a directory
     reports absence, and absence looked like a safe deletion. It was cited from
     src/torusfold/scheme2/aform_from_template.py all along.

Both were the same mistake in different clothes: deciding what to look at instead of
looking everywhere. This script looks everywhere, extracts every docs/ path by regex, and
asks the filesystem.

Run:  python tools/check_doc_links.py
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

SKIP_DIRS = {".git", "__pycache__", "_calib", "_strays", "results", "output_web",
             "output_2013nt", "_cgdata", "_civenv", "node_modules",
             "_empty_strays", "_docs_video", "_park_20261002"}
SCAN_EXT = {".py", ".md", ".txt", ".yml", ".yaml", ".toml", ".js", ".cmd", ".bat",
            ".json", ".mermaid", ".cfg", ".lock"}

# This file's own docstring and tools/retire_docs.py name the retired files on purpose --
# they are the record of the move. Excluding them keeps the check about the repository
# rather than about the tooling that edited it.
SELF = {"check_doc_links.py", "retire_docs.py", "finish_docs_retirement.py"}

# `docs/foo.md`, `docs/archive/foo.md`, optionally with a :line suffix.
PATH_RE = re.compile(r"docs/([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)(?::(\d+))?")


def _nested_repo_roots() -> set:
    """Subdirectories of ROOT that are themselves git repositories.

    WHY THIS RULE INSTEAD OF A LIST OF NAMES. This checkout has twice acquired a second
    copy of the repository inside it -- `_iso/` first, then `_repro_check/` -- each
    untracked, each not gitignored, each holding an older `scripts/ibi_armA_verdict.py`
    whose `docs/plan_c_basis_family.md` reference was already repaired here. The checker
    walked into both and reported those stale references as broken links in this
    repository. Adding the name `_iso` fixed that copy and not the next one, which is the
    failure mode of a blocklist: it only knows the copies someone already tripped over.

    A nested working copy is never part of this tree's links, whatever it is called, so
    the rule is the shape of the thing rather than its name. Its own files are read by
    whoever owns it, in its own checkout.
    """
    out = set()
    for d in ROOT.rglob(".git"):
        if d.is_dir() and d.parent != ROOT:
            out.add(d.parent)
    return out


_NESTED_REPOS = _nested_repo_roots()

found = {}
for f in sorted(ROOT.rglob("*")):
    if not f.is_file() or f.suffix not in SCAN_EXT:
        continue
    if SKIP_DIRS & set(f.parts) or f.name in SELF:
        continue
    if any(f == r or r in f.parents for r in _NESTED_REPOS):
        continue
    try:
        text = f.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    for m in PATH_RE.finditer(text):
        rel = m.group(1)
        if rel.startswith("..") or "*" in rel or rel.startswith("<"):
            continue
        line = text[:m.start()].count("\n") + 1
        found.setdefault(rel, []).append((f.relative_to(ROOT), line, m.group(2)))

print(f"=== {len(found)} distinct docs/ path(s) referenced ===\n")
missing, ok = [], 0
for rel in sorted(found):
    target = DOCS / rel
    refs = found[rel]
    if target.exists():
        ok += 1
        print(f"  OK     docs/{rel:52s} ({len(refs)} ref)")
    else:
        missing.append(rel)
        print(f"  BROKEN docs/{rel:52s} ({len(refs)} ref)")
        for f, line, ln in refs:
            suffix = f":{ln}" if ln else ""
            print(f"           {f}:{line}{suffix}")

print(f"\n=== {ok} resolve, {len(missing)} broken ===")
if missing:
    print("\nFiles named that do not exist:")
    for m in missing:
        print(f"  docs/{m}")
sys.exit(1 if missing else 0)
