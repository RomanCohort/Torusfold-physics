"""Print the edited force-field section of README.md verbatim, for review.

Read-only: extracts lines and prints them. Writes nothing.
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(r"C:\baidunetdiskdownload\torusfold-hybrid")
lines = (REPO / "README.md").read_text(encoding="utf-8").splitlines()

start = next(i for i, l in enumerate(lines)
             if l.startswith("## Force-field constants"))
end = next(i for i, l in enumerate(lines)
           if l.startswith("## Update log"))

print(f"README.md L{start + 1}-L{end}   ({end - start} lines)")
print("=" * 78)
for i in range(start, end):
    print(f"{i + 1:5d}| {lines[i]}")
print("=" * 78)
print(f"end of section (next line is L{end + 1}: {lines[end]})")
