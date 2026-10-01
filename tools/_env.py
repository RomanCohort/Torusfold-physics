# -*- coding: utf-8 -*-
"""Publish .env.local into the current process.

The setup step records where the external tools and the interpreter live, and
writes it to `.env.local`. The wrappers read their ROOT variables at import time,
so anything that imports them without those variables set builds relative paths
and reports every tool as missing — while the files sit exactly where the setup
recorded them.

`activate_deps.bat` covers the case where the server is started by `start.bat`.
It does not cover `python serve.py`, `python tools/configure_deps.py check`, or
any ad-hoc script, which is how the failure was found: the dependency report
printed "MISS RhoFold+ checkpoint  pretrained\\rhofold_pretrained_params.pt", a
relative path, while a 485 MB checkpoint was present at the recorded location.

Import this and call `apply()` before importing anything that reads those
variables. An explicit value already in the environment always wins, so a
command-line override keeps working.

Standard library only, and safe to import from a script that sits next to it.
"""
from __future__ import annotations

import io
import os
from typing import List, Tuple


def parse(path: str) -> Tuple[dict, List[str]]:
    """Read `.env.local` as a dict, plus any lines that could not be parsed.

    Written as UTF-8. Decoded as the locale's ANSI page only if that fails, which
    is what a hand-edited file on a Chinese Windows tends to be — the paths can
    contain non-ASCII user names, and a wrong guess yields a plausible-looking
    path that does not exist.
    """
    values: dict = {}
    unreadable: List[str] = []
    if not path or not os.path.isfile(path):
        return values, unreadable

    text = None
    for encoding in ("utf-8-sig", None):
        try:
            with io.open(path, encoding=encoding) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
        except OSError:
            return values, unreadable
    if text is None:
        return values, unreadable

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            unreadable.append(line)
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not value:
            unreadable.append(line)
            continue
        values[key] = value
    return values, unreadable


def apply(repo_root: str = None) -> List[str]:
    """Set every variable from `.env.local` that is not already set.

    Returns the names applied, so a caller can report what happened. Never
    raises: a missing or malformed file leaves the environment untouched.
    """
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    values, _ = parse(os.path.join(repo_root, ".env.local"))
    applied: List[str] = []
    for key, value in sorted(values.items()):
        if os.environ.get(key):
            continue
        os.environ[key] = value
        applied.append(key)
    return applied
