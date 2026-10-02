"""The two shipped skill copies must not drift apart.

WHY THIS FILE EXISTS -- one guidance document ships twice, at two paths, because
two harnesses look in different places for project skills:

    .claude/skills/torusfold-predict/SKILL.md   Claude Code
    .dsh/skills/torusfold-predict/SKILL.md      DeepSeek Harness

Nothing in this repository read either file. `grep -rn SKILL.md` across
scripts/, tools/, tests/, src/, docs/, README.md and .gitlab-ci.yml returns
nothing, so the two bodies were held in step by hand. Measured 2026-10-02: they
were identical, differing only in frontmatter -- which is the intended state,
not a defect. The risk is the next edit, which lands in one of them. Neither
harness reports a stale skill, and the reader is a biologist who cannot tell an
out-of-date instruction from a current one, so this has to fail a test instead.

The checks below are the ones whose violation makes a skill SILENTLY disappear
rather than raise. Both loaders skip a file whose frontmatter they reject:

    name         must match ^[a-z0-9]+(?:-[a-z0-9]+)*$
                 (C:\\dsh\\node_modules\\@deepseek-ai\\dsh-skill\\lib\\index.js:17)
    description  must be a non-empty string, at most 1024 characters
    ---          frontmatter opens on line 1 and closes with a bare ---

Stdlib only on purpose: `.gitlab-ci.yml` installs numpy and nothing else, so
PyYAML is not available on the machine that runs this.

Run: python -m pytest -q tests/test_skills_in_sync.py
"""

import difflib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR_NAME = "torusfold-predict"
CLAUDE_SKILL = ROOT / ".claude" / "skills" / SKILL_DIR_NAME / "SKILL.md"
DSH_SKILL = ROOT / ".dsh" / "skills" / SKILL_DIR_NAME / "SKILL.md"

# dsh-skill/lib/index.js:17, and the Claude Code equivalent.
SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_NAME = 64
MAX_DESCRIPTION = 1024


def _read(path):
    """Text of the file, asserting it has not acquired CRLF endings.

    A Windows editor rewrites line endings without being asked, and a CRLF body
    would be a whole-file diff hiding whatever real change was intended.
    """
    raw = path.read_bytes()
    assert b"\r" not in raw, (
        f"{path} has CRLF line endings; this file is LF in the index. "
        f"A line-ending change buries the real edit."
    )
    return raw.decode("utf-8")


def _parse_frontmatter(text):
    """Return (fields, body), or (None, None) if the frontmatter is malformed.

    A YAML subset: a plain scalar, a double-quoted scalar, and a folded `>-`
    block. That is every shape the two files use, and it is why this exists
    instead of `import yaml` -- see the module docstring.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, None
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return None, None

    fields, key, buf = {}, None, []
    for line in lines[1:end]:
        if key is not None:                       # inside a folded block
            if line.strip():
                buf.append(line.strip())
                continue
            fields[key] = " ".join(buf).strip()
            key = None
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if v in (">-", ">", "|", "|-"):
            key, buf = k, []
        elif len(v) >= 2 and v.startswith('"') and v.endswith('"'):
            fields[k] = v[1:-1]
        elif v:
            fields[k] = v
    if key is not None:
        fields[key] = " ".join(buf).strip()
    return fields, "\n".join(lines[end + 1:])


def _first_difference(left, right):
    for line in difflib.unified_diff(left.split("\n"), right.split("\n"),
                                     "claude", "dsh", lineterm="", n=0):
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            return line
    return "(no differing line found)"


def test_both_copies_exist():
    for path in (CLAUDE_SKILL, DSH_SKILL):
        assert path.is_file(), f"missing skill file: {path}"


def test_frontmatter_is_valid_and_within_the_loaders_limits():
    for path in (CLAUDE_SKILL, DSH_SKILL):
        fields, body = _parse_frontmatter(_read(path))
        assert fields is not None, f"{path}: missing or unterminated frontmatter"

        name = fields.get("name", "")
        description = fields.get("description", "")

        assert name, f"{path}: frontmatter requires a name"
        assert SKILL_NAME_RE.match(name), (
            f"{path}: skill name {name!r} is not lowercase/digits/hyphens; "
            f"both loaders ignore the file rather than reporting it"
        )
        assert len(name) <= MAX_NAME, f"{path}: name is {len(name)} chars, limit {MAX_NAME}"

        assert description, f"{path}: frontmatter requires a description"
        assert len(description) <= MAX_DESCRIPTION, (
            f"{path}: description is {len(description)} chars, limit {MAX_DESCRIPTION}"
        )
        assert body.strip(), f"{path}: the body is empty"


def test_directory_name_matches_the_skill_name():
    """Claude Code turns the directory name into the command, so
    /torusfold-predict depends on the two agreeing."""
    for path in (CLAUDE_SKILL, DSH_SKILL):
        fields, _ = _parse_frontmatter(_read(path))
        assert path.parent.name == fields["name"], (
            f"{path}: directory {path.parent.name!r} does not match "
            f"name {fields['name']!r}"
        )


def test_claude_copy_carries_only_the_two_documented_fields():
    """Commit 20e3564 decided this deliberately: Claude Code documents `name`
    and `description`, so the DSH file's `whenToUse` was folded into the
    description rather than carried over as an unknown key."""
    fields, _ = _parse_frontmatter(_read(CLAUDE_SKILL))
    assert set(fields) == {"name", "description"}, (
        f"the Claude Code copy carries {sorted(fields)}; if Claude Code has "
        f"started honouring another field, update this test and the header."
    )


def test_the_two_bodies_are_identical():
    """The load-bearing check. Every test above passes while the guidance itself
    has drifted, which is the failure this file exists to catch."""
    _, claude_body = _parse_frontmatter(_read(CLAUDE_SKILL))
    _, dsh_body = _parse_frontmatter(_read(DSH_SKILL))
    assert claude_body == dsh_body, (
        "The Claude Code and DSH skill bodies differ. They are two copies of one "
        "document -- edit both. First differing line: "
        + _first_difference(claude_body, dsh_body)
    )
