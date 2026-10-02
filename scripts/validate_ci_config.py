"""Validate both CI configurations before they reach a remote.


A pipeline file that does not parse fails silently from here -- GitLab shows a config
error on the site and GitHub shows nothing at all until someone opens the Actions tab.
Neither is visible from the machine that wrote the file, so it is checked locally.

The `on:` key is read back as boolean True by PyYAML because YAML 1.1 treats bare
`on`/`off`/`yes`/`no` as booleans. GitHub's own parser does not, and every Actions
workflow in existence writes `on:`, so this is expected rather than a defect -- but it
is worth asserting explicitly so a future reader does not "fix" it into `"on":`.
"""
import pathlib
import sys

import yaml

FAILED = False


def check(path, expect_keys, required_jobs=None):
    global FAILED
    p = pathlib.Path(path)
    if not p.is_file():
        print(f"{path}: MISSING")
        FAILED = True
        return
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"{path}: DOES NOT PARSE -> {type(exc).__name__}: {exc}")
        FAILED = True
        return

    print(f"{path}: parses")
    keys = [("on" if k is True else k) for k in doc]
    print(f"  top-level keys : {keys}")
    missing = [k for k in expect_keys if k not in keys]
    if missing:
        print(f"  MISSING KEYS   : {missing}")
        FAILED = True

    jobs = doc.get("jobs") or {}
    for name in jobs:
        job = jobs[name]
        steps = job.get("steps", [])
        print(f"  job {name!r}: {len(steps)} steps, runs-on={job.get('runs-on')}")
        bad = [i for i, s in enumerate(steps) if not (("run" in s) or ("uses" in s))]
        if bad:
            print(f"    STEPS WITH NOTHING TO DO at indices {bad}")
            FAILED = True

    if required_jobs:
        for name in required_jobs:
            if name not in jobs:
                print(f"  MISSING JOB    : {name}")
                FAILED = True

    # A `script:` (GitLab) must be a list of non-empty strings; a single string is a
    # common mistake that GitLab rejects at runtime rather than at parse time.
    for key, val in doc.items():
        if isinstance(val, dict) and "script" in val:
            s = val["script"]
            if not isinstance(s, list) or not all(isinstance(x, str) and x.strip() for x in s):
                print(f"  job {key!r}: `script` must be a list of non-empty strings")
                FAILED = True


print("=== GitLab (.gitlab-ci.yml) ===")
check(".gitlab-ci.yml", expect_keys=["stages", "test", "headline"], required_jobs=[])

print("\n=== GitHub (.github/workflows/ci.yml) ===")
check(".github/workflows/ci.yml", expect_keys=["on", "jobs"], required_jobs=["test", "headline"])

# The `on:` footgun, asserted rather than left to a comment.
gh = yaml.safe_load(pathlib.Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
if True not in gh:
    print("\nNOTE: the trigger key is not the bare `on:` boolean form; GitHub requires "
          "either `on:` or a quoted \"on\":. Both work, but confirm you meant it.")
print("\n" + ("FAILED" if FAILED else "both CI files are structurally valid"))
sys.exit(1 if FAILED else 0)
