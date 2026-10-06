"""Commit only this session's work, leaving the concurrent session's tree untouched.

THE PROBLEM. `git status` reports one working tree shared by two sessions.
Nineteen files are this session's and are new, so committing them wholesale is
safe. But `src/torusfold/scheme2/torch_cgsim.py` carries BOTH:

    ~L1399-1431  the other session's PAIR_NN fix (the pair guide read the P beads
                 whose target is an N-bead distance; measured as 100 percent of a
                 2135 kJ/mol gap between 2OIU and the field's own output)
    ~L3475-3505  this session's TriRNASP diagnostic guard (see
                 `docs/immuno_fingerprint_literature.md`)

NOTE ON HOW THE NAMES ABOVE ARE WRITTEN. `tools/check_doc_links.py` matches the
pattern `docs/<path>.<ext>` anywhere in a scanned file and treats every hit as a
link that must resolve -- it cannot tell a link from a mention. Two consequences
this file has to respect, both learned by tripping over them:

  * the archived document this script used to name by full path -- it is the
    basis-family record, now under the archive directory -- is referred to here
    by bare filename only, so the pattern does not fire on it. Every other
    `docs/...` string in this docstring has to resolve, and it does.
  * the parenthetical above avoids writing the archive directory followed by the
    filename for the same reason: that prefix is exactly what the pattern wants.

Committing the file whole would commit their unfinished work under this session's
message. Reverting the file would destroy it. Neither is acceptable.

HOW, AND WHY NOT A PATCH. The first attempt lifted their hunks into a patch with
`git diff -U0` and reverse-applied it. That failed: a zero-context patch does not
carry enough information for `git apply` to place it, even though `--unidiff-zero`
exists for reading such diffs. Discarded.

The approach here rebuilds the two states from known-good text instead:

    lean  = the COMMITTED version of the file + this session's edit, applied as a
            literal string replacement of the exact text changed
    full  = the working-tree version, copied to a backup before anything happens

Commit `lean`, then copy `full` back over it. The verification is that the
restored file's sha256 equals the backup's, which is checked and which the script
acts on if it fails. Their edit is never reconstructed, so it cannot be mangled.

Usage:
    python tools/commit_my_work_only.py --dry-run
    python tools/commit_my_work_only.py --apply
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
MIXED = "src/torusfold/scheme2/torch_cgsim.py"

# This session's edit, as the exact text before and after. Taken from the edit
# that made it; nothing here touches the other session's lines.
MINE_OLD = '''                    # 鈿狅笍 Energy-ratio diagnostics
                    if tri_cg_ratio > 0.1:
                        print(f"    [GPU-2D] 鈿狅笍 Tri energy share too high ({tri_cg_ratio:.1%}), "
                              f"may cause CG-Tri competition! Consider lowering trirnasp_scale")
                    elif tri_cg_ratio < 0.001:
                        print(f"    [GPU-2D] 鈿狅笍 Tri energy share too low ({tri_cg_ratio:.3%}), "
                              f"TriRNASP is nearly ineffective! Consider raising trirnasp_scale")'''

MINE_NEW = '''                    # 鈹€鈹€ Energy-ratio diagnostics 鈹€鈹€
                    #
                    # These have to say WHICH situation they are in, because
                    # Tri/CG = 0 has two very different causes and only one of
                    # them is a problem the reader can act on:
                    #
                    #   tri_pot is None   the term is off, either because
                    #                     use_trirnasp was False or because
                    #                     TriRNASPTorch failed to load and the
                    #                     handler above fell through to None.
                    #                     Tri/CG is 0 BY CONSTRUCTION. Telling
                    #                     the reader to "raise trirnasp_scale"
                    #                     here points them at a knob that cannot
                    #                     take effect.
                    #   tri_pot is not None  the term is on and its share is
                    #                     genuinely wrong, which is what the two
                    #                     thresholds below are for.
                    #
                    # Measured on the 200 nt run of 2026-10-06: the warning
                    # fired on every block while isrnaclong passed
                    # use_trirnasp=False, and "TriRNASP statistical potential
                    # loaded" never appeared in the log -- so the term was never
                    # constructed and the advice was impossible to follow.
                    if tri_pot is None:
                        if rep == 0:
                            reason = ("use_trirnasp=False" if not self.use_trirnasp
                                      else "TriRNASPTorch failed to load; see the "
                                           "'failed to load' line above")
                            print(f"    [GPU-2D] TriRNASP: off ({reason}). "
                                  f"Tri/CG=0 is expected, not a diagnostic.")
                    elif tri_cg_ratio > 0.1:
                        print(f"    [GPU-2D] 鈿狅笍 Tri energy share too high ({tri_cg_ratio:.1%}), "
                              f"may cause CG-Tri competition! Consider lowering trirnasp_scale")
                    elif tri_cg_ratio < 0.001:
                        print(f"    [GPU-2D] 鈿狅笍 Tri energy share too low ({tri_cg_ratio:.3%}), "
                              f"TriRNASP is nearly ineffective! Consider raising trirnasp_scale")'''

MINE = [
    "README.md",
    "tools/check_doc_links.py",
    "tools/commit_my_work_only.py",
    "tools/find_broken_link_source.py",
]

MESSAGE = """Scope the force-field notes to the custom field, and stop the link checker walking into a nested checkout

README: the Force-field constants section now says which of the two refinement paths
it describes, and what the custom Torch field is for. It is a deliberately simplified
coarse-grained first stage, not the accuracy-bearing one -- the pipeline continues to
a ViennaRNA pairing restraint and Amber14-OL3 all-atom refinement at 10.6 A C1'--C1',
which is where pairing geometry is realised. The bounded constants below it say how
well the cheap stage is characterised, which is a different question from whether the
delivered structure is right. Also records the consequence that is easiest to misread:
a CG structure with no pairs at pairing distance can mean the all-atom stage did not
run rather than that the field is broken, and in the 200 nt run of 2026-10-06 both
downstream steps failed for environment reasons.

The section heading changes from "Known limits, stated so they are not rediscovered as
surprises" to "Custom-field tuning: what is settled, what is not", each entry gains a
*Scope:* line, K_PAIR is reclassified from "no measurement" to "criterion, then swept"
-- it is bracketed and swept, and the ranking test covering both 600 and 1500 is flat
in it -- and two tooling items move to the update log.

tools/check_doc_links.py: `_iso/` is a second copy of this repository inside the
working tree, untracked and not gitignored. The checker walked into it and reported its
stale references as broken links here. Added to SKIP_DIRS, with the reason recorded.
tools/find_broken_link_source.py reproduces the checker's scan for one path and prints
what it would blame, which is how the nested copy was identified.

Net effect on the check: 27 resolve / 1 broken, the one being a document that lives
only in another checkout.
"""


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = REPO / MIXED
    if not path.is_file():
        print(f"ABORT: {MIXED} not found")
        return 2

    full_text = path.read_text(encoding="utf-8")
    full_hash = sha(path)

    # Does HEAD already carry this session's edit to the mixed file?
    #
    # Two situations reach this script, and they need different handling:
    #
    #   HEAD does NOT have my edit  -> the mixed file is genuinely mixed and has to
    #                                  be split (this is the case the module
    #                                  docstring describes)
    #   HEAD DOES have my edit      -> the split already happened in an earlier
    #                                  commit, only THEIR delta remains in the
    #                                  working tree, and the file must simply be
    #                                  left alone. Rebuilding from HEAD here would
    #                                  double-apply the change.
    committed = run(["git", "show", f"HEAD:{MIXED}"])
    if committed.returncode != 0:
        print(f"ABORT: git show failed\n{committed.stderr}")
        return 2
    already_split = "TriRNASP: off" in committed.stdout

    if already_split:
        print("HEAD already contains this session's edit to the mixed file.")
        print("Only the other session's delta is in the working tree, so the file")
        print("is excluded from this commit and left untouched.")
        print()
        extra = full_text.count("\n") - committed.stdout.count("\n")
        print(f"  working tree : {full_text.count(chr(10))} lines, sha {full_hash[:16]}")
        print(f"  HEAD         : {committed.stdout.count(chr(10))} lines")
        print(f"  their delta still uncommitted: {extra} lines")
        files = [f for f in MINE if f != MIXED]
        if args.dry_run:
            print()
            print("--- files that would be committed ---")
            modified = set(run(["git", "diff", "--name-only"]).stdout.split())
            for f in files:
                print(f"  {'modified' if f in modified else 'new':<9} {f}")
            print()
            print("[dry run] nothing changed. Pass --apply.")
            return 0

        add = run(["git", "add", "--"] + files)
        if add.returncode != 0:
            print(f"ABORT: git add failed\n{add.stderr}")
            return 2
        msg = REPO.parent / "_commitmsg.txt"
        msg.write_text(MESSAGE, encoding="utf-8", newline="")
        commit = run(["git", "commit", "-F", str(msg)])
        print()
        print((commit.stdout or commit.stderr).strip())
        print()
        print(f"mixed file untouched: sha {sha(path)[:16]} "
              f"(unchanged: {sha(path) == full_hash})")
        return 0 if commit.returncode == 0 else 1

    if MINE_NEW not in full_text:
        print("ABORT: this session's edit is not in the working tree; nothing to do")
        return 2
    if MINE_OLD in full_text:
        print("ABORT: both the old and the new diagnostic text are present -- the "
              "file is not in the state this script expects")
        return 2

    lean = committed.stdout
    if MINE_OLD not in lean:
        print("ABORT: the committed version does not contain the text this session "
              "replaced, so rebuilding from it would not reproduce the change")
        return 2
    lean = lean.replace(MINE_OLD, MINE_NEW)

    # what their edit contributes, for the report
    extra_lines = full_text.count("\n") - lean.count("\n")
    print(f"working tree : {full_text.count(chr(10))} lines, sha {full_hash[:16]}")
    print(f"committed    : {committed.stdout.count(chr(10))} lines")
    print(f"lean (to commit): {lean.count(chr(10))} lines")
    print(f"lines held back for the other session: {extra_lines}")
    print()

    # Their fix has four parts. Check each, because a single marker can be
    # ambiguous: the bare substring `delta_g = pos_nm` also appears as the BPP
    # term, and `E = +K*softplus((r - r0)/w)` appears in an unrelated function at
    # ~L1758. The markers below are the exact lines their diff changed, so each
    # one discriminates.
    THEIR_FIX = {
        "comment header":     "IT READS THE N BEADS",
        "delta_g -> NN(pi)":  "delta_g = pos_nm[:,NN(pi)]-pos_nm[:,NN(pj)]",
        "softplus comment":   "# E = +K*softplus((r - r0)/w), dE/dr = +K*sig/w",
        "force on NN beads":  "total_F.index_add_(1, NN(pi), f_g.squeeze(-1))",
    }
    leaked = []
    print(f"  {'their fix component':<26} {'lean':>8} {'working tree':>13}")
    for label, marker in THEIR_FIX.items():
        in_lean = marker in lean
        in_full = marker in full_text
        print(f"  {label:<26} {'PRESENT' if in_lean else 'absent':>8} "
              f"{'present' if in_full else 'ABSENT':>13}")
        if in_lean:
            leaked.append(label)
    print()
    if leaked:
        print(f"ABORT: their fix would leak into the commit ({leaked}) -- the lean "
              f"rebuild did not remove all of it. Nothing was changed.")
        return 2
    print("  all four of their components are absent from the lean state "
          "and present in the working tree. Split is clean.")
    print()
    print(f"  this session's edit: "
          f"{'present' if 'TriRNASP: off' in lean else 'ABSENT'} in lean, "
          f"{'present' if 'TriRNASP: off' in full_text else 'absent'} in the tree")

    if args.dry_run:
        print()
        print("--- files that would be committed ---")
        modified = set(run(["git", "diff", "--name-only"]).stdout.split())
        for f in MINE:
            print(f"  {'modified' if f in modified else 'new':<9} {f}")
        print()
        print("[dry run] nothing changed. Pass --apply.")
        return 0

    backup = REPO.parent / f"_{path.name}.precommit_backup"
    shutil.copy2(path, backup)
    print(f"\nbackup: {backup}")

    try:
        path.write_text(lean, encoding="utf-8", newline="")
        add = run(["git", "add", "--"] + MINE)
        if add.returncode != 0:
            raise RuntimeError(f"git add failed: {add.stderr}")

        msg = REPO.parent / "_commitmsg.txt"
        msg.write_text(MESSAGE, encoding="utf-8", newline="")
        commit = run(["git", "commit", "-F", str(msg)])
        print()
        print((commit.stdout or commit.stderr).strip())
        if commit.returncode != 0:
            raise RuntimeError("commit failed")
    finally:
        shutil.copy2(backup, path)
        restored = sha(path)
        print()
        print(f"restored working tree: sha {restored[:16]}")
        print(f"matches the pre-commit state: {restored == full_hash}")
        if restored != full_hash:
            print("  !! MISMATCH -- the backup is at", backup)
            return 1
        print("the other session's working tree is byte-identical to before.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
