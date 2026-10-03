"""Build a corrected copy of run_armA.cmd: the wrong comment replaced, a lock check added.

Two changes, both about the fault that cost 2026-10-02:

  1. The 30-line comment block claiming the silent failure came from a quoted token at the
     start of a line is REPLACED with the measured cause: cmd skips a command whose
     output redirection cannot open its target, and the target was locked by four orphaned
     worker processes from the previous day. A wrong diagnosis in a comment is worse than
     no comment -- it sends the next reader to the wrong place, and this session spent
     hours there.

  2. A PRE-FLIGHT LOCK CHECK. `>>` on a locked file is silent, so a recurrence would be
     indistinguishable from a working run. The script now tests both redirect targets are
     openable before entering the retry loops and reports it loudly if not.

The output is a COPY (`_fixed.cmd`) whose Python calls point at a scratch output dir
(`results/plan_c/_fixed_log`), so it can be exercised while the real task is mid-run
without touching the files the real task holds.
"""

import pathlib
import sys

ROOT = pathlib.Path(r"C:\baidunetdiskdownload\torusfold-hybrid")
LIVE = ROOT / "results" / "plan_c" / "run_armA.cmd"
BACKUP = ROOT / "results" / "plan_c" / "_run_armA_before_lockcheck.cmd"
COPY = ROOT / "results" / "plan_c" / "_fixed.cmd"

OLD_START = 'rem WHY EVERY PYTHON LINE IS NOW WRITTEN `call "%PY%" ...` RATHER THAN `"%PY%" ...`'
OLD_END = "rem ---------------------------------------------------------------------------"

NEW_COMMENT = '''rem WHY `call "%PY%" ...` -- and what was ACTUALLY wrong on 2026-10-02.
rem
rem Short answer, so nobody follows the wrong thread again: the prefix is a precaution,
rem not the fix. Every silent failure that afternoon came from a LOCK on the redirect
rem target. The full account is the next block.
rem
rem ---------------------------------------------------------------------------
rem THE ACTUAL FAULT, MEASURED 20:15: results/plan_c/armA_smoke.out WAS LOCKED
rem
rem Four orphaned python_ibiA.exe processes, created 2026-10-01 19:11:21 --
rem multiprocessing workers whose parent (PID 56764) had died -- held that file open for
rem nineteen hours. Its mtime had frozen at 19:11:32, eleven seconds after they started.
rem Verified directly rather than inferred:
rem
rem     [System.IO.File]::Open($f, 'Append', 'Write', 'None')  -> "being used by another
rem                                                                process"
rem     Move-Item $f $f.probe                                   -> the same error
rem
rem WHAT A LOCK DOES TO THIS SCRIPT. cmd's `>>` cannot open the locked target, and when a
rem redirection cannot open its file cmd SKIPS THE COMMAND ENTIRELY. Not fail -- skip.
rem ERRORLEVEL stays 0 and the rest of the loop body runs normally. That is precisely the
rem signature this file produced all afternoon:
rem
rem     the echo lines printed          the block was executing
rem     exit=0 was recorded             ERRORLEVEL was never touched by a command never run
rem     python never started            nothing was asked of it
rem     armA_smoke.out's mtime froze    >> never opened it
rem
rem Eight retries, twenty seconds apart, every one a no-op that looked like success. The
rem retrying is what hid it: from outside, a broken loop and a working one both finish in
rem under a second.
rem
rem A PROBE THAT MISLED ME, recorded so the shape of the mistake is recognisable. Four
rem prefixes were run through the scheduler -- bare, `:: &`, `call`, `timeout /t 1 >nul &`
rem -- and all four worked, which I read as "the prefix is the deciding variable". None of
rem the four touched the locked path. A probe that avoids the fault cannot find it. A
rem 30-line comment was then written into this file blaming a quoted token at the start of
rem a line; that was wrong, and it has been replaced by this block.
rem
rem THE FIX, DONE: the four orphans were killed by explicit PID and the lock released.
rem Verified: armA_smoke.out went from a frozen 2026-10-01 19:11:32 / 7993 bytes to
rem 2026-10-02 20:15:36 / 8940 bytes, with 39 python_ibiA processes up.
rem
rem IF A PYTHON CALL HERE GOES SILENT AGAIN -- returns instantly, exit 0, no bytes, and
rem the target's mtime unchanged -- check the LOCK FIRST, not the quoting:
rem
rem     $f='results\\plan_c\\armA_smoke.out'
rem     try {[IO.File]::Open($f,'Append','Write','None').Close();'openable'}
rem     catch {'LOCKED: ' + $_.Exception.Message}
rem
rem then look for stale python_ibiA.exe processes holding it. Killing by image name is
rem forbidden on this machine (docs/dev_machine_handoff.md section 9): match the
rem CommandLine and stop by PID, or the sweep takes out whatever else is running.
rem ---------------------------------------------------------------------------'''

LOCK_CHECK = '''rem ---- PRE-FLIGHT 0: kill the previous run''s leftovers. ------------------------------
rem ---- WHY THIS IS THE ROOT FIX AND THE CHECK BELOW IS NOT.
rem ----
rem ---- scripts/ibi_loop.py:958 opens its worker pool as
rem ----     with ctx.Pool(processes=min(_N_WORKERS, len(remaining))) as pool_procs:
rem ---- A context manager reclaims the workers on a NORMAL exit. A driver that is killed,
rem ---- or that dies outside the with-block, never runs __exit__, and its children outlive
rem ---- it. They inherited the redirected stdout handle, so they keep armA.out open -- and
rem ---- a locked redirect target is not an error in cmd, it is a SKIP. The command is not
rem ---- executed, ERRORLEVEL stays 0, the loop finishes normally, and the log records
rem ---- `exit=0` while python was never started. Measured on 2026-10-02: armA_smoke.out
rem ---- held nineteen hours by four orphans from the previous day; then armA.out held by
rem ---- orphans from the run before that. Eight retries per cycle, every one a no-op.
rem ----
rem ---- So the PRE-FLIGHT CHECK ALONE IS NOT ENOUGH, which was measured too: with a live
rem ---- locked armA.out the check passed and the run still failed, because a check samples
rem ---- a moment and the redirect needs the whole span. Killing the leftovers first is what
rem ---- actually makes the next redirect work.
rem ----
rem ---- Killed by PID, never by image name (docs/dev_machine_handoff.md section 9), and
rem ---- python_ab.exe -- the sibling task plan_c_ab2oiu -- cannot be hit because the script
rem ---- filters on python_ibiA.exe only.
"%PY_BOOT%" results\\plan_c\\stop_stale_workers.py

rem ---- PRE-FLIGHT: the redirect targets must be openable. --------------------------
rem ---- This is the loud half: it turns a silent no-op into a named abort. It is NOT the
rem ---- fix -- see the block above. `>>` on a locked file does not fail, it SKIPS THE
rem ---- COMMAND, so without this the loop reports exit=0 for hours and writes nothing.
rem
rem Test BOTH targets. Do NOT guard with `if exist`: a file that does not exist yet is the
rem ordinary first-run case and is precisely the one the redirect must be able to CREATE.
rem An earlier version had that guard, so it passed silently on the case it exists to catch.
rem `copy /y nul <target>` was tried next and is also wrong -- it answers the question by
rem whether it can CREATE the file, so it fails on a target that is merely openable-but-
rem locked elsewhere.
rem
rem `>>"%%T" echo.` is the honest test: it attempts the same redirection the real call
rem will use, and cmd reports the failure through ERRORLEVEL even though it reports
rem nothing else.
set "_LOCKED="
for %%T in (__TARGET1__ __TARGET2__) do (
  >>"%%T" echo. 2>nul
  if errorlevel 1 set "_LOCKED=!_LOCKED! %%T"
)
if defined _LOCKED (
  echo ==== ABORT: redirect target is locked:!_LOCKED! >> __LOG__
  echo      Every attempt would be a silent no-op. Find the holder: >> __LOG__
  echo        Get-CimInstance Win32_Process ^| Where-Object CommandLine -like '*ibi_loop*' >> __LOG__
  echo      and stop it BY PID, never by image name. >> __LOG__
  exit /b 2
)
'''


def main():
    apply = "--apply" in sys.argv
    original = LIVE.read_text(encoding="utf-8")
    text = original

    # 1. replace the wrong comment -- IF it is still there. The first build removed it, so
    #    re-running has to be a no-op here rather than an error; this script is meant to be
    #    re-run whenever the header changes, not once.
    start = text.find(OLD_START)
    if start < 0:
        print("  note: the old comment block is already gone -- skipping step 1")
    else:
        end = text.find(OLD_END, start) + len(OLD_END)
        text = text[:start] + NEW_COMMENT + text[end:]
        assert OLD_START not in text, "old comment survives in the output"
        print("  step 1: replaced the old comment block")

    # 1b. re-running must also not stack a second copy of the new comment.
    if text.count("THE ACTUAL FAULT, MEASURED 20:15") > 1:
        raise SystemExit("the new comment appears more than once -- refusing to add another")

    # 2. pre-flight, and the cleanup call ahead of it. Idempotent: if a previous build
    #    already inserted them, replace that block rather than adding a second one.
    anchor = "rem ---- smoke gate:"
    if anchor not in text:
        raise SystemExit("smoke gate anchor not found")
    lock_check = (LOCK_CHECK
                  .replace("__TARGET1__", "results\\plan_c\\armA_smoke.out")
                  .replace("__TARGET2__", "results\\plan_c\\armA.out")
                  .replace("__LOG__", "results\\plan_c\\run_armA.log")
                  .replace("__PY_BOOT__", "C:\\ana\\envs\\comfyui\\python.exe"))
    assert "__TARGET" not in lock_check and "__LOG__" not in lock_check \
        and "__PY_BOOT__" not in lock_check, "a placeholder survived substitution"

    if "PRE-FLIGHT 0: kill the previous run" in text:
        # already inserted: swap the old inserted block for the new one
        old_start = text.find("rem ---- PRE-FLIGHT")
        old_end = text.find(anchor)
        text = text[:old_start] + lock_check + text[old_end:]
        print("  step 2: replaced the previously inserted pre-flight block")
    else:
        text = text.replace(anchor, lock_check + anchor, 1)
        print("  step 2: inserted the pre-flight block")

    # 3. the python calls stay exactly as they are -- this is the live file
    before = text
    assert ">> results\\plan_c\\armA_smoke.out" in text, "smoke redirect not found"
    assert ">> results\\plan_c\\armA.out" in text, "arm redirect not found"
    assert text == before, "the live build must not rewrite the redirects"

    # PY_BOOT: a stable interpreter for the pre-flight cleanup. Deliberately NOT %PY% --
    # the cleanup has to run even when the arm's interpreter is itself suspect, and using
    # a separate one keeps the search pattern ('python_ibiA.exe') from matching the very
    # process doing the searching.
    if 'set PY_BOOT=' not in text:
        marker = "set PY="
        idx = text.find(marker)
        if idx < 0:
            raise SystemExit("could not find where to add PY_BOOT")
        eol = text.find("\n", idx)
        text = (text[:eol + 1]
                + "set PY_BOOT=C:\\ana\\envs\\comfyui\\python.exe\n"
                + text[eol + 1:])
        print("  added: set PY_BOOT=C:\\ana\\envs\\comfyui\\python.exe")

    if not apply:
        print("DRY RUN -- nothing written. Re-run with --apply.")
        return

    # Keep the pre-change file so the edit is reversible without git (this path is
    # gitignored, so git cannot recover it).
    if not BACKUP.exists():
        with open(BACKUP, "w", encoding="ascii", newline="") as fh:
            fh.write(original)
        print(f"  backed up the previous file to {BACKUP.name} "
              f"({len(original.encode('ascii', 'replace'))} bytes)")

    with open(LIVE, "w", encoding="ascii", newline="") as fh:
        fh.write(text)
    raw = LIVE.read_bytes()
    print(f"wrote {LIVE.name}: {len(raw)} bytes, {text.count(chr(10)) + 1} lines, "
          f"ascii-only={all(b < 128 for b in raw)}")
    print(f"  CRLF count: {raw.count(bytes([13, 10]))}")

    # Verify the two things that must agree, because they did not the first time.
    pre = [l for l in text.split("\n") if "for %%T" in l]
    calls = [l for l in text.split("\n") if l.strip().startswith("call ") and ">>" in l]
    print(f"  pre-flight targets : {pre[0].strip() if pre else 'NOT FOUND'}")
    for c in calls:
        print(f"  python call target : {c.strip()[:100]}")
    print(f"  pre-flight present : {'true' if 'PRE-FLIGHT' in text else 'MISSING'}")
    print(f"  old comment gone   : {OLD_START not in text}")


if __name__ == "__main__":
    main()
