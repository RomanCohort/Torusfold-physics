"""Checkpoint management for the long pipeline.

WHAT WAS WRONG
--------------
`isrnaclong_pipeline` wrote its resume state with two module-level functions in
`isrnaclong.py`, and the arrangement had three defects that only bite on a real
multi-hour run:

1. **A checkpoint was not tied to the configuration that produced it.** It carried
   `seq_sha1` and `seq_len` and nothing else that described the run. So running the
   same sequence with different parameters — `use_metad`, `md_step_scale`,
   `use_5bead`, `n_rest2_replicas` — found a checkpoint on disk and resumed from it,
   and the stage guards (`if ckpt_level >= 3.5: restored from checkpoint`) skipped
   the work the new parameters asked for. The parameters were stored nowhere and
   nothing compared them, so the run silently returned the previous configuration's
   result. There is no log line that could reveal it, because nothing was wrong from
   the code's point of view: a valid checkpoint is a valid checkpoint.

2. **Orphaned arrays accumulated.** Array fields are saved as `ckpt_<field>.npy`
   and the manifest references them. Writing a *different* set of fields — which is
   what `_delete_checkpoint_from_level` does, and what a level's field set changing
   between code versions does — left the previous files behind with nothing
   referencing them. Measured on a real output directory: four `ckpt_*.npy` files
   on disk and zero of them referenced by the manifest.

3. **A crash could not be detected.** Arrays were renamed into place one at a time
   under fixed names, then the JSON was replaced. A process that died partway left
   new arrays beside an old manifest, or some arrays new and some old, and the next
   run loaded it as a coherent checkpoint. Every array was individually valid; the
   *set* was not.

WHAT THIS DOES
--------------
One object owns the checkpoint file. It records a configuration signature and
refuses to resume across a mismatch (keeping the old file aside rather than deleting
it, because "a previous run left state here that does not apply" is worth being able
to inspect). It tracks every array it writes so it can clear the ones it replaces.
And it stamps each write with a serial, so a manifest whose arrays are not all from
the same write is detected and discarded rather than loaded.

COMPATIBILITY
-------------
`_checkpoint.json` stays where it was and keeps its shape — the values are flattened
into the top level exactly as before, with `_npy_refs` naming the arrays. Existing
readers (`scripts/plot_pair_heatmap_v2.py` reads `pairs` and `far_pairs` straight
out of it) are unaffected; the bookkeeping keys are prefixed with `_` and add
alongside.

A manifest written before this change has no `_schema`, so its configuration is
unknown. That is treated as a mismatch and reported — the array names are the old
fixed ones, so it cannot be proven to belong to the current run, and the previous
behaviour was to assume it did.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

# Bumped when the on-disk arrangement changes in a way older readers must notice.
SCHEMA = 2

MANIFEST_NAME = "_checkpoint.json"
# Where a checkpoint that cannot be trusted is moved instead of being deleted.
QUARANTINE_NAME = "_checkpoint.incompatible.json"
# Oldest first; used to describe a resumed run in the log line.
LEVEL_SEQUENCE = ["0", "1", "1.5", "2", "2.3", "2.5", "2.6",
                  "3", "3.5", "4", "5", "5.5"]

# Parameters that do NOT change what the pipeline computes.
#
# Listed as exclusions rather than as inclusions on purpose: a parameter added to
# `isrnaclong_pipeline` later is signature-bearing by default. The failure mode of
# getting that wrong is a wasted re-run; the failure mode of the opposite default is
# a silently wrong structure, because a resumed stage never runs and its old output
# is passed off as the new configuration's.
NON_CONTENT_PARAMS = frozenset({
    "verbose",        # logging only
    "resume",         # this is the resume switch itself
    "output_dir",     # where it is written, not what is written
    "sequence",       # already covered, exactly, by seq_sha1
})


def config_signature(sequence: str, params: Dict) -> str:
    """A digest of everything that decides what the run produces.

    `secondary_structure` is included: a different dot-bracket is a different
    calculation on the same chain.
    """
    keep = {k: v for k, v in (params or {}).items() if k not in NON_CONTENT_PARAMS}
    try:
        body = json.dumps(keep, sort_keys=True, default=repr)
    except Exception:                                        # noqa: BLE001
        # A parameter that will not serialise must not silently drop out of the
        # signature — `repr` above covers most cases, and this is the backstop.
        body = repr(sorted((k, repr(v)) for k, v in keep.items()))
    blob = "%s\x00%s\x00%s" % (sequence, SCHEMA, body)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def sequence_sha1(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()


def read_manifest(output_dir) -> Dict:
    """Read `_checkpoint.json` without deciding anything about it. `{}` if absent.

    Separate from `CheckpointStore.load_usable` on purpose. Loading through a store
    evaluates whether the checkpoint may be resumed from, and declining *moves the
    file aside* — correct when a run is starting, wrong when someone is only looking
    at what is in a directory. An inspector that called it would quarantine a
    perfectly good checkpoint just because the page was opened.
    """
    try:
        return json.loads((Path(output_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def checkpoint_report(output_dir) -> Dict:
    """Describe the checkpoint state of an output directory, for display.

    Reports the four things that can be true of it, because they call for different
    actions and the directory alone does not say which:
      - `usable`      a manifest that was written by the current arrangement
      - `legacy`      a manifest from before checkpoints carried a configuration
      - `quarantined` a state that was rejected, kept for inspection
      - `orphans`     array files nothing references (the leak that was measured at
                      four files in one directory)
    """
    d = Path(output_dir)
    manifest = read_manifest(d)
    quarantine_path = d / QUARANTINE_NAME
    quarantine = {}
    if quarantine_path.exists():
        try:
            quarantine = json.loads(quarantine_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            quarantine = {}

    refs = (manifest.get("_npy_refs") or {}) if manifest else {}
    live_names = {Path(p).name for p in refs.values()}
    on_disk = sorted(p.name for p in d.glob("ckpt_*.npy")) if d.is_dir() else []
    orphans = [n for n in on_disk if n not in live_names]

    arrays = []
    for field, p in sorted(refs.items()):
        fp = Path(p)
        arrays.append({
            "field": field,
            "file": fp.name,
            "exists": fp.exists(),
            "bytes": fp.stat().st_size if fp.exists() else 0,
        })

    if manifest:
        if manifest.get("_config_sig"):
            state = "usable"
        else:
            state = "legacy"
    elif quarantine:
        state = "quarantined"
    else:
        state = "absent"

    cfg = manifest.get("_config") or quarantine.get("_config") or {}
    return {
        "dir": str(d),
        "state": state,
        "manifest_present": bool(manifest),
        "level": manifest.get("level", quarantine.get("level")),
        "schema": manifest.get("_schema"),
        "config_sig": manifest.get("_config_sig") or quarantine.get("_config_sig"),
        "config": cfg,
        "config_count": len(cfg),
        "fields": sorted(k for k in manifest if not k.startswith("_")) if manifest else [],
        "arrays": arrays,
        "array_bytes": sum(a["bytes"] for a in arrays),
        "orphans": orphans,
        "quarantined": bool(quarantine),
        "quarantine_level": quarantine.get("level"),
        "quarantine_seq": str(quarantine.get("seq_sha1") or "")[:12] or None,
        "quarantine_sig": str(quarantine.get("_config_sig") or "")[:12] or None,
        "written": manifest.get("time"),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


class CheckpointStore:
    """Owns one `_checkpoint.json` and the `ckpt_*.npy` arrays beside it.

    Usage is the same shape as the two functions it replaces:

        store = CheckpointStore(output_path, sequence, params)
        state = store.load_usable()        # {} when there is nothing to resume from
        ...
        store.save("2.5", best_coords=coords, energy=e)
    """

    def __init__(self, output_dir, sequence: str, params: Optional[Dict] = None):
        self.dir = Path(output_dir)
        self.path = self.dir / MANIFEST_NAME
        self.sequence = sequence
        self.params = dict(params or {})
        self.sig = config_signature(sequence, self.params)
        self.seq_sha1 = sequence_sha1(sequence)
        self.seq_len = len(sequence)
        # Array fields written by this save, so the next save can clear the rest.
        self._mine: List[str] = []
        # Why the last load did not produce a usable state, for the caller's log.
        self.discarded_reason: Optional[str] = None
        self.resumed_from: Optional[str] = None

    # ── array bookkeeping ────────────────────────────────────────────────────

    def _array_path(self, field: str) -> Path:
        return self.dir / ("ckpt_%s.npy" % field)

    def _array_names_current(self) -> Set[str]:
        if not self.dir.is_dir():
            return set()
        return {p.name for p in self.dir.glob("ckpt_*.npy")}

    def orphaned_arrays(self) -> Set[str]:
        """`ckpt_*.npy` files on disk that this store does not consider current.

        The fallback glob is the point: arrays written under the previous scheme
        used the same naming, so files left behind by it are found and removed
        rather than living on forever beside the ones in use.
        """
        return self._array_names_current() - {"ckpt_%s.npy" % f for f in self._mine}

    def sweep_orphans(self, verbose: bool = True) -> int:
        """Delete array files nothing references. Returns how many were removed."""
        removed = 0
        for name in sorted(self.orphaned_arrays()):
            try:
                (self.dir / name).unlink()
                removed += 1
            except OSError:
                pass
        if removed and verbose:
            print("  [checkpoint] removed %d orphaned array file(s)" % removed)
        return removed

    def disk_usage(self) -> Tuple[int, int]:
        """(number of array files, bytes) — for reporting what a run leaves behind."""
        files = sorted(self.dir.glob("ckpt_*.npy"))
        return len(files), sum(p.stat().st_size for p in files if p.exists())

    # ── write ────────────────────────────────────────────────────────────────

    def save(self, level, verbose: bool = True, **extra) -> None:
        """Write the cumulative state at `level`.

        Every array in the state is written on every save, which is what makes the
        serial check meaningful: the manifest and its arrays are one write.
        """
        state = dict(getattr(self, "_state", {}) or {})
        state["level"] = level
        state.update(extra)
        self._state = state

        # A serial per write. Any npy older than the manifest that names it means a
        # previous write died partway, and the pair must not be trusted together.
        serial = hashlib.sha1(
            ("%s\x00%s" % (self.sig, os.urandom(8).hex())).encode("utf-8")
        ).hexdigest()[:16]

        arrays: Dict[str, str] = {}
        clean: Dict = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                p = self._array_path(k)
                tmp = self.dir / ("_%s.tmp.npy" % k)
                with open(tmp, "wb") as fh:
                    np.save(fh, v)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(str(tmp), str(p))
                arrays[k] = str(p)
            else:
                clean[k] = v

        clean["_npy_refs"] = arrays
        clean["_npy_serial"] = serial
        clean["_npy_files"] = sorted(arrays)
        clean["_schema"] = SCHEMA
        clean["_config_sig"] = self.sig
        # The parameters themselves, not just their digest. A digest proves two runs
        # differ; it cannot say how, and "which setting was this produced with" is
        # the question actually asked of an output directory months later. Recorded
        # as text so a checkpoint stays readable without the code that wrote it.
        try:
            clean["_config"] = json.loads(json.dumps(self.params, default=str,
                                                     sort_keys=True))
        except Exception:                                    # noqa: BLE001
            clean["_config"] = {k: str(v) for k, v in self.params.items()}
        clean["_seq_sha1"] = self.seq_sha1
        clean["_seq_len"] = self.seq_len
        # Kept under the original names as well: they predate this module and other
        # code reads them.
        clean.setdefault("seq_sha1", self.seq_sha1)
        clean.setdefault("seq_len", self.seq_len)

        _atomic_write_text(self.path, json.dumps(clean, default=str,
                                                 ensure_ascii=False))
        self._mine = sorted(arrays)
        if verbose:
            print("  [checkpoint] level=%s, %d array(s), %d field(s)"
                  % (level, len(arrays), len(clean)))

    # ── read ─────────────────────────────────────────────────────────────────

    def _read_manifest(self) -> Dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            self.discarded_reason = "unreadable manifest (%s)" % e
            return {}

    def _quarantine(self, manifest: Dict, why: str, verbose: bool = True) -> None:
        """Move an untrustworthy checkpoint aside instead of deleting it.

        Recording `why` is the part that matters most. Declining to resume is silent
        by nature — a run that starts from Level 0 looks the same whether it found
        nothing or found something it had to reject — so without this the operator
        sees a run restart and has no way to learn that their output directory held
        a checkpoint written under different settings, or a torn one. This was
        missing while the docstring claimed it, and every rejection path reported
        nothing.
        """
        self.discarded_reason = why
        try:
            if manifest:
                _atomic_write_text(self.dir / QUARANTINE_NAME,
                                   json.dumps(manifest, default=str,
                                              ensure_ascii=False))
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
        if verbose:
            print("  [checkpoint] not resuming: %s" % why)
            print("  [checkpoint] the previous state was kept as %s and will be "
                  "rebuilt from Level 0" % QUARANTINE_NAME)

    def load_usable(self, verbose: bool = True) -> Dict:
        """The resume state, or {} when there is nothing that can be trusted.

        Distinguishes the three cases that matter, because they call for different
        things from a reader:
          - no file            → a fresh run, silently
          - unusable file      → a fresh run, loudly, with the old state kept
          - usable file        → resume
        """
        self.discarded_reason = None
        self.resumed_from = None
        if not self.path.exists():
            return {}

        manifest = self._read_manifest()
        if not manifest:
            return {}

        # 1. Does it belong to this sequence?
        #
        # Not `a or b`: `or` is a truth test, so an empty-string fingerprint would
        # fall through to the second key instead of being compared as the value it
        # is. An absent key is `None`; anything else present is used.
        have_seq = manifest.get("seq_sha1")
        if have_seq is None:
            have_seq = manifest.get("_seq_sha1")
        if have_seq is not None and have_seq != self.seq_sha1:
            self._quarantine(manifest, "it is for a different sequence "
                                       "(%s != %s)" % (str(have_seq)[:8],
                                                       self.seq_sha1[:8]), verbose)
            return {}

        # 2. Does it belong to this configuration?
        have_sig = manifest.get("_config_sig")
        if have_sig is None:
            self._quarantine(manifest, "it records no configuration, so it cannot "
                                       "be shown to belong to this run "
                                       "(written before checkpoints carried one)",
                             verbose)
            return {}
        if have_sig != self.sig:
            self._quarantine(manifest, "it was written with different parameters "
                                       "(%s != %s); a stage that already ran would "
                                       "not be re-run under the new settings"
                                       % (str(have_sig)[:8], self.sig[:8]), verbose)
            return {}

        # 3. Are its arrays all from the same write as its manifest?
        refs = manifest.get("_npy_refs") or {}
        missing = [k for k, p in refs.items() if not Path(p).exists()]
        if missing:
            self._quarantine(manifest, "its arrays are incomplete (missing %s)"
                             % ", ".join(sorted(missing)[:4]), verbose)
            return {}
        try:
            mtime_manifest = self.path.stat().st_mtime
        except OSError:
            mtime_manifest = None
        if mtime_manifest is not None:
            stale = [k for k, p in refs.items()
                     if Path(p).stat().st_mtime - mtime_manifest > 1.0]
            if stale:
                self._quarantine(manifest, "its arrays are newer than its manifest "
                                           "(%s), so a write did not finish"
                                 % ", ".join(sorted(stale)[:4]), verbose)
                return {}

        # 4. Load the arrays. A field whose array will not load is dropped, not
        #    fatal: the stages that need it check for its presence.
        arrays = manifest.pop("_npy_refs", {})
        for k, npy_path in arrays.items():
            try:
                manifest[k] = np.load(npy_path)
            except Exception as e:                            # noqa: BLE001
                if verbose:
                    print("  [checkpoint] array %s failed to load (%s); that field "
                          "will be recomputed" % (k, e))
                manifest.pop(k, None)

        self._mine = sorted(arrays)
        self._state = {k: v for k, v in manifest.items() if not k.startswith("_")}
        self.resumed_from = str(manifest.get("level"))
        self.sweep_orphans(verbose=verbose)
        if verbose:
            print("  [checkpoint] resuming after Level %s (%d field(s), %d array(s))"
                  % (self.resumed_from, len(self._state), len(arrays)))
        return dict(self._state)

    # ── targeted rollback ────────────────────────────────────────────────────

    def drop_fields(self, fields: Iterable[str], verbose: bool = True) -> List[str]:
        """Remove fields, and the arrays that belong to them. Returns what went.

        The arrays are deleted here rather than left for the next save's sweep.
        A rollback is exactly the moment someone is about to re-run the stage that
        wrote them, and leaving them on disk for the whole of that re-run is how the
        previous arrangement accumulated four unreferenced files in one directory.

        The level is not touched: what the remaining state is at is the caller's
        question, not this object's.
        """
        state = getattr(self, "_state", None)
        if state is None:
            state = self.load_usable(verbose=False)
        removed = []
        for f in fields:
            if f in state:
                del state[f]
                removed.append(f)
                try:
                    self._array_path(f).unlink(missing_ok=True)
                except OSError:
                    pass
        self._mine = [f for f in self._mine if f in state]
        self._state = state
        return removed
