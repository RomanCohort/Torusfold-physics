"""Track the best finished structure the pipeline has written so far.

A run leaves a chain of progressively better PDBs in the output directory, so the
3D view can show real progress instead of staying empty until the end. Something
has to decide which of them is both (a) the furthest along and (b) actually
complete, and that is this.

"(b) complete" is not a formality: `_write_coords_pdb` opens the file and writes
records into it, so a reader can arrive mid-write and see a truncated structure.
A file is only offered once its final record is on disk.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

# Candidates in order of improvement, earliest first. Each entry is
# (level label, path relative to the output directory, what it is, scratch?).
# The label is what the viewer shows, so it names the stage rather than the file.
#
# `scratch` marks a file that is a working artifact of a run in progress rather
# than a result. Scratch files are shown only while a run is going; with no job
# running they are debris, and since they are written last they are also the newest
# files present, so any newest-wins rule hands them the panel. See
# newest_structure()'s `include_scratch`.
#
# Paths are relative because not every structure lands at the top level: Level 1
# writes into vfold3d/ and Level 2 works inside cg2aa/.
#
# This list is what the pipeline actually WRITES, which is fewer stages than it
# computes. Levels 3 and 4 keep their coordinates in memory as arrays and leave no
# file behind, so on a long run the newest displayable structure can be hours old.
# That is a property of the pipeline, not of this list, and inventing a file to
# fill the gap would be showing something that does not exist.
VIEWER_STAGES: List[Tuple[str, str, str, bool]] = [
    ("1", "vfold3d/assembled.pdb", "segmented prediction, assembled", True),
    ("1.5", "level1_5_relaxed.pdb", "coarse-grained, globally relaxed", True),
    # Rewritten every ~12 s from Level 1.5 onward, so the panel has something new to
    # show through the long stretches that write nothing else. It is a coarse-grained
    # trace early on and refined coordinates later, which is why its label is not
    # fixed here: the writer records the stage it was writing from in the sidecar
    # `latest_cg.meta`, and _live_level reads it. Hard-coding a level is what made a
    # run still at Level 2 display "Level 3.5" over its own trace.
    ("live", "latest_cg.pdb", "refining (live)", True),
    # `_final_cg_for_aa.pdb` is written as Level 2's own hand-off file, so it is
    # scratch too: it is superseded by the all-atom structure that follows it.
    ("2", "_final_cg_for_aa.pdb", "folding round complete", True),
    ("2.5", "final_allatom.pdb", "CG to all-atom placement", False),
    ("2.6", "final_allatom_refined.pdb", "PyRosetta refined", False),
    ("5", "level5_cg.pdb", "Amber refinement input", False),
    ("5", "level5_amber.pdb", "Amber14-OL3 refined", False),
    ("5.5", "level5_ppr.pdb", "base-pair repair applied", False),
    ("5.5", "isrnaclong_final.pdb", "final structure", False),
]

# Written by `_write_snapshot` beside `latest_cg.pdb`, holding the stage label for
# whichever coordinates are currently in that file.
_LIVE_META = "latest_cg.meta"

_PDB_COMPLETE_MARKERS = ("END", "ENDMDL")
_state_lock = threading.Lock()
_state: Dict[str, object] = {"signature": None, "result": None, "at": 0.0}


def _abs(output_dir: str, rel: str) -> str:
    """Join a registry path onto the output directory.

    Registry paths are written with forward slashes so they read the same on every
    platform; split and re-join rather than relying on both separators being
    accepted.
    """
    return os.path.join(output_dir, *rel.split("/"))


def _live_level(output_dir: str, fallback: str) -> str:
    """The stage label for `latest_cg.pdb`, from the sidecar the writer leaves.

    That file is rewritten by whichever stage is running, so a label fixed in this
    registry describes it wrongly for most of a run. The writer knows the answer and
    records it; this reads it. Anything unreadable or unexpected falls back to the
    label in the registry, which is honest about not knowing.
    """
    try:
        with open(_abs(output_dir, _LIVE_META), "r", encoding="utf-8",
                  errors="replace") as f:
            text = f.read(64).strip()
    except OSError:
        return fallback
    # Keep it short and plain: this goes straight into the viewer's label.
    if not text or len(text) > 16 or not all(c.isdigit() or c == "." for c in text):
        return fallback
    return text


def _is_complete(path: str) -> bool:
    """Has this PDB finished being written?

    Read the tail rather than the whole file: these reach tens of megabytes and
    this is called on every status poll.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    if size < 64:
        return False
    try:
        with open(path, "rb") as f:
            tail_len = min(512, size)
            f.seek(size - tail_len)
            tail = f.read(tail_len)
    except OSError:
        return False
    text = tail.decode("ascii", "replace")
    return any(m in text for m in _PDB_COMPLETE_MARKERS)


def _atom_count(path: str) -> int:
    """ATOM records in a file, or 0 if it cannot be read.

    Only used to reject a file that parsed as complete but holds nothing, so the
    cost is paid once per changed file rather than per poll.
    """
    n = 0
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if line.startswith("ATOM"):
                    n += 1
    except OSError:
        return 0
    return n


def _residue_count(path: str) -> int:
    """Distinct residues in a PDB, counted from the resSeq field.

    Paired with _atom_count this says whether a file is a coarse-grained trace or
    a full-atom structure, which is a more reliable signal than the filename: the
    filenames differ per level and the stage registry would have to be trusted.
    """
    seen = set()
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if line.startswith("ATOM"):
                    seen.add((line[21:22], line[22:27]))
    except OSError:
        return 0
    return len(seen)


def _digest(path: str) -> str:
    """SHA-1 of a file, or '' if unreadable.

    Stages hand the same coordinates forward: `_final_cg_for_aa.pdb` is written as
    a byte-for-byte copy of what Level 1.5 left behind, so a run can report a new
    stage whose geometry is identical to the last one. The digest lets the viewer
    tell "a new structure arrived" from "the same structure was renamed", which
    are very different things to be looking at.
    """
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def _is_all_atom(stage: Dict) -> bool:
    """Distinguish a full-atom structure from a coarse-grained trace.

    Not from the filename — from the ratio of atoms to residues, because that is
    what actually differs: a CG trace carries one P per residue, an all-atom
    structure around twenty atoms per residue. Anything above four per residue is
    unambiguously all-atom, and the gap between the two is wide enough that the
    threshold is not delicate.
    """
    residues = _residue_count(stage["path"])
    if residues < 1:
        return False
    return stage["atoms"] > 4 * residues


def _pick(candidates: List[Dict]) -> Optional[Dict]:
    """Which structure to show.

    Among files that are all-atom, the most recently written; only if there is no
    all-atom structure does the newest coarse-grained one win. Recency alone is not
    enough: `latest_cg.pdb` is rewritten every 45 seconds throughout refinement and
    would therefore always be the newest file, so a plain newest-first rule would
    replace a finished all-atom structure with a coarse trace and keep it there.
    Quality first, recency within quality.
    """
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda c: (c["mtime"], c["_order"]), reverse=True)
    all_atom = [c for c in ordered if _is_all_atom(c)]
    return (all_atom or ordered)[0]


def newest_structure(output_dir: str, run_started_at: Optional[float] = None,
                     include_scratch: bool = True) -> Optional[Dict]:
    """The furthest-along finished structure in `output_dir`, or None.

    `run_started_at` guards against a stale file from a previous run being shown
    as the current one: a file older than the run that is producing it cannot
    belong to that run.

    `include_scratch` decides whether the run's own working files are eligible.
    They are scratch: `latest_cg.pdb` is rewritten every few seconds and
    `level1_5_relaxed.pdb` holds a coarse-grained trace, and both are only
    meaningful as a picture of a run that is in progress. With no job running they
    are debris, and because they are written last they are also the newest files in
    the directory — so a plain newest-wins rule hands them the panel. That is what
    left the 3D view showing a ten-atom test trace instead of the 2,013 nt model
    after a run was stopped: the debris was newer than the result.
    """
    if not output_dir or not os.path.isdir(output_dir):
        return None

    # Cheap signature so the expensive checks only run when something changed.
    # Keyed on the relative path, not the basename: two stages can share a
    # filename in different subdirectories.
    #
    # `include_scratch` is part of the signature: the same directory yields a
    # different answer with it on and off, and a cached result from the other mode
    # would be served for a run that had started or stopped in between.
    try:
        entries = [("scratch", int(bool(include_scratch)), 0)]
        for _level, rel, _desc, _scratch in VIEWER_STAGES:
            if _scratch and not include_scratch:
                continue
            p = _abs(output_dir, rel)
            if os.path.isfile(p):
                st = os.stat(p)
                entries.append((rel, int(st.st_mtime), st.st_size))
    except OSError:
        return None
    if len(entries) < 2:
        return None
    signature = tuple(entries)

    with _state_lock:
        if _state.get("signature") == signature:
            return _state.get("result")  # type: ignore[return-value]

    # Newest first, not furthest-along first.
    #
    # The registry is ordered by pipeline depth, so the loop below used to keep
    # overwriting its choice with each later stage still present on disk. That is
    # the wrong question while a run is in progress: the stages are not written in
    # one burst, and between two of them nothing is written for a long time (Level
    # 3 and 4 keep their coordinates in memory). Showing the deepest file that
    # exists means the view sits still for that whole stretch, which is what makes
    # it look lagging.
    #
    # The file written most recently is the one that reflects what the run just
    # did, so that is what gets shown. Depth is only a tie-break, for the case of
    # several stages written in the same second.
    candidates = []
    for order, (level, rel, desc, scratch) in enumerate(VIEWER_STAGES):
        if scratch and not include_scratch:
            continue
        path = _abs(output_dir, rel)
        if not os.path.isfile(path):
            continue
        if run_started_at and os.path.getmtime(path) < run_started_at - 5:
            continue
        if not _is_complete(path):
            continue
        atoms = _atom_count(path)
        if atoms < 1:
            continue
        candidates.append({
            "level": (_live_level(output_dir, level) if rel == "latest_cg.pdb"
                      else level),
            "name": rel, "desc": desc, "path": path,
            "atoms": atoms, "bytes": os.path.getsize(path),
            "mtime": os.path.getmtime(path), "_order": order,
            "scratch": scratch,
        })

    best = _pick(candidates)
    if best is not None:
        best.pop("_order", None)

    # Hashed once, for the winner only: only the returned structure is compared.
    if best is not None:
        best["digest"] = _digest(best["path"])

    with _state_lock:
        _state.update({"signature": signature, "result": best, "at": time.time()})
    return best


# The delivered model, shown when no run has produced anything yet.
#
# This is not a pipeline stage: it is the committed 2,013 nt result
# (artifacts/2013nt/isrnaclong_final.pdb, 42,831 atoms), decoded from the shipped
# viewer rather than re-run. It is offered so the panel has something real in it
# on first load instead of an empty box, and it is labelled as delivered so it is
# never mistaken for the output of the run on screen.
DELIVERED_STRUCTURES = [
    {
        "level": "delivered",
        "name": "artifacts/2013nt/isrnaclong_final.pdb",
        "desc": "delivered model, 2013 nt (not from this run)",
    },
]


def delivered_structure(repo_root: str) -> Optional[Dict]:
    """The committed model for the empty-state view, or None if absent.

    Level "delivered" rather than a number on purpose: it did not come from a
    level of the current run, and giving it a number would put it on the progress
    ladder where it does not belong.
    """
    for entry in DELIVERED_STRUCTURES:
        path = os.path.join(repo_root, *entry["name"].split("/"))
        if not os.path.isfile(path):
            continue
        atoms = _atom_count(path)
        if atoms < 1 or not _is_complete(path):
            continue
        return {
            "level": entry["level"], "name": entry["name"], "desc": entry["desc"],
            "path": path, "atoms": atoms, "bytes": os.path.getsize(path),
            "mtime": os.path.getmtime(path), "digest": _digest(path),
            "delivered": True,
        }
    return None
