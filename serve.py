# -*- coding: utf-8 -*-
"""TorusFold Web Server — SSE streaming + Predict API + static files

Usage: python serve.py [port]
Default port: 8877
"""
import os
import sys
import io
import json
import re
import time
import uuid
import hashlib
import email.utils
import threading
import tempfile
import numpy as np
from http.server import HTTPServer, ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

# Changes on every process start. The browser compares it against the value it
# remembered, so "your job is gone because the server restarted" is a different
# message from "nothing was running".
_SERVER_GENERATION = "%d-%d" % (os.getpid(), int(time.time()))

# The console streams a prediction replaces sys.stderr with. Kept so the HTTP
# access log can keep going to the real one instead of into the job log.
_PROCESS_STDERR = sys.stderr

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
WEB_DIR = os.path.join(SRC, "torusfold", "web")

# Publish .env.local into this process, so the pipeline can see it.
#
# The setup button and `configure_deps.py write` record where the external tools
# and the interpreter live, and `start.bat` loads that file before launching the
# server. Starting serve.py directly bypasses that, and the consequence is silent
# rather than loud: the wrappers read their ROOT variables at import time, so with
# nothing set they build relative paths and every predictor fails with
# "checkpoint not found" — while the files are sitting exactly where the setup
# recorded them.
#
# The reader lives in tools/_env.py because the dependency report needs the same
# behaviour, and two copies of this had already drifted apart.
def _load_env_local(path):
    script_dir = os.path.join(ROOT, "tools")
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        import _env
        return _env.apply(ROOT)
    except ImportError:
        return []


_APPLIED_ENV = _load_env_local(os.path.join(ROOT, ".env.local"))

# Optional: env TF_SCHEME2_SRC points to an additional scheme2 source directory.
# This repo's src/ already contains most scheme2 modules; inject an extra path only when needed.
SCHEME2_SRC = os.environ.get("TF_SCHEME2_SRC", "")

# ── SSE log buffer (thread-safe) ────────────────────────────────
_log_entries = []
_log_lock = threading.Lock()

# ── PDB session storage ─────────────────────────────────────────
_pdb_sessions = {}  # session_id → {pdb_text, pdb_path}

def _emit_log(level, message):
    """Thread-safe log emission for SSE streaming."""
    with _log_lock:
        _log_entries.append({
            "timestamp": time.time(),
            "level": level,
            "message": message,
        })

def _clear_logs():
    with _log_lock:
        _log_entries.clear()

# ════════════════════════════════════════════════════════════════════
# RUN PROGRESS
#
# The pipeline prints one `[Level X.Y] <action>` banner each time it enters a
# stage, and those banners are the only structured signal it emits. Rather than
# reaching into 2000 lines of pipeline to add callbacks, the tee writer watches
# for them: every banner is a free progress tick, and a resumed run that skips
# stages simply emits fewer ticks.
#
# Progress is therefore stage-quantised, not smooth. The per-run stage weights
# are what turn "which stage am I in" into "how far through am I" — without them
# a bar would sit near 1% through the whole of Level 2, which is most of the wall
# time on a default run.
# ════════════════════════════════════════════════════════════════════

# Stage ladder: the ORDER is fixed, the WEIGHTS are not. Weights are computed per
# run from the parameters by _stage_weights(), because a constant table is what
# made the first version of this lie: it assumed coarse-grained folding dominated
# every run, so a run configured with cheap folding and expensive REST2 shot to
# 85% before its real bottleneck had started and then sat there for the whole of
# it.
LEVEL_ORDER = [
    ("0",   "Secondary structure"),
    ("1",   "3D prediction"),
    ("1.5", "Global restraint relaxation"),
    ("2",   "Coarse-grained folding"),
    ("2.3", "5-bead refinement"),
    ("2.5", "CG to all-atom"),
    ("2.6", "PyRosetta refinement"),
    ("3",   "RL fine-tuning"),
    ("3.5", "Metadynamics"),
    ("4",   "REST2 refinement"),
    ("5",   "Amber refinement"),
    ("5.5", "PPR repair"),
]
_LEVEL_INDEX = {name: i for i, (name, _) in enumerate(LEVEL_ORDER)}

# ── How the per-run stage weights are built ──────────────────────────────────
#
# Only one wall-time fact exists in this repository: docs/REPRODUCTION_RESOURCES
# section 2 states the ~7 h reference run "is dominated by the Level-2 REMD
# sampling stage (2 replicas x 8 relax rounds) together with the Level-4 REST2
# run (8 replicas x 20,000 steps)". Nothing states any other stage's share, and
# one data point cannot solve for twelve shares.
#
# So the model does not pretend to. Level 2 and Level 4 together take
# _HEAD_SHARE of every run — that much is documented — and are split between
# themselves by the relative step counts this run will execute. Everything else
# divides the remainder by step count.
#
# An earlier version set per-step factors for each stage individually and let
# Level 4's weight follow raw REST2 steps. It was wrong by a factor of ~150: a
# run whose REST2 stage was the whole bottleneck got a 2% weight for that stage,
# so the bar parked at 41% for the entire run. Anchoring the head is what fixes
# it, and the cost is that the non-head shares are order-of-magnitude. The UI
# says so rather than implying they were measured.
#
# The weights drive the PROGRESS BAR only. The time estimate uses measured wall
# time. The two deliberately take different inputs: the bar has to be smooth,
# the estimate has to be honest.
_HEAD_SHARE = 0.85        # Level 2 + Level 4, per the documented reference run
_FIVE_BEAD_STEPS_AT_2000 = 130_845.0


def _stage_weights(params, sequence_length=None):
    """Per-stage share of a run, computed from that run's own parameters.

    A constant table was tried first and measured wrong: it put coarse-grained
    folding at 45% of every run, so a run configured with cheap folding and
    expensive REST2 reached 85% before its real bottleneck had started and then
    sat there for the whole of it. Deriving the split per run is what fixes that.
    """
    p = params or {}

    def num(key, default):
        v = p.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def flag(key, default=True):
        """A boolean parameter as submitted, falling back to its real default.

        The caller passes only what the user overrode, so a parameter left at its
        default is absent from `p`, not False. Reading `p.get("use_metad")`
        therefore answered None for a run that was going to run metadynamics — and
        metadynamics was modelled at 0% of the run as a result. The defaults come
        from the pipeline's own signature.
        """
        if key in p and p[key] is not None:
            v = p[key]
            if isinstance(v, str):
                return v.strip().lower() not in ("0", "false", "no", "off", "")
            return bool(v)
        d = _pipeline_defaults().get(key, default)
        if isinstance(d, str):
            return d.strip().lower() not in ("0", "false", "no", "off", "")
        return bool(d)

    # Parameters the pipeline's signature gives a real value to; used when the
    # caller passed nothing for them.
    def pipe_num(key, fallback):
        if key in p and p[key] is not None:
            try:
                return float(p[key])
            except (TypeError, ValueError):
                pass
        d = _pipeline_defaults().get(key, fallback)
        try:
            return float(d)
        except (TypeError, ValueError):
            return float(fallback)

    L = sequence_length or 0
    rounds = max(1.0, pipe_num("n_relax_rounds", 6))

    # Every stage is modelled as a number of MD steps it will execute, and the
    # weights are those numbers normalised. One currency, so no stage can be given
    # a share by hand that has nothing to do with its cost.
    #
    # This replaced a scheme that reserved "_HEAD_SHARE = 0.85" of the run for
    # Levels 2 and 4 and divided the rest among the others by step count. The
    # reserved share was not derived from anything: with the default parameters
    # Level 4 came out at 84% of the modelled run while metadynamics came out at
    # 0.00%, and metadynamics is the longest stage there is — measured here at
    # about 5,000 of its 200,000 steps in four minutes, so roughly 2.6 hours.
    #
    # The consequences were visible and were reported as two separate faults: the
    # progress bar sat at 8% for hours because the stage actually running had been
    # given no width, and the estimate kept reporting "0s left" because the
    # fraction done could not move.
    steps = {}

    # Level 2: 8 rounds x 5,000 steps, with 64 replicas above 1,000 nt and
    # nrep/n_rest2_replicas below (isrnaclong.py:1568-1571).
    if L > 1000:
        l2_reps = 64.0
    else:
        l2_reps = max(pipe_num("nrep", 1) or 6.0, pipe_num("n_rest2_replicas", 1))
    steps["2"] = 8.0 * 5000.0 * rounds / 20.0 * (l2_reps / 8.0)

    # Level 4: the replicas run batched in one tensor, so the cost is the step
    # budget, not the replica count.
    steps["4"] = max(1.0, pipe_num("rest2_nsteps", 300000))

    # The rest are direct step counts, or a wall-clock-equivalent estimate where
    # the stage does not run MD at all. The constants for 0/1 are in the same
    # units by construction: they are the rough seconds those stages take, which
    # makes them comparable with a step count only because the normalisation
    # below is by total, so what matters is the ratio.
    steps["0"] = 600.0 + L * 0.4
    steps["1"] = (1200.0 + L * 0.6) * (2.0 if flag("use_rhofold", False) else 1.0)
    if flag("use_msa", True):
        steps["1"] += 800.0
    steps["1"] += 1500.0 * max(0.0, pipe_num("n_candidates", 1) - 1)
    steps["1.5"] = 1000.0
    steps["2.5"] = 600.0 * max(1.0, L / 200.0)      # scales with the conversion
    if flag("use_5bead", True):
        steps["2.3"] = _FIVE_BEAD_STEPS_AT_2000 * max(1.0, L / 2000.0) * (rounds / 20.0)
    else:
        steps["2.3"] = 0.0
    steps["2.6"] = 20000.0 if flag("use_pyrosetta", False) else 0.0
    steps["3"] = (5000.0 * max(1.0, pipe_num("rl_n_simulations", 50)) / 50.0
                  if flag("use_rl_mcts", True) else 0.0)
    steps["3.5"] = (max(0.0, pipe_num("metad_n_steps", 200000))
                    if flag("use_metad", True) else 0.0)
    steps["5"] = 3000.0
    steps["5.5"] = (400.0 * max(1.0, pipe_num("ppr_max_rounds", 5))
                    if flag("use_ppr", True) else 0.0)

    steps_total = sum(steps.values()) or 1.0
    weights = {name: steps.get(name, 0.0) / steps_total for name, _ in LEVEL_ORDER}
    total = sum(weights.values()) or 1.0
    return {k: max(0.0, v / total) for k, v in weights.items()}


# A run is described by anchor points, each (fraction_done, seconds_elapsed).
_STAGE_BANNER = re.compile(r"\[Level ([0-9]+(?:\.[0-9]+)?)\]\s*(.*)")
# In-stage progress, as the samplers report it: "[GPU-MetaD] 5000/200000".
# Without this the bar only ever moved at a `[Level X.Y]` banner, so through
# Levels 3.5 and 4 — the longest stages, printing a step line every few seconds —
# the terminal scrolled while the progress bar sat still, and the two looked like
# they were describing different runs.
_STEP_PROGRESS = re.compile(r"\b([0-9]{3,})\s*/\s*([0-9]{3,})\b")
_JOB_LOCK = threading.Lock()
_JOB_ID = {"current": None}
# The environment-setup sidecar (tools/install_deps.py). Tracked separately from
# a prediction because the two can be started independently and one must not
# block the other.
_install_lock = threading.Lock()
_install_state = {"running": False, "started_at": None, "finished_at": None,
                  "result": None}
# Timestamp of the last keep-alive progress line, so a quiet stage still shows
# the run is alive without flooding the log.
_PROGRESS_BEAT = {"at": 0.0, "stage": None}


def _register_job(job_id, sequence, params):
    """Publish a job the browser can re-attach to after a refresh.

    server_generation lets a page that reloaded after a server restart tell
    "no job is running" apart from "the job you were watching is gone".
    """
    with _JOB_LOCK:
        _JOB_ID["current"] = job_id
    weights = _stage_weights(params, len(sequence))
    _predict_state.update({
        "job_id": job_id,
        "sequence": sequence,
        "sequence_length": len(sequence),
        "params": dict(params),
        "status": "running",
        "progress": 0,
        "current_level": -1,
        "level_name": "Queued",
        "stage_label": "Queued",
        "message": "Starting...",
        "started_at": time.time(),
        "finished_at": None,
        "elapsed": 0.0,
        "weights": weights,
        "stage": {"fraction": 0.0, "label": "Starting", "anchors": [(0.0, 0.0)],
                  "spans": [], "entered": None, "spent": {}, "level": None},
        "stages": [],
        "eta": {"state": "measuring", "confidence": "none", "remaining_low": None,
                "remaining_high": None, "total_est": None,
                "basis": "waiting for the first stage to complete"},
        "result_ready": False,
        "log_count": 0,
        "server_generation": _SERVER_GENERATION,
        "levels": [{"level": n, "label": lab, "cost": weights.get(n, 0.0),
                    "pct": round(weights.get(n, 0.0) * 100, 1)} for n, lab in LEVEL_ORDER],
    })


def _note_stage(level_name, label):
    """Record arrival at a stage and recompute progress plus a time estimate."""
    idx = _LEVEL_INDEX.get(level_name)
    if idx is None:
        return
    now = time.time()
    started = _predict_state.get("started_at") or now
    elapsed = max(0.0, now - started)

    weights = _predict_state.get("weights") or _stage_weights(None)
    ordered = [n for n, _ in LEVEL_ORDER]

    done_cost = sum(weights.get(n, 0.0) for n in ordered[:idx])
    here_cost = weights.get(level_name, 0.0)
    # Credit the stage as NOT yet started, not half done.
    #
    # This used to add `0.5 * here_cost` on the reasoning that a stage is entered
    # rather than left. That put the bar ahead of the truth and, worse, put it
    # ahead of the within-stage reader: on entering Level 3.5 the bar jumped to
    # 32.6% (done_cost + half of 3.5's 33.5% weight), while a step line reading
    # 5000/200000 computed 16.8% from the start of that stage. The reader only
    # ever moves the bar forward, so every reading was discarded as a step
    # backwards and the bar sat at 32.6% for the whole of a two-hour stage.
    #
    # Now both use the same formula — the sum of finished stages plus the current
    # stage's own completed fraction — so entering a stage shows the work already
    # done, and the step lines carry it forward from there.
    fraction = min(1.0, done_cost)

    stage = _predict_state.get("stage") or {"anchors": [(0.0, 0.0)]}
    anchors = list(stage.get("anchors") or [(0.0, 0.0)])
    for i in range(len(anchors) - 1, 0, -1):
        if abs(anchors[i][0] - fraction) < 1e-6:
            anchors[i] = (fraction, elapsed)
            break
    else:
        anchors.append((fraction, elapsed))

    # Close the previous span so its real duration is measured, not guessed.
    spans = list(stage.get("spans") or [])
    spent = dict(stage.get("spent") or {})
    entered = stage.get("entered")
    prev_level = stage.get("level")
    if entered is not None and prev_level:
        dur = max(0.0, now - entered)
        spent[prev_level] = spent.get(prev_level, 0.0) + dur
        spans.append({"level": prev_level, "seconds": round(dur, 1)})

    # Publish a per-level view of the plan: what has run, what is running, what
    # is still ahead, and how long each finished stage actually took. The UI
    # renders its timeline from this, so the stage list is derived from the
    # server rather than being a copy of the ladder hard-coded in the HTML —
    # which is how the previous hard-coded list ended up empty after a change.
    level_labels = dict(LEVEL_ORDER)
    plan = []
    for i, name in enumerate(ordered):
        spent_here = spent.get(name)
        if i == idx and entered is not None:
            state = "running"
            spent_here = max(0.0, now - entered)
        elif i < idx:
            state = "done"
        else:
            state = "pending"
        plan.append({
            "level": name,
            "label": level_labels.get(name, name),
            "state": state,
            "weight_pct": round(weights.get(name, 0.0) * 100, 2),
            "seconds": round(spent_here, 1) if spent_here is not None else None,
        })

    stages = list(_predict_state.get("stages") or [])
    stages.append({"level": level_name, "label": label, "at": round(elapsed, 1)})

    _predict_state.update({
        "current_level": idx,
        "level_name": level_name,
        "stage_label": label,
        "message": label,
        "progress": round(fraction * 100, 1),
        "fraction": round(fraction, 4),
        "elapsed": round(elapsed, 1),
        "plan": plan,
        # Reset when a stage is entered: this is the within-stage fraction tracked
        # by _watch_for_step_progress, and carrying the previous stage's value into
        # the next one would make its first step reading look like a step backwards
        # and be discarded. `stage_progress` is cleared for the same reason — a
        # stale step count from the previous stage would be shown against this one.
        "inner_fraction": None,
        "stage_progress": None,
        "stage": {"fraction": fraction, "label": label, "anchors": anchors,
                  "spans": spans[-24:], "entered": now, "spent": spent,
                  "level": level_name},
        "stages": stages[-64:],
        "eta": _estimate_remaining(anchors, elapsed, current_level=level_name,
                                   entered=now, weights=weights),
    })


def _estimate_remaining(anchors, elapsed, current_level=None, entered=None, weights=None):
    """Seconds left, from the measured wall time at each stage boundary.

    Two independent estimates are combined, because either alone is wrong in a
    way the other catches:

      * the median of (elapsed / fraction) over the boundaries reached so far.
        This is what actually predicts total duration.
      * the elapsed time of the stage currently running, divided by its share of
        the work. That is a LOWER BOUND, since only part of the stage is done.
        Without it the median collapses the moment a run of cheap stages
        completes and an expensive one starts — the failure that made the first
        version of this report "2s left" for eighty seconds.

    The result is a range: one or two boundaries cannot support more precision
    than that, and a confident wrong ETA is worse than an honest wide one.
    """
    usable = [(f, s) for f, s in anchors if f > 0.001]
    if not usable:
        return {"state": "measuring", "confidence": "none", "remaining_low": None,
                "remaining_high": None, "total_est": None,
                "basis": "no stage has been reached yet"}

    rates = [s / f for f, s in usable if s > 0.5]
    if not rates:
        return {"state": "measuring", "confidence": "none", "remaining_low": None,
                "remaining_high": None, "total_est": None,
                "basis": "elapsed time is too small to extrapolate from"}

    rates.sort()
    mid = rates[len(rates) // 2]

    # Lower bound from the stage currently running: only part of it is done, so
    # elapsed/share underestimates its total, never overestimates it.
    bound = 0.0
    if current_level and entered and weights and weights.get(current_level, 0.0) > 1e-6:
        inside = max(0.0, time.time() - entered)
        bound = inside / weights[current_level]

    total = max(mid, bound)
    spread = (rates[-1] - rates[0]) / mid if len(rates) > 1 else 1.6
    spread = min(spread, 3.0)
    # Wide early, narrowing as boundaries accumulate.
    shrink = 1.0 + 0.5 * len(rates)
    lo = max(0.25, 1.0 - 0.5 * spread / shrink)
    hi = min(3.0, 1.0 + 0.5 * spread / shrink)

    remaining = max(0.0, total - elapsed)
    basis = "extrapolated from %d stage boundary(ies)" % len(rates)
    if bound > mid:
        basis += "; the running stage already exceeds that, so the estimate was raised"
    return {
        "state": "estimated",
        "confidence": "low" if len(rates) < 3 else ("medium" if len(rates) < 6 else "high"),
        "remaining_low": round(remaining * lo),
        "remaining_high": round(remaining * hi),
        "total_est": round(total),
        "basis": basis,
        "progress_basis": round(max(f for f, _ in usable), 3),
    }


def _maybe_emit_progress(pub, every=60.0):
    """Write a keep-alive line for a stage that prints nothing of its own.

    The pipeline banners only fire on ENTERING a stage, so a long stage is
    silent. Level 4 (REST2) is the worst case: hours of work with no output, so
    both the job log and the terminal look hung while the run is in fact fine.
    One line a minute is enough to tell "working" apart from "stuck", and it
    lands in the same log the browser and the terminal both read.

    Rate-limited here rather than in the caller so every path that reports state
    gets the same behaviour.
    """
    if pub.get("status") != "running":
        return
    stage = pub.get("level_name")
    if not stage:
        return
    now = time.time()
    last = _PROGRESS_BEAT.get("at", 0.0)
    if now - last < every:
        return
    _PROGRESS_BEAT["at"] = now
    _PROGRESS_BEAT["stage"] = stage

    elapsed = pub.get("elapsed") or 0.0
    eta = pub.get("eta") or {}
    if eta.get("state") == "estimated" and eta.get("remaining_low") is not None:
        left = "still running — about %s left" % _human_seconds(
            (eta["remaining_low"] + eta["remaining_high"]) / 2.0)
    else:
        left = "still running"
    line = "  [%s] %s · %s · %s · %.1f%%" % (
        stage, pub.get("stage_label") or "", _human_seconds(elapsed), left,
        pub.get("progress") or 0.0)

    # Print rather than _emit_log. _emit_log only appends to the in-memory buffer
    # that feeds the browser; the LOG FILE is written by the tee wrapping stdout,
    # so a buffer-only line never reaches the file and the terminal following it
    # stays silent. Printing goes to both: the tee mirrors it to the file and
    # echoes it back through _emit_log for the browser.
    stream = sys.stdout
    if stream is not None and hasattr(stream, "write"):
        try:
            stream.write(line + "\n")
            stream.flush()
            return
        except (OSError, ValueError):
            pass
    _emit_log("progress", line)


def _human_seconds(seconds):
    s = int(max(0, seconds))
    if s >= 3600:
        return "%dh %dm" % (s // 3600, (s % 3600) // 60)
    if s >= 60:
        return "%dm %ds" % (s // 60, s % 60)
    return "%ds" % s


def _publish_final(status, message, error=None):
    now = time.time()
    started = _predict_state.get("started_at") or now
    _predict_state.update({
        "status": status,
        "message": message,
        "error": error,
        "finished_at": now,
        "elapsed": round(now - started, 1),
        "result_ready": status == "done",
        "eta": {"state": "done" if status == "done" else "stopped",
                "confidence": "high", "remaining_low": 0, "remaining_high": 0,
                "total_est": round(now - started), "basis": "finished"},
    })


# The state the UI is allowed to see. The prediction result carries the whole
# PDB as a string, so it is never included here; it is fetched separately from
# /api/result/{job_id}.
_PUBLIC_KEYS = (
    "job_id", "status", "progress", "current_level", "level_name", "stage_label",
    "message", "error", "elapsed", "eta", "stages", "levels", "started_at",
    "finished_at", "sequence_length", "result_ready", "server_generation",
    "fraction", "weights", "plan", "log_path", "stage_progress",
)

def _model_source_ids():
    """The model hosts the installer accepts, plus "auto".

    Read from the installer so the endpoint and the CLI cannot disagree about
    what is valid. Cached after the first read.
    """
    cached = _MODEL_SOURCE_CACHE.get("ids")
    if cached:
        return cached
    ids = {"auto"}
    try:
        script_dir = os.path.join(ROOT, "tools")
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        import install_deps as _id_mod
        ids.update(s["id"] for s in _id_mod.MODEL_SOURCES)
    except Exception:                                    # noqa: BLE001
        ids.update({"huggingface", "hf-mirror", "modelscope"})
    _MODEL_SOURCE_CACHE["ids"] = ids
    return ids


_MODEL_SOURCE_CACHE = {}


def _viewer_module():
    """The checkpoint-tracking module, imported from tools/ on first use."""
    global _VIEWER_MOD
    if _VIEWER_MOD is None:
        script_dir = os.path.join(ROOT, "tools")
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        import viewer_stage as _vs
        _VIEWER_MOD = _vs
    return _VIEWER_MOD


_VIEWER_MOD = None


def _viewer_stage(output_dir):
    """The newest finished checkpoint, or None. Never raises: the 3D panel is not
    worth failing a status poll over."""
    try:
        mod = _viewer_module()
        started = _predict_state.get("started_at")
        return mod.newest_structure(output_dir, run_started_at=started)
    except Exception:                                    # noqa: BLE001
        return None


# Below this, a "structure" is not a fold: a handful of atoms is a run that was
# interrupted early, or a stub. Showing the delivered 2,013 nt model is more use
# than showing ten phosphate atoms, so this decides when that happens.
_MIN_MEANINGFUL_ATOMS = 40


def _display_structure(output_dir):
    """What the 3D panel should be showing.

    A checkpoint from the current run wins, and so does a substantial result left
    by a previous one — that is what someone reopening the page wants to see. The
    committed 2,013 nt model is the fallback for the empty state: no run in
    progress and nothing substantive on disk. It carries level "delivered" rather
    than a pipeline level, so it is never read as a stage of the run being watched.
    """
    stage = _viewer_stage(output_dir)
    running = _predict_state.get("status") == "running"
    if stage and (running or stage.get("atoms", 0) >= _MIN_MEANINGFUL_ATOMS):
        return stage
    try:
        delivered = _viewer_module().delivered_structure(ROOT)
    except Exception:                                    # noqa: BLE001
        delivered = None
    return delivered or stage


def _deps_payload():
    """GET /api/deps — what the setup button should offer, and its state.

    Reports the same three-way split the installer uses, so the UI can be honest
    about it: things that can be fetched automatically, things already present,
    and things that have no download source and need a human.
    """
    with _install_lock:
        state = dict(_install_state)
    # The model hosts are published so the UI can offer them rather than
    # hard-coding a list that would drift from the installer's.
    sources = []
    try:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import install_deps as _id_mod
        sources = [{"id": s["id"], "label": s["label"], "note": s["note"]}
                   for s in _id_mod.MODEL_SOURCES]
        default_source = _id_mod.DEFAULT_SOURCE
    except Exception:                                    # noqa: BLE001
        default_source = "auto"
    return {
        "install_running": state["running"],
        "started_at": state["started_at"],
        "finished_at": state["finished_at"],
        "last_result": state["result"],
        "tools_dir": os.path.join(os.path.dirname(ROOT), "TorusFold-tools"),
        "has_installer": os.path.isfile(os.path.join(ROOT, "tools", "install_deps.py")),
        "model_sources": sources,
        "default_source": default_source,
    }


def _public_state():
    state = _predict_state
    out = {k: state.get(k) for k in _PUBLIC_KEYS}
    # The ladder is always published, with the weights of the run in progress so
    # the UI can draw the timeline before anything happens and does not have to
    # hard-code either the level names or their relative costs.
    weights = state.get("weights") or _stage_weights(None)
    out["levels"] = [{"level": n, "label": lab, "cost": weights.get(n, 0.0),
                      "pct": round(weights.get(n, 0.0) * 100, 1)} for n, lab in LEVEL_ORDER]
    # Before any run, synthesise the plan so the UI has something to draw rather
    # than an empty timeline.
    if not out.get("plan"):
        out["plan"] = [{"level": n, "label": lab, "state": "pending",
                        "weight_pct": round(weights.get(n, 0.0) * 100, 2),
                        "seconds": None} for n, lab in LEVEL_ORDER]

    if state.get("status") == "running" and state.get("started_at"):
        now = time.time()
        out["elapsed"] = round(now - state["started_at"], 1)
        # Recompute the estimate here rather than serving the one stored when the
        # stage was entered. Two reasons: the "current stage is overrunning"
        # lower bound needs the time spent inside the stage, which is zero at
        # entry, and a long stage would otherwise keep showing the figure
        # calculated before it started — the exact failure that had a four-hour
        # stage reporting "8s left" for two minutes.
        stage = state.get("stage") or {}
        anchors = stage.get("anchors") or [(0.0, 0.0)]
        out["eta"] = _estimate_remaining(
            anchors, out["elapsed"],
            current_level=stage.get("level"),
            entered=stage.get("entered"),
            weights=weights,
        )

    out["log_count"] = len(_log_entries)
    # Which checkpoint the 3D view should be showing. Published with the run state
    # so the viewer advances on the heartbeat it already receives, rather than
    # polling a second endpoint for it.
    stage = _display_structure(os.path.join(ROOT, "output_web"))
    if stage:
        out["structure"] = {"level": stage["level"], "name": stage["name"],
                            "desc": stage["desc"], "atoms": stage["atoms"],
                            "mtime": stage["mtime"],
                            "digest": stage.get("digest", "")}
        # Measurements of whatever structure is on screen, so the readout panels
        # have numbers during a run instead of only after it finishes. Cached on
        # the structure's digest, so this costs nothing until the file changes.
        out["metrics"] = _live_metrics(stage)
    else:
        out["structure"] = None
        out["metrics"] = None
    return out


def _current_job_payload():
    """/api/current — what a freshly loaded page asks for.

    A page refresh loses the job id, so the client re-attaches from here rather
    than starting a second run. If the id it remembered belongs to an earlier
    server process, this says so explicitly instead of letting the browser
    poll a job that no longer exists.
    """
    with _JOB_LOCK:
        job_id = _JOB_ID["current"]
    payload = _public_state()
    payload["has_job"] = bool(job_id)
    payload["job_running"] = _predict_state.get("status") == "running"
    return payload


# ════════════════════════════════════════════════════════════════════
# PARAMETER REGISTRY — the single source of truth for what the web UI may set.
#
# This table exists because the same parameter list used to live in three
# places (the HTML inputs, app.js, and this file's call into the pipeline) and
# they had drifted apart: the UI offered five numbers, the pipeline accepts
# thirty-four options, and several values this file passed disagreed with the
# demo invocation in run_2013nt.py.
#
# Defaults are NOT written here. They are read from the real signature of
# isrnaclong_pipeline() by _pipeline_defaults(), so this table can never state a
# default the pipeline does not use. Add a knob here and it appears in the UI;
# nothing else needs editing.
# ════════════════════════════════════════════════════════════════════

# kind: "int" | "float" | "bool" | "text"
# group decides which collapsible section of the parameters card it lands in.
_PARAM_SPEC = [
    # ── segmentation and relaxation ──────────────────────────────────
    dict(name="max_seg_len", kind="int", group="Segmentation",
         label="Max segment length", unit="nt", min=50, max=1000, step=10,
         help="Level 1 splits the chain into chunks of at most this many nt."),
    dict(name="overlap", kind="int", group="Segmentation",
         label="Segment overlap", unit="nt", min=0, max=100, step=5,
         help="Overlap between chunks; larger gives more context for the Kabsch alignment."),
    dict(name="n_relax_rounds", kind="int", group="Segmentation",
         label="Relaxation rounds", min=0, max=60, step=1,
         help="Level 2 iterations. Early stopping usually truncates this."),
    dict(name="n_parallel", kind="int", group="Segmentation",
         label="Parallel segments", min=0, max=64, step=1,
         help="0 lets the pipeline pick from the CPU count."),

    # ── sampling budget ──────────────────────────────────────────────
    dict(name="n_rest2_replicas", kind="int", group="Sampling",
         label="REST2 replicas", min=2, max=64, step=1,
         help="Replica-exchange replicas. Costs one core each."),
    dict(name="rest2_nsteps", kind="int", group="Sampling",
         label="REST2 steps", min=5000, max=2000000, step=5000),
    dict(name="md_step_scale", kind="float", group="Sampling",
         label="MD step scale", min=0.01, max=1.0, step=0.05,
         help="Multiplies the per-round Level 2 step count. The dominant cost."),
    dict(name="nrep", kind="int", group="Sampling",
         label="REMD replicas", min=1, max=64, step=1,
         help="Concurrent Level 2 replicas, one process each."),
    dict(name="platform", kind="text", group="Sampling", label="Platform",
         placeholder="auto",
         help="OpenMM/LAMMPS platform: auto, CPU, CUDA, OpenCL."),

    # ── enhanced sampling ────────────────────────────────────────────
    dict(name="use_5bead", kind="bool", group="Enhanced sampling",
         label="5-bead CG refinement",
         help="Level 2.3. P/S/B1/B2/B3 per nucleotide instead of 3 beads."),
    dict(name="use_metad", kind="bool", group="Enhanced sampling",
         label="Metadynamics",
         help="Level 3.5. Wells-tempered hills along BSJ distance, contacts and Rg."),
    dict(name="metad_n_steps", kind="int", group="Enhanced sampling",
         label="Metadynamics steps", min=0, max=2000000, step=10000),

    # ── reinforcement learning ───────────────────────────────────────
    dict(name="use_rl_relax", kind="bool", group="Guidance & repair",
         label="RL relaxation guidance", help="Level 2. Off = fixed-parameter ablation."),
    dict(name="use_rl_mcts", kind="bool", group="Guidance & repair",
         label="RL-MCTS closing", help="Level 3. Off = ablation."),
    dict(name="rl_n_simulations", kind="int", group="Guidance & repair",
         label="MCTS iterations", min=0, max=500, step=5),

    # ── Level 1 prediction source ────────────────────────────────────
    dict(name="use_rhofold", kind="bool", group="Prediction source",
         label="RhoFold+ ensemble",
         help="On: RhoFold+/trRosettaRNA2/RNAbpFlow ensemble. Off: Vfold3D only."),
    dict(name="n_candidates", kind="int", group="Prediction source",
         label="Candidates per segment", min=1, max=10, step=1),
    dict(name="use_msa", kind="bool", group="Prediction source",
         label="Adaptive MSA",
         help="Feeds the predictor a pseudo-MSA. Recommended with RhoFold+."),

    # ── optional stages ──────────────────────────────────────────────
    dict(name="use_pyrosetta", kind="bool", group="Optional stages",
         label="PyRosetta refinement (Level 2.6)",
         help="Needs WSL with PyRosetta. Skips in well under a second when absent."),
    dict(name="use_ppr", kind="bool", group="Optional stages",
         label="PPR base-pair repair (Level 5.5)"),
    dict(name="ppr_max_rounds", kind="int", group="Optional stages",
         label="PPR rounds", min=0, max=20, step=1),
    dict(name="resume", kind="bool", group="Optional stages",
         label="Resume from checkpoint",
         help="Reuses any checkpoint in the output directory."),

    # ── structRFM heads (opt-in, off by default) ─────────────────────
    dict(name="use_multi_task_heads", kind="bool", group="structRFM heads",
         label="Multi-task heads"),
    dict(name="use_structrfm", kind="bool", group="structRFM heads",
         label="structRFM scoring"),
    dict(name="multitask_head_weights", kind="text", group="structRFM heads",
         label="Head weights path", placeholder="(unset)"),
    dict(name="ss_head_weight", kind="float", group="structRFM heads",
         label="SS head weight", min=0, max=10, step=0.1),
    dict(name="pair_head_weight", kind="float", group="structRFM heads",
         label="Pair head weight", min=0, max=10, step=0.1),
    dict(name="bsj_head_weight", kind="float", group="structRFM heads",
         label="BSJ head weight", min=0, max=10, step=0.1),
    dict(name="clash_head_weight", kind="float", group="structRFM heads",
         label="Clash head weight", min=0, max=10, step=0.1),

    # ── method variants ──────────────────────────────────────────────
    dict(name="use_rcm_reweight", kind="bool", group="Method variants",
         label="RCM pair reweighting",
         help="Overwrites the method-agreement pair weights with RCM confidence. Off by default."),
]

# Keys the old frontend sent, and the real parameter each one meant. Kept so an
# older cached page still drives the pipeline correctly.
_LEGACY_KEYS = {
    "rounds": "n_relax_rounds",
    "replicas": "n_rest2_replicas",
    "rest2steps": "rest2_nsteps",
    "use_rl": None,          # expands to both RL switches
    "max_seg_len": "max_seg_len",
    "overlap": "overlap",
}

_schema_cache = {}
_schema_lock = threading.Lock()


def _pipeline_defaults():
    """Read the real defaults off isrnaclong_pipeline's signature.

    Importing the pipeline pulls in torch, so this is done once and cached, and
    main() warms it on a background thread to keep the first request fast.
    """
    with _schema_lock:
        if _schema_cache:
            return _schema_cache
        try:
            if SRC not in sys.path:
                sys.path.insert(0, SRC)
            from torusfold.scheme2.isrnaclong import isrnaclong_pipeline as f
            defaults = dict(f.__kwdefaults__ or {})
        except Exception as exc:                      # pragma: no cover
            _emit_log("warn", f"parameter defaults unavailable: {exc}")
            defaults = {}
        _schema_cache.update(defaults)
        return _schema_cache


def _default_for(name):
    return _pipeline_defaults().get(name)


def parameter_schema():
    """The payload GET /api/schema returns: groups, knobs, defaults, bounds."""
    defaults = _pipeline_defaults()
    groups = []
    by_group = {}
    for spec in _PARAM_SPEC:
        name = spec["name"]
        if name not in defaults:
            # A knob the pipeline no longer accepts. Surface it rather than
            # dropping it silently, so a rename cannot hide here.
            _emit_log("warn", f"parameter '{name}' is not a pipeline argument")
        item = dict(spec)
        item["default"] = defaults.get(name)
        g = item.pop("group")
        if g not in by_group:
            by_group[g] = []
            groups.append({"name": g, "params": by_group[g]})
        by_group[g].append(item)
    return {"groups": groups,
            "open_by_default": ["Segmentation", "Sampling"],
            "path": "torusfold.scheme2.isrnaclong.isrnaclong_pipeline",
            "knob_count": sum(len(g["params"]) for g in groups),
            "pipeline_option_count": len(defaults)}


def normalise_params(raw):
    """Turn a request body into pipeline kwargs.

    Accepts the schema-shaped {"params": {...}} and the legacy flat body, so a
    browser holding an older app.js keeps working. Values are coerced to the
    declared kind and clamped to the declared range; unknown keys are rejected
    rather than forwarded, because the pipeline raises TypeError on them.
    """
    if not isinstance(raw, dict):
        return {}, []
    incoming = raw.get("params") if isinstance(raw.get("params"), dict) else raw
    defaults = _pipeline_defaults()
    spec_by_name = {s["name"]: s for s in _PARAM_SPEC}
    out, notes = {}, []

    for key, value in incoming.items():
        name = _LEGACY_KEYS.get(key, key)
        if name is None:                       # legacy "use_rl": both switches
            for twin in ("use_rl_relax", "use_rl_mcts"):
                out[twin] = bool(value)
            notes.append("use_rl -> use_rl_relax + use_rl_mcts")
            continue
        spec = spec_by_name.get(name)
        if spec is None:
            notes.append(f"ignored unknown parameter '{key}'")
            continue
        kind = spec["kind"]
        try:
            if kind == "bool":
                if isinstance(value, str):
                    value = value.strip().lower() not in ("", "0", "false", "off", "no")
                value = bool(value)
            elif kind == "int":
                value = int(float(value))
            elif kind == "float":
                value = float(value)
            else:
                value = str(value)
        except (TypeError, ValueError):
            notes.append(f"'{name}': {value!r} is not a valid {kind}, using default")
            continue
        if kind in ("int", "float"):
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and value < lo:
                notes.append(f"{name}={value} below min {lo}; clamped")
                value = lo
            if hi is not None and value > hi:
                notes.append(f"{name}={value} above max {hi}; clamped")
                value = hi
        out[name] = value

    return out, notes

# ── stdout/stderr capture for pipeline output ────────────────────

class _LogMirror:
    """Append every line the process writes to a file, as it is written.

    Two reasons this exists rather than `tee`:

      * Python block-buffers stdout when it is not a terminal (8 KB by default),
        so a run redirected to a file or a pipe shows nothing for minutes and
        then dumps everything at once. A run in progress is exactly when the
        output is worth watching, so the file is opened line-buffered and the
        process is started unbuffered as well.
      * One file per job means a finished run can be read back afterwards, and
        its failures compared against a later run's.

    Writes are best-effort: a log that cannot be written must never take the
    pipeline down with it.
    """

    def __init__(self, original, path):
        self._orig = original
        self._path = path
        self._fh = None
        self.encoding = getattr(original, "encoding", None) or "utf-8"
        self.errors = getattr(original, "errors", None) or "replace"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
        except OSError:
            self._fh = None

    @property
    def path(self):
        return self._path

    def write(self, s):
        if isinstance(s, (bytes, bytearray)):
            s = bytes(s).decode(self.encoding, self.errors)
        if self._orig:
            self._orig.write(s)
        if self._fh:
            try:
                self._fh.write(s)
            except (OSError, ValueError):
                self._fh = None
        return len(s)

    def flush(self):
        if self._orig:
            self._orig.flush()
        if self._fh:
            try:
                self._fh.flush()
            except (OSError, ValueError):
                self._fh = None

    def close(self):
        self.flush()

    def fileno(self):
        if self._orig and hasattr(self._orig, "fileno"):
            return self._orig.fileno()
        raise io.UnsupportedOperation("fileno")

    def isatty(self):
        return bool(getattr(self._orig, "isatty", lambda: False)())

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    def __getattr__(self, name):
        return getattr(self.__dict__["_orig"], name)


class _TeeWriter:
    """Wraps a file-like object, echoing each line to _emit_log."""
    def __init__(self, original, level="info"):
        self._orig = original
        self._level = level
        self._buf = ""
        # A stand-in for sys.stdout must answer the text-stream protocol.
        # Libraries (and subprocess wrappers) read .encoding / .errors off
        # sys.stdout; a bare write/flush object raises AttributeError there.
        # utf-8 first: the log buffer goes out as SSE JSON, so a gbk console
        # default would only turn wide characters into decode errors later.
        self.encoding = "utf-8"
        self.errors = "replace"

    def write(self, s):
        if isinstance(s, (bytes, bytearray)):
            s = bytes(s).decode(self.encoding, self.errors)
        if self._orig:
            self._orig.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                self._watch_for_stage(line)
                _emit_log(self._level, line)

    def flush(self):
        if self._orig:
            self._orig.flush()
        if self._buf.strip():
            line = self._buf.strip()
            self._watch_for_stage(line)
            _emit_log(self._level, line)
            self._buf = ""

    @staticmethod
    def _watch_for_stage(line):
        """Turn a `[Level X.Y]` banner into a progress tick.

        The pipeline already prints one of these at every stage boundary, so the
        progress bar can be driven without adding a callback to 2000 lines of
        pipeline code. Anything that is not a banner falls through to the
        within-stage reader below.
        """
        m = _STAGE_BANNER.search(line)
        if m:
            level_name = m.group(1)
            label = m.group(2).strip().rstrip(".")
            if not label:
                label = dict(LEVEL_ORDER).get(level_name, "Level " + level_name)
            # Keep it short: this string goes into the progress header.
            if len(label) > 72:
                label = label[:69].rstrip() + "..."
            _note_stage(level_name, label)
            return
        _watch_for_step_progress(line)


def _watch_for_step_progress(line):
    """Advance the bar from a within-stage step count such as `5000/200000`.

    The banner handler only fires at stage boundaries, so the bar used to sit
    still for the whole of Levels 3.5 and 4 while the terminal showed thousands of
    step lines. This moves it in between: the stage's own fraction of the run is
    interpolated by how far through its step budget it is.

    Deliberately conservative. A `N/M` pair appears in many unrelated lines (a
    count of hills, a fraction of a molecule), so the pair is only believed when it
    is internally consistent — M large, N <= M — and the value is only moved
    forward, never backward. A wrong reading therefore cannot walk the bar
    backwards, and the anchors used for the ETA are keyed to stage entry, not to
    this, so the time estimate is unaffected.
    """
    m = _STEP_PROGRESS.search(line)
    if not m:
        return
    try:
        done, total = int(m.group(1)), int(m.group(2))
    except ValueError:
        return
    if total < 1000 or done > total or done < 0:
        return
    inner = done / float(total)

    state = _predict_state
    if state.get("status") != "running":
        return
    stage = state.get("stage") or {}
    entered = stage.get("entered")
    level_name = stage.get("level")
    if not level_name or entered is None:
        return
    weights = state.get("weights") or {}
    # The head of the run (Levels 0-2) is where _HEAD_SHARE of the modelled cost
    # sits; a stage's share of the whole is its weight. Moving within a stage must
    # therefore move the overall fraction by that stage's own width, not by the
    # raw step ratio.
    reached = 1.0 - sum(v for k, v in weights.items()
                        if _LEVEL_INDEX.get(k, -1) > _LEVEL_INDEX.get(level_name, -1))
    width = weights.get(level_name, 0.0)
    fraction = reached - width + width * inner

    # Publish the pipeline's OWN numbers, updated on every reading.
    #
    # These are not the same quantity as the percentage below and must not be
    # presented as if they were. `done/total` is what the stage itself reports; the
    # percentage is `width` — this stage's share of the whole run, from a
    # hand-written weight table, which is an estimate — multiplied by that ratio.
    # Measured here, metadynamics was modelled at 33% of the run and actually takes
    # about 7%, so the percentage was wrong by a factor of five regardless of how
    # the ratio advanced, and no care in the parser could have made the two agree.
    #
    # Updated before the monotonicity check below, so a reading that does not move
    # the bar still updates the number shown next to it.
    state["stage_progress"] = {
        "step": done,
        "total": total,
        "ratio": round(inner, 4),
        "level": level_name,
        "modelled_pct": round(width * 100.0, 2),
    }

    prev = state.get("inner_fraction")
    if prev is not None and fraction <= prev:
        return
    state["inner_fraction"] = fraction
    state["progress"] = round(min(99.0, max(state.get("progress") or 0.0,
                                            fraction * 100.0)), 1)

    def fileno(self):
        # No real descriptor when the original is absent or is itself a mock.
        if self._orig and hasattr(self._orig, "fileno"):
            return self._orig.fileno()
        raise io.UnsupportedOperation("fileno")  # io is imported at module top

    def isatty(self):
        return bool(getattr(self._orig, "isatty", lambda: False)())

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    def close(self):
        self.flush()

    def __getattr__(self, name):
        # Anything else the caller expects of a stream (buffer, name, ...)
        # is delegated to the stream being wrapped.
        return getattr(self.__dict__["_orig"], name)

# ── Global prediction state ─────────────────────────────────────
_predict_state = {
    "status": "idle",       # idle | running | done | error
    "progress": 0,          # 0-100
    "current_level": -1,
    "message": "",
    "result": None,
    "error": None,
    "start_time": 0,
    # Published from the first request onward, so a page that has never seen a
    # run can still tell which server process it is talking to.
    "server_generation": _SERVER_GENERATION,
}


class TorusFoldHandler(SimpleHTTPRequestHandler):
    """Serves static files + POST /predict + GET /status + SSE streaming"""

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data, code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._set_cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("/api/predict", "/predict"):
            self._handle_predict()
        elif path in ("/api/upload", "/upload"):
            self._handle_upload()
        elif path in ("/api/score-pdb", "/api/score-pdb"):
            self._handle_score_pdb()
        elif path in ("/api/feedback", "/feedback"):
            self._handle_feedback()
        elif path in ("/api/install-deps", "/install-deps"):
            self._handle_install_deps()
        else:
            self._send_json({"error": f"Unknown POST endpoint: {path}"}, 404)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("/api/health", "/health"):
            # "status" here answers "is the server answering", and the job state is
            # reported separately under its own name. They used to share the field,
            # so a healthy server with nothing running reported `status: "idle"` and
            # the interface displayed that as the backend's state — reading, quite
            # reasonably, as "the backend is not started".
            self._send_json({"ok": True, "status": "ready",
                             "job_status": _predict_state["status"]})
            return
        elif path in ("/api/schema", "/schema"):
            self._send_json(parameter_schema())
            return
        elif path in ("/api/current", "/current"):
            self._send_json(_current_job_payload())
            return
        elif path in ("/api/deps", "/deps"):
            self._send_json(_deps_payload())
            return
        elif path in ("/api/structure", "/structure") or path.startswith("/api/structure/"):
            # Both forms: without a name it reports which checkpoint is current,
            # with one it serves those bytes. Matching only the trailing-slash form
            # sent the bare path to the static file handler.
            self._handle_structure(path)
            return
        elif path in ("/api/log", "/log"):
            self._handle_log()
            return
        elif path in ("/api/status", "/status"):
            self._handle_status()
            return
        elif path.startswith("/api/jobs/"):
            self._handle_job_status(path)
            return
        elif path.startswith("/api/result/"):
            self._handle_job_result(path)
            return
        elif path.startswith("/api/sse/"):
            self._handle_sse(path)
            return
        elif path.startswith("/api/score-pdb/sse/"):
            self._handle_score_pdb_sse(path)
            return

        # Static files: served from WEB_DIR or ROOT
        if path.startswith("/web/"):
            file_path = os.path.join(WEB_DIR, path[5:])
        elif path == "" or path == "/":
            file_path = os.path.join(WEB_DIR, "index.html")
        else:
            file_path = os.path.join(ROOT, path.lstrip("/"))

        if os.path.isfile(file_path):
            self._serve_file(file_path)
        else:
            self.send_error(404, f"File not found: {path}")

    def _serve_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        ct_map = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".pdb": "chemical/x-pdb",
            ".fa": "text/plain",
            ".fasta": "text/plain",
            ".woff2": "font/woff2",
            ".woff": "font/woff",
        }
        content_type = ct_map.get(ext, "application/octet-stream")
        with open(file_path, "rb") as f:
            data = f.read()

        # Revalidate every time instead of letting the browser guess.
        #
        # Without validators a browser applies heuristic caching and can serve a
        # stale app.js long after the file changed, with no way to notice — which
        # is exactly what happened while developing this: an edit appeared to do
        # nothing because the old script was still in the cache. `no-cache` means
        # "check with me first", not "do not store", so unchanged files still cost
        # only a 304.
        digest = hashlib.sha1(data).hexdigest()
        etag = '"%s"' % digest
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        try:
            self.send_header("Last-Modified",
                             email.utils.formatdate(os.path.getmtime(file_path),
                                                    usegmt=True))
        except OSError:
            pass
        self._set_cors()
        self.end_headers()
        self.wfile.write(data)

    def _handle_status(self):
        self._send_json(_public_state())

    def _handle_job_status(self, path):
        """GET /api/jobs/{jid} — return task status"""
        jid = path.split("/")[-1]
        self._send_json(_public_state())

    def _handle_job_result(self, path):
        """GET /api/result/{jid} — return prediction result"""
        jid = path.split("/")[-1]
        if _predict_state["status"] == "done" and _predict_state.get("result"):
            self._send_json(_predict_state["result"])
        elif _predict_state["status"] == "running":
            self._send_json({"status": "running", "progress": _predict_state["progress"]})
        elif _predict_state["status"] == "error":
            self._send_json({"error": _predict_state["error"]}, 500)
        else:
            self._send_json({"error": "No result available"}, 404)

    # ── SSE endpoint ─────────────────────────────────────────────
    def _handle_sse(self, path):
        """GET /api/sse/{job_id} — Server-Sent Events streaming"""
        job_id = path.split("/")[-1]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._set_cors()
        self.end_headers()

        last_idx = 0
        last_beat = 0.0
        try:
            while True:
                with _log_lock:
                    new_entries = _log_entries[last_idx:]
                    last_idx = len(_log_entries)
                # _public_state recomputes elapsed while running, so the header
                # clock ticks even when no stage boundary has been crossed.
                pub = _public_state()
                status = pub["status"]

                for entry in new_entries:
                    data = json.dumps(entry, ensure_ascii=False)
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()

                # A heartbeat every 0.5s for a run measured in hours is ~14k
                # frames, nearly all identical. Send one whenever the state
                # actually moves (which also carries a new log line), and a
                # keepalive only every few seconds so a quiet stage stays quiet.
                now = time.time()
                if new_entries or (now - last_beat) >= 3.0:
                    last_beat = now
                    beat = dict(pub)
                    beat["level"] = "heartbeat"
                    self.wfile.write(("data: " + json.dumps(beat, ensure_ascii=False) + "\n\n")
                                     .encode("utf-8"))
                    self.wfile.flush()

                    # Keep-alive line for a stage that prints nothing of its own.
                    # Level 4 (REST2) can run for hours with no output at all, and
                    # the pipeline only prints on entering a stage — so without
                    # this the log and the terminal look frozen exactly when the
                    # run is working hardest.
                    _maybe_emit_progress(pub)

                if status in ("done", "error"):
                    # Send final event
                    event_type = "done" if status == "done" else "error"
                    final = json.dumps({
                        "event": event_type,
                        "status": status,
                        "message": pub.get("message"),
                        "job_id": pub.get("job_id"),
                        "elapsed": pub.get("elapsed"),
                    })
                    self.wfile.write(f"event: {event_type}\ndata: {final}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    break

                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # Client disconnected

    def _handle_score_pdb(self):
        """POST /api/score-pdb — upload a PDB, start analysis, return a session_id"""
        body = self._read_body()
        if not body:
            self._send_json({"error": "Empty body"}, 400)
            return

        import uuid
        session_id = str(uuid.uuid4())[:8]
        pdb_text = body.decode('utf-8', errors='replace')

        # Save PDB
        out_dir = os.path.join(ROOT, "output_web")
        os.makedirs(out_dir, exist_ok=True)
        pdb_path = os.path.join(out_dir, f"scored_{session_id}.pdb")
        with open(pdb_path, "w", encoding="utf-8") as f:
            f.write(pdb_text)

        # Store PDB text for SSE handler
        _pdb_sessions[session_id] = {
            "pdb_text": pdb_text,
            "pdb_path": pdb_path,
        }

        self._send_json({"ok": True, "session_id": session_id, "size": len(body)})

    def _handle_score_pdb_sse(self, path):
        """GET /api/score-pdb/sse/{session_id} — SSE streaming PDB analysis."""
        session_id = path.split("/")[-1]
        session = _pdb_sessions.get(session_id)
        if not session:
            self._send_json({"error": "Session not found"}, 404)
            return

        pdb_text = session["pdb_text"]
        pdb_path = session["pdb_path"]

        # SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._set_cors()
        self.end_headers()

        def send_event(event_type, data):
            try:
                payload = json.dumps(data, ensure_ascii=False)
                self.wfile.write(f"event: {event_type}\ndata: {payload}\n\n".encode('utf-8'))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise

        try:
            # Ensure import path
            if SRC not in sys.path:
                sys.path.insert(0, SRC)

            # Step 1: Parsing
            send_event("step", {"step": "parse", "message": "Parsing PDB records..."})
            from torusfold.scheme2.pdb_analyzer import parse_pdb
            parsed_pdb = parse_pdb(pdb_text)
            send_event("step", {
                "step": "parsed",
                "message": f"Parsed {parsed_pdb['n_atoms']} atoms, {parsed_pdb['n_residues']} residues, {parsed_pdb['n_chains']} chains",
                "data": {
                    "n_atoms": parsed_pdb["n_atoms"],
                    "n_residues": parsed_pdb["n_residues"],
                    "n_chains": parsed_pdb["n_chains"],
                    "is_nucleic": parsed_pdb["is_nucleic"],
                },
            })

            if parsed_pdb["n_atoms"] == 0:
                send_event("error", {"message": "No ATOM records found in PDB"})
                return

            # Step 2: Clash
            send_event("step", {"step": "clash", "message": "Computing clash score..."})
            from torusfold.scheme2.pdb_analyzer import (
                compute_clash_score, compute_radius_of_gyration,
                compute_bond_rmsd, compute_sasa_estimate, compute_end_to_end,
                compute_shape_descriptors, compute_backbone_angles,
            )
            clash = compute_clash_score(parsed_pdb["coords"], parsed_pdb["atom_names"], parsed_pdb["residue_ids"])
            send_event("metric", {"key": "clash", "message": f"Clash score: {clash['clash_score']}", "data": clash})

            # Step 3: RoG
            send_event("step", {"step": "rog", "message": "Computing radius of gyration..."})
            rog = compute_radius_of_gyration(parsed_pdb["coords"])
            send_event("metric", {"key": "rog", "message": f"RoG: {rog:.2f} A", "data": {"rog": rog}})

            # Step 4: Bond
            send_event("step", {"step": "bond", "message": "Computing bond geometry..."})
            bond = compute_bond_rmsd(parsed_pdb["coords"], parsed_pdb["atom_names"], parsed_pdb["residue_ids"])
            send_event("metric", {"key": "bond", "message": f"Bond RMSD: {bond['bond_rmsd']:.3f} A", "data": bond})

            # Step 5: SASA
            send_event("step", {"step": "sasa", "message": "Estimating solvent accessibility..."})
            sasa = compute_sasa_estimate(parsed_pdb["coords"], parsed_pdb["atom_names"])
            send_event("metric", {"key": "sasa", "message": f"SASA: {sasa['mean_sasa']:.4f}", "data": sasa})

            # Step 6: E2E
            send_event("step", {"step": "e2e", "message": "Computing end-to-end distance..."})
            e2e = compute_end_to_end(parsed_pdb["coords"])
            send_event("metric", {"key": "e2e", "message": f"End-to-end: {e2e:.2f} A", "data": {"e2e": e2e}})

            # Step 7: Shape
            send_event("step", {"step": "shape", "message": "Computing shape descriptors..."})
            shape = compute_shape_descriptors(parsed_pdb["coords"])
            send_event("metric", {"key": "shape", "message": f"Asphericity: {shape['asphericity']:.4f}", "data": shape})

            # Step 8: Backbone
            send_event("step", {"step": "backbone", "message": "Computing backbone angles..."})
            backbone = compute_backbone_angles(parsed_pdb["coords"], parsed_pdb["atom_names"], parsed_pdb["residue_ids"])
            send_event("metric", {"key": "backbone", "message": f"Mean angle: {backbone['mean_angle']:.1f} deg", "data": backbone})

            # Step 9: Pair satisfaction
            if parsed_pdb["is_nucleic"]:
                send_event("step", {"step": "pairs", "message": "Computing pair satisfaction..."})
                from torusfold.scheme2.pdb_analyzer import compute_pair_satisfaction
                pairs = compute_pair_satisfaction(parsed_pdb["coords"], parsed_pdb["residue_ids"],
                                                  parsed_pdb["atom_names"], parsed_pdb["residue_names"])
                send_event("metric", {"key": "pairs", "message": f"Pair rate: {pairs['satisfaction_rate']:.1%}", "data": pairs})

                # Step 10: A-form score
                send_event("step", {"step": "aform", "message": "Computing A-form geometry score..."})
                from torusfold.scheme2.pdb_analyzer import compute_aform_score
                aform = compute_aform_score(parsed_pdb["coords"], parsed_pdb["atom_names"])
                send_event("metric", {"key": "aform", "message": f"A-form score: {aform['aform_score']:.3f}", "data": aform})

                # Step 11: Stacking
                send_event("step", {"step": "stacking", "message": "Computing stacking analysis..."})
                from torusfold.scheme2.pdb_analyzer import compute_stacking_analysis
                stacking = compute_stacking_analysis(parsed_pdb["coords"], parsed_pdb["atom_names"])
                send_event("metric", {"key": "stacking", "message": f"Stacking: {stacking['stacking_fraction']:.1%}", "data": stacking})

            # Step 12: B-factor
            b_factors = parsed_pdb["b_factors"]
            if b_factors:
                b_arr = np.array(b_factors)
                b_stats = {"mean": round(float(np.mean(b_arr)), 2), "std": round(float(np.std(b_arr)), 2), "max": round(float(np.max(b_arr)), 2)}
                send_event("metric", {"key": "b_factor", "message": f"Mean B: {b_stats['mean']:.2f}", "data": b_stats})

            # Done
            send_event("done", {"message": "Analysis complete", "pdb_path": pdb_path, "n_atoms": parsed_pdb["n_atoms"]})

            # Clean up session
            _pdb_sessions.pop(session_id, None)

            # Close connection so client knows we're done
            try:
                self.wfile.write(b"\n")
                self.wfile.flush()
            except Exception:
                pass

        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected

    def _handle_feedback(self):
        """POST /api/feedback — save user feedback"""
        body = self._read_body()
        try:
            feedback = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        fb_path = os.path.join(ROOT, "feedback.json")
        existing = []
        if os.path.exists(fb_path):
            try:
                with open(fb_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []

        feedback["server_timestamp"] = time.time()
        existing.append(feedback)

        with open(fb_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        self._send_json({"ok": True, "count": len(existing)})

    def _handle_install_deps(self):
        """POST /api/install-deps — fetch what the external tools need.

        A sidecar to tools/install_deps.py rather than a reimplementation, and
        deliberately narrow: it installs pip packages, clones the two tool
        repositories that have upstream URLs, and fetches the one checkpoint that
        has a working direct link. It does NOT touch anything already on the
        machine, and it does not claim to do the parts that have no download
        source — those come back in the summary as manual steps.

        Runs detached so the browser is not held open by a multi-GB transfer; the
        installer's JSON events land in the same log the console panel already
        reads. POST {"cancel": true} stops a running one.
        """
        body = self._read_body()
        try:
            opts = json.loads(body) if body else {}
        except json.JSONDecodeError:
            opts = {}

        if opts.get("cancel"):
            self._handle_install_cancel()
            return

        if _install_state.get("running"):
            self._send_json({"error": "An install is already running",
                             "started_at": _install_state.get("started_at")}, 409)
            return

        skip_downloads = bool(opts.get("skip_downloads"))
        skip_git = bool(opts.get("skip_git"))
        # Installing packages changes the interpreter the user runs the pipeline
        # with, so it is opt-in: the default pass surveys and reports instead.
        install_missing = bool(opts.get("install_missing"))
        # Where model weights come from. Validated against the installer's own
        # list rather than a pattern: an allow-list cannot be talked around, and
        # it also turns a typo into a visible rejection instead of a silent
        # fallback that hides the mistake the user is trying to fix.
        source = str(opts.get("source") or "auto")
        if source not in _model_source_ids():
            self._send_json({"error": "unknown model source: %r" % source,
                             "allowed": sorted(_model_source_ids())}, 400)
            return
        script = os.path.join(ROOT, "tools", "install_deps.py")
        if not os.path.isfile(script):
            self._send_json({"error": "tools/install_deps.py not found"}, 500)
            return

        with _install_lock:
            _install_state.update({"running": True, "started_at": time.time(),
                                   "finished_at": None, "result": None})
        thread = threading.Thread(target=self._run_install,
                                  args=(script, sys.executable, skip_downloads, skip_git,
                                        source, install_missing),
                                  daemon=True)
        thread.start()
        self._send_json({"status": "started", "script": script,
                         "python": sys.executable, "source": source,
                         "install_missing": install_missing,
                         "skip_downloads": skip_downloads, "skip_git": skip_git})

    def _handle_install_cancel(self):
        """POST /api/install-deps {"cancel": true} — stop a running setup.

        A setup can legitimately take a long time (a 532 MB checkpoint over a slow
        mirror) and the browser has no other way to stop it. The child process tree
        is terminated, and the state is cleared here as well as in the pump thread
        so the flag cannot survive the cancellation.
        """
        with _install_lock:
            proc = _install_state.get("proc")
            running = _install_state.get("running")
        if not running:
            self._send_json({"ok": True, "note": "nothing was running"})
            return
        killed = False
        if proc is not None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                killed = True
            except (OSError, subprocess.SubprocessError):
                killed = False
        _emit_log("warn", "Environment setup cancelled by request")
        with _install_lock:
            _install_state.update({"running": False, "finished_at": time.time(),
                                   "proc": None,
                                   "result": {"installed": [], "skipped": [],
                                              "failed": [], "cancelled": True,
                                              "manual": []}})
        self._send_json({"ok": True, "killed": killed})

    def _run_install(self, script, python, skip_downloads, skip_git, source="auto",
                     install_missing=False):
        cmd = [python, "-u", script, "--python", python]
        if skip_downloads:
            cmd.append("--skip-downloads")
        if skip_git:
            cmd.append("--skip-git")
        if install_missing:
            cmd.append("--install-missing")
        if source and source != "auto":
            cmd += ["--source", source]

        # A sidecar's output has to reach BOTH the browser's console panel and the
        # server's own console. _emit_log alone only fills the in-memory buffer
        # that feeds SSE, so a terminal watching the server would show nothing for
        # the whole install. Writing to sys.stdout fixes that (the tee mirrors it
        # back through _emit_log), and _emit_log is still called when stdout is
        # not a usable stream.
        def say(level, message):
            stream = sys.stdout
            written = False
            if stream is not None and hasattr(stream, "write"):
                try:
                    stream.write(message + "\n")
                    stream.flush()
                    written = True
                except (OSError, ValueError):
                    written = False
            if not written:
                _emit_log(level, message)

        say("step", "Environment setup started")
        say("info", " ".join(cmd))
        summary = None
        try:
            try:
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace", bufsize=1)
                # Published so a cancel request can reach it. The installer can
                # legitimately run for many minutes on a slow mirror.
                with _install_lock:
                    _install_state["proc"] = proc
                for line in proc.stdout:
                    line = line.rstrip()
                    if not line:
                        continue
                    # The installer emits one JSON object per event; anything else
                    # is its human-readable tail and is passed through as-is.
                    if line.startswith("{"):
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            say("info", line)
                            continue
                        kind = event.get("event")
                        if kind == "progress":
                            pct = event.get("pct")
                            say("progress", "  %s: %s MB%s" % (
                                event.get("id"), event.get("mb"),
                                "" if pct is None else " (%.1f%%)" % pct))
                        elif kind == "step":
                            lvl = {"fail": "error", "skip": "info"}.get(event.get("status"), "info")
                            say(lvl, "  %s %s%s" % (
                                event.get("status", "").upper(), event.get("label", ""),
                                "" if not event.get("note") else " — " + event["note"]))
                        elif kind == "summary":
                            summary = event
                    else:
                        say("info", line)
                proc.wait(timeout=30)
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                say("error", "Environment setup failed to run: %s" % exc)
        finally:
            # This reset MUST happen on every path. It used to sit after the
            # try/except, so any exception the handlers did not name — or a
            # thread death — left install_running True forever and every later
            # request was answered "An install is already running" with no way
            # back short of restarting the server.
            ok = bool(summary) and not summary.get("failed")
            try:
                say("success" if ok else "warn",
                    "Environment setup finished%s" % ("" if ok else " with problems"))
            except Exception:                            # noqa: BLE001
                pass
            with _install_lock:
                _install_state.update({"running": False, "finished_at": time.time(),
                                       "proc": None, "result": summary})

    def _handle_predict(self):
        if _predict_state["status"] == "running":
            self._send_json({"error": "Prediction already running"}, 409)
            return

        body = self._read_body()
        try:
            params = json.loads(body) if body else {}
        except json.JSONDecodeError:
            params = {}

        sequence = params.get("sequence", "").strip().upper().replace("T", "U")
        if not sequence:
            self._send_json({"error": "No sequence provided"}, 400)
            return

        bad = [c for c in sequence if c not in "ACGNU"]
        if bad:
            self._send_json({"error": f"Invalid characters: {set(bad)}"}, 400)
            return

        # Coerce and bound-check against the registry before anything is queued,
        # so a malformed body is rejected here rather than as a TypeError 40s
        # into a run. The notes go to the SSE log so the user sees what changed.
        pipeline_params, notes = normalise_params(params)

        # Clear log buffer for new prediction
        _clear_logs()
        for note in notes:
            _emit_log("warn", f"parameter: {note}")

        job_id = str(uuid.uuid4())[:8]
        # Register before the thread starts, so a browser that refreshes the
        # instant after submitting can still find the job.
        _register_job(job_id, sequence, pipeline_params)
        thread = threading.Thread(
            target=self._run_prediction,
            args=(sequence, pipeline_params),
            daemon=True,
        )
        thread.start()
        self._send_json({"status": "started", "job_id": job_id, "length": len(sequence),
                         "params": sorted(pipeline_params), "notes": notes})

    def _run_prediction(self, sequence, params):
        global _predict_state
        _note_stage("0", "Secondary structure prediction")

        _emit_log("step", "=== TorusFold Pipeline Started ===")
        _emit_log("info", f"Sequence length: {len(sequence)} nt")

        orig_stdout = sys.stdout
        orig_stderr = sys.stderr

        # Capture stdout/stderr for SSE streaming, and mirror both to a per-job
        # log file. The mirror is what makes the run watchable from a second
        # terminal: `Get-Content -Wait output_web/logs/<job>.log`, or `tail -f`.
        # It goes INSIDE the tee so the file sees exactly what the browser sees.
        _mirror_path = os.path.join(ROOT, "output_web", "logs",
                                    "%s.log" % (_predict_state.get("job_id") or "job"))
        _stdout_mirror = _LogMirror(orig_stdout, _mirror_path)
        _stderr_mirror = _LogMirror(orig_stderr, _mirror_path)
        _predict_state["log_path"] = _mirror_path
        sys.stdout = _TeeWriter(_stdout_mirror, "info")
        sys.stderr = _TeeWriter(_stderr_mirror, "warn")
        _emit_log("info", "live log: %s" % _mirror_path)

        try:
            sys.path.insert(0, SRC)
            if SCHEME2_SRC:
                sys.path.insert(0, SCHEME2_SRC)

            # Monkey-patch OpenCL
            try:
                import openmm as _mm
                _orig = _mm.Platform.getPlatformByName
                def _safe_get(name):
                    if name in ("OpenCL", "CUDA"):
                        raise RuntimeError(f"Disabled: {name}")
                    return _orig(name)
                _mm.Platform.getPlatformByName = staticmethod(_safe_get)
            except ImportError:
                pass

            os.environ["OPENMM_CPU_THREADS"] = os.environ.get("OPENMM_CPU_THREADS", "16")

            # Level 0: Secondary structure
            _predict_state["current_level"] = 0
            _predict_state["message"] = "ViennaRNA secondary structure..."
            _predict_state["progress"] = 5
            _emit_log("step", "Level 0: ViennaRNA Partition Function BPP + confidence tiering")

            import RNA
            ss, mfe = RNA.fold(sequence)
            fc = RNA.fold_compound(sequence)
            fc.pf()

            _predict_state["message"] = f"MFE={mfe:.1f} kcal/mol"
            _predict_state["progress"] = 10
            _emit_log("info", f"MFE = {mfe:.1f} kcal/mol")
            _emit_log("info", f"SS length = {len(ss)}")

            out_dir = os.path.join(ROOT, "output_web")
            os.makedirs(out_dir, exist_ok=True)

            # Level 1+: Full pipeline.
            #
            # Every knob comes from the registry via normalise_params(), which
            # already dropped unknown keys and clamped the rest. Nothing is
            # hard-coded here on purpose: this block used to pin use_msa=False,
            # use_pyrosetta=False, use_ppr=False and md_step_scale=0.1, which
            # silently disabled two headline levels and disagreed with the demo
            # invocation in run_2013nt.py.
            #
            # verbose is forced on: the SSE log stream is built by teeing stdout,
            # so verbose=False would leave the console panel empty.
            call_kwargs = dict(params)
            call_kwargs["verbose"] = True
            _emit_log("info", "parameters: " + ", ".join(
                f"{k}={call_kwargs[k]}" for k in sorted(call_kwargs) if k != "verbose"))

            from torusfold.scheme2.isrnaclong import isrnaclong_pipeline
            result = isrnaclong_pipeline(
                sequence=sequence,
                secondary_structure=ss,
                output_dir=out_dir,
                **call_kwargs,
            )

            elapsed = time.time() - _predict_state["start_time"]
            _emit_log("success", f"Pipeline complete in {elapsed:.1f}s")

            # Read PDB file
            pdb_path = os.path.join(out_dir, "isrnaclong_final.pdb")
            pdb_text = ""
            if os.path.isfile(pdb_path):
                with open(pdb_path, "r", encoding="utf-8", errors="replace") as f:
                    pdb_text = f.read()

            # Build comprehensive result JSON
            details = getattr(result, "details", {}) or {}
            result_dict = _build_result_dict(
                result=result, details=details, pdb_text=pdb_text,
                sequence=sequence, ss=ss, mfe=mfe, elapsed=elapsed,
                pdb_path=pdb_path, out_dir=out_dir,
            )

            _publish_final("done", "Complete")
            _predict_state["progress"] = 100.0
            _predict_state["current_level"] = len(LEVEL_ORDER) - 1
            _predict_state["level_name"] = LEVEL_ORDER[-1][0]
            _predict_state["stage_label"] = "Complete"
            _predict_state["result"] = result_dict

        except Exception as exc:
            _emit_log("error", f"Pipeline failed: {exc}")
            _publish_final("error", str(exc), error=str(exc))
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    def _handle_log(self):
        """GET /api/log — the current job's log file.

        The browser already receives the same text over SSE, so this is for the
        case SSE cannot cover: a page opened after the run started, or one that
        reloaded and wants the part it missed. `?tail=N` limits it to the last N
        lines, which is what a 7-hour run needs — the whole file is megabytes.
        """
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            tail = int((query.get("tail") or ["0"])[0])
        except ValueError:
            tail = 0

        path = _predict_state.get("log_path")
        if not path or not os.path.isfile(path):
            self._send_json({"error": "No log for the current job yet",
                             "job_id": _predict_state.get("job_id")}, 404)
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as exc:
            self._send_json({"error": f"Could not read the log: {exc}"}, 500)
            return

        lines = text.splitlines()
        truncated = False
        if tail > 0 and len(lines) > tail:
            lines = lines[-tail:]
            truncated = True
        self._send_json({
            "job_id": _predict_state.get("job_id"),
            "log_path": path,
            "lines": len(lines),
            "truncated": truncated,
            "text": "\n".join(lines),
        })

    def _handle_structure(self, path):
        """GET /api/structure[/{name}] — the best finished structure so far.

        Without a name this reports which checkpoint is current, so the viewer can
        follow a run: a prediction leaves progressively better PDBs in the output
        directory, and showing them turns the empty 3D panel into the one thing
        that actually communicates progress.

        With a name it serves that file's bytes. The name is checked against the
        known stage list rather than joined onto a path: this endpoint reads from
        disk, and a caller-supplied path would make it a file-read primitive for
        anything on the machine.
        """
        name = ""
        if path.startswith("/api/structure/"):
            # The caller percent-escapes the name; stage paths can contain a
            # directory separator, and the allowlist below is what actually
            # constrains the request.
            name = unquote(path[len("/api/structure/"):]).strip()
        out_dir = os.path.join(ROOT, "output_web")

        if not name:
            stage = _display_structure(out_dir)
            if not stage:
                self._send_json({"available": False,
                                 "reason": "no finished structure yet"})
                return
            self._send_json({"available": True, "level": stage["level"],
                             "name": stage["name"], "desc": stage["desc"],
                             "atoms": stage["atoms"], "bytes": stage["bytes"],
                             "mtime": stage["mtime"],
                             "digest": stage.get("digest", "")})
            return

        mod = _viewer_module()
        allowed = {s[1] for s in mod.VIEWER_STAGES}
        delivered = {d["name"] for d in mod.DELIVERED_STRUCTURES}
        if name not in allowed and name not in delivered:
            self._send_json({"error": "unknown structure: %r" % name,
                             "allowed": sorted(allowed | delivered)}, 404)
            return
        # The name came from the URL, so confirm it still resolves inside the
        # directory it is allowed to come from before opening it. A delivered model
        # lives under the repository, the stages live under the output directory;
        # neither may reach outside its own root.
        if name in delivered:
            target = mod._abs(ROOT, name)
            root_real = os.path.realpath(ROOT)
        else:
            target = mod._abs(out_dir, name)
            root_real = os.path.realpath(out_dir)
        if not os.path.realpath(target).startswith(root_real + os.sep):
            self._send_json({"error": "path escapes its directory"}, 400)
            return
        if not os.path.isfile(target):
            self._send_json({"error": "not written yet", "name": name}, 404)
            return
        try:
            with open(target, "rb") as f:
                data = f.read()
        except OSError as exc:
            self._send_json({"error": "could not read: %s" % exc}, 500)
            return
        self.send_response(200)
        self.send_header("Content-Type", "chemical/x-pdb")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._set_cors()
        self.end_headers()
        self.wfile.write(data)

    def _handle_upload(self):
        body = self._read_body()
        if not body:
            self._send_json({"error": "Empty upload"}, 400)
            return
        tmp = os.path.join(ROOT, "output_web", "uploaded.pdb")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(body)
        self._send_json({"ok": True, "path": tmp, "size": len(body)})

    def log_message(self, format, *args):
        # Write to the process's original stderr, NOT to sys.stderr. A prediction
        # swaps sys.stderr for the tee that feeds the job log, so using sys.stderr
        # here interleaved HTTP request lines into the middle of the pipeline's
        # own log — the very file someone is following in a terminal.
        (_PROCESS_STDERR or sys.stderr).write(
            f"[{self.log_date_time_string()}] {format % args}\n")


# ── Result builder ───────────────────────────────────────────────
def _build_result_dict(result, details, pdb_text, sequence, ss, mfe, elapsed, pdb_path, out_dir):
    """Build the comprehensive result JSON from pipeline output."""
    L = len(sequence)

    # Physical signals
    physical = {
        "closure_distance_Ang": details.get("closure_distance", getattr(result, "closure_error", 0)),
        "bond_rmsd_Ang": details.get("bond_rmsd", 0),
        "bond_mean_Ang": details.get("bond_mean", 5.9),
        "sasa_mean": details.get("sasa_mean", 0),
        "sasa_bsj": details.get("sasa_bsj", 0),
        "bsj_closure_tightness": details.get("bsj_closure_tightness", 0),
        "dsRNA_fraction": details.get("dsRNA_fraction", 0),
        "mean_pair_prob": details.get("mean_pair_prob", 0),
        "long_range_pair_fraction": details.get("long_range_pair_fraction", 0),
    }

    # Immune
    immune = {
        "buried_motif_count": details.get("buried_motif_count", 0),
        "ires_3d_accessibility": details.get("ires_3d_accessibility", 0),
        "motif_accessibility": details.get("motif_accessibility", {}),
    }

    # circDesign
    circdesign = {
        "mfe_kcal_mol": mfe,
        "mfe_per_nt": round(mfe / L, 3) if L else 0,
        "cai_human": details.get("cai_human", 0),
        "ires_deviation_L2_clamped": details.get("ires_deviation", 0),
        "ires_length": details.get("ires_length", 0),
        "cds_length": details.get("cds_length", 0),
    }

    # Stem loops
    stem_loops = {
        "count": details.get("stem_loop_count", 0),
        "stem_lengths": details.get("stem_lengths", []),
        "loop_lengths": details.get("loop_lengths", []),
        "mean_stem_length": details.get("mean_stem_length", 0),
        "mean_loop_length": details.get("mean_loop_length", 0),
    }

    # rsRNASP1. The pipeline computes this on the final all-atom structure and
    # reports it under details["final"]["rsrnasp1"]. It previously read a flat
    # `rsrasp1_energy` key that nothing ever wrote, and the summary field was
    # hard-coded to None, so this row could not display a value at all.
    #
    # Absent stays None rather than defaulting to 0: the pass thresholds in
    # web/modules/panels.js would render a 0 as a hard FAIL. More negative is
    # better, and the value is only comparable within one sequence length.
    _final = details.get("final") or {}
    _rsrnasp1 = _final.get("rsrnasp1")
    rsRNASP1 = {
        "score_all_atom": _rsrnasp1,
        "score_per_nt": (round(_rsrnasp1 / details["input"]["sequence_length"], 4)
                         if _rsrnasp1 is not None
                         and (details.get("input") or {}).get("sequence_length")
                         else None),
    }

    # rnadvisor scores (same rule: absent means absent, not zero).
    # DFIRE and 3dRNAscore are not computed anywhere in this tree; they are keyed
    # here so the interface shows "--" rather than a fabricated number.
    rnadvisor = {
        "rsRNASP_docker": _rsrnasp1,
        "DFIRE": details.get("DFIRE"),
        "3drnascore": details.get("3drnascore"),
    }

    # Structural 3D
    structural_3d = {
        "radius_of_gyration_A": details.get("radius_of_gyration", 0),
        "pair_satisfaction_rate": getattr(result, "pair_rate", 0),
        "contact_order_pct": details.get("contact_order_pct", 0),
        "backbone_p_pp_angle_deg": details.get("backbone_p_pp_angle", 0),
        "contour_length_A": details.get("contour_length", 0),
        "end_to_end_distance_A": physical["closure_distance_Ang"],
        "pair_distance_distribution": details.get("pair_distance_distribution", {}),
    }

    # Shape
    shape_3d = {
        "asphericity": details.get("asphericity", 0),
        "prolateness": details.get("prolateness", 0),
        "rog_A": structural_3d["radius_of_gyration_A"],
        "eigenvalues": details.get("eigenvalues", [0, 0, 0]),
    }

    # Pairing
    pairing_quality = {
        "wc_pairs": details.get("wc_pairs", 0),
        "wobble_pairs": details.get("wobble_pairs", 0),
        "total": details.get("total_pairs", 0),
    }
    pairing_quality["wc_pct"] = round(
        100 * pairing_quality["wc_pairs"] / max(1, pairing_quality["total"]), 1
    )
    pairing_quality["wobble_pct"] = round(
        100 * pairing_quality["wobble_pairs"] / max(1, pairing_quality["total"]), 1
    )

    # Sequence composition
    seq_comp = {
        "gc_pct": round(100 * sum(1 for c in sequence if c in "GC") / max(1, L), 1),
        "A": sequence.count("A"),
        "U": sequence.count("U"),
        "G": sequence.count("G"),
        "C": sequence.count("C"),
        "length": L,
    }

    # Per-residue
    per_residue = {
        "top_penalized": details.get("top_penalized", []),
        "region_positive_energy": details.get("region_positive_energy", {}),
    }

    # IRES/CDS bounds
    ires_start = details.get("ires_start", int(L * 0.4))
    ires_end = details.get("ires_end", int(L * 0.74))
    cds_start = ires_end
    cds_end = L

    return {
        "pdb": pdb_text,
        "structure_note": "Level 4 REST2 output",
        "length": L,
        "viennarna_version": "2.7.2",
        "ires_bounds": {"start": ires_start, "end": ires_end, "length": ires_end - ires_start},
        "cds_bounds": {"start": cds_start, "end": cds_end, "length": cds_end - cds_start},
        "physical": physical,
        "immune": immune,
        "circdesign": circdesign,
        "stem_loops": stem_loops,
        "rsRNASP1": rsRNASP1,
        "per_residue": per_residue,
        "rnadvisor": rnadvisor,
        "structural_3d": structural_3d,
        "shape_3d": shape_3d,
        "pairing_quality": pairing_quality,
        "sequence_composition": seq_comp,
        "method": getattr(result, "method", "rhofoldcirclong"),
        "runtime": elapsed,
        "ss": ss,
        "mfe": mfe,
        "pdb_path": pdb_path,
    }


def _shape_anisotropy(eigenvalues):
    """Anisotropy of the gyration tensor, from its eigenvalues.

    The panel plots a "Shape Moment" and nothing in this repository ever supplied
    one, so there is no existing definition to match. Rather than invent a formula
    and label it with a name that implies a convention, this computes a quantity
    with a stated meaning: the variance of the normalised eigenvalues, which is the
    standard measure of how far a shape departs from spherical.

        0    a sphere (all three eigenvalues equal)
        1    a fully extended rod (one eigenvalue carries everything)

    The previous attempt mapped 1 - 3*sum(p^2), which is the same idea written with
    the wrong offset: it returns 0 for a sphere but goes negative for anything
    elongated — it produced -0.56 for a 139 nt trace while claiming to be 0..1.
    This form cannot leave the range.

    Returns None when it cannot be computed, so the panel shows "--" rather than a
    plausible-looking number.
    """
    try:
        vals = sorted((abs(float(v)) for v in (eigenvalues or [])
                       if v is not None), reverse=True)
        if len(vals) != 3:
            return None
        total = sum(vals)
        if total <= 0:
            return None
        p = [v / total for v in vals]
        mean = sum(p) / 3.0
        variance = sum((x - mean) ** 2 for x in p) / 3.0
        # Variance of three non-negative values summing to 1: 0 for
        # (1/3,1/3,1/3) and 2/9 for (1,0,0), which are its minimum and maximum.
        # Dividing by 2/9 maps those onto 0 and 1 exactly.
        return round(variance / (2.0 / 9.0), 4)
    except (TypeError, ValueError):
        return None


_LIVE_METRICS = {"digest": None, "payload": None, "at": 0.0, "computing": None}
_LIVE_METRICS_LOCK = threading.Lock()


def _live_metrics(stage):
    """Measure the structure on screen, for the readout panels — without blocking.

    Why this exists: every panel on the right is built from `result`, and the server
    produces `result` only when a run finishes. So for the whole of a run — hours —
    the right-hand tabs held nothing but their static labels and a row of "--",
    which is what they looked like: broken.

    This must never run on the request path. `analyze_pdb` costs 10.8 s on the
    committed 2,013 nt model (42,831 atoms — the Shrake-Rupley SASA dominates), and
    /api/current is polled every few seconds. Called synchronously it stalled the
    status endpoint for 24 s on a quiet page and would have done so repeatedly.

    So: return whatever is already computed and start a background pass when the
    structure has changed. The first poll after a new checkpoint shows the previous
    numbers for a moment; every poll after that is instant. A stale-but-real number
    with a known source beats a responsive measurement that freezes the interface.
    """
    if not stage or not stage.get("path"):
        return None
    digest = stage.get("digest") or ""

    with _LIVE_METRICS_LOCK:
        fresh = bool(digest) and _LIVE_METRICS["digest"] == digest
        cached = _LIVE_METRICS["payload"]
        already = _LIVE_METRICS["computing"] == digest
        if not fresh and not already:
            _LIVE_METRICS["computing"] = digest
            threading.Thread(
                target=_compute_live_metrics,
                args=(dict(stage), digest),
                daemon=True,
                name="torusfold-metrics",
            ).start()
    if fresh:
        return cached
    # Not measured yet: hand back the previous measurement, tagged with what it was
    # measured from, so the panel can say so rather than imply it is current.
    if cached:
        return cached
    return {"live": True, "pending": True,
            "source": {"name": stage.get("name"), "level": stage.get("level")}}


def _compute_live_metrics(stage, digest):
    """The expensive half. Runs on its own thread; never raises."""
    t0 = time.time()
    try:
        payload = _measure_structure(stage)
    except Exception as exc:                                 # noqa: BLE001
        import traceback as _tb
        payload = {"live": True,
                   "error": "%s: %s" % (type(exc).__name__, exc),
                   "trace": _tb.format_exc()[-900:]}
    with _LIVE_METRICS_LOCK:
        _LIVE_METRICS.update({"digest": digest, "payload": payload,
                              "at": time.time(), "computing": None})
    # Say so in the run log. This thread is invisible otherwise, and a silent
    # worker that never reports is indistinguishable from one that never ran —
    # which is exactly how it presented the first time: the panels stayed on their
    # placeholder values and nothing anywhere said why.
    try:
        if payload.get("error"):
            _emit_log("warn", "metrics: could not measure %s (%s)"
                      % (stage.get("name"), payload["error"]))
        else:
            _emit_log("info", "metrics: measured %s in %.1fs"
                      % (stage.get("name"), time.time() - t0))
    except Exception:
        pass


def _measure_structure(stage):
    """Parse and measure one structure file. Returns the panel payload.

    Runs on a worker thread, never on the request path — see _live_metrics.
    """
    payload = {"live": True, "error": "measurement did not run"}
    try:
        if SRC not in sys.path:
            sys.path.insert(0, SRC)
        from torusfold.scheme2 import pdb_analyzer as pa
        import numpy as _np

        with open(stage["path"], "r", errors="replace") as f:
            text = f.read()

        # One parse, shared. analyze_pdb parses the file itself, so calling both it
        # and parse_pdb parsed the same 42,831-atom file twice — 11 s of duplicated
        # work on the delivered model.
        parsed = pa.parse_pdb(text)
        a = pa.analyze_pdb(text)
        coords = parsed["coords"]
        names = parsed["atom_names"]

        # Residue order and index, built once. Several steps below need it and each
        # used to rebuild it.
        order = list(dict.fromkeys(parsed["residue_ids"]))
        pos_of = {rid: i for i, rid in enumerate(order)}
        seq = []
        _one = {"A": "A", "ADE": "A", "U": "U", "URA": "U",
                "G": "G", "GUA": "G", "C": "C", "CYT": "C"}
        _last = object()
        for rn, rid in zip(parsed["residue_names"], parsed["residue_ids"]):
            if rid != _last:
                seq.append(_one.get(rn, "N"))
                _last = rid

        # Phosphorus positions: the coarse-grained trace, and the atom the pipeline's
        # own BSJ metric uses.
        p_atoms = [i for i, n in enumerate(names) if n == "P"]

        # Closure: the gap that has to be closed for a circular RNA.
        closure = None
        if len(p_atoms) >= 2:
            closure = float(_np.linalg.norm(coords[p_atoms[0]] - coords[p_atoms[-1]]))

        # Pair satisfaction, and the pair statistics derived from the same pass.
        pair_rate = None
        pair_breakdown = None
        pairing_quality = None
        pair_range = None
        pair_dist = None
        per_residue = None
        try:
            ps = pa.compute_pair_satisfaction(coords, parsed["residue_ids"],
                                              names, parsed["residue_names"])
            # The function returns `satisfaction_rate`, not a key named after the
            # panel's field. Reading the wrong name yielded None while the metric
            # itself was fine.
            pair_rate = ps.get("satisfaction_rate")
            pair_breakdown = {
                "total_pairs": ps.get("total_pairs"),
                "wc_eligible": ps.get("wc_eligible_count"),
                "satisfied": ps.get("satisfied_count"),
                "mean_pair_distance": ps.get("mean_pair_distance"),
            }

            wc = ps.get("satisfied_count") or 0
            eligible = ps.get("wc_eligible_count") or 0

            # One pass over the phosphorus pairs, distance-bucketed, counting the
            # wobble pairs as it goes. This was three separate nested loops over the
            # same pairs.
            satisfied_lt = partial = unsatisfied = wobble = 0
            nP = len(p_atoms)
            for x in range(nP):
                px = coords[p_atoms[x]]
                rx = parsed["residue_ids"][p_atoms[x]]
                ix = pos_of.get(rx)
                for y in range(x + 1, nP):
                    iy = pos_of.get(parsed["residue_ids"][p_atoms[y]])
                    if ix is None or iy is None or abs(ix - iy) < 4:
                        continue
                    bx, by = seq[ix], seq[iy]
                    pair = {bx, by}
                    if pair not in ({"A", "U"}, {"G", "C"}, {"G", "U"}):
                        continue
                    d = float(_np.linalg.norm(px - coords[p_atoms[y]]))
                    if d < 15.0:
                        satisfied_lt += 1
                    elif d <= 30.0:
                        partial += 1
                    else:
                        unsatisfied += 1
                    if d <= 12.0 and pair == {"G", "U"}:
                        wobble += 1

            pairing_quality = {
                "wc_pairs": wc,
                "wc_pct": round(100.0 * wc / eligible, 1) if eligible else 0.0,
                "wobble_pairs": wobble,
                "wobble_pct": round(100.0 * wobble / eligible, 1) if eligible else 0.0,
                "total": eligible,
            }

            # Pair distance ranges, straight off the analyzer's own breakdown.
            br = ps.get("by_range") or {}
            total_pairs = sum((br.get(k) or {}).get("count", 0)
                              for k in ("local", "medium", "long")) or 0
            pair_range = {}
            for key, base in (("local", "local"), ("medium", "medium"), ("long", "long")):
                cnt = (br.get(key) or {}).get("count", 0)
                pair_range[base + ("_lt50" if key == "local"
                                   else "_50_500" if key == "medium" else "_gt500")] = cnt
                pair_range[base + "_pct"] = (round(100.0 * cnt / total_pairs, 1)
                                             if total_pairs else 0.0)

            pair_dist = {
                "satisfied_lt15A": {"count": satisfied_lt},
                "partial_15_30A": {"count": partial},
                "unsatisfied_gt30A": {"count": unsatisfied},
            }

            # Per-residue series for the strip charts: SASA and the B-factor column
            # are per-atom in the file, reduced here to one value per residue.
            try:
                per_atom_sasa = (a.get("sasa") or {}).get("per_atom_sasa") or []
                bf = parsed.get("b_factors") or []
                sasa_by_res = [[] for _ in order]
                b_by_res = [[] for _ in order]
                for i, rid in enumerate(parsed["residue_ids"]):
                    k = pos_of.get(rid)
                    if k is None:
                        continue
                    if i < len(per_atom_sasa):
                        sasa_by_res[k].append(float(per_atom_sasa[i]))
                    if i < len(bf):
                        b_by_res[k].append(float(bf[i]))

                def _mean(xs):
                    return (sum(xs) / len(xs)) if xs else 0.0

                per_residue = {
                    "sasa_per_residue": [round(_mean(v), 3) for v in sasa_by_res],
                    "bfactor_per_residue": [round(_mean(v), 3) for v in b_by_res],
                }
            except Exception:
                per_residue = None
        except Exception:
            # A metric that cannot be computed stays absent; the panels show "--"
            # rather than a number that means nothing.
            pass

        payload = {
            "live": True,
            "source": {"level": stage.get("level"), "name": stage.get("name"),
                       "atoms": stage.get("atoms"),
                       "delivered": bool(stage.get("delivered"))},
            "physical": {
                "closure_distance_Ang": closure,
                "bond_rmsd_Ang": (a.get("bond") or {}).get("bond_rmsd"),
                "sasa_mean": (a.get("sasa") or {}).get("mean_sasa"),
                "radius_of_gyration_A": a.get("rog"),
                "end_to_end_distance_A": a.get("end_to_end"),
            },
            "structural_3d": {
                "clash_count": (a.get("clash") or {}).get("clash_count"),
                "clash_score": (a.get("clash") or {}).get("clash_score"),
                "pair_satisfaction_rate": pair_rate,
                "pair_breakdown": pair_breakdown,
                "pair_distance_distribution": pair_dist,
                "contact_order_pct": None,
                "contour_length_A": None,
                "radius_of_gyration_A": a.get("rog"),
                "asphericity": (a.get("shape") or {}).get("asphericity"),
                "prolateness": (a.get("shape") or {}).get("prolateness"),
                "eigenvalues": (a.get("shape") or {}).get("eigenvalues"),
                "aform_score": (a.get("aform") or {}).get("aform_score"),
                "stacking": a.get("stacking"),
            },
            "shape_3d": {
                "asphericity": (a.get("shape") or {}).get("asphericity"),
                "prolateness": (a.get("shape") or {}).get("prolateness"),
                "rog_A": a.get("rog"),
                "eigenvalues": (a.get("shape") or {}).get("eigenvalues"),
                # How far the shape departs from spherical, 0..1. Named
                # shape_moment because that is the key the panel reads.
                "shape_moment": _shape_anisotropy((a.get("shape") or {}).get("eigenvalues")),
            },
            "pairing_quality": pairing_quality,
            "pair_range": pair_range,
            "per_residue": per_residue,
            "sequence": {"length": a.get("n_residues"), "n_atoms": a.get("n_atoms")},
        }
    except Exception as exc:                             # noqa: BLE001
        # Include the traceback: a bare message from this block cost a round trip to
        # diagnose once already.
        import traceback as _tb
        payload = {"live": True,
                   "error": "%s: %s" % (type(exc).__name__, exc),
                   "trace": _tb.format_exc()[-900:]}
    # The caller caches this; see _compute_live_metrics. Returning rather than
    # caching here keeps the two jobs apart.
    return payload


_TOOL_SUMMARY = [
    # (label, env var holding the root, relative path that must exist under it,
    #  whether the pipeline is meaningfully degraded without it)
    ("RhoFold+",       "RHOFOLD_ROOT",       os.path.join("pretrained", "rhofold_pretrained_params.pt"), True),
    ("RNAbpFlow",      "RNABPFLOW_ROOT",     os.path.join("checkpoint", "RNA3DB.ckpt"),                 True),
    ("DivideFold",     "TF_DIVIDEFOLD_ROOT", None,                                                      True),
    ("trRosettaRNA2",  "TRRNA2_RUNNER",      None,                                                      False),
    ("isRNAcirc",      "ISRNACIRC_BIN_DIR",  None,                                                      True),
]


def _print_tool_summary():
    """One line per external tool: is it configured, and does the path exist.

    This exists because an unset ROOT variable does not stop the pipeline. Each
    predictor is called inside a try, so a missing one is skipped and the run
    produces a worse ensemble with no visible error on this side — the only trace
    was a line in the job log that appears minutes into a run. Reporting it at
    startup turns a silent quality loss into something the operator sees before
    pressing Predict.
    """
    if _APPLIED_ENV:
        print("  .env.local: %d variable(s) applied (%s)"
              % (len(_APPLIED_ENV), ", ".join(sorted(_APPLIED_ENV)[:4])
                 + (", ..." if len(_APPLIED_ENV) > 4 else "")))
    else:
        print("  .env.local: not present (or already in the environment)")

    for label, var, subpath, required in _TOOL_SUMMARY:
        root = os.environ.get(var, "").strip()
        if not root:
            state = "NOT SET"
        else:
            probe = os.path.join(root, subpath) if subpath else root
            if os.path.exists(probe):
                state = "ok"
            elif os.path.isdir(root) or os.path.isfile(root):
                # The directory is right but the expected file under it is not.
                state = "root ok, %s MISSING" % (subpath or "path")
            else:
                state = "path does not exist"
        if state == "ok":
            print("    %-15s ok        %s" % (label, root))
        else:
            print("    %-15s %-9s %s" % (label, state, root or "(%s unset)" % var))

    # isRNAcirc needs a second path, and its absence does not look like one. The
    # binary loads cleanly and then exits with "Wrong coeffDIR" — naming the
    # directory, never the missing files — when the five AA_*.dat templates are
    # not in the directory it is handed. On the distributed package they are in
    # Data/data/IsRNA2/, not Data/. Checking them here costs nothing and is the
    # difference between "configured" and "will actually convert".
    coeff = os.environ.get("CG_TO_ALLATOM_COEFF", "").strip()
    if coeff:
        needed = ("AA_baseA.dat", "AA_baseG.dat", "AA_baseC.dat",
                  "AA_baseU.dat", "AA_backbone.dat")
        missing = [n for n in needed if not os.path.isfile(os.path.join(coeff, n))]
        if missing:
            print("    %-15s %-9s %s" % ("CG coeff", "INCOMPLETE",
                                         "missing " + ", ".join(missing)))
        else:
            print("    %-15s ok        %s" % ("CG coeff", coeff))
    elif os.environ.get("ISRNACIRC_BIN_DIR"):
        print("    %-15s %-9s %s" % ("CG coeff", "NOT SET",
                                     "CG_TO_ALLATOM_COEFF unset; the wrapper will"
                                     " search the isRNAcirc tree for the templates"))

    # PyRosetta, with its licence terms. It is not a download — it has to be
    # obtained under a licence the user arranges — so "absent" and "you must
    # license this" are different messages, and only the second is actionable.
    # Reported here because it appeared in no report at all before: Level 2.6 could
    # be quietly skipped and nothing said why.
    try:
        sys.path.insert(0, os.path.join(ROOT, "tools"))
        import _pyrosetta as _pj
        _p = _pj.probe()
        if _p.get("available"):
            print("    %-15s ok        %s" % ("PyRosetta", _p.get("path") or ""))
        else:
            print("    %-15s %-9s %s" % ("PyRosetta", "not found",
                                         _p.get("why", "")))
        for _line in _wrap_text(_p.get("licence", ""), 66):
            print("      ! %s" % _line)
    except Exception:
        pass


def _wrap_text(text, width):
    """Wrap a note onto lines of at most `width`, on word boundaries."""
    words = str(text).split()
    lines = []
    current = ""
    for w in words:
        if current and len(current) + 1 + len(w) > width:
            lines.append(current)
            current = w
        else:
            current = (current + " " + w).strip()
    if current:
        lines.append(current)
    return lines or [""]


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8877
    os.chdir(ROOT)

    # Reading the pipeline's defaults means importing it, which pulls in torch and
    # takes seconds. Do it on a background thread so the first schema request and
    # the first prediction do not each pay for it while the UI waits.
    def _warm():
        n = len(_pipeline_defaults())
    threading.Thread(target=_warm, daemon=True).start()

    # ThreadingHTTPServer, not HTTPServer. The SSE log stream holds its
    # connection open for the whole run, and a single-threaded server serves one
    # request at a time — so while a browser was streaming, every other request
    # (including the status polls that drive the progress bar) queued behind it
    # and the UI appeared frozen. Each connection now gets its own thread.
    server = ThreadingHTTPServer(("0.0.0.0", port), TorusFoldHandler)
    server.daemon_threads = True
    print(f"TorusFold server: http://127.0.0.1:{port}/")
    print(f"  Static root: {ROOT}")
    print(f"  Web dir: {WEB_DIR}")
    # Printing which external tools resolved, at startup, is the point: every
    # predictor fails quietly when its ROOT variable is unset — the run continues
    # and simply produces a worse ensemble — so the only place this was visible
    # was a line in the job log, minutes in.
    _print_tool_summary()
    # After the tool table, so a background thread cannot interleave its output
    # into the middle of it. This line used to be printed by the warm-up thread
    # and landed between the tools and the PyRosetta note.
    try:
        print(f"  parameter defaults loaded: {len(_pipeline_defaults())}")
    except Exception:
        pass
    print(f"  GET  /api/schema   — tunable parameters, defaults and bounds")
    print(f"  POST /api/predict  — run pipeline")
    print(f"  GET  /api/sse/{{jid}} — SSE streaming")
    print(f"  POST /api/feedback — save feedback")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
