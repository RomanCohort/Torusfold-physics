# -*- coding: utf-8 -*-
"""Run ONE TorusFold prediction as a child process, so that /api/stop can really kill it.

WHY A PROCESS AND NOT A THREAD. serve.py used to call isrnaclong_pipeline in a thread of its own, and a
stop then had to be cooperative: a flag the pipeline notices at a stage boundary. That works for a stage
that prints, and it CANNOT work for Level 1, because the heavy work -- RhoFold+ inference, the segment
ensembles -- runs inside a ThreadPoolExecutor and the calling thread waits on it in a C-level wait where a
pending async exception is not delivered. Measured on this machine, with the correct thread id:

    24 x PyThreadState_SetAsyncExc(worker_thread, _StopRequested) -> rc=1, spread over 20 s
    live threads at the time: MainThread, Thread-8 (_run_prediction), 4 x ThreadPoolExecutor-0_N, ...
    result: the run never left Level 1, and only killing the server stopped it.

A child process has no such problem: taskkill /T /F takes the process, its thread pool and anything it
spawned, while the server that owns the port, the panel and the SSE stream stays up.

WHAT THIS FILE DOES. Reads a job spec (JSON, argv[1]) -- sequence, dot-bracket, pipeline kwargs, output
directory, result path, stop-file path -- runs the pipeline exactly as serve.py used to call it, prints the
pipeline's own output on stdout (the parent streams that into the panel), and writes the pieces of the
result the parent cannot recompute to the result path. It also watches for the parent's stop file at every
print, so the ordinary stop leaves the run at a level boundary with its checkpoint consistent; only a child
that does not reach a print gets killed.

EXIT CODES. 0 done, 2 failed (traceback on stderr, one-line reason in <result>.error), 3 stopped on request.
"""
import json
import os
import sys
import time
import traceback


class _Stop(Exception):
    """Raised inside this process when the parent's stop file appears."""


class _StopWatch:
    """sys.stdout/sys.stderr wrapper that checks the stop file once per write.

    The pipeline prints at every stage boundary and every few thousand MD steps, so this is where a
    cooperative stop is noticed -- the same place serve.py used to put it, one process out.
    """

    def __init__(self, stream, stop_file):
        self._s = stream
        self._stop_file = stop_file or ""
        self.encoding = getattr(stream, "encoding", "utf-8")
        self.errors = getattr(stream, "errors", "replace")

    def write(self, s):
        if self._stop_file and os.path.exists(self._stop_file):
            raise _Stop()
        return self._s.write(s)

    def flush(self):
        try:
            return self._s.flush()
        except Exception:
            return None

    def __getattr__(self, name):
        return getattr(self._s, name)


def main():
    spec_path = sys.argv[1]
    # utf-8-sig: tolerate a BOM, which Windows PowerShell's Out-File writes by default. The server writes
    # this file with plain utf-8, so this only matters when a human writes a spec by hand.
    with open(spec_path, "r", encoding="utf-8-sig") as fh:
        spec = json.load(fh)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (os.path.join(root, "src"), os.path.join(root, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    extra = os.environ.get("TF_SCHEME2_SRC", "")
    if extra:
        sys.path.insert(0, extra)

    # The same two things serve.py did before calling the pipeline in-process, kept identical on purpose:
    # OpenCL/CUDA are refused (the ROCm OpenMM platform aborts on this machine) and the CPU thread count is
    # pinned so a level does not try to use every core it can see.
    try:
        import openmm as _mm
        _orig = _mm.Platform.getPlatformByName
        def _safe(name):
            if name in ("OpenCL", "CUDA"):
                raise RuntimeError("Disabled: %s" % name)
            return _orig(name)
        _mm.Platform.getPlatformByName = staticmethod(_safe)
    except ImportError:
        pass
    os.environ["OPENMM_CPU_THREADS"] = os.environ.get("OPENMM_CPU_THREADS", "16")

    stop_file = spec.get("stop_file") or ""
    result_path = spec["result_path"]
    out_stdout, out_stderr = sys.stdout, sys.stderr
    sys.stdout = _StopWatch(out_stdout, stop_file)
    sys.stderr = _StopWatch(out_stderr, stop_file)

    t0 = time.time()
    try:
        from torusfold.scheme2.isrnaclong import isrnaclong_pipeline
        kwargs = dict(spec.get("params") or {})
        kwargs["verbose"] = True
        print("[runner] child pid %d, %d nt, output %s"
              % (os.getpid(), len(spec["sequence"]), spec["output_dir"]), flush=True)
        result = isrnaclong_pipeline(
            sequence=spec["sequence"],
            secondary_structure=spec["ss"],
            output_dir=spec["output_dir"],
            **kwargs,
        )
    except _Stop:
        sys.stdout, sys.stderr = out_stdout, out_stderr
        print("[runner] stop file seen: leaving at the level boundary, checkpoint kept", flush=True)
        return 3
    except BaseException as exc:                     # noqa: BLE001 -- the parent needs a reason either way
        sys.stdout, sys.stderr = out_stdout, out_stderr
        traceback.print_exc()
        try:
            with open(result_path + ".error", "w", encoding="utf-8") as fh:
                fh.write("%s: %s\n" % (type(exc).__name__, exc))
        except OSError:
            pass
        return 2
    finally:
        sys.stdout, sys.stderr = out_stdout, out_stderr

    payload = {
        "details": getattr(result, "details", {}) or {},
        "closure_error": getattr(result, "closure_error", 0),
        "pair_rate": getattr(result, "pair_rate", 0),
        "method": getattr(result, "method", "rhofoldcirclong"),
        "seconds": round(time.time() - t0, 1),
    }
    with open(result_path, "w", encoding="utf-8") as fh:
        # default=str: details carries numpy scalars and Paths, and the parent only reads it as JSON.
        json.dump(payload, fh, default=str)
    print("[runner] result written to %s" % result_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
