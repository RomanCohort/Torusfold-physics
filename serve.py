# -*- coding: utf-8 -*-
"""TorusFold Web Server — SSE streaming + Predict API + static files

Usage: python serve.py [port]
Default port: 8877
"""
import os
import sys
import io
import json
import time
import threading
import tempfile
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
WEB_DIR = os.path.join(SRC, "torusfold", "web")

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

# ── stdout/stderr capture for pipeline output ────────────────────
class _TeeWriter:
    """Wraps a file-like object, echoing each line to _emit_log."""
    def __init__(self, original, level="info"):
        self._orig = original
        self._level = level
        self._buf = ""

    def write(self, s):
        if self._orig:
            self._orig.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                _emit_log(self._level, line)

    def flush(self):
        if self._orig:
            self._orig.flush()
        if self._buf.strip():
            _emit_log(self._level, self._buf.strip())
            self._buf = ""

# ── Global prediction state ─────────────────────────────────────
_predict_state = {
    "status": "idle",       # idle | running | done | error
    "progress": 0,          # 0-100
    "current_level": -1,
    "message": "",
    "result": None,
    "error": None,
    "start_time": 0,
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
        else:
            self._send_json({"error": f"Unknown POST endpoint: {path}"}, 404)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path in ("/api/health", "/health"):
            self._send_json({"ok": True, "status": _predict_state["status"]})
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
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self._set_cors()
        self.end_headers()
        self.wfile.write(data)

    def _handle_status(self):
        self._send_json(_predict_state)

    def _handle_job_status(self, path):
        """GET /api/jobs/{jid} — return task status"""
        jid = path.split("/")[-1]
        self._send_json(_predict_state)

    def _handle_job_result(self, path):
        """GET /api/result/{jid} — return prediction result"""
        jid = path.split("/")[-1]
        if _predict_state["status"] == "done" and _predict_state["result"]:
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
        try:
            while True:
                with _log_lock:
                    new_entries = _log_entries[last_idx:]
                    last_idx = len(_log_entries)
                    status = _predict_state["status"]
                    progress = _predict_state.get("progress", 0)
                    current_level = _predict_state.get("current_level", -1)
                    message = _predict_state.get("message", "")

                for entry in new_entries:
                    data = json.dumps(entry, ensure_ascii=False)
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()

                # Heartbeat with status
                heartbeat = json.dumps({
                    "level": "heartbeat",
                    "status": status,
                    "progress": progress,
                    "current_level": current_level,
                    "message": message,
                })
                self.wfile.write(f"data: {heartbeat}\n\n".encode("utf-8"))
                self.wfile.flush()

                if status in ("done", "error"):
                    # Send final event
                    event_type = "done" if status == "done" else "error"
                    final = json.dumps({
                        "event": event_type,
                        "status": status,
                        "message": message,
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

        # Clear log buffer for new prediction
        _clear_logs()

        import uuid
        job_id = str(uuid.uuid4())[:8]
        _predict_state["job_id"] = job_id
        thread = threading.Thread(
            target=self._run_prediction,
            args=(sequence, params),
            daemon=True,
        )
        thread.start()
        self._send_json({"status": "started", "job_id": job_id, "length": len(sequence)})

    def _run_prediction(self, sequence, params):
        global _predict_state
        _predict_state.update({
            "status": "running", "progress": 0,
            "current_level": 0, "message": "Starting...",
            "result": None, "error": None, "start_time": time.time(),
        })
        _emit_log("step", "=== TorusFold Pipeline Started ===")
        _emit_log("info", f"Sequence length: {len(sequence)} nt")

        # Capture stdout/stderr for SSE streaming
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = _TeeWriter(orig_stdout, "info")
        sys.stderr = _TeeWriter(orig_stderr, "warn")

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

            # Pipeline params
            max_seg = int(params.get("max_seg_len", 200))
            overlap = int(params.get("overlap", 20))
            rounds = int(params.get("rounds", 1))
            replicas = int(params.get("replicas", 4))
            rest2steps = int(params.get("rest2steps", 50000))
            use_rl = params.get("use_rl", True)
            use_rhofold = params.get("use_rhofold", True)

            out_dir = os.path.join(ROOT, "output_web")
            os.makedirs(out_dir, exist_ok=True)

            # Level 1+: Full pipeline
            _predict_state["current_level"] = 1
            _predict_state["message"] = "3D structure prediction..."
            _predict_state["progress"] = 20
            _emit_log("step", "Level 1: Segmented Vfold3D/RhoFold+ + Kabsch assembly")

            from torusfold.scheme2.isrnaclong import isrnaclong_pipeline
            result = isrnaclong_pipeline(
                sequence=sequence,
                secondary_structure=ss,
                output_dir=out_dir,
                max_seg_len=max_seg,
                overlap=overlap,
                n_relax_rounds=rounds,
                use_rl_relax=use_rl,
                use_rl_mcts=use_rl,
                rl_n_simulations=20,
                n_rest2_replicas=replicas,
                rest2_nsteps=rest2steps,
                md_step_scale=0.1,
                nrep=max(2, replicas),
                platform="auto",
                use_rhofold=use_rhofold,
                n_candidates=1,
                use_msa=False,
                resume=False,
                verbose=False,
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

            _predict_state.update({
                "status": "done",
                "progress": 100,
                "current_level": 5,
                "message": "Complete",
                "result": result_dict,
            })

        except Exception as exc:
            _emit_log("error", f"Pipeline failed: {exc}")
            _predict_state.update({
                "status": "error",
                "message": str(exc),
                "error": str(exc),
            })
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

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
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


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

    # rsRNASP1
    rsRNASP1 = {
        "score_all_atom": details.get("rsrasp1_energy", 0),
        "score_per_nt": details.get("rsrasp1_energy_per_nt", 0),
    }

    # rnadvisor scores
    rnadvisor = {
        "rsRNASP_docker": details.get("rsRNASP_docker", 0),
        "DFIRE": details.get("dfire_energy", 0),
        "3drnascore": details.get("score_3drnascore", 0),
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


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8877
    os.chdir(ROOT)
    server = HTTPServer(("0.0.0.0", port), TorusFoldHandler)
    print(f"TorusFold server: http://127.0.0.1:{port}/")
    print(f"  Static root: {ROOT}")
    print(f"  Web dir: {WEB_DIR}")
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
