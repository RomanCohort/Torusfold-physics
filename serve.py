# -*- coding: utf-8 -*-
"""TorusFold Web Server — 静态文件 + Predict POST API

用法: python serve.py [port]
默认端口: 8877
"""
import os
import sys
import json
import time
import threading
import tempfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
WEB_DIR = os.path.join(SRC, "torusfold", "web")

# 后端源码在 scheme2-rl 仓库 (干净仓库只含前端+部分模块)
SCHEME2_SRC = r"C:\Users\颜子壹\TorusFold-scheme2-rl\src"

# 全局预测状态
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
    """处理静态文件 + POST /predict + GET /status"""

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
        elif path.startswith("/api/score-pdb"):
            self._handle_score_pdb()
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

        # 静态文件: 从 WEB_DIR 或 ROOT 提供
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
        """GET /api/jobs/{jid} — 返回任务状态"""
        jid = path.split("/")[-1]
        self._send_json(_predict_state)

    def _handle_job_result(self, path):
        """GET /api/result/{jid} — 返回预测结果"""
        jid = path.split("/")[-1]
        if _predict_state["status"] == "done" and _predict_state["result"]:
            self._send_json(_predict_state["result"])
        elif _predict_state["status"] == "running":
            self._send_json({"status": "running", "progress": _predict_state["progress"]})
        elif _predict_state["status"] == "error":
            self._send_json({"error": _predict_state["error"]}, 500)
        else:
            self._send_json({"error": "No result available"}, 404)

    def _handle_score_pdb(self):
        """POST /api/score-pdb — 上传 PDB 并评分"""
        body = self._read_body()
        if not body:
            self._send_json({"error": "Empty body"}, 400)
            return
        # 保存 PDB
        out_dir = os.path.join(ROOT, "output_web")
        os.makedirs(out_dir, exist_ok=True)
        pdb_path = os.path.join(out_dir, "scored.pdb")
        with open(pdb_path, "wb") as f:
            f.write(body)
        self._send_json({"ok": True, "path": pdb_path, "size": len(body)})

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

        # 后台线程跑管线
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

        try:
            sys.path.insert(0, SRC)
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

            # 二级结构
            _predict_state["current_level"] = 0
            _predict_state["message"] = "ViennaRNA secondary structure..."
            _predict_state["progress"] = 5

            import RNA
            ss, mfe = RNA.fold(sequence)
            fc = RNA.fold_compound(sequence)
            fc.pf()

            _predict_state["message"] = f"MFE={mfe:.1f} kcal/mol"
            _predict_state["progress"] = 10

            # 管线参数
            max_seg = int(params.get("max_seg_len", 200))
            overlap = int(params.get("overlap", 20))
            rounds = int(params.get("rounds", 1))
            replicas = int(params.get("replicas", 4))
            rest2steps = int(params.get("rest2steps", 50000))
            use_rl = params.get("use_rl", True)
            use_rhofold = params.get("use_rhofold", True)

            # 输出目录
            out_dir = os.path.join(ROOT, "output_web")
            os.makedirs(out_dir, exist_ok=True)

            _predict_state["current_level"] = 1
            _predict_state["message"] = "3D structure prediction..."
            _predict_state["progress"] = 20

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

            _predict_state.update({
                "status": "done",
                "progress": 100,
                "current_level": 5,
                "message": "Complete",
                "result": {
                    "pair_rate": result.pair_rate,
                    "cross_segment_ok_rate": result.cross_segment_ok_rate,
                    "energy_cg": result.energy_cg,
                    "n_segments": result.n_segments,
                    "runtime": elapsed,
                    "ss": ss,
                    "mfe": mfe,
                    "pdb_path": os.path.join(out_dir, "isrnaclong_final.pdb"),
                    "sequence_length": len(sequence),
                },
            })

        except Exception as exc:
            _predict_state.update({
                "status": "error",
                "message": str(exc),
                "error": str(exc),
            })

    def _handle_upload(self):
        body = self._read_body()
        if not body:
            self._send_json({"error": "Empty upload"}, 400)
            return
        # 保存到临时文件
        tmp = os.path.join(ROOT, "output_web", "uploaded.pdb")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(body)
        self._send_json({"ok": True, "path": tmp, "size": len(body)})

    def log_message(self, format, *args):
        # 简化日志
        if "/status" not in str(args):
            sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8877
    os.chdir(ROOT)
    server = HTTPServer(("0.0.0.0", port), TorusFoldHandler)
    print(f"TorusFold server: http://127.0.0.1:{port}/web/index.html")
    print(f"  Static root: {ROOT}")
    print(f"  Web dir: {WEB_DIR}")
    print(f"  POST /predict — run pipeline")
    print(f"  GET  /status  — prediction status")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
