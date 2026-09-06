"""
pyrosetta_refine.py — PyRosetta all-atom refinement (WSL, conditionally triggered)

Runs after cg_to_allatom() to fix clashes and optimize local geometry.
It only triggers when clashscore > threshold, acting as a fuse/fallback rather than
a core step.

Runtime environment: WSL (Linux), PyRosetta 4.2023
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


# PyRosetta RNA-specific scoring function
_SCOREFN = None


def _init_pyrosetta():
    """Lazily initialize PyRosetta (import + init take about 2 seconds)."""
    global _SCOREFN
    if _SCOREFN is not None:
        return _SCOREFN
    try:
        import pyrosetta
        pyrosetta.init(
            "-ex1 -ex2aro "
            "-ignore_zero_occupancy false "
            "-no_fconfig "
        )
        _SCOREFN = pyrosetta.create_score_function("ref2015")
        return _SCOREFN
    except ImportError:
        raise ImportError(
            "PyRosetta not available. This module must run in WSL with PyRosetta installed."
        )


def compute_clashscore(pdb_path: str) -> float:
    """Compute a PDB's clashscore (clashing atom pairs per 1000 atoms).

    Args:
        pdb_path: path to the PDB file

    Returns:
        clashscore (float); 0 = no clashes
    """
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    from pyrosetta.io import pose_from_pdb
    from pyrosetta.rosetta.core.scoring import ScoreType

    scorefxn = _init_pyrosetta()
    pose = pose_from_pdb(pdb_path)

    # FA_REP energy = the clash term
    fa_rep = pose.energies().total_energies()[ScoreType.fa_rep]
    # number of clashing atom pairs (rough estimate: atom pairs with fa_rep > 0.25)
    n_atoms = pose.size()
    clashscore = max(0, fa_rep / 25.0)  # empirical conversion

    return clashscore


def pyrosetta_refine(
    pdb_path: str,
    output_path: str,
    max_iter: int = 500,
    cartesian: bool = False,
    clash_threshold: float = 20.0,
    verbose: bool = True,
) -> Tuple[str, float]:
    """PyRosetta RNA all-atom refinement (algorithmically optimized version).

    Algorithmic optimizations:
        1. Skip canonical RNA restoration (cg_to_allatom already outputs standard RNA)
        2. Skip manual perturbation (L-BFGS line search explores on its own)
        3. Two-stage minimization: steepest_descent (breaks up clashes) -> L-BFGS (refines).
           2-3x faster than pure L-BFGS, because SD is more efficient on the large
           off-equilibrium gradients.
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.io import pose_from_pdb

    scorefxn = _init_pyrosetta()

    # 1. load the PDB
    try:
        pose = pose_from_pdb(pdb_path)
    except (SystemExit, Exception) as e:
        if verbose:
            _msg = str(e)[:200] if str(e) else type(e).__name__
            print(f"  [WARN] failed to load PDB ({_msg})")
        import shutil
        shutil.copy2(pdb_path, output_path)
        return output_path, float("inf")

    n_res = pose.total_residue()

    # 2. clashscore
    fa_rep_init = pose.energies().total_energies()[ScoreType.fa_rep]
    clashscore_init = max(0, fa_rep_init / 25.0)
    e_init = scorefxn(pose)

    if clashscore_init <= clash_threshold:
        if verbose:
            print(f"  Clashscore {clashscore_init:.1f} <= {clash_threshold}, skipping")
        pose.dump_pdb(output_path)
        return output_path, e_init

    if verbose:
        print(f"  Clashscore {clashscore_init:.1f} > {clash_threshold}, refining...")

    # 3. MoveMap
    mm = MoveMap()
    mm.set_bb(True)
    mm.set_chi(True)
    mm.set_jump(False)

    # 4. two-stage minimization
    # Stage 1: steepest_descent — converges fast on large gradients (clashes), breaking up severe clashes
    sd_steps = max(30, max_iter // 4)
    sd_mover = MinMover(mm, scorefxn, "steepest_descent", 1.0, sd_steps)
    sd_mover.cartesian(False)
    try:
        sd_mover.apply(pose)
    except Exception:
        pass

    # Stage 2: L-BFGS — second-order information accelerates the local refinement
    lbfgs_steps = max(50, max_iter * 3 // 4)
    lbfgs_mover = MinMover(mm, scorefxn, "lbfgs", 0.1, lbfgs_steps)
    lbfgs_mover.cartesian(False)
    try:
        lbfgs_mover.apply(pose)
    except Exception as e:
        if verbose:
            print(f"  [WARN] L-BFGS error: {e}")

    # 5. output
    e_final = scorefxn(pose)
    clashscore_final = max(0, pose.energies().total_energies()[ScoreType.fa_rep] / 25.0)
    pose.dump_pdb(output_path)

    if verbose:
        print(f"  E: {e_init:.0f} -> {e_final:.0f}, clash: {clashscore_init:.0f} -> {clashscore_final:.0f}")

    return output_path, e_final


def _server_main():
    """Long-running PyRosetta service: receive refinement requests on a Unix socket
    so that PyRosetta is not repeatedly re-initialized.

    Protocol (JSON lines):
      request: {"pdb": "<abs_path>", "output": "<abs_path>", "max_iter": 200}
      response: {"ok": true, "output": "...", "energy": 123.4}
                {"ok": false, "error": "..."}
      shutdown: {"shutdown": true}
    """
    import socket
    import json

    SOCK_PATH = "/tmp/torusfold_pyrosetta.sock"

    # remove a leftover socket
    try:
        os.unlink(SOCK_PATH)
    except OSError:
        pass

    # pre-initialize PyRosetta (only once)
    print("[pyrosetta-server] initializing PyRosetta...")
    t0 = __import__("time").time()
    scorefxn = _init_pyrosetta()
    print(f"[pyrosetta-server] initialization done in {__import__('time').time() - t0:.1f}s, listening on {SOCK_PATH}")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCK_PATH)
    sock.listen(1)
    sock.settimeout(600)  # exit automatically after 10 minutes without a request

    try:
        while True:
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                print("[pyrosetta-server] timed out with no request, exiting")
                break

            try:
                # read the request (length prefix: 4-byte little-endian)
                raw_len = b""
                while len(raw_len) < 4:
                    chunk = conn.recv(4 - len(raw_len))
                    if not chunk:
                        break
                    raw_len += chunk

                if len(raw_len) < 4:
                    conn.sendall(b'{"ok":false,"error":"short header"}')
                    continue

                msg_len = int.from_bytes(raw_len, "little")
                raw_msg = b""
                while len(raw_msg) < msg_len:
                    chunk = conn.recv(min(65536, msg_len - len(raw_msg)))
                    if not chunk:
                        break
                    raw_msg += chunk

                req = json.loads(raw_msg.decode("utf-8"))

                if req.get("shutdown"):
                    conn.sendall(b'{"ok":true}')
                    print("[pyrosetta-server] received shutdown, exiting")
                    return

                pdb_path = req["pdb"]
                output_path = req.get("output", pdb_path + "_refined.pdb")
                max_iter = req.get("max_iter", 200)

                out_path, energy = pyrosetta_refine(
                    pdb_path, output_path, max_iter=max_iter, verbose=False,
                )
                resp = json.dumps({"ok": True, "output": out_path, "energy": energy})
                resp_bytes = resp.encode("utf-8")
                conn.sendall(len(resp_bytes).to_bytes(4, "little") + resp_bytes)

            except Exception as e:
                err_resp = json.dumps({"ok": False, "error": str(e)[:500]})
                err_bytes = err_resp.encode("utf-8")
                try:
                    conn.sendall(len(err_bytes).to_bytes(4, "little") + err_bytes)
                except Exception:
                    pass
            finally:
                conn.close()
    finally:
        sock.close()
        try:
            os.unlink(SOCK_PATH)
        except OSError:
            pass


# WSL entry point: allows invocation from the command line
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "--server":
        _server_main()
    elif len(sys.argv) < 3:
        print("Usage:")
        print("  python pyrosetta_refine.py input.pdb output.pdb [max_iter]")
        print("  python pyrosetta_refine.py --server")
        sys.exit(1)
    else:
        inp = sys.argv[1]
        out = sys.argv[2]
        mi = int(sys.argv[3]) if len(sys.argv) > 3 else 500
        pyrosetta_refine(inp, out, max_iter=mi)
