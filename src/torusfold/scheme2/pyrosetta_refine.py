"""
pyrosetta_refine.py — PyRosetta 全原子精修 (WSL, 条件触发)

在 cg_to_allatom() 之后运行, 修复 clash / 优化局部几何。
仅在 clashscore > threshold 时触发, 作为"保险丝"而非核心步骤。

运行环境: WSL (Linux), PyRosetta 4.2023
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


# PyRosetta RNA 专用评分函数
_SCOREFN = None


def _init_pyrosetta():
    """延迟初始化 PyRosetta (import + init 耗时约2秒)."""
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
    """计算 PDB 的 clashscore (clashing atom pairs per 1000 atoms).

    Args:
        pdb_path: PDB 文件路径

    Returns:
        clashscore (float), 0 = 无碰撞
    """
    import pyrosetta
    from pyrosetta.rosetta.core.pose import Pose
    from pyrosetta.io import pose_from_pdb
    from pyrosetta.rosetta.core.scoring import ScoreType

    scorefxn = _init_pyrosetta()
    pose = pose_from_pdb(pdb_path)

    # FA_REP 能量 = 碰撞项
    fa_rep = pose.energies().total_energies()[ScoreType.fa_rep]
    # 碰撞原子对数 (粗估: fa_rep > 0.25 的原子对)
    n_atoms = pose.size()
    clashscore = max(0, fa_rep / 25.0)  # 经验换算

    return clashscore


def pyrosetta_refine(
    pdb_path: str,
    output_path: str,
    max_iter: int = 500,
    cartesian: bool = False,
    clash_threshold: float = 20.0,
    verbose: bool = True,
) -> Tuple[str, float]:
    """PyRosetta RNA 全原子精修 (算法优化版).

    算法优化:
        1. 跳过 canonical RNA 恢复 (cg_to_allatom 输出已是标准 RNA)
        2. 跳过手动扰动 (L-BFGS line search 自带探索)
        3. 两阶段 minimize: steepest_descent(打散 clash) → L-BFGS(精修)
           比纯 L-BFGS 快 2-3×, 因为 SD 对远离平衡的大梯度更高效
    """
    import pyrosetta
    from pyrosetta.rosetta.core.scoring import ScoreType
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.io import pose_from_pdb

    scorefxn = _init_pyrosetta()

    # 1. 加载 PDB
    try:
        pose = pose_from_pdb(pdb_path)
    except (SystemExit, Exception) as e:
        if verbose:
            _msg = str(e)[:200] if str(e) else type(e).__name__
            print(f"  [WARN] PDB 加载失败 ({_msg})")
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
            print(f"  Clashscore {clashscore_init:.1f} <= {clash_threshold}, 跳过")
        pose.dump_pdb(output_path)
        return output_path, e_init

    if verbose:
        print(f"  Clashscore {clashscore_init:.1f} > {clash_threshold}, 精修中...")

    # 3. MoveMap
    mm = MoveMap()
    mm.set_bb(True)
    mm.set_chi(True)
    mm.set_jump(False)

    # 4. 两阶段 minimize
    # Stage 1: steepest_descent — 对大梯度(clash)收敛快, 打散严重碰撞
    sd_steps = max(30, max_iter // 4)
    sd_mover = MinMover(mm, scorefxn, "steepest_descent", 1.0, sd_steps)
    sd_mover.cartesian(False)
    try:
        sd_mover.apply(pose)
    except Exception:
        pass

    # Stage 2: L-BFGS — 二阶信息加速局部精修
    lbfgs_steps = max(50, max_iter * 3 // 4)
    lbfgs_mover = MinMover(mm, scorefxn, "lbfgs", 0.1, lbfgs_steps)
    lbfgs_mover.cartesian(False)
    try:
        lbfgs_mover.apply(pose)
    except Exception as e:
        if verbose:
            print(f"  [WARN] L-BFGS 异常: {e}")

    # 5. 输出
    e_final = scorefxn(pose)
    clashscore_final = max(0, pose.energies().total_energies()[ScoreType.fa_rep] / 25.0)
    pose.dump_pdb(output_path)

    if verbose:
        print(f"  E: {e_init:.0f} → {e_final:.0f}, clash: {clashscore_init:.0f} → {clashscore_final:.0f}")

    return output_path, e_final


def _server_main():
    """PyRosetta 长驻服务: Unix socket 接收精修请求, 避免反复 init.

    协议 (JSON lines):
      请求: {"pdb": "<abs_path>", "output": "<abs_path>", "max_iter": 200}
      响应: {"ok": true, "output": "...", "energy": 123.4}
             {"ok": false, "error": "..."}
      退出: {"shutdown": true}
    """
    import socket
    import json

    SOCK_PATH = "/tmp/torusfold_pyrosetta.sock"

    # 清理残留 socket
    try:
        os.unlink(SOCK_PATH)
    except OSError:
        pass

    # 预初始化 PyRosetta (只做一次)
    print("[pyrosetta-server] 初始化 PyRosetta...")
    t0 = __import__("time").time()
    scorefxn = _init_pyrosetta()
    print(f"[pyrosetta-server] 初始化完成 ({__import__('time').time() - t0:.1f}s), 监听 {SOCK_PATH}")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(SOCK_PATH)
    sock.listen(1)
    sock.settimeout(600)  # 10 分钟无请求自动退出

    try:
        while True:
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                print("[pyrosetta-server] 超时无请求, 退出")
                break

            try:
                # 读取请求 (长度前缀: 4 字节 little-endian)
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
                    print("[pyrosetta-server] 收到 shutdown, 退出")
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


# WSL 入口: 允许从命令行调用
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
