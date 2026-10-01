"""DivideFold CPU runner — runs in a clean subprocess with the GPU hidden.

DivideFold is launched as a separate process rather than imported, so that the
ROCm/CUDA environment of the parent cannot leak into it: DivideFold is a pure-CPU
tool and the parent may be holding a GPU context for the coarse-grained MD.

Invoked by isrnaclong.py's Level 0 consensus step as:

    <python> scripts/_dd_runner.py --seq <sequence> --max-frag 200

and is expected to print the predicted dot-bracket string on stdout.

Where DivideFold itself lives is resolved by the caller through
TF_DIVIDEFOLD_ROOT; this script only has to put that checkout's `src/` on the
path before importing `dividefold.predict`. The locations below are fallbacks for
a checkout that sits beside this repository.
"""
import argparse
import os
import sys


def _candidate_src_dirs():
    """Directories that may contain the dividefold package."""
    seen = []
    env_root = os.environ.get("TF_DIVIDEFOLD_ROOT", "")
    if env_root:
        seen.append(os.path.join(env_root, "src"))
    # A DivideFold-main checkout sitting beside this repository.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(here)
    seen.append(os.path.join(os.path.dirname(repo_root), "DivideFold-main", "src"))
    seen.append(os.path.join(repo_root, "DivideFold-main", "src"))
    # No machine-specific absolute paths: this file ships with the repository, so
    # a hard-coded user directory would be wrong for everyone else.
    return [p for p in seen if p]


def main(argv=None):
    # Hide every GPU before DivideFold is imported, so nothing tries to use one.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = ""

    ap = argparse.ArgumentParser(description="DivideFold CPU runner")
    ap.add_argument("--seq", required=True, help="RNA sequence")
    ap.add_argument("--max-frag", type=int, default=200,
                    help="maximum fragment length for the recursive split")
    args = ap.parse_args(argv)

    for path in _candidate_src_dirs():
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    import RNA as _RNA

    def _rnafold(seq):
        md = _RNA.md()
        fc = _RNA.fold_compound(seq, md)
        ss, _ = fc.mfe()
        return ss

    from dividefold.predict import dividefold_predict

    ss = dividefold_predict(args.seq, max_fragment_length=args.max_frag,
                            predict_fnc=_rnafold)
    print(ss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
