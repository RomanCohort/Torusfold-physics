"""Run ibi_round0.py with named force-field constants patched, and label the run with them.

Two measured facts make this necessary rather than convenient.

1. README, "Known limits": on a LINEAR chain the three BSJ terms should be zero -- they act on
   P(0)-P(L-1), which only a circular molecule has, and on a linear reference they are 91.55
   percent of the energy. Every structure in the IBI pool is a linear deposited chain, so every
   IBI round-0 residual so far carries them. scripts/attribute_compaction.py measured what they
   do: with them the chain compacts from Rg 1.229 to 1.067 nm in 80 ps; without them it sits at
   1.152 and does not move.

2. The chain never says which field it ran on beyond the constant fingerprint, and a patch would
   not show up in a filename. So the patch is echoed into the log, and the guide-shape probe in
   ibi_round0.py independently reports the guide's shape.

Patching is safe here for the same reason it is in scan_k_pair.py: the constants are module
globals read at call time, not default arguments. A name that does not exist raises instead of
being ignored -- a typo would otherwise produce a silent run of the unpatched field.

Run: python scripts/ibi_patched_field.py --set=K_BSJ=0,K_BSJ_GUIDE=0,K_BSJ_CONTACT=0 \
                                         [n_rep] [n_steps] [idx] [friction] [stride] [burn] \
                                         [--blocks=N]
"""
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import torusfold.scheme2.torch_cgsim as C       # noqa: E402

_set = {}
_positional = []
for a in sys.argv[1:]:
    if a.startswith("--set="):
        for kv in a[len("--set="):].split(","):
            kv = kv.strip()
            if not kv:
                continue
            name, _, val = kv.partition("=")
            _set[name.strip()] = float(val)
    else:
        _positional.append(a)

for _n, _v in _set.items():
    if not hasattr(C, _n):
        raise SystemExit(f"no such constant in torch_cgsim: {_n!r}; refusing to run, because a "
                         f"silently ignored name would produce a run of the unpatched field")
    setattr(C, _n, _v)

print("PATCHED FIELD: " + (", ".join(f"{k}={v:g}" for k, v in _set.items()) or "(none)"))
print("Every constant below is read after this patch, so the fingerprint line that follows "
      "already reflects it.")

sys.argv = ["ibi_round0.py"] + (_positional or ["8", "100000", "0", "0.1", "25", "20000"])
runpy.run_path(str(REPO / "scripts" / "ibi_round0.py"), run_name="__main__")
