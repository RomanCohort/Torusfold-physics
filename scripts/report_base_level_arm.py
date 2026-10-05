"""Base-level reporting for a scored arm: the per-coordinate sim/ref ratios and joint_J_base."""
import json, pathlib, sys, statistics as st
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import boltzmann_bonded as B

COORDS = B.scored_coords()
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/ibi_base")
print("scored coordinates (%d): %s" % (len(COORDS), ", ".join(COORDS)))
for f in sorted(root.glob("round*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    ps = d.get("per_structure") or []
    jb = [p["joint_J_base"] for p in ps if p.get("joint_J_base") is not None]
    j = [p["joint_J"] for p in ps if p.get("joint_J") is not None]
    print("\n== %s  (%d structures, %.0f s)" % (f.name, len(ps), d.get("seconds", float("nan"))))
    print("   joint_J (trace, controlled) median %.4f | joint_J_base median %s"
          % (st.median(j) if j else float("nan"), ("%.4f" % st.median(jb)) if jb else "n/a"))
    # pooled |ln(sim/ref)| per coordinate, from the per-structure sim_ref_table entries
    for k, c in enumerate(COORDS):
        vals = [p["sim_ref_table"][k] for p in ps
                if p.get("sim_ref_table") and k < len(p["sim_ref_table"])
                and p["sim_ref_table"][k] is not None]
        if not vals:
            continue
        print("   %-10s sim/ref mean %.4f  median %.4f  n=%d" % (c, st.mean(vals), st.median(vals), len(vals)))
