"""Short-trajectory comparison: pre-fix vs post-fix guide forces on far pairs.

Builds a small CG helix with three far (sequence-distant) pairs pulled to
1.5 / 2.0 / 2.5 nm, then runs a short overdamped trajectory with the OLD
(8eaaf7f) and NEW (450605d) cg_energy_forces and reports pair-distance
trends. Old guides repelled (pathological potential); new guides attract
within ~r0 + 3w ~ 1.6 nm.
"""
import importlib.util, pathlib, sys
sys.path.insert(0, 'D:/torusfold-hybrid/src')
import torch, math
import torusfold.scheme2.torch_cgsim as tc_new

def load_old(path):
    spec = importlib.util.spec_from_file_location('tc_old_ref', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['tc_old_ref'] = mod
    spec.loader.exec_module(mod)
    return mod

tc_old = load_old('D:/torusfold-hybrid/old_torch_cgsim_ref.py')

def build_system(L=24, pair_offsets=(12, 12, 12), pulls=(1.5, 2.0, 2.5), seed=3):
    torch.manual_seed(seed)
    pts = []
    for i in range(L):
        a = i * 0.55
        p = [a * 0.55, 0.42 * math.sin(a), 0.42 * math.cos(a)]
        pts += [p, [p[0], p[1] + 0.15, p[2] + 0.1], [p[0], p[1] - 0.15, p[2] - 0.1]]
    pos = torch.tensor([pts], dtype=torch.float32)
    pairs = [(2, 2 + pair_offsets[0]), (6, 6 + pair_offsets[1]), (10, 10 + pair_offsets[2])]
    # pull pair ends apart along the line between them
    for (i, j), pull in zip(pairs, pulls):
        pi, pj = 3 * i, 3 * j
        d = pos[0, pi] - pos[0, pj]
        cur = float(d.norm())
        if cur < 1e-6:
            d = torch.tensor([1.0, 0.0, 0.0])
            cur = 1.0
        target = cur + (pull - cur)
        shift = d / cur * (pull - cur)
        pos[0, pi] = pos[0, pi] + shift * 0.5
        pos[0, pj] = pos[0, pj] - shift * 0.5
    pt = torch.tensor([[i, j] for i, j in pairs], dtype=torch.long)
    return pos, pt

def run(fn, pos0, pairs, steps=200, dt=2e-5):
    pw = torch.ones(pairs.shape[0], dtype=torch.float32)
    p = pos0.clone()
    dists = []
    for s in range(steps):
        E, F = fn(p, pairs, pw)
        p = p + dt * F
        if s % 50 == 0 or s == steps - 1:
            dd = [float((p[0, 3*i] - p[0, 3*j]).norm()) for (i, j) in pairs.tolist()]
            dists.append((s, dd))
    return dists

pos0, pairs = build_system()
pw = torch.ones(pairs.shape[0], dtype=torch.float32)

# Pre-equilibrate with the NEW force field (no far pairs yet, guides inert),
# so the trajectory below starts from a balanced helix and only the pulled
# pairs are out of equilibrium.
print('pre-equilibrating (500 overdamped steps, new force field)...')
peq = pos0.clone()
for s in range(500):
    E, F = tc_new.cg_energy_forces(peq, pairs, pw)
    fmax = float(F.abs().max())
    peq = peq + F / (1.0 + fmax) * 1e-3
    if fmax < 5.0:
        break

# Pull the three pairs apart along their connecting line to 1.5 / 2.0 nm
pos = peq.clone()
for (i, j), pull in zip(pairs.tolist(), (1.5, 2.0, 2.5)):
    pi, pj = 3 * i, 3 * j
    d = pos[0, pi] - pos[0, pj]
    cur = float(d.norm())
    shift = d / cur * (pull - cur)
    pos[0, pi] = pos[0, pi] + shift * 0.5
    pos[0, pj] = pos[0, pj] - shift * 0.5

d0 = [float((pos[0, 3*i] - pos[0, 3*j]).norm()) for (i, j) in pairs.tolist()]
print('after pull, pair distances (nm):', [round(x, 3) for x in d0])
print('running old (8eaaf7f) trajectory...')
d_old = run(lambda p, pa, pw: tc_old.cg_energy_forces(p, pa, pw), pos, pairs)
print('running new (450605d) trajectory...')
d_new = run(lambda p, pa, pw: tc_new.cg_energy_forces(p, pa, pw), pos, pairs)
print()
print('step | pair dists OLD -> NEW')
for so, do in d_old:
    dn = dict(d_new)[so]
    print(f'{so:4d} | old {[round(x, 3) for x in do]}  new {[round(x, 3) for x in dn]}')

print()
print('== isolated guide terms only (all other force constants zeroed) ==')
def zero_all(mod, keep):
    for k in list(mod.__dict__):
        if k.startswith('K_') or k.startswith('_K_'):
            if k not in keep:
                try: setattr(mod, k, 0.0)
                except Exception: pass
    for k, v in keep.items():
        setattr(mod, k, v)

keep = {'K_PAIR_GUIDE': 100.0, 'K_BSJ_GUIDE': 100.0,
        '_K_PAIR_GUIDE': 100.0, '_K_BSJ_GUIDE': 100.0}
# unified 路径用 K_*；batched 用 _K_*；cg_energy_forces 是 unified
zero_all(tc_new, keep)
zero_all(tc_old, keep)

# 短螺旋，一对残基拉到 1.3 nm（guide 边缘内）
torch.manual_seed(5)
L = 10
pts = []
for i in range(L):
    a = i * 0.55
    pts += [[a*0.55, 0.42*math.sin(a), 0.42*math.cos(a)],
            [a*0.55, 0.42*math.sin(a)+0.15, 0.42*math.cos(a)+0.1],
            [a*0.55, 0.42*math.sin(a)-0.15, 0.42*math.cos(a)-0.1]]
pos = torch.tensor([pts], dtype=torch.float32)
pr = torch.tensor([[2, 7]], dtype=torch.long)
d = pos[0, 6] - pos[0, 21]
cur = float(d.norm())
d0 = float(d.norm())
print(f'pair (2,7) initial distance: {d0:.3f} nm')
pos = pos.clone()
pos[0, 6] = pos[0, 6] + d / cur * (1.3 - cur) * 0.5
pos[0, 21] = pos[0, 21] - d / cur * (1.3 - cur) * 0.5
print('after pulling to 1.3 nm, running 150 steps (isolated guides)...')
for name, mod in (('OLD(8eaaf7f)', tc_old), ('NEW(450605d)', tc_new)):
    p = pos.clone()
    traj = []
    for s in range(150):
        E, F = mod.cg_energy_forces(p, pr, torch.ones(1, dtype=torch.float32))
        p = p + F * 1e-4
        if s % 50 == 0 or s == 149:
            traj.append(round(float((p[0,6]-p[0,21]).norm()), 3))
    print(f'  {name}: pair distance over steps -> {traj}')
