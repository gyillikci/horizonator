#!/usr/bin/env python3
"""E3: scale the search box two orders of magnitude — 100 km x 100 km —
and measure search performance, using the native C ray-marcher
(fastmarch.c) as the candidate-skyline engine.

Box: 100 km x 100 km centered at (36.60, 26.90) — Dodecanese/SE Aegean.
Observations: GL renders (the independent implementation) at random at-sea
ground-truth positions, z = 5 m.

Search: sea-masked hierarchical coarse-to-fine:
  L0  2 km grid over all at-sea candidates in the box
      -> non-max-suppressed top-K seeds (min separation 6 km)
  L1  5x5 at 500 m around each seed -> keep best 5
  L2  5x5 at 125 m around each      -> keep best 1
  L3  5x5 at 25 m + quadratic sub-grid refinement

Reports per-trial: error, timing per level, evaluation counts, the L0 rank
of the true basin (-> miss rate vs K without re-solving), and the cost
margin between the best and second-best L0 basin (multimodality).

Run headlessly:   xvfb-run -a python3 e3_scale.py
"""

import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import skyline as S

DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)

Z = 5.0
LAT_C, LON_C = 36.60, 26.90
BOX = 100e3
L0_STEP = 2e3
NMS_SEP = 6e3
K_SEEDS = 15
NGT = 15
WIDTH, HEIGHT = 3600, 400
AZ = np.arange(-180.0, 180.0, 0.1) + 0.05

mlat, mlon = S.meters_per_degree(LAT_C)

print('building mosaic + C marcher...', flush=True)
cm = S.CMarcher(DIR3, (35.0, 38.0), (25.0, 28.0))
dip = S.horizon_dip_rad(Z)


def cand_skyline(dn, de):
    el, r = cm.skyline(LAT_C + dn / mlat, LON_C + de / mlon, Z, AZ)
    return el


def is_sea(dn, de):
    lat = LAT_C + dn / mlat
    lon = LON_C + de / mlon
    y = (cm.lat_nw - lat) / cm.dpp
    x = (lon - cm.lon_nw) / cm.dpp
    return cm.mosaic[int(round(y)), int(round(x))] <= 0.0


# ---------------- candidate grid (sea only)
g = np.arange(-BOX / 2, BOX / 2 + 1, L0_STEP)
cand = np.array([(dn, de) for dn in g for de in g if is_sea(dn, de)])
print(f'L0 grid: {g.size}x{g.size} = {g.size**2}, at sea: {len(cand)} '
      f'({len(cand)/g.size**2*100:.0f}%)', flush=True)

# ---------------- ground truth at sea + GL observations
rng = np.random.default_rng(20260816)
gts = []
while len(gts) < NGT:
    dn, de = rng.uniform(-BOX / 2, BOX / 2, 2)
    if is_sea(dn, de):
        gts.append((dn, de))
gts = np.array(gts)

print('rendering GL ground-truth observations...', flush=True)
t0 = time.time()
obs = []
for dn, de in gts:
    lat, lon = LAT_C + dn / mlat, LON_C + de / mlon
    gl = S.GlSkyline(lat, lon, width=WIDTH, height=HEIGHT,
                     render_radius_m=45000., dir_dems=DIR3)
    _, el, r = gl.skyline(lat, lon, Z)
    land_frac = float(np.mean(S.seahorizon_fill(el, Z) > (-dip + 0.5e-3)))
    obs.append((S.seahorizon_fill(el, Z), land_frac))
    del gl
print(f'  {time.time()-t0:.0f}s for {NGT} observations', flush=True)


def refine_stage(el_obs, seeds, step):
    """5x5 grid at `step` around each seed; returns per-seed (best_cost,
    (dn,de)) and the evaluation count"""
    out = []
    ev = 0
    for dn0, de0 in seeds:
        best = (np.inf, (dn0, de0))
        for i in range(-2, 3):
            for j in range(-2, 3):
                dn, de = dn0 + i * step, de0 + j * step
                c = S.cost(el_obs, cand_skyline(dn, de))
                ev += 1
                if c < best[0]:
                    best = (c, (dn, de))
        out.append(best)
    return out, ev


trials = []
for t, ((gdn, gde), (el_obs, land_frac)) in enumerate(zip(gts, obs)):
    tt = {}
    t0 = time.time()
    c0 = np.array([S.cost(el_obs, cand_skyline(dn, de)) for dn, de in cand])
    tt['L0'] = time.time() - t0

    # non-max suppression -> seed list, and the L0 rank of the true basin
    order = np.argsort(c0)
    seeds, seed_costs = [], []
    for o in order:
        p = cand[o]
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= NMS_SEP for q in seeds):
            seeds.append(p)
            seed_costs.append(c0[o])
        if len(seeds) >= K_SEEDS:
            break
    d_gt = [np.hypot(s[0] - gdn, s[1] - gde) for s in seeds]
    rank_true = next((i for i, d in enumerate(d_gt) if d < 4e3), None)

    t0 = time.time()
    r1, ev1 = refine_stage(el_obs, seeds, 500.0)
    r1.sort(key=lambda x: x[0])
    r2, ev2 = refine_stage(el_obs, [p for _, p in r1[:5]], 125.0)
    r2.sort(key=lambda x: x[0])
    r3, ev3 = refine_stage(el_obs, [r2[0][1]], 25.0)
    tt['refine'] = time.time() - t0

    # sub-grid quadratic on the winning 3x3
    bdn, bde = r3[0][1]
    cq = np.array([[S.cost(el_obs, cand_skyline(bdn + i * 25., bde + j * 25.))
                    for j in (-1, 0, 1)] for i in (-1, 0, 1)])
    ddx, ddy = S.quadratic_refine(np.array([-25., 0., 25.]), cq)
    edn, ede = bdn + ddy, bde + ddx

    err = float(np.hypot(edn - gdn, ede - gde))
    trials.append(dict(
        gt=[float(gdn), float(gde)], est=[float(edn), float(ede)], err=err,
        land_frac=land_frac, rank_true=rank_true,
        evals=int(len(cand) + ev1 + ev2 + ev3 + 9),
        t_L0=tt['L0'], t_refine=tt['refine'],
        margin=float((seed_costs[1] - seed_costs[0]) / seed_costs[0])
        if len(seed_costs) > 1 else np.inf,
        c0=c0 if t == 0 else None))
    print(f'  gt#{t}: err {err:8.1f} m  land {land_frac*100:4.0f}%  '
          f'rank_true {rank_true}  evals {trials[-1]["evals"]}  '
          f'{tt["L0"]+tt["refine"]:.1f}s', flush=True)

# ---------------- summary + save
err = np.array([x['err'] for x in trials])
ok = err < 100.0
print(f'\nsuccess (<100 m): {ok.sum()}/{NGT}   '
      f'CEP50(all) {np.percentile(err,50):.1f} m   '
      f'CEP50(success) {np.percentile(err[ok],50) if ok.any() else np.nan:.1f} m')
tL0 = np.mean([x['t_L0'] for x in trials])
trf = np.mean([x['t_refine'] for x in trials])
print(f'timing/fix: L0 {tL0:.1f}s + refine {trf:.1f}s = {tL0+trf:.1f}s '
      f'({np.mean([x["evals"] for x in trials]):.0f} evals)')
ranks = [x['rank_true'] for x in trials]
print('L0 rank of true basin:', ranks)

c0map = trials[0].pop('c0')
for x in trials:
    x.pop('c0', None)
np.savez_compressed(os.path.join(OUT, 'e3_l0map.npz'),
                    cand=cand, c0=c0map, gt0=gts[0], grid=g)
with open(os.path.join(OUT, 'e3_results.json'), 'w') as f:
    json.dump(dict(trials=trials, box_m=BOX, l0_step=L0_STEP,
                   k_seeds=K_SEEDS, nms_sep=NMS_SEP), f, indent=1)
print('wrote', os.path.join(OUT, 'e3_results.json'))
