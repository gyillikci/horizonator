#!/usr/bin/env python3
"""E2: noise/nuisance ablations on the E1 closed loop.

Per the study doc (doc/skyline-matching-study.md, section 6). Reuses the E1
candidate lattices (out/e1_<site>_lattice.npz) and ground-truth positions
(out/e1_<site>.npz), so most configs are solve-only and run in seconds. New
renders are needed only for the height-mismatch axis (observations rendered
at z != 5) and the DEM cross-test (observations rendered from 1" SRTM,
solved against the 3" lattice).

Axes:
  noise      gaussian skyline noise, sigma in {0.5, 1, 2, 4} mrad
  bias       heading bias in {0.05, 0.1, 0.2, 0.5, 1, 2} deg, solved both
             naively and with azimuth-shift co-estimation (the mitigation)
  dz         observer-height / tide mismatch: obs rendered at 5+dz m,
             dz in {-2,-1,+1,+2}, solver assumes 5 m
  fov        azimuth sector in {360, 180, 90, 40} deg
  cloud      cloud base truncating the top {10, 25, 50} % of land skyline
  refraction actual refraction k' in {0.10, 0.16, 0.20} vs the model's 0.13,
             applied to the observation as del = -r(az)*(k0-k')/(2R) using
             the per-bin skyline range (exact to first order)
  dem        observation rendered from 1" SRTM, matched against 3" lattice

Run headlessly:   xvfb-run -a python3 e2_ablations.py
(e1_closed_loop.py must have been run first.)
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
DIR1 = os.path.expanduser('~/.horizonator/DEMs_SRTM1')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')

Z = 5.0
BOX, LATTICE = 1000.0, 25.0
WIDTH, HEIGHT = 3600, 400

SITES = {
    'A-strait':   dict(lat=36.95, lon=27.25, sector_center_deg=180.0),
    'B-offshore': dict(lat=36.60, lon=26.85, sector_center_deg=60.0),
}

rng = np.random.default_rng(20260815)
results = {}

for name, site in SITES.items():
    print(f'== site {name} ==', flush=True)
    lat_c, lon_c = site['lat'], site['lon']
    mlat, mlon = S.meters_per_degree(lat_c)

    # E1 artifacts: lattice + ground truth
    el_lattice = np.load(os.path.join(OUT, f'e1_{name}_lattice.npz'))['el']
    d1 = np.load(os.path.join(OUT, f'e1_{name}.npz'))
    gt, az = d1['gt'], d1['az']
    n2 = int(BOX / 2 / LATTICE)
    idx = np.arange(-n2, n2 + 1)
    cache = {(i, j): el_lattice[a, b]
             for a, i in enumerate(idx) for b, j in enumerate(idx)}

    def lattice_skyline(dn, de):
        i = int(np.clip(round(dn / LATTICE), -n2, n2))
        j = int(np.clip(round(de / LATTICE), -n2, n2))
        return cache[(i, j)]

    def solve(el_obs, weights=None, cost_fn=None):
        dn, de, _ = S.solve_position(lattice_skyline, el_obs, Z, box_m=BOX,
                                     coarse_n=9, fine_step_m=LATTICE,
                                     weights=weights, cost_fn=cost_fn)
        return dn, de

    def errs(rows):
        rows = np.array(rows)
        e = np.hypot(rows[:, 2] - rows[:, 0], rows[:, 3] - rows[:, 1])
        return dict(cep50=float(np.percentile(e, 50)),
                    cep95=float(np.percentile(e, 95)), max=float(e.max()))

    # ---- observations: nominal z, the dz variants, and the SRTM1 variants
    gl = S.GlSkyline(lat_c, lon_c, width=WIDTH, height=HEIGHT,
                     render_radius_m=45000., dir_dems=DIR3)
    obs = {}          # dz -> list of (el, r) per GT
    for dz in (0.0, -2.0, -1.0, 1.0, 2.0):
        obs[dz] = []
        for g in gt:
            _, el, r = gl.skyline(lat_c + g[0] / mlat, lon_c + g[1] / mlon,
                                  Z + dz)
            obs[dz].append((S.seahorizon_fill(el, Z + dz).astype(np.float64),
                            r))
    del gl

    print('  rendering SRTM1 observations...', flush=True)
    t0 = time.time()
    gl1 = S.GlSkyline(lat_c, lon_c, width=WIDTH, height=HEIGHT,
                      render_radius_m=45000., dir_dems=DIR1, SRTM1=True)
    obs1 = []
    for g in gt:
        _, el, r = gl1.skyline(lat_c + g[0] / mlat, lon_c + g[1] / mlon, Z)
        obs1.append(S.seahorizon_fill(el, Z).astype(np.float64))
    del gl1
    print(f'  ... {time.time()-t0:.0f}s', flush=True)

    res = {}

    # ---- noise sweep
    res['noise'] = {}
    for sig in (0.5e-3, 1e-3, 2e-3, 4e-3):
        rows = [list(g) + list(solve(el + rng.normal(0, sig, el.shape)))
                for g, (el, _) in zip(gt, obs[0.0])]
        res['noise'][f'{sig*1e3:g}'] = errs(rows)

    # ---- heading-bias sweep, naive and with azimuth co-estimation
    res['bias'] = {}
    res['bias_azfit'] = {}
    pxdeg = 360.0 / WIDTH
    for bd in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        sh = int(round(bd / pxdeg))
        rows, rows2 = [], []
        for g, (el, _) in zip(gt, obs[0.0]):
            el_b = np.roll(el, sh)
            rows.append(list(g) + list(solve(el_b)))
            rows2.append(list(g) + list(solve(el_b, cost_fn=S.cost_azshift)))
        res['bias'][f'{bd:g}'] = errs(rows)
        res['bias_azfit'][f'{bd:g}'] = errs(rows2)

    # ---- height/tide mismatch
    res['dz'] = {}
    for dz in (-2.0, -1.0, 1.0, 2.0):
        rows = [list(g) + list(solve(el))
                for g, (el, _) in zip(gt, obs[dz])]
        res['dz'][f'{dz:+g}'] = errs(rows)

    # ---- FOV sweep
    res['fov'] = {}
    sc = site['sector_center_deg']
    for fov in (360.0, 180.0, 90.0, 40.0):
        dd = (az - sc + 180.0) % 360.0 - 180.0
        w = (np.abs(dd) <= fov / 2).astype(float)
        rows = [list(g) + list(solve(el, weights=w))
                for g, (el, _) in zip(gt, obs[0.0])]
        res['fov'][f'{fov:g}'] = errs(rows)

    # ---- cloud truncation of the top n% of the land skyline
    res['cloud'] = {}
    dip = S.horizon_dip_rad(Z)
    for frac in (0.10, 0.25, 0.50):
        rows = []
        for g, (el, _) in zip(gt, obs[0.0]):
            land = el > (-dip + 0.5e-3)
            if not land.any():
                continue
            base = np.quantile(el[land], 1.0 - frac)
            rows.append(list(g) + list(solve(np.minimum(el, base))))
        res['cloud'][f'{int(frac*100)}'] = errs(rows)

    # ---- refraction-coefficient mismatch: true k' vs the model's 0.13.
    # el shift = -r * (k0 - k')/(2 R), first order, using the per-bin range
    res['refraction'] = {}
    for kp in (0.10, 0.16, 0.20):
        delta = (S.K_REFRACTION - kp) / (2.0 * S.REARTH)
        rows = []
        for g, (el, r) in zip(gt, obs[0.0]):
            rr = np.where(np.isfinite(r), r, S.horizon_distance_m(Z))
            rows.append(list(g) + list(solve(el - rr * delta)))
        res['refraction'][f'{kp:g}'] = errs(rows)

    # ---- DEM cross-test: 1" observation vs 3" lattice
    res['dem'] = {'SRTM1obs': errs([list(g) + list(solve(el))
                                    for g, el in zip(gt, obs1)])}

    results[name] = res
    for axis, vals in res.items():
        print(f'  {axis:<11}',
              '  '.join(f'{k}:{v["cep50"]:.0f}/{v["cep95"]:.0f}m'
                        for k, v in vals.items()), flush=True)

with open(os.path.join(OUT, 'e2_results.json'), 'w') as f:
    json.dump(results, f, indent=1)
print('wrote', os.path.join(OUT, 'e2_results.json'))
