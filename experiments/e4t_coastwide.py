#!/usr/bin/env python3
"""E4t: the whole-coast fix — is CrossLocate-style retrieval needed?

CrossLocate (WACV'22) exists because mountain photos at country scale
need learned retrieval to prune candidates before verification. The
coastal regime may not: an at-sea observer's candidates are only SEA
cells with land in view (a huge natural prune), and the FFT cost makes
each candidate ~4 ms. This experiment localizes ONE photo against the
ENTIRE southeast Aegean — lat 36–38, lon 26–28.5, a 222 x 222 km
region, ~50x the E3 box — with no position prior at all (heading from
compass only).

Pipeline: 1 km coarse grid over sea cells with land within the horizon
-> FFT cost per candidate -> top basins (20 km NMS) -> hierarchical
refine 250 m -> 50 m at the best. Reports the fix error vs the true
camera position of the composite, the basin margin at coast scale, and
wall-clock — the number that decides whether learned retrieval has a
job here.

Run:   python3 e4t_coastwide.py [photo]   (default out/synth/strait2)
"""

import os
import sys
import time
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S
import extract
from skyfix import AZ, BETAS, fast_photo_cost, observation

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
DIR3 = os.path.expanduser('~/.horizonator/DEMs_SRTM3')

LAT0, LAT1, LON0, LON1 = 36.0, 38.0, 26.0, 28.5
TRUTH = dict(strait2=(36.9622, 27.2384, 25.0, -0.5, 0.5),
             strait1=(36.9500, 27.2500, 180.0, 0.5, 0.0),
             offshore=(36.6050, 26.8590, 60.0, 0.0, 0.0))
F35 = 24
FOV = np.degrees(2 * np.arctan(0.8 * 21.63 / F35))
Z = 5.0

if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'strait2'
    lat_t, lon_t, heading, pitch, roll = TRUTH[name]
    img = extract.load_image(os.path.join(OUT, 'synth', name + '.jpg'))
    el_obs, w, _ = observation(img, FOV, heading, roll, pitch)
    shifts = np.arange(-60, 61, 2)          # compass prior only

    t0 = time.time()
    cm = S.CMarcher(DIR3, (LAT0, LAT1), (LON0, LON1), d_min=150.)
    print(f'mosaic {cm.mosaic.shape} loaded {time.time()-t0:.1f}s')

    # ---- candidate mask on a 1 km grid: sea cell AND land within 45 km
    dec = 12                                # ~1.1 km at 3"
    sea = cm.mosaic[::dec, ::dec] <= 0.5
    land = ~sea
    from scipy.ndimage import binary_dilation
    reach = binary_dilation(land, iterations=45)   # ~45 km reach
    cand = sea & reach
    ys, xs = np.where(cand)
    lats = cm.lat_nw - ys * dec * cm.dpp
    lons = cm.lon_nw + xs * dec * cm.dpp
    inb = (lats > LAT0 + .05) & (lats < LAT1 - .05) \
        & (lons > LON0 + .05) & (lons < LON1 - .05)
    lats, lons = lats[inb], lons[inb]
    print(f'{len(lats)} coastal-sea candidates '
          f'({100 * len(lats) / cand.size:.0f}% of the region)')

    # ---- coarse sweep
    t1 = time.time()
    costs = np.empty(len(lats))
    for k in range(len(lats)):
        el, _ = cm.skyline(lats[k], lons[k], Z, AZ)
        costs[k] = fast_photo_cost(el_obs, w, el, shifts, BETAS)[0]
        if k % 5000 == 4999:
            print(f'  {k + 1}/{len(lats)}  {time.time() - t1:.0f}s',
                  flush=True)
    t_coarse = time.time() - t1

    # ---- basins by NMS at 20 km, then refine the best
    order = np.argsort(costs)
    kept = []
    for o in order:
        p = (lats[o], lons[o])
        if all(np.hypot((p[0] - q[0]) * 111000,
                        (p[1] - q[1]) * 88000) > 20000 for q, _ in kept):
            kept.append((p, costs[o]))
        if len(kept) == 5:
            break
    margin = (kept[1][1] - kept[0][1]) / max(kept[0][1], 1e-12)
    (blat, blon), bcost = kept[0]
    mlat, mlon = S.meters_per_degree(blat)
    dn0 = de0 = 0.0
    for step in (250.0, 50.0):
        best = (np.inf, dn0, de0)
        for di in range(-3, 4):
            for dj in range(-3, 4):
                el, _ = cm.skyline(blat + (dn0 + di * step) / mlat,
                                   blon + (de0 + dj * step) / mlon, Z, AZ)
                c = fast_photo_cost(el_obs, w, el, shifts, BETAS)[0]
                if c < best[0]:
                    best = (c, dn0 + di * step, de0 + dj * step)
        _, dn0, de0 = best
    lat_e = blat + dn0 / mlat
    lon_e = blon + de0 / mlon
    err = float(np.hypot((lat_e - lat_t) * mlat, (lon_e - lon_t) * mlon))
    t_all = time.time() - t0
    res = dict(photo=name, n_candidates=int(len(lats)),
               err_m=err, margin=float(margin),
               coarse_s=t_coarse, total_s=t_all,
               fix=[float(lat_e), float(lon_e)],
               basins=[[float(a), float(b), float(c)]
                       for (a, b), c in kept])
    print(f'\nFIX {lat_e:.5f},{lon_e:.5f}  err {err:.0f} m  '
          f'margin {margin:.2f}')
    print(f'runners-up: ' + '  '.join(
        f'({a:.3f},{b:.3f})' for (a, b), _ in kept[1:3]))
    print(f'coarse {t_coarse:.0f}s over {len(lats)} candidates, '
          f'total {t_all:.0f}s')
    with open(os.path.join(OUT, f'e4t_{name}.json'), 'w') as f:
        json.dump(res, f, indent=1)
