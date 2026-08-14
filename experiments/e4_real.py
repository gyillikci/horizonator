#!/usr/bin/env python3
"""E4: localization from a real photograph (Aegean coast, Gulf of Akbuk
area, 2026-08-10 15:42 local).

Instruments (a Theodolite-app frame taken at the same spot):
  GPS      +37.476847 / +27.414204, altitude 62 ft = 18.9 m  <- used ONLY
           to center the search box (5 km x 5 km = 25 km^2)
  heading  037 deg true at frame center (prior; co-estimated below)
  pitch    +5.3 deg, roll -0.9 deg (not needed: the observed skyline is
           anchored to the sea-horizon line instead)

The observation here is a HAND-DIGITIZED skyline: (x, height-above-sea-
horizon) points read visually off the raw photo (2016x1512, ~70 deg
horizontal FOV phone camera). Digitization accuracy is ~+-10 px vertical
(~+-6 mrad), so expect degraded accuracy vs automatic extraction (study
doc E2: 4 mrad noise costs ~2x). Pass --csv points.csv to substitute a
properly extracted skyline (columns: x_px, h_px) once the original file
is available.

Unknowns searched: position (5 km box, coarse-to-fine), azimuth offset
(+-6 deg, co-estimated per candidate), horizontal FOV (64-78 deg outer
loop). Elevation zero is anchored: el = h_px * scale - horizon_dip(18.9 m).

Run:   python3 e4_real.py     (no GL needed: native marcher only)
"""

import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S

DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)

GPS_LAT, GPS_LON = 37.476847, 27.414204   # box center ONLY
Z = 18.9                                   # m, from the app (62 ft)
HEADING_PRIOR = 37.0                       # deg true, frame center
IMG_W = 2016.0
BOX = 5000.0
AZ = np.arange(-180.0, 180.0, 0.1) + 0.05

# ---- hand-digitized skyline: (x_px, height_above_sea_horizon_px)
# Read visually off the raw photo. Left half: sunlit near hills with a
# distinctive conical peak; right half: hazy distant ranges with a large
# massif right of center.
POINTS = np.array([
    [0, 60], [100, 55], [200, 50], [300, 48], [400, 52], [500, 55],
    [600, 62], [700, 78], [800, 55], [850, 40], [900, 30], [1000, 22],
    [1050, 20], [1100, 24], [1150, 28], [1200, 26], [1250, 30],
    [1300, 34], [1350, 40], [1400, 48], [1450, 60], [1500, 72],
    [1550, 82], [1600, 88], [1650, 80], [1700, 70], [1750, 60],
    [1800, 52], [1850, 46], [1900, 42], [1950, 40], [2015, 38],
], dtype=float)
if len(sys.argv) > 2 and sys.argv[1] == '--csv':
    POINTS = np.loadtxt(sys.argv[2], delimiter=',')
    print(f'using {len(POINTS)} digitized points from {sys.argv[2]}')

mlat, mlon = S.meters_per_degree(GPS_LAT)
dip = S.horizon_dip_rad(Z)

print('building mosaic + C marcher...', flush=True)
# d_min = 1 km: CRITICAL for a land-based observer. SRTM's 90 m cells smear
# the coastal bluff the photographer stood on into terrain that blocks the
# predicted view at ranges of 150-500 m, which poisons the cost at the true
# position (25 mrad RMS vs 6 mrad with the mask). A camera on a bluff edge
# sees over its local ground; the DEM does not know that.
cm = S.CMarcher(DIR3, (37.0, 38.0), (26.0, 28.0), d_min=1000.0)


def observation(fov_deg):
    """Digitized points -> (el_obs, weights) on the global 0.1-deg azimuth
    grid, with the photo centered on HEADING_PRIOR (residual heading error
    is absorbed by the azimuth-shift co-estimation)."""
    scale = np.radians(fov_deg) / IMG_W          # rad per pixel
    az_pt = HEADING_PRIOR + np.degrees((POINTS[:, 0] - IMG_W / 2) * scale)
    el_pt = POINTS[:, 1] * scale - dip
    el = np.full(AZ.size, np.nan)
    m = (AZ >= az_pt.min()) & (AZ <= az_pt.max())
    el[m] = np.interp(AZ[m], az_pt, el_pt)
    w = np.isfinite(el).astype(float)
    return np.where(np.isfinite(el), el, -dip), w


def cost_shift(el_obs, el_syn, weights, max_shift_px=120):
    """Huber cost minimized over a global azimuth shift (0.2-deg steps,
    +-6 deg): heading-prior error co-estimation. The weight mask shifts
    with the observation."""
    best, sbest = np.inf, 0
    for s in range(-max_shift_px, max_shift_px + 1, 2):
        c = S.cost(np.roll(el_obs, s), el_syn,
                   weights=np.roll(weights, s))
        if c < best:
            best, sbest = c, s
    for s in (sbest - 1, sbest + 1):
        best = min(best, S.cost(np.roll(el_obs, s), el_syn,
                                weights=np.roll(weights, s)))
    return best, sbest


def solve(fov_deg):
    el_obs, w = observation(fov_deg)

    def C(dn, de):
        el, _ = cm.skyline(GPS_LAT + dn / mlat, GPS_LON + de / mlon, Z, AZ)
        return cost_shift(el_obs, el, w)[0]

    g = np.arange(-BOX / 2, BOX / 2 + 1, 250.0)      # 21x21 coarse
    cc = np.array([[C(dn, de) for de in g] for dn in g])
    i, j = np.unravel_index(np.argmin(cc), cc.shape)
    dn0, de0 = g[i], g[j]
    for step in (50.0, 12.5):                        # 5x5 refinements
        best = (np.inf, dn0, de0)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                c = C(dn0 + di * step, de0 + dj * step)
                if c < best[0]:
                    best = (c, dn0 + di * step, de0 + dj * step)
        _, dn0, de0 = best
    return best[0], dn0, de0, cc, g


t0 = time.time()
results = {}
for fov in (62.0, 66.0, 70.0, 74.0, 78.0):
    c, dn, de, cc, g = solve(fov)
    results[fov] = (c, dn, de, cc, g)
    print(f'FOV {fov:.0f} deg: cost {c:.3e}  ->  dn {dn:+7.1f}  de {de:+7.1f} m',
          flush=True)

fov_best = min(results, key=lambda k: results[k][0])
c, dn, de, cc, g = results[fov_best]
lat_e = GPS_LAT + dn / mlat
lon_e = GPS_LON + de / mlon
err = float(np.hypot(dn, de))
el_obs, w = observation(fov_best)
el_est, _ = cm.skyline(lat_e, lon_e, Z, AZ)
_, shift = cost_shift(el_obs, el_est, w)
print(f'\nbest FOV {fov_best:.0f} deg, heading offset {shift*0.1:+.1f} deg')
print(f'estimate: {lat_e:.6f}, {lon_e:.6f}')
print(f'GPS:      {GPS_LAT:.6f}, {GPS_LON:.6f}')
print(f'ERROR: {err:.0f} m   ({time.time()-t0:.0f}s total)')

el_gps, _ = cm.skyline(GPS_LAT, GPS_LON, Z, AZ)
np.savez_compressed(os.path.join(OUT, 'e4_result.npz'),
                    az=AZ, el_obs=el_obs, w=w, el_est=el_est,
                    el_gps=el_gps, cc=cc, g=g, dn=dn, de=de,
                    fov=fov_best, shift_deg=shift * 0.1, err=err)
with open(os.path.join(OUT, 'e4_result.json'), 'w') as f:
    json.dump(dict(lat_est=lat_e, lon_est=lon_e, err_m=err,
                   fov_deg=fov_best, heading_offset_deg=shift * 0.1,
                   fov_estimates={f'{k:.0f}': dict(cost=v[0], dn=v[1], de=v[2])
                                  for k, v in results.items()},
                   digitized='hand (visual), +-6 mrad'), f, indent=1)
