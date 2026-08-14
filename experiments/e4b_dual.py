#!/usr/bin/env python3
"""E4b: joint localization from TWO real photos taken from the same spot on
the south shore of Lake Bafa (2026-08-10 15:41-15:42), hand-digitized.

Photo A (iPhone 17 Pro 1.0x, ~71.6 deg horizontal FOV from the 24mm-equiv
main camera): heading prior 037 true. Photo B (0.5x ultrawide, 13mm equiv
-> 106.1 deg horizontal FOV in 4:3): heading prior 352 true. GPS
(37.476788, 27.414212; used ONLY to center the 5 km x 5 km box), altitude
readings 46-62 ft -> z = 16 m.

Improvements over e4_real.py:
- proper pinhole pixel->angle mapping (az = atan(u/f)), required at 106 deg
- FOV fixed from the lens spec, not fitted
- per-photo nuisances co-estimated: azimuth offset (heading error, +-6 deg)
  AND a global elevation offset (pitch/waterline datum, +-40..+20 mrad) --
  the scene is a lake, so the waterline is the far shore, not a sea horizon
- joint cost: bin-count-weighted average over both photos (~130 deg total)

Skylines are hand-digitized (x_px, height_above_waterline_px), ~+-10 px.
Run:   python3 e4b_dual.py
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

GPS_LAT, GPS_LON = 37.476788, 27.414212
Z = 16.0
IMG_W = 2016.0
BOX = 5000.0
AZ = np.arange(-180.0, 180.0, 0.1) + 0.05
DAZ = 0.1

PHOTOS = {
    'A-1.0x': dict(
        heading=37.0, fov_deg=71.6,
        # digitized off the 1.0x photo (NE view: near hills with conical
        # peak left of center, saddle, big Besparmak massif right)
        points=np.array([
            [0, 60], [100, 55], [200, 50], [300, 48], [400, 52], [500, 55],
            [600, 62], [700, 78], [800, 55], [850, 40], [900, 30],
            [1000, 22], [1050, 20], [1100, 24], [1150, 28], [1200, 26],
            [1250, 30], [1300, 34], [1350, 40], [1400, 48], [1450, 60],
            [1500, 72], [1550, 82], [1600, 88], [1650, 80], [1700, 70],
            [1750, 60], [1800, 52], [1850, 46], [1900, 42], [1950, 40],
            [2015, 38]], dtype=float)),
    'B-0.5x': dict(
        heading=352.0, fov_deg=106.1,
        # digitized off the 0.5x photo (N view across the lake: hazy hills
        # far left, north-shore hills, prominent cone at x~1080, turbine
        # saddle, rising rugged hills to the right edge)
        points=np.array([
            [0, 30], [100, 28], [200, 25], [300, 25], [400, 30], [500, 38],
            [600, 42], [700, 45], [800, 42], [900, 48], [1000, 65],
            [1080, 95], [1160, 60], [1240, 45], [1320, 50], [1400, 55],
            [1500, 60], [1600, 70], [1700, 75], [1800, 70], [1900, 78],
            [2015, 80]], dtype=float)),
}

mlat, mlon = S.meters_per_degree(GPS_LAT)

print('building mosaic + C marcher...', flush=True)
# d_min = 1 km: mask the observer's own roadside bluff (see e4_real.py)
cm = S.CMarcher(DIR3, (37.0, 38.0), (26.0, 28.0), d_min=1000.0)


def make_obs(photo):
    """Pinhole mapping of digitized points onto the global azimuth grid.
    Returns (el_obs, weights); el_obs still needs the per-candidate
    co-estimated global elevation offset added."""
    f = (IMG_W / 2) / np.tan(np.radians(photo['fov_deg']) / 2)
    u = photo['points'][:, 0] - IMG_W / 2
    az_pt = photo['heading'] + np.degrees(np.arctan2(u, f))
    el_pt = np.arctan2(photo['points'][:, 1], np.hypot(u, f))
    el = np.full(AZ.size, np.nan)
    # handle the 360-wrap (photo B spans 299..45 deg)
    d = (AZ[:, None] - az_pt[None, :] + 180.0) % 360.0 - 180.0
    az_rel = np.degrees(np.arctan2(u, f))
    lo, hi = az_rel.min(), az_rel.max()
    rel = (AZ - photo['heading'] + 180.0) % 360.0 - 180.0
    m = (rel >= lo) & (rel <= hi)
    el[m] = np.interp(rel[m], az_rel, el_pt)
    w = np.isfinite(el).astype(float)
    return np.where(np.isfinite(el), el, 0.0), w


OBS = {k: make_obs(p) for k, p in PHOTOS.items()}
NBINS = {k: int(OBS[k][1].sum()) for k in OBS}
print({k: f'{NBINS[k]} bins' for k in OBS})

BETAS = np.arange(-0.040, 0.0201, 0.004)     # elevation-offset search
SHIFTS = range(-60, 61, 2)                   # azimuth-offset search, 0.2 deg


def photo_cost(el_obs, w, el_syn):
    """min over (azimuth shift, elevation offset) of the Huber cost"""
    best = np.inf
    for s in SHIFTS:
        eo = np.roll(el_obs, s)
        ww = np.roll(w, s)
        m = ww > 0
        r = el_syn[m] - eo[m]
        # elevation offset: closed-form-ish via the beta grid
        for b in BETAS:
            rb = np.abs(r - b)
            h = np.where(rb <= 3e-3, 0.5 * rb * rb, 3e-3 * (rb - 1.5e-3))
            c = float(np.mean(h))
            if c < best:
                best = c
    return best


def joint_cost(dn, de):
    el, _ = cm.skyline(GPS_LAT + dn / mlat, GPS_LON + de / mlon, Z, AZ)
    num = den = 0.0
    for k in OBS:
        c = photo_cost(*OBS[k], el)
        num += c * NBINS[k]
        den += NBINS[k]
    return num / den


t0 = time.time()
g = np.arange(-BOX / 2, BOX / 2 + 1, 250.0)
cc = np.array([[joint_cost(dn, de) for de in g] for dn in g])
i, j = np.unravel_index(np.argmin(cc), cc.shape)
dn0, de0 = g[i], g[j]
print(f'coarse: dn {dn0:+.0f} de {de0:+.0f}  ({time.time()-t0:.0f}s)', flush=True)

for step in (50.0, 12.5):
    best = (np.inf, dn0, de0)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            c = joint_cost(dn0 + di * step, de0 + dj * step)
            if c < best[0]:
                best = (c, dn0 + di * step, de0 + dj * step)
    _, dn0, de0 = best

lat_e = GPS_LAT + dn0 / mlat
lon_e = GPS_LON + de0 / mlon
err = float(np.hypot(dn0, de0))
print(f'\njoint estimate: {lat_e:.6f}, {lon_e:.6f}')
print(f'GPS:            {GPS_LAT:.6f}, {GPS_LON:.6f}')
print(f'ERROR: {err:.0f} m  (cost {best[0]:.3e}, {time.time()-t0:.0f}s total)')

# per-photo solo solves for comparison
solo = {}
for k in OBS:
    def C1(dn, de, k=k):
        el, _ = cm.skyline(GPS_LAT + dn / mlat, GPS_LON + de / mlon, Z, AZ)
        return photo_cost(*OBS[k], el)
    cc1 = np.array([[C1(dn, de) for de in g] for dn in g])
    i, j = np.unravel_index(np.argmin(cc1), cc1.shape)
    d1, e1 = g[i], g[j]
    for step in (50.0, 12.5):
        b1 = (np.inf, d1, e1)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                c = C1(d1 + di * step, e1 + dj * step)
                if c < b1[0]:
                    b1 = (c, d1 + di * step, e1 + dj * step)
        _, d1, e1 = b1
    solo[k] = float(np.hypot(d1, e1))
    print(f'photo {k} alone: err {solo[k]:.0f} m (dn {d1:+.0f} de {e1:+.0f})')

el_est, _ = cm.skyline(lat_e, lon_e, Z, AZ)
el_gps, _ = cm.skyline(GPS_LAT, GPS_LON, Z, AZ)
np.savez_compressed(os.path.join(OUT, 'e4b_result.npz'),
                    az=AZ, cc=cc, g=g, dn=dn0, de=de0, err=err,
                    el_est=el_est, el_gps=el_gps,
                    obsA=OBS['A-1.0x'][0], wA=OBS['A-1.0x'][1],
                    obsB=OBS['B-0.5x'][0], wB=OBS['B-0.5x'][1])
with open(os.path.join(OUT, 'e4b_result.json'), 'w') as f:
    json.dump(dict(lat_est=lat_e, lon_est=lon_e, err_m=err, solo=solo,
                   note='joint 2-photo, pinhole, FOV from lens spec'),
              f, indent=1)
