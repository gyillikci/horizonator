#!/usr/bin/env python3
"""E4h: the instrumented real-photo pair — Lake Bafa, Theodolite attitude.

Two photos taken seconds apart from the same bluff on the south shore of
Lake Bafa (GPS 37.476847 N 27.414204 E, used ONLY to center the box),
with full Theodolite attitude: photo A heading 037 deg true, pitch
+5.3 deg, roll -0.9 deg; photo B heading 014, pitch +5.8, roll 0.0;
GPS altitude ~60 ft -> z 18.3 m. Camera: iPhone 1.0x (71.6 deg H-FOV;
confirmed by the level-line position implied by the pitch, and by the
FOV sweep below). The photos arrive through a channel that strips
files, so the skylines are hand-digitized (POINTS_*, fractional image
coordinates) like E4a; digitization noise ~3-8 mrad dominates the error
budget. This is the in-envelope test: attitude priors present, heading
co-estimated +-6 deg, elevation offset +-10 mrad around the pitch.

Solves each photo alone and the two jointly (cost sum, independent
heading/offset nuisances per photo). Results to out/e4h_results.json.
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S
from skyfix import basin_margin

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
DIR3 = os.path.expanduser('~/.horizonator/DEMs_SRTM3')
AZ = np.arange(-180.0, 180.0, 0.1) + 0.05
BETAS = np.arange(-0.010, 0.0101, 0.002)
LAT_GT, LON_GT = 37.476847, 27.414204
Z = 18.3
BOX = 5000.0
DMIN = 1000.0

# (x_frac, y_frac) of the sky/terrain boundary, full-frame 4:3 images
POINTS_A = [  # heading 037, pitch +5.3, roll -0.9
    (0.000, 0.535), (0.020, 0.510), (0.050, 0.545), (0.100, 0.548),
    (0.150, 0.545), (0.200, 0.545), (0.250, 0.542), (0.300, 0.527),
    (0.325, 0.508), (0.345, 0.503), (0.370, 0.520), (0.400, 0.545),
    (0.450, 0.550), (0.500, 0.548), (0.550, 0.548), (0.600, 0.545),
    (0.650, 0.540), (0.700, 0.525), (0.750, 0.490), (0.800, 0.455),
    (0.850, 0.435), (0.880, 0.430), (0.920, 0.445), (0.960, 0.460),
    (1.000, 0.470),
]
POINTS_B = [  # heading 014, pitch +5.8, roll 0.0
    (0.000, 0.435), (0.030, 0.450), (0.060, 0.500), (0.100, 0.545),
    (0.150, 0.550), (0.200, 0.550), (0.250, 0.548), (0.300, 0.545),
    (0.350, 0.550), (0.400, 0.540), (0.430, 0.530), (0.455, 0.522),
    (0.480, 0.535), (0.520, 0.548), (0.570, 0.550), (0.620, 0.548),
    (0.670, 0.542), (0.720, 0.530), (0.770, 0.510), (0.820, 0.490),
    (0.870, 0.470), (0.900, 0.465), (0.940, 0.470), (1.000, 0.480),
]
PHOTOS = [
    ('A', POINTS_A, 37.0, 5.3, -0.9),
    ('B', POINTS_B, 14.0, 5.8, 0.0),
]
ASPECT = 4.0 / 3.0

# the waterline (far shore of the lake) digitized per photo: a crisp
# edge at a physically known elevation, ~-atan(h_above_lake/d_shore)
# ~ -3 mrad for d ~ 5 km. Referencing the skyline to it (a sextant-style
# differential measurement) cancels the digitizer's absolute-y bias and
# most of the pitch-chain error, which the first (absolute) solve showed
# as a consistent ~1.5 km displacement shared by both photos
WATERLINE_A = [(0.00, 0.590), (0.30, 0.588), (0.60, 0.588), (1.00, 0.585)]
WATERLINE_B = [(0.00, 0.582), (0.50, 0.585), (1.00, 0.585)]
EL_WL = -0.003
WATERLINES = dict(A=WATERLINE_A, B=WATERLINE_B)


def _map(points, fov_deg, pitch_deg, roll_deg):
    p = np.array(points)
    t = np.tan(np.radians(fov_deg) / 2)
    u = (p[:, 0] - 0.5) * 2 * t                # x in tan units, f=1
    v = (0.5 - p[:, 1]) * 2 * t / ASPECT
    cr, sr = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    ur = u * cr - v * sr
    vr = u * sr + v * cr
    az_rel = np.degrees(np.arctan2(ur, 1.0))
    el_pt = np.arctan2(vr, np.hypot(ur, 1.0)) + np.radians(pitch_deg)
    return az_rel, el_pt


def observation(points, fov_deg, heading, pitch_deg, roll_deg,
                waterline=None):
    """Digitized points -> (el_obs, weights) on the global azimuth grid.
    With waterline given, skyline elevations are referenced to the
    digitized waterline at EL_WL (differential measurement)."""
    az_rel, el_pt = _map(points, fov_deg, pitch_deg, roll_deg)
    if waterline is not None:
        az_wl, el_wl = _map(waterline, fov_deg, pitch_deg, roll_deg)
        o = np.argsort(az_wl)
        el_pt = el_pt - np.interp(az_rel, az_wl[o], el_wl[o]) + EL_WL
    el = np.zeros(AZ.size)
    wt = np.zeros(AZ.size)
    rel = (AZ - heading + 180.0) % 360.0 - 180.0
    m = (rel >= az_rel.min()) & (rel <= az_rel.max())
    order = np.argsort(az_rel)
    el[m] = np.interp(rel[m], az_rel[order], el_pt[order])
    wt[m] = 1.0
    return el, wt


def masked_cost(el_syn, el_obs, wt, shifts):
    best = np.inf
    for s in shifts:
        eo = np.roll(el_obs, s)
        mm = np.roll(wt, s) > 0
        r = el_syn[mm] - eo[mm]
        rb = np.abs(r[None, :] - BETAS[:, None])
        h = np.where(rb <= 3e-3, .5 * rb * rb, 3e-3 * (rb - 1.5e-3))
        c = h.mean(1).min()
        if c < best:
            best = c
    return best


def solve(obs_list, fov_note=''):
    """Coarse-to-fine over the box for the summed cost of obs_list."""
    cm = S.CMarcher(DIR3, (LAT_GT - .6, LAT_GT + .6),
                    (LON_GT - .8, LON_GT + .8), d_min=DMIN)
    mlat, mlon = S.meters_per_degree(LAT_GT)
    shifts = range(-60, 61, 2)

    def C(dn, de):
        la, lo = LAT_GT + dn / mlat, LON_GT + de / mlon
        el, _ = cm.skyline(la, lo, Z, AZ)
        return sum(masked_cost(el, eo, w, shifts) for eo, w in obs_list)

    step0 = 250.0
    g = np.arange(-BOX / 2, BOX / 2 + 1, step0)
    cc = np.array([[C(dn, de) for de in g] for dn in g])
    i, j = np.unravel_index(np.argmin(cc), cc.shape)
    dn0, de0 = g[i], g[j]
    margin = basin_margin(cc, g, min_sep=4 * step0)
    boundary = max(abs(dn0), abs(de0)) >= BOX / 2 - step0
    for step in (50.0, 12.5):
        best = (np.inf, dn0, de0)
        for di in range(-3, 4):
            for dj in range(-3, 4):
                c = C(dn0 + di * step, de0 + dj * step)
                if c < best[0]:
                    best = (c, dn0 + di * step, de0 + dj * step)
        _, dn0, de0 = best
    c0 = C(dn0, de0)
    rms = float(np.sqrt(2 * c0 / len(obs_list)) * 1e3)
    err = float(np.hypot(dn0, de0))
    return dict(err_m=err, dn_m=float(dn0), de_m=float(de0),
                rms_mrad=rms, margin=float(margin),
                boundary=bool(boundary), note=fov_note)


if __name__ == '__main__':
    fov = 71.6
    out = {}
    for mode in ('abs', 'rel'):
        wl = (lambda n: None) if mode == 'abs' else (lambda n: WATERLINES[n])
        obs = {name: observation(pts, fov, h, p, r, waterline=wl(name))
               for name, pts, h, p, r in PHOTOS}
        for name in obs:
            out[f'{name}_{mode}'] = r = solve([obs[name]])
            print(f"photo {name} ({mode}): err {r['err_m']:6.0f} m  "
                  f"rms {r['rms_mrad']:.1f} mrad  margin {r['margin']:.2f}"
                  f"{'  BOUNDARY' if r['boundary'] else ''}", flush=True)
        out[f'joint_{mode}'] = r = solve(list(obs.values()))
        print(f"joint   ({mode}): err {r['err_m']:6.0f} m  rms "
              f"{r['rms_mrad']:.1f} mrad  margin {r['margin']:.2f}"
              f"{'  BOUNDARY' if r['boundary'] else ''}", flush=True)

    # FOV sweep on the waterline-referenced joint solve
    out['fov_sweep'] = {}
    for f in (60.0, 65.0, 71.6, 78.0, 90.0, 106.1):
        o = [observation(pts, f, h, p, r, waterline=WATERLINES[n])
             for n, pts, h, p, r in PHOTOS]
        rs = solve(o, fov_note=f'fov {f}')
        out['fov_sweep'][str(f)] = rs
        print(f"fov {f:5.1f}: err {rs['err_m']:6.0f} m  "
              f"rms {rs['rms_mrad']:.1f} mrad  margin {rs['margin']:.2f}",
              flush=True)
    with open(os.path.join(OUT, 'e4h_results.json'), 'w') as f:
        json.dump(out, f, indent=1)
