#!/usr/bin/env python3
"""E5e: the echo-sounder as a second, decorrelated TRN channel.

The skadi tiles keep ocean bathymetry as negative elevations — a
charted-depth grid the pipeline has always thrown away. A boat already
measures depth continuously (NMEA DPT), so depth-vs-chart is a
position observable that shares NOTHING with the camera: it works in
fog, at night, and under a featureless sky. Four runs of the E5
passage (same sensor stream, +3% log, +1.5 deg compass bias, depth
sounder 3% + 0.5 m noise each leg):

    DR                dead reckoning only
    DR+depth          the fog/night regime: no camera at all
    DR+skyline        the E5c baseline
    DR+skyline+depth  full suite

Run:   python3 e5e_depth.py       (writes out/e5e_results.json)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S
from skynav import SkyNav, AZ

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))
DIR1 = os.path.expanduser('~/.horizonator/DEMs_SRTM1')
LAT0, LON0, Z = 36.95, 27.25, 5.0
DT, N, SPEED = 300.0, 24, 3.0
LOG_BIAS, CB = 1.03, np.radians(1.5)
FIX_EVERY = 3

mlat, mlon = S.meters_per_degree(LAT0)
cm_truth = S.CMarcher(DIR3, (36.4, 37.4), (26.6, 27.9))
bathy_truth = S.Dem(DIR1, clamp_negative=False)


def true_heading(k):
    return np.radians(80.0 if k < N // 2 else 110.0)


truth = [np.array([-5000.0, -2200.0])]
for k in range(N):
    h = true_heading(k)
    truth.append(truth[-1] + SPEED * DT * np.array([np.sin(h), np.cos(h)]))
truth = np.array(truth)


def true_depth(p):
    v = float(bathy_truth.sample(np.array([LAT0 + p[1] / mlat]),
                                 np.array([LON0 + p[0] / mlon]))[0])
    return max(-v, 0.0)


def run(use_skyline, use_depth):
    rng = np.random.default_rng(20260818)
    nav = SkyNav(LAT0, LON0, Z, DIR3, lat_range=(36.4, 37.4),
                 lon_range=(26.6, 27.9), start_pos=tuple(truth[0]),
                 start_heading=true_heading(0),
                 start_sigma=(50.0, 50.0, np.radians(3)))
    if use_depth:
        nav.enable_bathymetry(DIR1)
    errs = []
    for k in range(N):
        dist = SPEED * DT * LOG_BIAS * (1 + rng.normal(0, 0.01))
        hdg = true_heading(k) + CB + rng.normal(0, np.radians(0.3))
        nav.add_odometry(dist, hdg)
        d_true = true_depth(truth[k + 1])
        d_meas = d_true * (1 + rng.normal(0, 0.03)) + rng.normal(0, 0.5)
        if use_depth and d_true > 3.0:
            nav.add_depth(max(d_meas, 0.5))
        if use_skyline and (k + 1) % FIX_EVERY == 0:
            el, _ = cm_truth.skyline(LAT0 + truth[k + 1][1] / mlat,
                                     LON0 + truth[k + 1][0] / mlon,
                                     Z, AZ)
            el = el + rng.normal(0, 1e-3, el.size)
            el = np.roll(el, int(round(np.degrees(CB) / 0.1)))
            nav.take_fix(el)
        lat, lon, _ = nav.current()
        errs.append(np.hypot((lat - (LAT0 + truth[k + 1][1] / mlat)) * mlat,
                             (lon - (LON0 + truth[k + 1][0] / mlon)) * mlon))
    return np.array(errs)


if __name__ == '__main__':
    out = {}
    for name, sky, dep in (('DR', False, False),
                           ('DR+depth', False, True),
                           ('DR+skyline', True, False),
                           ('DR+skyline+depth', True, True)):
        e = run(sky, dep)
        out[name] = dict(mean=float(e.mean()), final=float(e[-1]),
                         worst=float(e.max()))
        print(f'{name:18s}: mean {e.mean():6.1f} m  final {e[-1]:6.1f} m'
              f'  worst {e.max():6.1f} m', flush=True)
    with open(os.path.join(OUT, 'e5e_results.json'), 'w') as f:
        json.dump(out, f, indent=1)
