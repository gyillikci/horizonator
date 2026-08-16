#!/usr/bin/env python3
"""E5f: the night watch — charted-light bearings replace the skyline.

After dark the skyline instrument is blind, but the coast lights up
with surveyed, self-identifying landmarks: lights whose flash
characteristics (period/pattern/color, from any List of Lights or OSM
seamark data) let a camera identify WHICH charted point it sees. An
identified light bearing is a line of position through a known point —
fused here via light_bearing_factor, sharing the compass-bias variable
so the lights also calibrate the compass at night.

Same E5 passage, at night (no skyline fixes possible), three charted
lights along the strait, bearings measured when within 18 km (compass
bias + 0.4 deg noise). Runs:

    DR                 dead reckoning through the night
    DR+lights          the night channel
    DR+lights+depth    full night suite (echo sounder too)

Run:   python3 e5f_lights.py       (writes out/e5f_results.json)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S
from skynav import SkyNav

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))
DIR1 = os.path.expanduser('~/.horizonator/DEMs_SRTM1')
LAT0, LON0, Z = 36.95, 27.25, 5.0
DT, N, SPEED = 300.0, 24, 3.0
LOG_BIAS, CB = 1.03, np.radians(1.5)

mlat, mlon = S.meters_per_degree(LAT0)
bathy = S.Dem(DIR1, clamp_negative=False)

# charted lights (synthetic stand-ins at plausible Aegean positions),
# ENU relative to (LAT0, LON0); real deployment reads these from a
# List of Lights / OSM seamarks with their flash characteristics
LIGHTS = [(( (27.170 - LON0) * mlon), ((36.980 - LAT0) * mlat)),
          (( (27.300 - LON0) * mlon), ((36.890 - LAT0) * mlat)),
          (( (27.430 - LON0) * mlon), ((37.020 - LAT0) * mlat))]
VIS_M = 18000.0


def true_heading(k):
    return np.radians(80.0 if k < N // 2 else 110.0)


truth = [np.array([-5000.0, -2200.0])]
for k in range(N):
    h = true_heading(k)
    truth.append(truth[-1] + SPEED * DT * np.array([np.sin(h), np.cos(h)]))
truth = np.array(truth)


def run(use_lights, use_depth):
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
        p = truth[k + 1]
        if use_depth:
            v = float(bathy.sample(np.array([LAT0 + p[1] / mlat]),
                                   np.array([LON0 + p[0] / mlon]))[0])
            d_true = max(-v, 0.0)
            if d_true > 3.0:
                nav.add_depth(max(d_true * (1 + rng.normal(0, 0.03))
                                  + rng.normal(0, 0.5), 0.5))
        if use_lights:
            for le, ln in LIGHTS:
                r = np.hypot(le - p[0], ln - p[1])
                if r < VIS_M:
                    brg_true = np.arctan2(le - p[0], ln - p[1])
                    nav.add_light_bearing(
                        le, ln,
                        brg_true + CB + rng.normal(0, np.radians(0.4)))
        lat, lon, _ = nav.current()
        errs.append(np.hypot((lat - (LAT0 + p[1] / mlat)) * mlat,
                             (lon - (LON0 + p[0] / mlon)) * mlon))
    return np.array(errs), nav.compass_bias_deg()


if __name__ == '__main__':
    out = {}
    for name, li, de in (('DR', False, False),
                         ('DR+lights', True, False),
                         ('DR+lights+depth', True, True)):
        e, bias = run(li, de)
        out[name] = dict(mean=float(e.mean()), final=float(e[-1]),
                         worst=float(e.max()), bias_deg=bias)
        print(f'{name:16s}: mean {e.mean():6.1f} m  final {e[-1]:6.1f} m'
              f'  worst {e.max():6.1f} m  bias {bias:+.2f} deg',
              flush=True)
    with open(os.path.join(OUT, 'e5f_results.json'), 'w') as f:
        json.dump(out, f, indent=1)
