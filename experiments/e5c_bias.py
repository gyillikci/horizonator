#!/usr/bin/env python3
"""E5c: the compass bias as a graph variable — A/B on the E5 passage.

Imported from the parallel study branch (celestial-navigation,
claude/iphone-celestial-sighting-imu-ctwbnf): systematic sensor biases
belong in the factor graph as estimated variables, not smeared into
per-measurement sigmas. Same 2-hour Bodrum-Kos stream as e5b_live.py
(+3% log, 1.5 deg compass bias), run twice:

    A (legacy)  bias priced into the odometry lateral sigma (0.045*d)
    B (bias)    bias estimated via heading factors; lateral sigma 0.015*d

Reports fused error and the recovered bias vs the true 1.5 deg.
Run:   python3 e5c_bias.py       (no GL needed)
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
LAT0, LON0, Z = 36.95, 27.25, 5.0
DT, N, SPEED = 300.0, 24, 3.0
LOG_BIAS, COMPASS_BIAS = 1.03, np.radians(1.5)
FIX_EVERY = 3

mlat, mlon = S.meters_per_degree(LAT0)
cm_truth = S.CMarcher(DIR3, (36.4, 37.4), (26.6, 27.9))


def true_heading(k):
    return np.radians(80.0 if k < N // 2 else 110.0)


truth = [np.array([-5000.0, -2200.0])]
for k in range(N):
    h = true_heading(k)
    truth.append(truth[-1] + SPEED * DT * np.array([np.sin(h), np.cos(h)]))
truth = np.array(truth)


def run(estimate_bias):
    rng = np.random.default_rng(20260818)     # identical sensor stream
    nav = SkyNav(LAT0, LON0, Z, DIR3, lat_range=(36.4, 37.4),
                 lon_range=(26.6, 27.9),
                 start_pos=tuple(truth[0]), start_heading=true_heading(0),
                 start_sigma=(50.0, 50.0, np.radians(3)),
                 estimate_compass_bias=estimate_bias)
    errs = []
    for k in range(N):
        dist = SPEED * DT * LOG_BIAS * (1 + rng.normal(0, 0.01))
        hdg = true_heading(k) + COMPASS_BIAS + rng.normal(0, np.radians(0.3))
        nav.add_odometry(dist, hdg)
        if (k + 1) % FIX_EVERY == 0:
            el_obs, _ = cm_truth.skyline(LAT0 + truth[k + 1][1] / mlat,
                                         LON0 + truth[k + 1][0] / mlon,
                                         Z, AZ)
            el_obs = el_obs + rng.normal(0, 1e-3, el_obs.size)
            el_obs = np.roll(el_obs,
                             int(round(np.degrees(COMPASS_BIAS) / 0.1)))
            nav.take_fix(el_obs)
        lat, lon, _ = nav.current()
        errs.append(np.hypot((lat - (LAT0 + truth[k + 1][1] / mlat)) * mlat,
                             (lon - (LON0 + truth[k + 1][0] / mlon)) * mlon))
    return np.array(errs), nav.compass_bias_deg()


if __name__ == '__main__':
    ea, _ = run(False)
    eb, bias = run(True)
    print(f'A (bias in sigmas):   mean {ea.mean():5.1f} m  '
          f'final {ea[-1]:5.1f} m  worst {ea.max():5.1f} m')
    print(f'B (bias in graph):    mean {eb.mean():5.1f} m  '
          f'final {eb[-1]:5.1f} m  worst {eb.max():5.1f} m')
    print(f'recovered compass bias {bias:+.2f} deg (true +1.50)')
    with open(os.path.join(OUT, 'e5c_results.json'), 'w') as f:
        json.dump(dict(legacy=dict(mean=float(ea.mean()),
                                   final=float(ea[-1]),
                                   worst=float(ea.max())),
                       bias_var=dict(mean=float(eb.mean()),
                                     final=float(eb[-1]),
                                     worst=float(eb.max()),
                                     recovered_bias_deg=bias)),
                  f, indent=1)
