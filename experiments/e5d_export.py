#!/usr/bin/env python3
"""E5d: export the E5c passage as a measurement stream for the C++ port.

Runs the same simulated 2-hour Bodrum-Kos passage through SkyNav (bias
estimation on), captures every sensor-level measurement the factor
graph consumes — odometry legs, accepted skyline fixes with their
anisotropic covariances, and the per-fix azimuth-shift compass-bias
measurements — and writes them as a plain CSV the gtsam C++ example
(examples/SkylineNavExample.cpp in the gtsam fork) can replay without
Python, a DEM, or the solver. Also solves the identical batch graph in
python-gtsam and records the reference solution for parity checking.

Outputs:  out/skyline_nav_stream.csv, out/e5d_reference.json
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gtsam
import skyline as S
from skynav import SkyNav, AZ, X, B
from skyline_factor import skyline_factor, heading_bias_factor
from skyfix import fast_photo_cost

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))
LAT0, LON0, Z = 36.95, 27.25, 5.0
DT, N, SPEED = 300.0, 24, 3.0
LOG_BIAS, COMPASS_BIAS = 1.03, np.radians(1.5)
FIX_EVERY = 3
START = (-5000.0, -2200.0)
START_SIGMA = (50.0, 50.0, np.radians(3))

mlat, mlon = S.meters_per_degree(LAT0)
cm_truth = S.CMarcher(DIR3, (36.4, 37.4), (26.6, 27.9))


def true_heading(k):
    return np.radians(80.0 if k < N // 2 else 110.0)


truth = [np.array(START)]
for k in range(N):
    h = true_heading(k)
    truth.append(truth[-1] + SPEED * DT * np.array([np.sin(h), np.cos(h)]))
truth = np.array(truth)

rng = np.random.default_rng(20260818)
nav = SkyNav(LAT0, LON0, Z, DIR3, lat_range=(36.4, 37.4),
             lon_range=(26.6, 27.9), start_pos=START,
             start_heading=true_heading(0), start_sigma=START_SIGMA,
             estimate_compass_bias=True)

rows = [f'PRIOR,{START[0]},{START[1]},'
        f'{np.pi / 2 - true_heading(0)},{START_SIGMA[0]},'
        f'{START_SIGMA[1]},{START_SIGMA[2]}']
for k in range(N):
    dist = SPEED * DT * LOG_BIAS * (1 + rng.normal(0, 0.01))
    hdg = true_heading(k) + COMPASS_BIAS + rng.normal(0, np.radians(0.3))
    nav.add_odometry(dist, hdg)
    rows.append(f'ODO,{dist},{hdg}')
    if (k + 1) % FIX_EVERY == 0:
        el_obs, _ = cm_truth.skyline(LAT0 + truth[k + 1][1] / mlat,
                                     LON0 + truth[k + 1][0] / mlon, Z, AZ)
        el_obs = el_obs + rng.normal(0, 1e-3, el_obs.size)
        el_obs = np.roll(el_obs,
                         int(round(np.degrees(COMPASS_BIAS) / 0.1)))
        fix, cov, margin, accepted, _ = nav.take_fix(el_obs)
        if accepted:
            el_fix, _ = nav.cm.skyline(LAT0 + fix[1] / mlat,
                                       LON0 + fix[0] / mlon, Z, AZ)
            _, s_best, _ = fast_photo_cost(el_obs, np.ones(AZ.size),
                                           el_fix, np.arange(-100, 101, 1),
                                           betas=np.array([0.0]))
            b_meas = -np.radians(s_best * 0.1)
            rows.append(f'FIX,{k + 1},{fix[0]},{fix[1]},'
                        f'{cov[0, 0]},{cov[0, 1]},{cov[1, 1]},{b_meas}')
    rows.append(f'TRUTH,{k + 1},{truth[k + 1][0]},{truth[k + 1][1]}')

csv_path = os.path.join(OUT, 'skyline_nav_stream.csv')
with open(csv_path, 'w') as f:
    f.write('# skyline_nav stream: PRIOR e n theta se sn sth | '
            'ODO dist heading | FIX k e n cee cen cnn bias_meas | '
            'TRUTH k e n\n')
    f.write('\n'.join(rows) + '\n')
print('wrote', csv_path, f'({len(rows)} rows)')

# ---- reference: solve the identical batch graph in python-gtsam
graph = gtsam.NonlinearFactorGraph()
vals = gtsam.Values()
keep = []
k = 0
theta_prev = None
for line in rows:
    p = line.split(',')
    if p[0] == 'PRIOR':
        e, n, th, se, sn, sth = map(float, p[1:])
        graph.add(gtsam.PriorFactorPose2(
            X(0), gtsam.Pose2(e, n, th),
            gtsam.noiseModel.Diagonal.Sigmas([se, sn, sth])))
        vals.insert(X(0), gtsam.Pose2(e, n, th))
        graph.add(gtsam.PriorFactorVector(
            B(0), np.zeros(1),
            gtsam.noiseModel.Isotropic.Sigma(1, np.radians(5.0))))
        vals.insert(B(0), np.zeros(1))
        theta_prev = th
    elif p[0] == 'ODO':
        dist, hdg = float(p[1]), float(p[2])
        th = np.pi / 2 - hdg
        graph.add(gtsam.BetweenFactorPose2(
            X(k), X(k + 1), gtsam.Pose2(dist, 0.0, th - theta_prev),
            gtsam.noiseModel.Diagonal.Sigmas(
                [0.04 * dist + 5.0, 0.015 * dist + 5.0, np.radians(1.0)])))
        f = heading_bias_factor(X(k + 1), B(0), hdg, np.radians(0.5))
        keep.append(f)
        graph.add(f)
        prev = vals.atPose2(X(k))
        vals.insert(X(k + 1), gtsam.Pose2(
            prev.x() + dist * np.sin(hdg),
            prev.y() + dist * np.cos(hdg), th))
        k += 1
        theta_prev = th
    elif p[0] == 'FIX':
        kk = int(p[1])
        e, n, cee, cen, cnn, bm = map(float, p[2:])
        f = skyline_factor(X(kk), e, n,
                           np.array([[cee, cen], [cen, cnn]]))
        keep.append(f)
        graph.add(f)
        graph.add(gtsam.PriorFactorVector(
            B(0), np.array([bm]),
            gtsam.noiseModel.Isotropic.Sigma(1, np.radians(0.3))))
res = gtsam.LevenbergMarquardtOptimizer(graph, vals).optimize()
last = res.atPose2(X(N))
bias = float(res.atVector(B(0))[0])
errs = []
for line in rows:
    p = line.split(',')
    if p[0] == 'TRUTH':
        kk, te, tn = int(p[1]), float(p[2]), float(p[3])
        pk = res.atPose2(X(kk))
        errs.append(float(np.hypot(pk.x() - te, pk.y() - tn)))
ref = dict(final_e=float(last.x()), final_n=float(last.y()),
           final_theta=float(last.theta()),
           bias_deg=float(np.degrees(bias)),
           mean_err_m=float(np.mean(errs)), final_err_m=errs[-1])
with open(os.path.join(OUT, 'e5d_reference.json'), 'w') as f:
    json.dump(ref, f, indent=1)
print('python batch reference:', json.dumps(ref, indent=1))
