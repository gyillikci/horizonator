#!/usr/bin/env python3
"""E5: skyline fixes fused with dead reckoning in a GTSAM factor graph.

A simulated 2-hour coastal passage through the Bodrum-Kos strait (E1 site
A): the vessel dead-reckons with realistic errors (+3% log bias, 1.5 deg
compass bias, per-leg noise), and every 30 minutes takes a skyline fix --
solved by the native marcher over a 2 km box centered on the CURRENT DR
estimate (as a real system would), with 1 mrad skyline noise and the same
compass bias applied to the observation (absorbed by azimuth
co-estimation). Each fix enters the graph as a SkylineFactor with its
Laplace covariance.

Compared: dead reckoning alone vs the optimized graph.
Run:   python3 e5_fusion.py       (no GL needed)
"""

import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gtsam
import skyline as S
from skyline_factor import skyline_factor, laplace_cov

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))

LAT0, LON0 = 36.95, 27.25          # ENU origin, mid-strait
Z = 5.0
AZ = np.arange(-180.0, 180.0, 0.1) + 0.05
DT = 300.0                          # s per leg
N = 24                              # 2 hours
SPEED = 3.0                         # m/s (~6 kn)
LOG_BIAS = 1.03                     # +3% speed error
COMPASS_BIAS = np.radians(1.5)
FIX_EVERY = 3                       # a skyline fix every 15 min

rng = np.random.default_rng(20260818)
mlat, mlon = S.meters_per_degree(LAT0)

print('building marcher...', flush=True)
cm = S.CMarcher(DIR3, (36.4, 37.4), (26.6, 27.9))


def true_heading(k):
    # an east-bound passage through the strait south of the Bodrum
    # peninsula, staying in open water
    return np.radians(80.0 if k < N // 2 else 110.0)


# ---------------- truth track (ENU meters, heading)
truth = [np.array([-5000.0, -2200.0])]
for k in range(N):
    h = true_heading(k)
    truth.append(truth[-1] + SPEED * DT * np.array([np.sin(h), np.cos(h)]))
truth = np.array(truth)

# ---------------- dead reckoning (measured odometry)
legs = []
for k in range(N):
    dist = SPEED * DT * LOG_BIAS * (1 + rng.normal(0, 0.01))
    hdg = true_heading(k) + COMPASS_BIAS + rng.normal(0, np.radians(0.3))
    legs.append((dist, hdg))
dr = [truth[0].copy()]
for dist, hdg in legs:
    dr.append(dr[-1] + dist * np.array([np.sin(hdg), np.cos(hdg)]))
dr = np.array(dr)

# ---------------- skyline fixes at the DR positions' times
def solve_fix(true_pos, center):
    """Simulate an observation at true_pos, solve a 2 km box around
    `center` (the DR estimate), return (fix_enu, cov, err_m)."""
    el_obs, _ = cm.skyline(LAT0 + true_pos[1] / mlat,
                           LON0 + true_pos[0] / mlon, Z, AZ)
    el_obs = el_obs + rng.normal(0, 1e-3, el_obs.size)
    el_obs = np.roll(el_obs, int(round(np.degrees(COMPASS_BIAS) / 0.1)))

    def C(de, dn):
        el, _ = cm.skyline(LAT0 + (center[1] + dn) / mlat,
                           LON0 + (center[0] + de) / mlon, Z, AZ)
        return S.cost_azshift(el_obs, el)

    g = np.arange(-1000.0, 1001.0, 250.0)
    cc = np.array([[C(de, dn) for de in g] for dn in g])
    i, j = np.unravel_index(np.argmin(cc), cc.shape)
    dn0, de0 = g[i], g[j]
    for step in (50.0, 12.5):
        best = (np.inf, de0, dn0)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                c = C(de0 + dj * step, dn0 + di * step)
                if c < best[0]:
                    best = (c, de0 + dj * step, dn0 + di * step)
        _, de0, dn0 = best
    fix = center + np.array([de0, dn0])
    cov = laplace_cov(lambda e, n: C(e - center[0], n - center[1]),
                      fix[0], fix[1], scale=0.02)
    return fix, cov, float(np.hypot(*(fix - true_pos)))


fixes = {}
t0 = time.time()
for k in range(FIX_EVERY, N + 1, FIX_EVERY):
    fix, cov, err = solve_fix(truth[k], dr[k])
    sig = np.sqrt(np.diag(cov))
    fixes[k] = (fix, cov)
    print(f'  fix at t={k*DT/60:.0f} min: err {err:5.1f} m  '
          f'sigma ({sig[0]:.0f},{sig[1]:.0f}) m  '
          f'[DR was {np.hypot(*(dr[k]-truth[k])):.0f} m off]', flush=True)
print(f'{len(fixes)} fixes in {time.time()-t0:.0f}s')

# ---------------- factor graph
graph = gtsam.NonlinearFactorGraph()
X = gtsam.symbol_shorthand.X
graph.add(gtsam.PriorFactorPose2(
    X(0), gtsam.Pose2(truth[0][0], truth[0][1], true_heading(0)),
    gtsam.noiseModel.Diagonal.Sigmas([50.0, 50.0, np.radians(3)])))
for k, (dist, hdg) in enumerate(legs):
    dtheta = hdg - (legs[k - 1][1] if k else true_heading(0))
    graph.add(gtsam.BetweenFactorPose2(
        X(k), X(k + 1), gtsam.Pose2(dist, 0.0, dtheta),
        gtsam.noiseModel.Diagonal.Sigmas(
            # sigmas price in the unmodeled log/compass biases: ~4% along,
            # dist*sin(2.5 deg) lateral
            [0.04 * dist + 5.0, 0.045 * dist + 5.0, np.radians(1.0)])))
for k, (fix, cov) in fixes.items():
    graph.add(skyline_factor(X(k), fix[0], fix[1], cov))

init = gtsam.Values()
for k in range(N + 1):
    hdg = legs[min(k, N - 1)][1]
    # initialize fix epochs at their fixes: LM otherwise converges
    # prematurely from a badly drifted DR initialization
    p = fixes[k][0] if k in fixes else dr[k]
    init.insert(X(k), gtsam.Pose2(p[0], p[1], hdg))
params = gtsam.LevenbergMarquardtParams()
params.setMaxIterations(200)
params.setRelativeErrorTol(1e-8)
params.setAbsoluteErrorTol(1e-8)
result = gtsam.LevenbergMarquardtOptimizer(graph, init, params).optimize()
fused = np.array([[result.atPose2(X(k)).x(), result.atPose2(X(k)).y()]
                  for k in range(N + 1)])

err_dr = np.hypot(*(dr - truth).T)
err_fu = np.hypot(*(fused - truth).T)
print(f'\nDR alone: mean {err_dr.mean():.0f} m, final {err_dr[-1]:.0f} m')
print(f'fused:    mean {err_fu.mean():.0f} m, final {err_fu[-1]:.0f} m')
json.dump(dict(err_dr=err_dr.tolist(), err_fused=err_fu.tolist(),
               truth=truth.tolist(), dr=dr.tolist(), fused=fused.tolist(),
               fixes={str(k): dict(fix=f.tolist(), cov=c.tolist())
                      for k, (f, c) in fixes.items()}),
          open(os.path.join(OUT, 'e5_results.json'), 'w'), indent=1)

# ---------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                              gridspec_kw=dict(width_ratios=[1.2, 1]))
fig.suptitle('E5 — skyline fixes fused with dead reckoning '
             '(GTSAM, Bodrum–Kos strait, 2 h passage)', fontsize=12)

# land background
gk = np.linspace(-6000, 6000, 301)
dem = S.Dem(DIR3)
LA = LAT0 + gk[:, None] / mlat + 0 * gk[None, :]
LO = LON0 + 0 * gk[:, None] + gk[None, :] / mlon
land = dem.sample(LA, LO) > 0
ax.imshow(np.where(land, 1.0, np.nan), extent=[-6, 6, -6, 6],
          origin='lower', cmap='Greys', vmin=0, vmax=1.6, zorder=1)
ax.plot(truth[:, 0] / 1e3, truth[:, 1] / 1e3, '-', color='#111111', lw=2,
        label='truth', zorder=3)
ax.plot(dr[:, 0] / 1e3, dr[:, 1] / 1e3, '--', color='#d55e00', lw=1.6,
        label='dead reckoning', zorder=3)
ax.plot(fused[:, 0] / 1e3, fused[:, 1] / 1e3, '-', color='#0072b2', lw=1.6,
        label='fused (DR + skyline factors)', zorder=4)
th = np.linspace(0, 2 * np.pi, 60)
for k, (fix, cov) in fixes.items():
    w, V = np.linalg.eigh(cov)
    E = V @ np.diag(np.sqrt(w) * 2) @ np.vstack([np.cos(th), np.sin(th)])
    ax.plot((fix[0] + E[0]) / 1e3, (fix[1] + E[1]) / 1e3, '-',
            color='#009e73', lw=1.0, zorder=5)
    ax.plot(fix[0] / 1e3, fix[1] / 1e3, '+', color='#009e73', ms=9,
            mew=1.8, zorder=5)
ax.plot([], [], '+', color='#009e73', label='skyline fixes (2σ)')
ax.set_xlabel('east (km)')
ax.set_ylabel('north (km)')
ax.set_aspect('equal')
ax.legend(frameon=True, framealpha=0.85, edgecolor='none', fontsize=9,
          loc='upper left')
ax.set_title('(a) track (land gray)', fontsize=10)

tmin = np.arange(N + 1) * DT / 60
ax2.plot(tmin, err_dr, '--', color='#d55e00', lw=1.8, label='dead reckoning')
ax2.plot(tmin, err_fu, '-', color='#0072b2', lw=1.8, label='fused')
for k in fixes:
    ax2.axvline(k * DT / 60, color='#009e73', lw=0.7, alpha=0.5)
ax2.set_xlabel('time (min)')
ax2.set_ylabel('position error (m)')
ax2.set_title('(b) error vs time — green lines: skyline fixes', fontsize=10)
ax2.legend(frameon=False, fontsize=9)
ax2.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'e5_fusion.png'), dpi=110)
print('wrote', os.path.join(OUT, 'e5_fusion.png'))
