#!/usr/bin/env python3
"""E5b: the live instrument loop (skynav.SkyNav) on the E5 scenario.

Same 2-hour Bodrum-Kos passage and sensor errors as e5_fusion.py, but
processed as a STREAM: each odometry leg and each skyline fix is fused
incrementally with iSAM2 the moment it arrives, and the fused position is
emitted as NMEA sentences (written to out/e5b.nmea -- feed it to OpenCPN
to watch the track). Reports per-update latency, the numbers that matter
on the CM5.

Run:   python3 e5b_live.py       (no GL needed)
"""

import os
import sys
import time
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

rng = np.random.default_rng(20260818)
mlat, mlon = S.meters_per_degree(LAT0)
cm_truth = S.CMarcher(DIR3, (36.4, 37.4), (26.6, 27.9))


def true_heading(k):
    return np.radians(80.0 if k < N // 2 else 110.0)


truth = [np.array([-5000.0, -2200.0])]
for k in range(N):
    h = true_heading(k)
    truth.append(truth[-1] + SPEED * DT * np.array([np.sin(h), np.cos(h)]))
truth = np.array(truth)

nav = SkyNav(LAT0, LON0, Z, DIR3, lat_range=(36.4, 37.4),
             lon_range=(26.6, 27.9),
             start_pos=tuple(truth[0]), start_heading=true_heading(0),
             start_sigma=(50.0, 50.0, np.radians(3)))

nmea = open(os.path.join(OUT, 'e5b.nmea'), 'w')
errs, t_odo, t_fix = [], [], []
for k in range(N):
    # ---- sensors for this leg (simulated)
    dist = SPEED * DT * LOG_BIAS * (1 + rng.normal(0, 0.01))
    hdg = true_heading(k) + COMPASS_BIAS + rng.normal(0, np.radians(0.3))

    t0 = time.time()
    nav.add_odometry(dist, hdg)
    t_odo.append(time.time() - t0)

    if (k + 1) % FIX_EVERY == 0:
        el_obs, _ = cm_truth.skyline(LAT0 + truth[k + 1][1] / mlat,
                                     LON0 + truth[k + 1][0] / mlon, Z, AZ)
        el_obs = el_obs + rng.normal(0, 1e-3, el_obs.size)
        el_obs = np.roll(el_obs,
                         int(round(np.degrees(COMPASS_BIAS) / 0.1)))
        t0 = time.time()
        nav.take_fix(el_obs)
        t_fix.append(time.time() - t0)

    lat, lon, cov = nav.current()
    err = np.hypot((lat - (LAT0 + truth[k + 1][1] / mlat)) * mlat,
                   (lon - (LON0 + truth[k + 1][0] / mlon)) * mlon)
    errs.append(err)
    tsim = (k + 1) * DT
    hms = f'{12 + int(tsim // 3600):02d}{int(tsim % 3600 // 60):02d}' \
          f'{int(tsim % 60):02d}.00'
    nmea.write(nav.nmea_gga(hms) + '\r\n')
    nmea.write(nav.nmea_rmc(hms, '100826', SPEED * 1.9438,
                            np.degrees(true_heading(k))) + '\r\n')
    tag = ' [fix]' if (k + 1) % FIX_EVERY == 0 else ''
    print(f't={tsim/60:5.0f} min  err {err:6.1f} m  '
          f'sigma {np.sqrt(np.trace(cov)):5.1f} m{tag}', flush=True)
nmea.close()

errs = np.array(errs)
print(f'\nlive fused: mean {errs.mean():.0f} m, final {errs[-1]:.0f} m '
      f'(batch E5 reference: 46 m / 9 m)')
print(f'latency: odometry update {np.mean(t_odo)*1e3:.1f} ms, '
      f'fix (solve+fuse) {np.mean(t_fix):.2f} s')
print('NMEA log -> out/e5b.nmea')
