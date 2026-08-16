#!/usr/bin/env python3
"""E5h: wind turbines as a daytime landmark channel.

Turbines are surveyed, 100 m tall, ridge-crowning, and everywhere on
the Aegean coast — a charted landmark set the camera can use by day
the way it uses lighthouses by night. Three stages:

  1  blade flicker: a turbine's glint modulates at the blade-pass
     rate (0.5-1 Hz); a mast or building does not. Detector must fire
     on the turbine trace and stay silent on static points.
  2  constellation fix: two synthetic farms (8-turbine ridge line +
     3-turbine cape cluster) seen from sea with UNKNOWN correspondence
     and UNKNOWN compass bias. The presence filter (scene has
     turbines -> candidates without visible turbines are impossible)
     prunes the grid; the bearing-set Hough localizes within it and
     recovers the compass bias as a by-product.
  3  the E5 passage by day with the turbine channel in the loop
     (per-leg Hough match at the DR position, matched bearings into
     the shared-bias factor):  DR / DR+turbines / DR+skyline /
     DR+skyline+turbines.

Run:   python3 e5h_turbines.py      (writes out/e5h_results.json)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S
from skynav import SkyNav, AZ
from lightscan import blade_flicker_hz, classify_trace
from turbines import TurbineDB, demo_farm, hough_align, \
    constellation_score, wrap

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))
LAT0, LON0, Z = 36.95, 27.25, 5.0
DT, N, SPEED = 300.0, 24, 3.0
LOG_BIAS, CB = 1.03, np.radians(1.5)
VIS_M = 15000.0
TOL = np.radians(1.0)

# two farms: the demo ridge line plus a small cape cluster
_capes = demo_farm(lat0=36.845, lon0=27.155, n=3, spacing_m=380.0,
                   bearing_deg=115.0)
DB = TurbineDB(turbines=demo_farm().turbines + _capes.turbines)


def stage1():
    rng = np.random.default_rng(20260816)
    fs, T = 5.0, 200
    t = np.arange(T) / fs
    f_bp = 0.85
    glint = np.maximum(0.0, np.sin(2 * np.pi * f_bp * t)) ** 6
    tr_turbine = 60 + 25 * glint + rng.normal(0, 2, T)
    tr_mast = 60 + rng.normal(0, 2, T)
    tr_slowvar = 60 + 8 * np.sin(2 * np.pi * 0.02 * t) \
        + rng.normal(0, 2, T)
    got_t = blade_flicker_hz(t, tr_turbine)
    got_m = blade_flicker_hz(t, tr_mast)
    got_s = blade_flicker_hz(t, tr_slowvar)
    ok = got_t is not None and abs(got_t - f_bp) < 0.1 \
        and got_m is None and got_s is None
    print(f'  turbine trace -> {got_t and round(got_t, 2)} Hz '
          f'(true {f_bp}), mast -> {got_m}, slow drift -> {got_s}  '
          f'{"OK" if ok else "FAIL"}')
    return dict(turbine_hz=got_t, mast=got_m, slow=got_s, ok=bool(ok))


def stage2():
    rng = np.random.default_rng(20260817)
    mlat, mlon = S.meters_per_degree(LAT0)
    lat_t, lon_t = 36.905, 27.285          # true viewpoint, at sea
    pred_true, keep = DB.bearings_from(lat_t, lon_t, VIS_M)
    meas = wrap(pred_true + CB + rng.normal(0, np.radians(0.3),
                                            pred_true.size))
    print(f'  {meas.size} turbine bearings seen '
          f'(of {len(DB)} charted), bias {np.degrees(CB):+.1f} deg')

    step, half = 750.0, 12000.0
    g = np.arange(-half, half + 1, step)
    scores = np.full((g.size, g.size), -1.0)
    offs = np.zeros_like(scores)
    for i, dn in enumerate(g):
        for j, de in enumerate(g):
            la = lat_t + dn / mlat
            lo = lon_t + de / mlon
            scores[i, j], offs[i, j] = constellation_score(
                DB, meas, la, lo, VIS_M, TOL)
    pruned = int((scores < 0).sum())
    i, j = np.unravel_index(np.argmax(scores), scores.shape)
    err = float(np.hypot(g[i], g[j]))
    bias_rec = float(np.degrees(offs[i, j]))
    print(f'  presence filter pruned {pruned}/{scores.size} cells '
          f'({100 * pruned / scores.size:.0f}%)')
    print(f'  best cell err {err:.0f} m, recovered compass offset '
          f'{bias_rec:+.2f} deg (true {np.degrees(CB):+.1f})')
    ok = err <= 2 * step and abs(bias_rec - np.degrees(CB)) < 0.5
    print(f'  {"OK" if ok else "FAIL"}')
    return dict(n_seen=int(meas.size), pruned=pruned,
                cells=int(scores.size), err_m=err,
                bias_deg=bias_rec, ok=bool(ok))


def true_heading(k):
    return np.radians(80.0 if k < N // 2 else 110.0)


def stage3(use_skyline, use_turbines, cm=None):
    mlat, mlon = S.meters_per_degree(LAT0)
    truth = [np.array([-5000.0, -2200.0])]
    for k in range(N):
        h = true_heading(k)
        truth.append(truth[-1] + SPEED * DT
                     * np.array([np.sin(h), np.cos(h)]))
    rng = np.random.default_rng(20260818)
    nav = SkyNav(LAT0, LON0, Z, DIR3, lat_range=(36.4, 37.4),
                 lon_range=(26.6, 27.9), start_pos=tuple(truth[0]),
                 start_heading=true_heading(0),
                 start_sigma=(50.0, 50.0, np.radians(3)))
    errs, used = [], 0
    ten = DB.enu(LAT0, LON0)
    for k in range(N):
        dist = SPEED * DT * LOG_BIAS * (1 + rng.normal(0, 0.01))
        hdg = true_heading(k) + CB + rng.normal(0, np.radians(0.3))
        nav.add_odometry(dist, hdg)
        p = truth[k + 1]
        if use_turbines:
            r = np.hypot(ten[:, 0] - p[0], ten[:, 1] - p[1])
            vis = np.where(r < VIS_M)[0]
            if vis.size >= 2:
                brg_true = np.arctan2(ten[vis, 0] - p[0],
                                      ten[vis, 1] - p[1])
                meas = wrap(brg_true + CB
                            + rng.normal(0, np.radians(0.3), vis.size))
                lat_dr, lon_dr, _ = nav.current()
                pred, keep = DB.bearings_from(lat_dr, lon_dr,
                                              VIS_M + 3000)
                off, n_in, rms, pairs = hough_align(meas, pred, TOL)
                # correspondence gates (the picket-fence defence): a
                # line farm admits a shifted pairing that aligns n-1
                # bearings, so demand >=3 inliers, a plausible compass
                # offset, and a tight post-alignment rms before ANY
                # bearing enters the graph
                if n_in >= 3 and abs(off) <= np.radians(4.0) \
                        and rms <= 0.5 * TOL:
                    for i, jj in pairs:
                        t = DB.turbines[int(keep[jj])]
                        nav.add_light_bearing(
                            (t['lon'] - LON0) * mlon,
                            (t['lat'] - LAT0) * mlat, float(meas[i]))
                        used += 1
        if use_skyline and (k + 1) % 3 == 0:
            el, _ = cm.skyline(LAT0 + p[1] / mlat, LON0 + p[0] / mlon,
                               Z, AZ)
            el = el + rng.normal(0, 1e-3, el.size)
            el = np.roll(el, int(round(np.degrees(CB) / 0.1)))
            nav.take_fix(el)
        lat, lon, _ = nav.current()
        errs.append(np.hypot((lat - (LAT0 + p[1] / mlat)) * mlat,
                             (lon - (LON0 + p[0] / mlon)) * mlon))
    e = np.array(errs)
    return dict(mean=float(e.mean()), final=float(e[-1]),
                worst=float(e.max()), bias_deg=nav.compass_bias_deg(),
                bearings_used=used)


if __name__ == '__main__':
    print('stage 1: blade-flicker detection')
    s1 = stage1()
    print('stage 2: constellation fix, unknown correspondence + bias')
    s2 = stage2()
    print('stage 3: the passage with the turbine channel')
    cm = S.CMarcher(DIR3, (36.4, 37.4), (26.6, 27.9))
    out = dict(stage1=s1, stage2=s2)
    for name, sky, tb in (('DR', False, False),
                          ('DR+turbines', False, True),
                          ('DR+skyline', True, False),
                          ('DR+skyline+turbines', True, True)):
        r = stage3(sky, tb, cm)
        out[name] = r
        print(f"  {name:20s}: mean {r['mean']:6.1f} m  final "
              f"{r['final']:6.1f} m  worst {r['worst']:6.1f} m  "
              f"bias {r['bias_deg']:+.2f}"
              + (f"  bearings {r['bearings_used']}" if tb else ''))
    with open(os.path.join(OUT, 'e5h_results.json'), 'w') as f:
        json.dump(out, f, indent=1, default=str)
    print('\nOK' if s1['ok'] and s2['ok'] else '\nSTAGE FAILURES — see above')
