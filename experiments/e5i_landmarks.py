#!/usr/bin/env python3
"""E5i: pylon rows and comm masts join the landmark web.

Builds on E5h (turbine constellation) and E5g (light identification).
The full daytime landmark set is now four classes — turbines, pylons,
masts, lighthouses(unlit by day) — and the night set gains the masts'
red obstruction lights. Stages:

  1  mixed-class constellation: 8 turbines + 10 pylons + 2 masts
     seen together (21 anonymous bearings, unknown correspondence,
     unknown compass bias). Class-aware assignment (blade flicker
     separates turbines from static towers) vs classless: same
     Hough, measurably tighter ambiguity.
  2  night crossover: the two masts' Fl.R.1.5s obstruction lights
     join the sea-light DB; the E5g night passage re-runs with the
     enlarged DB — identification still via the flash classifier,
     radius gate disambiguates the two identical mast characters.
  3  day passage with the whole web (turbines + pylons + masts)
     vs turbines only (E5h stage 3).

Run:   python3 e5i_landmarks.py     (writes out/e5i_results.json)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S
from skynav import SkyNav
from lights import demo_db, LightDB
from lightscan import classify_trace
from turbines import demo_farm, hough_align, constellation_score, \
    wrap, wrap as wrapa
from landmarks import demo_pylon_line, demo_masts, as_light_entries, \
    combined_db
from e5g_lightscan import flash_wave, FS, WATCH_S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))
LAT0, LON0, Z = 36.95, 27.25, 5.0
DT, N, SPEED = 300.0, 24, 3.0
LOG_BIAS, CB = 1.03, np.radians(1.5)
VIS_M = 15000.0
TOL = np.radians(1.0)

DAY_DB = combined_db(demo_farm(), demo_pylon_line(), demo_masts())
NIGHT_DB = LightDB(lights=demo_db().lights
                   + as_light_entries(demo_masts()))


def true_heading(k):
    return np.radians(80.0 if k < N // 2 else 110.0)


def stage1():
    rng = np.random.default_rng(20260819)
    lat_t, lon_t = 36.905, 27.285
    pred, keep = DAY_DB.bearings_from(lat_t, lon_t, VIS_M)
    cls_true = [DAY_DB.turbines[int(k)]['cls'] for k in keep]
    meas = wrap(pred + CB + rng.normal(0, np.radians(0.3), pred.size))
    cls_meas = ['turbine' if c == 'turbine' else 'static'
                for c in cls_true]
    print(f'  {meas.size} bearings in view '
          f'({cls_true.count("turbine")} turbine, '
          f'{cls_true.count("pylon")} pylon, '
          f'{cls_true.count("mast")} mast)')
    mlat, mlon = S.meters_per_degree(LAT0)
    step, half = 750.0, 12000.0
    g = np.arange(-half, half + 1, step)
    out = {}
    for name, cm in (('classless', None), ('class-aware', cls_meas)):
        scores = np.full((g.size, g.size), -1.0)
        offs = np.zeros_like(scores)
        for i, dn in enumerate(g):
            for j, de in enumerate(g):
                scores[i, j], offs[i, j] = constellation_score(
                    DAY_DB, meas, lat_t + dn / mlat,
                    lon_t + de / mlon, VIS_M, TOL, cls_meas=cm)
        i, j = np.unravel_index(np.argmax(scores), scores.shape)
        err = float(np.hypot(g[i], g[j]))
        # ambiguity: cells scoring within 1 inlier of the best
        near = int((scores >= scores[i, j] - 1.0).sum())
        out[name] = dict(err_m=err, bias_deg=float(np.degrees(offs[i, j])),
                         near_best_cells=near)
        print(f'  {name:11s}: err {err:5.0f} m  bias '
              f'{np.degrees(offs[i, j]):+.2f} deg  cells within 1 '
              f'inlier of best: {near}')
    ok = out['class-aware']['err_m'] <= 2 * step \
        and out['class-aware']['near_best_cells'] \
        <= out['classless']['near_best_cells']
    print(f'  {"OK" if ok else "FAIL"}')
    out['ok'] = bool(ok)
    return out


def night_passage(db):
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
    t = np.arange(int(WATCH_S * FS)) / FS
    errs, used = [], 0
    for k in range(N):
        dist = SPEED * DT * LOG_BIAS * (1 + rng.normal(0, 0.01))
        hdg = true_heading(k) + CB + rng.normal(0, np.radians(0.3))
        nav.add_odometry(dist, hdg)
        p = truth[k + 1]
        lat_dr, lon_dr, _ = nav.current()
        for L in db.lights:
            le = (L['lon'] - LON0) * mlon
            ln = (L['lat'] - LAT0) * mlat
            r = np.hypot(le - p[0], ln - p[1])
            if r >= VIS_M:
                continue
            on = flash_wave(L['char'], t, rng.uniform(0, 15))
            y = 8 + 2 * rng.normal(size=t.size) \
                + 120 * on * (rng.random(t.size) > 0.06)
            c = classify_trace(t, y)
            brg = np.arctan2(le - p[0], ln - p[1]) + CB \
                + rng.normal(0, np.radians(0.4))
            cand = db.match(c, lat_dr, lon_dr, VIS_M + 3000)
            if len(cand) > 1:
                # two masts share the generic aviation character —
                # disambiguate by the DR-predicted bearing (gate 5 deg)
                dr_e = (lon_dr - LON0) * mlon
                dr_n = (lat_dr - LAT0) * mlat
                cand = [cc for cc in cand if abs(wrapa(
                    np.arctan2((cc['lon'] - LON0) * mlon - dr_e,
                               (cc['lat'] - LAT0) * mlat - dr_n)
                    - brg)) < np.radians(5.0)]
            if len(cand) != 1 or cand[0]['id'] != L['id']:
                continue
            nav.add_light_bearing(le, ln, float(brg))
            used += 1
        lat, lon, _ = nav.current()
        errs.append(np.hypot((lat - (LAT0 + p[1] / mlat)) * mlat,
                             (lon - (LON0 + p[0] / mlon)) * mlon))
    e = np.array(errs)
    return dict(mean=float(e.mean()), final=float(e[-1]),
                worst=float(e.max()), bias_deg=nav.compass_bias_deg(),
                used=used)


def day_passage(db):
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
    ten = db.enu(LAT0, LON0)
    cls_all = [t['cls'] for t in db.turbines]
    errs, used = [], 0
    for k in range(N):
        dist = SPEED * DT * LOG_BIAS * (1 + rng.normal(0, 0.01))
        hdg = true_heading(k) + CB + rng.normal(0, np.radians(0.3))
        nav.add_odometry(dist, hdg)
        p = truth[k + 1]
        r = np.hypot(ten[:, 0] - p[0], ten[:, 1] - p[1])
        vis = np.where(r < VIS_M)[0]
        if vis.size >= 2:
            brg_true = np.arctan2(ten[vis, 0] - p[0],
                                  ten[vis, 1] - p[1])
            meas = wrap(brg_true + CB
                        + rng.normal(0, np.radians(0.3), vis.size))
            cls_meas = ['turbine' if cls_all[v] == 'turbine'
                        else 'static' for v in vis]
            lat_dr, lon_dr, _ = nav.current()
            pred, keep = db.bearings_from(lat_dr, lon_dr, VIS_M + 3000)
            cls_pred = [cls_all[int(kk)] for kk in keep]
            off, n_in, rms, pairs = hough_align(
                meas, pred, TOL, cls_meas, cls_pred)
            if n_in >= 3 and abs(off) <= np.radians(4.0) \
                    and rms <= 0.5 * TOL:
                for i, jj in pairs:
                    tt = db.turbines[int(keep[jj])]
                    nav.add_light_bearing(
                        (tt['lon'] - LON0) * mlon,
                        (tt['lat'] - LAT0) * mlat, float(meas[i]))
                    used += 1
        lat, lon, _ = nav.current()
        errs.append(np.hypot((lat - (LAT0 + p[1] / mlat)) * mlat,
                             (lon - (LON0 + p[0] / mlon)) * mlon))
    e = np.array(errs)
    return dict(mean=float(e.mean()), final=float(e[-1]),
                worst=float(e.max()), bias_deg=nav.compass_bias_deg(),
                used=used)


if __name__ == '__main__':
    print('stage 1: mixed-class constellation, class gate value')
    s1 = stage1()
    print('stage 2: night crossover — mast obstruction lights')
    sea = night_passage(LightDB(lights=demo_db().lights))
    full = night_passage(NIGHT_DB)
    for name, r in (('sea lights only', sea), ('+ mast lights', full)):
        print(f"  {name:16s}: mean {r['mean']:6.1f} m  final "
              f"{r['final']:6.1f} m  worst {r['worst']:6.1f} m  "
              f"bias {r['bias_deg']:+.2f}  identified {r['used']}")
    print('stage 3: day passage, whole web vs turbines only')
    turb = day_passage(demo_farm())
    web = day_passage(DAY_DB)
    for name, r in (('turbines only', turb), ('whole web', web)):
        print(f"  {name:16s}: mean {r['mean']:6.1f} m  final "
              f"{r['final']:6.1f} m  worst {r['worst']:6.1f} m  "
              f"bias {r['bias_deg']:+.2f}  bearings {r['used']}")
    with open(os.path.join(OUT, 'e5i_results.json'), 'w') as f:
        json.dump(dict(stage1=s1, night_sea=sea, night_full=full,
                       day_turbines=turb, day_web=web), f, indent=1,
                  default=str)
    print('\nOK' if s1['ok'] else '\nSTAGE-1 FAILURE — see above')
