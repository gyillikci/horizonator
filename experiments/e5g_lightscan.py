#!/usr/bin/env python3
"""E5g: close the night-channel loop — video in, identified bearings
out, fixes through the graph.

E5f assumed an oracle told the camera WHICH charted light it saw.
This experiment removes the oracle. Three stages, tested end to end:

  1  A synthetic night video (40 s at 5 Hz, 90x160 px) with the three
     demo lights on the horizon — Fl(3)W.15s, Fl.W.5s, Iso.W.4s —
     plus an UNCHARTED decoy (a fishing boat's working light,
     Fl.W.2.5s), sensor noise, and wave-occlusion dropouts.
     track_points + classify_trace must recover each character.
  2  LightDB.match within an 18 km gate must identify the three
     charted lights uniquely and REJECT the decoy (no chart entry).
  3  The E5f night passage re-run with identification in the loop:
     at each leg the visible lights' blink traces are synthesized,
     classified, matched, and only IDENTIFIED bearings enter
     SkyNav.add_light_bearing. Compare against E5f's oracle runs.

Run:   python3 e5g_lightscan.py      (writes out/e5g_results.json)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S
from skynav import SkyNav
from lights import demo_db, parse_character_string, LightDB
from lightscan import track_points, classify_trace, identify

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))
LAT0, LON0, Z = 36.95, 27.25, 5.0
DT, N, SPEED = 300.0, 24, 3.0
LOG_BIAS, CB = 1.03, np.radians(1.5)
VIS_M = 18000.0
FS, WATCH_S = 5.0, 40.0
DB = demo_db()


def flash_wave(char, t, phase=0.0):
    """Lit/dark wave of a light character over time vector t."""
    p = char['period_s'] or 1.0
    tt = (t + phase) % p
    if char['pattern'] == 'Iso':
        return (tt < p / 2).astype(float)
    if char['pattern'] in ('Fl', 'LFl'):
        w = 2.0 if char['pattern'] == 'LFl' else 0.8
        on = np.zeros_like(t)
        for g in range(char.get('group', 1)):
            on = np.maximum(on, ((tt >= g * 2.0)
                                 & (tt < g * 2.0 + w)).astype(float))
        return on
    if char['pattern'] == 'Oc':
        return (tt >= 1.5).astype(float)
    return np.ones_like(t)


def synth_video(lights_px, rng, T=int(WATCH_S * FS), H=90, W=160):
    """Night frames with point lights (u, v, char, phase), noise and
    wave-occlusion dropouts."""
    t = np.arange(T) / FS
    frames = rng.normal(8.0, 2.0, (T, H, W))     # sensor noise floor
    for u, v, char, phase in lights_px:
        on = flash_wave(char, t, phase)
        occl = rng.random(T) > 0.06              # 6% wave dropouts
        amp = 120.0 * on * occl
        for k in range(T):
            if amp[k] > 0:
                frames[k, v - 1:v + 2, u - 1:u + 2] += amp[k] * 0.5
                frames[k, v, u] += amp[k] * 0.5
    return t, frames


def stage12():
    rng = np.random.default_rng(20260816)
    chars = [DB.lights[0]['char'], DB.lights[1]['char'],
             DB.lights[2]['char'], parse_character_string('Fl.W.2.5s')]
    names = ['demo:1', 'demo:2', 'demo:3', 'DECOY']
    px = [(20, 44, chars[0], 1.0), (70, 45, chars[1], 2.2),
          (120, 44, chars[2], 0.4), (150, 46, chars[3], 3.1)]
    t, frames = synth_video(px, rng)
    tracks = track_points(frames)
    got = {}
    for tr in tracks:
        c = classify_trace(t, tr['trace'])
        if c is None:
            continue
        near = min(px, key=lambda p: abs(p[0] - tr['u']))
        got[names[px.index(near)]] = c
    report = {}
    okall = True
    for name, true_c in zip(names, chars):
        c = got.get(name)
        # correct = would match its own character (same rule the real
        # identification uses, harmonic equivalence included)
        ref = LightDB(lights=[dict(id='x', name='', lat=LAT0, lon=LON0,
                                   char=true_c, synthetic=True)])
        ok = c is not None and len(ref.match(c, LAT0, LON0, 1000.0)) == 1
        okall &= ok
        report[name] = dict(true=true_c, classified=c, ok=bool(ok))
        print(f"  {name:8s} true {fmt(true_c):14s} -> "
              f"classified {fmt(c):14s} {'OK' if ok else 'MISS'}")
    # stage 2: unique-match identification, decoy rejection
    ids = {}
    for name in names:
        c = got.get(name)
        cand = DB.match(c, LAT0, LON0, 60000.0) if c else []
        ids[name] = [L['id'] for L in cand]
        print(f"  {name:8s} matches {ids[name]}")
    id_ok = all(ids[f'demo:{i}'] == [f'demo:{i}'] for i in (1, 2, 3)) \
        and ids['DECOY'] == []
    print(f'  identification: {"OK" if id_ok else "FAIL"} '
          f'(three unique, decoy rejected)')
    return dict(classification=report, identification=ids,
                ok=bool(okall and id_ok))


def fmt(c):
    if c is None:
        return 'None'
    g = f"({c['group']})" if c.get('group', 1) > 1 else ''
    p = f".{c['period_s']:.0f}s" if c.get('period_s') else ''
    return f"{c['pattern']}{g}{p}"


def true_heading(k):
    return np.radians(80.0 if k < N // 2 else 110.0)


def run_passage(identified_only):
    """The E5f night passage; identified_only routes every sighting
    through synth trace -> classify -> match instead of the oracle."""
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
    errs, n_used, n_rej = [], 0, 0
    t = np.arange(int(WATCH_S * FS)) / FS
    for k in range(N):
        dist = SPEED * DT * LOG_BIAS * (1 + rng.normal(0, 0.01))
        hdg = true_heading(k) + CB + rng.normal(0, np.radians(0.3))
        nav.add_odometry(dist, hdg)
        p = truth[k + 1]
        lat_dr, lon_dr, _ = nav.current()
        for L in DB.lights:
            le = (L['lon'] - LON0) * mlon
            ln = (L['lat'] - LAT0) * mlat
            r = np.hypot(le - p[0], ln - p[1])
            if r >= VIS_M:
                continue
            brg_true = np.arctan2(le - p[0], ln - p[1])
            brg_meas = brg_true + CB + rng.normal(0, np.radians(0.4))
            if identified_only:
                on = flash_wave(L['char'], t, rng.uniform(0, 15))
                y = 8 + 2 * rng.normal(size=t.size) \
                    + 120 * on * (rng.random(t.size) > 0.06)
                c = classify_trace(t, y)
                cand = DB.match(c, lat_dr, lon_dr, VIS_M + 3000)
                if len(cand) != 1 or cand[0]['id'] != L['id']:
                    n_rej += 1
                    continue
                n_used += 1
            nav.add_light_bearing(le, ln, brg_meas)
        lat, lon, _ = nav.current()
        errs.append(np.hypot((lat - (LAT0 + p[1] / mlat)) * mlat,
                             (lon - (LON0 + p[0] / mlon)) * mlon))
    e = np.array(errs)
    return dict(mean=float(e.mean()), final=float(e[-1]),
                worst=float(e.max()), bias_deg=nav.compass_bias_deg(),
                sightings_used=n_used, sightings_rejected=n_rej)


if __name__ == '__main__':
    print('stage 1+2: video -> tracks -> characters -> identification')
    s12 = stage12()
    print('\nstage 3: the E5f passage with identification in the loop')
    oracle = run_passage(identified_only=False)
    closed = run_passage(identified_only=True)
    for name, r in (('oracle-ID (E5f)', oracle), ('classified-ID', closed)):
        print(f"  {name:16s}: mean {r['mean']:6.1f} m  final "
              f"{r['final']:6.1f} m  worst {r['worst']:6.1f} m  "
              f"bias {r['bias_deg']:+.2f} deg"
              + (f"  used {r['sightings_used']}, rejected "
                 f"{r['sightings_rejected']}" if name != 'oracle-ID (E5f)'
                 else ''))
    with open(os.path.join(OUT, 'e5g_results.json'), 'w') as f:
        json.dump(dict(stage12=s12, oracle=oracle, closed=closed), f,
                  indent=1, default=str)
    print('\nOK' if s12['ok'] else '\nSTAGE-1/2 FAILURES — see above')
