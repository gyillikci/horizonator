#!/usr/bin/env python3
"""E4v: the instrument on its own field data.

The curated Theodolite set (83 complete pairs, five sites, July-August
2026) is the first time this pipeline sees photographs taken FOR it:
clean camera frame, app-recorded attitude with its accuracies, GPS
good to a few meters for scoring only.

Each prime sighting (paired, sea horizon detectable, telephoto) is
solved with the auto-levelled solver inside a 6 km box centred on the
GPS, then scored three ways:

  error         distance from the GPS truth, decomposed ALONG and
                ACROSS the line of sight — a single narrow-FOV frame
                of distant terrain is close to a bearing measurement,
                so the split says whether a miss is geometry or a
                genuine mismatch
  gate          basin margin vs the E4k tiers, i.e. would the
                instrument have offered this fix
  calibration   reported sigma vs actual error, the ratio that tells
                whether the Laplace covariance (scale calibrated on
                synthetic runs in E1) is honest on real data

Run:   python3 e4v_theodolite.py [--index PATH]
"""

import os
import sys
import json
import subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
PHOTOS = os.environ.get(
    'THEODOLITE_DIR',
    os.path.join(os.path.dirname(os.path.dirname(HERE)),
                 'celestial-navigation', 'theodolite'))
INDEX = os.path.join(OUT, 'theodolite', 'index.json')


def run_one(s):
    e, a = s['exif'], s['attitude']
    img = os.path.join(PHOTOS, s['raw'])
    cmd = ['python3', os.path.join(HERE, 'skyfix.py'), img,
           '--center', f"{e['lat']:.6f},{e['lon']:.6f}",
           '--fov', f"{a['fov_deg']:.3f}",
           '--heading', f"{a['heading_deg']:.2f}",
           '--z', f"{max(e.get('alt_m') or 5.0, 2.0):.1f}",
           '--auto-level', '--box', '6000']
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    txt = p.stdout
    if '{' not in txt:
        return None
    j = json.loads(txt[txt.index('{'):])
    mlat = 111132.0
    mlon = 111320.0 * np.cos(np.radians(e['lat']))
    dn = (j['lat'] - e['lat']) * mlat
    de = (j['lon'] - e['lon']) * mlon
    h = np.radians(a['heading_deg'] + j.get('heading_offset_deg', 0.0))
    return dict(
        id=s['id'], site=f"{e['lat']:.4f},{e['lon']:.4f}",
        fov=a['fov_deg'], heading=a['heading_deg'],
        az_acc=a.get('azimuth_acc_deg'), gps_acc=a.get('gps_horz_m'),
        err_m=float(np.hypot(dn, de)), dn=float(dn), de=float(de),
        along_m=float(dn * np.cos(h) + de * np.sin(h)),
        across_m=float(-dn * np.sin(h) + de * np.cos(h)),
        margin=j.get('basin_margin'), rms_mrad=j.get('rms_mrad'),
        fix_ok=j.get('fix_ok'),
        sigma_n=j.get('sigma_n_m'), sigma_e=j.get('sigma_e_m'),
        heading_offset=j.get('heading_offset_deg'),
        level_pitch=j['photos'][0].get('pitch_deg'),
        level_roll=j['photos'][0].get('roll_deg'),
        theo_pitch=a.get('pitch_deg'), theo_roll=a.get('roll_deg'))


if __name__ == '__main__':
    idx = sys.argv[sys.argv.index('--index') + 1] \
        if '--index' in sys.argv else INDEX
    with open(idx) as f:
        S = json.load(f)['sightings']
    prime = [s for s in S if s.get('horizon')
             and (s['attitude'].get('fov_deg') or 99) < 25
             and s['attitude'].get('heading_deg') is not None]
    print(f'{len(prime)} prime sightings')
    rows = []
    for s in prime:
        try:
            r = run_one(s)
        except Exception as ex:
            print(f"  {s['id']}: {ex}")
            continue
        if r is None:
            print(f"  {s['id']}: no solution")
            continue
        rows.append(r)
        print(f"  {r['id']:12s} err {r['err_m']:6.0f} m "
              f"(along {r['along_m']:+6.0f} across {r['across_m']:+6.0f})"
              f"  margin {r['margin']:5.2f}  rms {r['rms_mrad']:.2f}"
              f"  sigma {r['sigma_n']:.0f}/{r['sigma_e']:.0f} m"
              f"  hdg_off {r['heading_offset']:+.1f}", flush=True)

    if rows:
        e = np.array([r['err_m'] for r in rows])
        sg = np.array([np.hypot(r['sigma_n'], r['sigma_e'])
                       for r in rows])
        acc = [r for r in rows if r['fix_ok']]
        print(f'\n{len(rows)} solved: median error {np.median(e):.0f} m, '
              f'best {e.min():.0f} m')
        print(f'  accepted by the gates: {len(acc)}, of which within '
              f'500 m: {sum(1 for r in acc if r["err_m"] < 500)}')
        print(f'  error / reported sigma: median '
              f'{np.median(e / sg):.1f}x  (1.0 = honest covariance)')
        dp = np.array([r['level_pitch'] - r['theo_pitch'] for r in rows
                       if r['theo_pitch'] is not None])
        dr = np.array([r['level_roll'] - r['theo_roll'] for r in rows
                       if r['theo_roll'] is not None])
        if dp.size:
            print(f'  auto-level vs Theodolite inclinometer: pitch '
                  f'median {np.median(dp):+.2f} deg, roll median '
                  f'{np.median(dr):+.2f} deg')
    with open(os.path.join(OUT, 'e4v_results.json'), 'w') as f:
        json.dump(rows, f, indent=1)
