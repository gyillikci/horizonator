#!/usr/bin/env python3
"""E5m: what breaks the fix most? The perfect-compass bound.

Every single-frame field fix co-estimates the compass offset inside
+-6 deg of a +-10 deg phone compass, and E4v measured the trade:
widening the window improves the fit and worsens the position. This
isolates heading's share directly: per sighting, (1) find the
REFERENCE heading by letting the offset float in a tiny box at the
GPS truth, (2) re-solve the full 6 km box with that heading and the
window clamped to +-0.5 deg. The error that REMAINS is what heading
cannot explain — model error plus geometry.

Run:  python3 e5m_blame.py      (writes out/e5m_results.json)
"""
import os, sys, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      'celestial-navigation', 'theodolite')
IDS = ['MYQR7719', 'KWHC9160', 'PQBC6867', 'EWAC7374', 'INKX2521',
       'SRYK4301']


def skyfix(img, la, lo, fov, hd, z, box, hw, extra=()):
    cmd = ['python3', os.path.join(HERE, 'skyfix.py'), img,
           '--center', f'{la:.6f},{lo:.6f}', '--fov', f'{fov:.3f}',
           '--heading', f'{hd:.2f}', '--z', f'{z:.1f}',
           '--auto-level', '--box', str(box),
           '--heading-window', str(hw)] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True,
                       timeout=1200)
    t = p.stdout
    return json.loads(t[t.index('{'):]) if '{' in t else None


if __name__ == '__main__':
    idx = {x['id']: x for x in json.load(open(os.path.join(
        HERE, 'out', 'theodolite', 'index.json')))['sightings']}
    base = {x['id']: x for x in json.load(open(os.path.join(
        HERE, 'out', 'e4v_results_all.json')))}
    rows = []
    for sid in IDS:
        s = idx[sid]; e, a = s['exif'], s['attitude']
        img = os.path.join(PHOTOS, s['raw'])
        z = max(e.get('alt_m') or 5.0, 2.0)
        # (1) reference heading: offset floats +-15 deg in a 400 m box
        j0 = skyfix(img, e['lat'], e['lon'], a['fov_deg'],
                    a['heading_deg'], z, 400, 15.0)
        if j0 is None or 'heading_offset_deg' not in j0:
            print(f'{sid}: no reference'); continue
        href = a['heading_deg'] + j0['heading_offset_deg']
        # (2) full box, heading clamped
        j1 = skyfix(img, e['lat'], e['lon'], a['fov_deg'], href, z,
                    6000, 0.5)
        if j1 is None:
            print(f'{sid}: no solve'); continue
        mlat = 111132.0
        mlon = 111320.0 * np.cos(np.radians(e['lat']))
        err = float(np.hypot((j1['lat'] - e['lat']) * mlat,
                             (j1['lon'] - e['lon']) * mlon))
        b = base.get(sid, {})
        rows.append(dict(id=sid, err_perfect=err,
                         err_compass=b.get('err_m'),
                         href_offset=j0['heading_offset_deg'],
                         margin=j1.get('basin_margin'),
                         rms=j1.get('rms_mrad')))
        print(f"{sid}: compass-heading err {b.get('err_m', float('nan')):6.0f} m"
              f" -> perfect-heading err {err:6.0f} m   (device was "
              f"{j0['heading_offset_deg']:+.1f} deg off)", flush=True)
    with open(os.path.join(HERE, 'out', 'e5m_results.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    if rows:
        ec = np.array([r['err_compass'] for r in rows
                       if r['err_compass']])
        ep = np.array([r['err_perfect'] for r in rows])
        off = np.array([abs(r['href_offset']) for r in rows])
        print(f'\nmedian: compass {np.median(ec):.0f} m -> perfect '
              f'{np.median(ep):.0f} m; device heading was a median '
              f'{np.median(off):.1f} deg off')
