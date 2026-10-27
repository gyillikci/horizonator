#!/usr/bin/env python3
"""Where do the cruise fixes go wrong? Reads solve_cruise.py's summary and, per frame, what the
camera saw at the truth (COP30 ridges in the field of view: median range, relief) and what the
solver used (levelled pitch and roll against the truth, compass error, parallax), then bins
the position error by each.

    python3 analyze_cruise.py DIR [--solve solve]
"""
import argparse, json, math, os, re, sys
import numpy as np
EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP)
import skyfix as F, skyline as S

ap = argparse.ArgumentParser()
ap.add_argument('dir'); ap.add_argument('--solve', default='solve')
a = ap.parse_args()
d = os.path.abspath(a.dir)
res = json.load(open(os.path.join(d, a.solve, 'summary.json')))
DEM = os.path.expanduser('~/.horizonator/DEMs_COP30')
rows, cms = [], {}
for r in res:
    p = r['pose']
    region = re.match(r'cruise_(.+)_\d+$', r['tag']).group(1) if re.match(r'cruise_(.+)_\d+$', r['tag']) else 'run'
    key = (round(p['lat'], 1), round(p['lon'], 1))
    if key not in cms:
        cms[key] = S.CMarcher(DEM, (p['lat'] - .6, p['lat'] + .6), (p['lon'] - .8, p['lon'] + .8), d_min=1000.0, d_max=40000.0,
                              refraction_k=0.0)
    el, rng = cms[key].skyline(p['lat'], p['lon'], p['z_msl'], F.AZ)
    rel = (F.AZ - p['heading_deg'] + 180) % 360 - 180
    m = (np.abs(rel) <= p['hfov_deg'] / 2) & np.isfinite(rng) & (rng > 0) & (el > -2e-3)
    x = dict(tag=r['tag'], region=region, status=r['status'], err=r.get('err'), along=r.get('along'), cross=r.get('cross'),
             compass=abs(r['compass_err']), land_frac=float(m.mean()),
             r_med=float(np.median(rng[m]) / 1e3) if m.sum() > 20 else None,
             relief=float(np.std(el[m]) * 1e3) if m.sum() > 20 else None)
    jf = os.path.join(d, a.solve, r['tag'] + '.json')
    if os.path.exists(jf):
        j = json.load(open(jf)); x['parallax'] = j.get('parallax_inv_km'); x['rms'] = j.get('rms_mrad')
    lg = os.path.join(d, a.solve, r['tag'] + '.log')
    if os.path.exists(lg):
        mm = re.search(r'pitch ([-+\d.]+), roll ([-+\d.]+)', open(lg).read())
        if mm:
            pr = (p['roll_deg'] + 180) % 360 - 180
            x['dpitch'], x['droll'] = abs(float(mm.group(1)) - p['pitch_deg']), abs(float(mm.group(2)) - pr)
    rows.append(x)


def table(title, key, edges, fmt):
    print('\n%s' % title)
    print('  %-14s %4s %5s %7s %7s %7s %7s %6s' % ('bin', 'n', 'acc', 'median', '<300', '>1km', '|along|', '|cross|'))
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = [x for x in rows if x.get(key) is not None and lo <= x[key] < hi and x['err'] is not None]
        if not g:
            continue
        e = np.array([x['err'] for x in g])
        print('  %-14s %4d %5d %7.0f %7d %7d %7.0f %6.0f' % (fmt % (lo, hi), len(g), sum(x['status'] == 'ok' for x in g), np.median(e),
              (e < 300).sum(), (e > 1000).sum(), np.median([abs(x['along']) for x in g]), np.median([abs(x['cross']) for x in g])))


print('%d frames, %d solved, %d accepted' % (len(rows), sum(x['err'] is not None for x in rows), sum(x['status'] == 'ok' for x in rows)))
print('\nby region')
print('  %-10s %4s %6s %5s %7s %7s %6s  %s' % ('region', 'n', 'solved', 'acc', 'median', '<300', '>1km', 'refusals'))
for reg in sorted({x['region'] for x in rows}):
    g = [x for x in rows if x['region'] == reg]
    e = np.array([x['err'] for x in g if x['err'] is not None])
    acc = [x for x in g if x['status'] == 'ok']
    print('  %-10s %4d %6d %5d %7.0f %7d %6d  %s' % (reg, len(g), e.size, len(acc), np.median(e) if e.size else np.nan,
          sum(x['err'] < 300 for x in acc), sum(x['err'] > 1000 for x in acc),
          ', '.join('%s %d' % (k, v) for k, v in sorted({s: sum(x['status'] == s for x in g) for s in {x['status'] for x in g} if s != 'ok'}.items()))))
table('by median range of the ridges in view (km)', 'r_med', [0, 3, 5, 8, 12, 20, 41], '%2.0f-%2.0f km')
table('by skyline relief (std, mrad)', 'relief', [0, 5, 10, 20, 40, 1e9], '%3.0f-%3.0f mrad')
table('by compass error (deg)', 'compass', [0, 2.5, 5, 7.5, 10, 30], '%4.1f-%4.1f deg')
table('by levelled roll error (deg)', 'droll', [0, 0.25, 0.5, 1, 10], '%4.2f-%4.2f deg')
table('by levelled pitch error (deg)', 'dpitch', [0, 0.1, 0.25, 0.5, 10], '%4.2f-%4.2f deg')
table('by parallax (1/r spread per km)', 'parallax', [0, 0.03, 0.1, 0.2, 0.4, 10], '%4.2f-%4.2f')
json.dump(rows, open(os.path.join(d, a.solve, 'analysis.json'), 'w'), indent=1)
