#!/usr/bin/env python3
"""The same routes sailed looking dead ahead and looking where look_planner.py said, route by
route: single-frame skyfix (accepted, within 300 m, median error) and a tracker's median.

    python3 look_compare.py out/aegean/speed_B out/aegean_look/solve_B [--track pfb_known_bias,pfk]
      --track A,B: the tracker results <dir>/A.json (dead ahead) and <dir>/B.json (looking)
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('ahead'); ap.add_argument('look')
ap.add_argument('--track', default=None)
a = ap.parse_args()


def single(p):
    out = {}
    for r in json.load(open(os.path.join(p, 'summary.json'))):
        out.setdefault(r['tag'].split('_')[1], []).append(r)
    return out


def tracker(p, name):
    f = os.path.join(os.path.dirname(os.path.normpath(p)), name + '.json')
    return json.load(open(f))['routes'] if os.path.exists(f) else {}


A, L = single(a.ahead), single(a.look)
TA = TL = {}
if a.track:
    na, nl = a.track.split(',')
    TA, TL = tracker(a.ahead, na), tracker(a.look, nl)
print('%-9s | %-29s | %-29s | %s' % ('route', 'dead ahead: acc, <300, median', 'looking: acc, <300, median', 'tracker median: ahead -> looking'))
tot = {k: [] for k in ('a', 'l')}
for route in sorted(set(A) | set(L)):
    row = []
    for key, d in (('a', A), ('l', L)):
        rs = d.get(route, [])
        ok = [r for r in rs if r['status'] == 'ok']
        e = np.array([r['err'] for r in ok])
        tot[key] += rs
        row.append('%3d/%-3d %3d %8.0f m' % (len(ok), len(rs), (e < 300).sum(), np.median(e) if e.size else np.nan))
    tr = ''
    if route in TA or route in TL:
        tr = '%5.0f -> %5.0f m' % (np.median([r['err'] for r in TA.get(route, [{'err': np.nan}])]),
                                   np.median([r['err'] for r in TL.get(route, [{'err': np.nan}])]))
    print('%-9s | %-29s | %-29s | %s' % (route, row[0], row[1], tr))
for key, name in (('a', 'all, dead ahead'), ('l', 'all, looking')):
    ok = [r for r in tot[key] if r['status'] == 'ok']
    e = np.array([r['err'] for r in ok])
    print('%-16s %d/%d accepted, %d under 300 m, %d over 1 km, median %.0f m' % (
        name, len(ok), len(tot[key]), (e < 300).sum(), (e > 1000).sum(), np.median(e)))
