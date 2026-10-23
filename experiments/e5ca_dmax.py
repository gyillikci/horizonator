#!/usr/bin/env python3
"""E5ca: cut the DEM skyline at a fixed range (haze), 44 frames.

E5bz found the photo skyline sitting ever lower than the DEM as the
DEM crest recedes (-25 to -37 mrad beyond 15 km): hazy far ridges are
in the DEM but the segmenter calls them sky. If the matcher keeps them,
it can only explain the photo by moving the camera. skyfix --dmax D
ray-casts the DEM only to D metres. Same frames and flags as the
stored E5bq huber arm (hard +-6 deg window), plus --dmax.

    python3 e5ca_dmax.py run [--dmax 15000] [--jobs 4]
    python3 e5ca_dmax.py report [--dmax 15000]
"""

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from assets import celnav

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'dmax')
BASE = os.path.join(HERE, 'out', 'baseline')
HAZE = os.path.join(HERE, 'out', 'compass', 'haze.json')
FRAMES = os.path.join(HERE, 'data', 'campaign_frames.json')
BAD, GOOD = 500.0, 300.0


def frames():
    return json.load(open(FRAMES))['frames']


def solve(fr, dmax, flags):
    out = os.path.join(OUT, '%s_d%d' % (fr['tag'], dmax // 1000))
    if os.path.exists(out + '.json'):
        return out
    cmd = [sys.executable, os.path.join(HERE, 'skyfix.py'),
           celnav(fr['path']),
           '--center', '%.6f,%.6f' % (fr['lat'], fr['lon']),
           '--box', str(fr['box']), '--z', str(fr['z']),
           '--fov', str(fr['fov']), '--heading', str(fr['heading']),
           '--dmax', str(dmax), '--out', out] + flags.split()
    with open(out + '.log', 'w') as lg:
        subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=HERE,
                       env=dict(os.environ, OMP_NUM_THREADS='1'))
    return out


def run(a):
    os.makedirs(OUT, exist_ok=True)
    flags = json.load(open(FRAMES))['flags'].replace(
        '~', os.path.expanduser('~'))
    fs = frames()
    with ThreadPoolExecutor(a.jobs) as ex:
        for k, out in enumerate(ex.map(lambda f: solve(f, a.dmax, flags),
                                       fs), 1):
            print('%3d/%d %s %s' % (k, len(fs), os.path.basename(out),
                                    '' if os.path.exists(out + '.json')
                                    else 'NO RESULT'), flush=True)


def load_one(f):
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    return dict(err=math.hypot(j['dn_m'], j['de_m']), dn=j['dn_m'],
                de=j['de_m'], ok=j['status'] == 'ok',
                hoff=j['photos'][0]['heading_offset_deg'],
                rms=j['rms_mrad'])


def summary(rows, k):
    e = np.array([r[k]['err'] for r in rows])
    acc = [r for r in rows if r[k]['ok']]
    return dict(median=float(np.median(e)), within300=int((e < GOOD).sum()),
                within500=int((e < BAD).sum()), accepted=len(acc),
                good=sum(r[k]['err'] < GOOD for r in acc),
                wrong=sum(r[k]['err'] >= BAD for r in acc),
                wrong_tags=[r['tag'] for r in acc if r[k]['err'] >= BAD])


def report(a):
    haze = json.load(open(HAZE)) if os.path.exists(HAZE) else {}
    cut = 'd%d' % (a.dmax // 1000)
    rows = []
    for fr in frames():
        r = dict(tag=fr['tag'],
                 far=haze.get(fr['tag'], {}).get('far_frac'),
                 hard6=load_one(os.path.join(BASE, fr['tag'] + '_huber.json')))
        r[cut] = load_one(os.path.join(OUT, '%s_%s.json' % (fr['tag'], cut)))
        if r['hard6'] and r[cut]:
            rows.append(r)
    print('frames with both: %d\n' % len(rows))
    print('%-28s %5s | %-22s | %-22s' % ('frame', 'far%', '40 km (stored)',
                                         '%s km' % cut[1:]))
    for r in rows:
        print('%-28s %5s | %s | %s' % (
            r['tag'][:28], '%3.0f' % (100 * r['far']) if r['far'] is not None
            else '  -',
            *['%6.0fm %s rms %4.1f' % (r[k]['err'], 'ACC' if r[k]['ok']
                                        else ' - ', r[k]['rms'])
              for k in ('hard6', cut)]))
    out = {}
    for name, sub in (('all', rows),
                      ('far>15%', [r for r in rows if (r['far'] or 0) > .15]),
                      ('far<=15%', [r for r in rows
                                    if (r['far'] or 0) <= .15])):
        if not sub:
            continue
        print('\n%s (%d frames)' % (name, len(sub)))
        for k in ('hard6', cut):
            s = summary(sub, k)
            out.setdefault(name, {})[k] = s
            print('  %-6s median %5.0f m  <300 %2d  <500 %2d | accepted %2d '
                  'good %d wrong %d %s' % (
                      k, s['median'], s['within300'], s['within500'],
                      s['accepted'], s['good'], s['wrong'],
                      ', '.join(t[3:] for t in s['wrong_tags'])))
        better = sum(r[cut]['err'] < r['hard6']['err'] - 50 for r in sub)
        worse = sum(r[cut]['err'] > r['hard6']['err'] + 50 for r in sub)
        print('  cut closer by >50 m on %d, farther on %d' % (better, worse))
    json.dump(dict(summary=out, rows=rows),
              open(os.path.join(OUT, 'report_%s.json' % cut), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    for n in ('run', 'report'):
        p = sp.add_parser(n)
        p.add_argument('--dmax', type=int, default=15000)
        p.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    dict(run=run, report=report)[a.cmd](a)
