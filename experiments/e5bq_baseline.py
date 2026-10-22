#!/usr/bin/env python3
"""E5bq: this instrument against the literature's matcher, same frames.

A reviewer's first question is "against what?". CrossLocate (WACV 2022),
the open learned baseline, cannot be run here: its trained models and
rendered databases are only on a host this environment cannot resolve,
and they cover the Alps. The baseline that CAN be reproduced exactly is
the one nearly every horizon paper uses for a levelled camera: correlate
the 1D elevation-vs-azimuth horizon with DEM horizons over a position
grid, heading by circular shift (Cozman & Krotkov 2000; Chiodini et al.
2017; Grelsson et al. 2020). skyfix --cost ncc is that matcher, inside
the identical pipeline - same extractor, DEM, box, grid, refinement,
heading window - so the only difference is the misfit:

  huber  robust residual in absolute elevation, offset beta bounded
         to +-10 mrad (this instrument);
  ncc    1 - weighted Pearson correlation: offset- and scale-free.

Frames: the 44 field frames of E5bm, data/campaign_frames.json, truth =
EXIF GPS. Two comparisons, because accuracy and integrity are different
questions:

  * accuracy: error of every solve, no gates;
  * integrity at equal availability: the literature has no acceptance
    gate, so rank each method's solves by its own confidence (basin
    margin of its own cost surface) and compare the k most confident,
    for k = the number this instrument's full gate set accepts. How
    many of those k are wrong by >= 500 m?

    python3 e5bq_baseline.py run [--jobs 4]
    python3 e5bq_baseline.py report
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
OUT = os.path.join(HERE, 'out', 'baseline')
FRAMES = os.path.join(HERE, 'data', 'campaign_frames.json')
BAD = 500.0      # a wrong fix, metres (E5bm's false-accept line)
GOOD = 300.0


def frames():
    return json.load(open(FRAMES))['frames']


def solve(fr, cost, flags):
    out = os.path.join(OUT, '%s_%s' % (fr['tag'], cost))
    if os.path.exists(out + '.json'):
        return out
    cmd = [sys.executable, os.path.join(HERE, 'skyfix.py'),
           celnav(fr['path']),
           '--center', '%.6f,%.6f' % (fr['lat'], fr['lon']),
           '--box', str(fr['box']), '--z', str(fr['z']),
           '--fov', str(fr['fov']), '--heading', str(fr['heading']),
           '--cost', cost, '--out', out] + flags.split()
    with open(out + '.log', 'w') as lg:
        subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=HERE,
                       env=dict(os.environ, OMP_NUM_THREADS='1'))
    return out


def run(a):
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(FRAMES))
    flags = meta['flags'].replace('~', os.path.expanduser('~'))
    jobs = [(fr, c) for fr in frames() for c in ('huber', 'ncc')]
    with ThreadPoolExecutor(a.jobs) as ex:
        for k, out in enumerate(ex.map(lambda j: solve(j[0], j[1], flags),
                                       jobs), 1):
            ok = os.path.exists(out + '.json')
            print('%3d/%d %s %s' % (k, len(jobs), os.path.basename(out),
                                    '' if ok else 'NO RESULT'), flush=True)


def load():
    rows = []
    for fr in frames():
        r = dict(tag=fr['tag'], stored=fr['e5bm_w6'])
        for c in ('huber', 'ncc'):
            f = os.path.join(OUT, '%s_%s.json' % (fr['tag'], c))
            if not os.path.exists(f):
                continue
            j = json.load(open(f))
            r[c] = dict(err=math.hypot(j['dn_m'], j['de_m']),
                        dn=j['dn_m'], de=j['de_m'], ok=j['status'] == 'ok',
                        margin=j['basin_margin'],
                        hoff=j['photos'][0]['heading_offset_deg'],
                        ncc=j['photos'][0].get('ncc'))
        rows.append(r)
    return [r for r in rows if 'huber' in r and 'ncc' in r]


def report(a):
    rows = load()
    n = len(rows)
    print('frames with both solves: %d of %d\n' % (n, len(frames())))

    # reproduction check against the stored E5bm +-6 run
    same = sum(1 for r in rows
               if math.hypot(r['huber']['dn'] - r['stored']['dn_m'],
                             r['huber']['de'] - r['stored']['de_m']) < 1
               and r['huber']['ok'] == (r['stored']['status'] == 'ok'))
    print('huber arm reproduces the stored E5bm run (same fix, same '
          'verdict) on %d of %d frames' % (same, n))

    print('\n%-36s %8s %5s %6s | %8s %6s %6s' % (
        'frame', 'huber', 'acc', 'margin', 'ncc', 'margin', 'r'))
    for r in rows:
        h, c = r['huber'], r['ncc']
        print('%-36s %7.0fm %5s %6.2f | %7.0fm %6.2f %6.3f' % (
            r['tag'][:36], h['err'], 'yes' if h['ok'] else 'no',
            h['margin'], c['err'], c['margin'], c['ncc'] or float('nan')))

    eh = np.array([r['huber']['err'] for r in rows])
    ec = np.array([r['ncc']['err'] for r in rows])
    print('\naccuracy, every solve, no gates:')
    for name, e in (('huber', eh), ('ncc  ', ec)):
        print('  %s median %5.0f m | within %d m: %2d | within %d m: %2d '
              '| >= %d m: %2d' % (name, np.median(e), GOOD,
                                  (e < GOOD).sum(), BAD, (e < BAD).sum(),
                                  BAD, (e >= BAD).sum()))
    better = int((eh < ec).sum())
    print('  huber closer than ncc on %d of %d frames' % (better, n))

    acc = [r for r in rows if r['huber']['ok']]
    k = len(acc)
    print('\nintegrity at equal availability (k = %d, what the full gate '
          'set accepts):' % k)
    wrong_gate = sum(r['huber']['err'] >= BAD for r in acc)
    right_gate = sum(r['huber']['err'] < GOOD for r in acc)
    print('  huber, full gate set     : %2d accepted, %d within %d m, '
          '%d wrong >= %d m' % (k, right_gate, GOOD, wrong_gate, BAD))
    for name in ('huber', 'ncc'):
        top = sorted(rows, key=lambda r: -r[name]['margin'])[:k]
        w = sum(r[name]['err'] >= BAD for r in top)
        g = sum(r[name]['err'] < GOOD for r in top)
        print('  %s, top-%d by own margin: %d within %d m, %d wrong >= %d '
              'm' % (name.ljust(5), k, g, GOOD, w, BAD))
    json.dump(rows, open(os.path.join(OUT, 'report.json'), 'w'), indent=1)
    draw(rows, eh, ec)


def draw(rows, eh, ec):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    acc = np.array([r['huber']['ok'] for r in rows])
    ax.scatter(ec[~acc], eh[~acc], s=34, facecolors='none',
               edgecolors='#8A8F98', label='refused by the gates')
    ax.scatter(ec[acc], eh[acc], s=40, color='#1E7F5C',
               label='accepted by the gates')
    lim = [20, max(eh.max(), ec.max()) * 1.4]
    ax.plot(lim, lim, color='#1A2732', lw=1, alpha=.6)
    for v in (GOOD, BAD):
        ax.axhline(v, color='#C0392B', lw=.6, ls=':', alpha=.6)
        ax.axvline(v, color='#C0392B', lw=.6, ls=':', alpha=.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('position error, literature NCC matcher (m)')
    ax.set_ylabel('position error, this instrument (Huber) (m)')
    ax.set_title('Same %d frames, extractor, DEM and search; only the misfit '
                 'differs' % len(rows), fontsize=10.5)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'baseline.png'), dpi=110)
    print('wrote', os.path.join(OUT, 'baseline.png'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['run', 'report'])
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    run(a) if a.cmd == 'run' else report(a)
