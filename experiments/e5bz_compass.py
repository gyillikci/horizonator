#!/usr/bin/env python3
"""E5bz: hard compass window against a soft compass prior, 44 frames.

The default solve searches the heading offset psi inside a HARD window
(+-6 deg) around the compass and treats every psi in it as equally
likely. When the compass is off by more than the window (Bodrum 9:
+12.8 deg at the true position) the best psi sits on the bound and the
position absorbs the rest - a confident false accept (E5by video).

The soft prior instead widens the search to +-4 sigma and charges each
psi 0.5 (psi/sigma)^2 in the units of the photo's own likelihood
(skyfix --compass-sigma). Arms, same frames, same flags otherwise:

  hard6   the stored E5bq huber arm (out/baseline), +-6 deg window
  soft3 / soft5 / soft8   --compass-sigma 3 / 5 / 8

Also reported per frame: psi_truth, the heading offset that best fits
the DEM at the TRUE position over +-30 deg (how wrong the compass
really was), so each arm's failures can be split into compass-caused
and not.

The haze question (is far terrain in the DEM skyline that the photo
does not show?) is answered at the truth, per frame, by the residual
of the columns whose DEM crest lies in each range band:

    python3 e5bz_compass.py run [--jobs 4]
    python3 e5bz_compass.py report
    python3 e5bz_compass.py haze
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
OUT = os.path.join(HERE, 'out', 'compass')
BASE = os.path.join(HERE, 'out', 'baseline')
FRAMES = os.path.join(HERE, 'data', 'campaign_frames.json')
SIGMAS = (3, 5, 8)
BAD, GOOD = 500.0, 300.0


def frames():
    return json.load(open(FRAMES))['frames']


def solve(fr, sig, flags):
    out = os.path.join(OUT, '%s_soft%d' % (fr['tag'], sig))
    if os.path.exists(out + '.json'):
        return out
    cmd = [sys.executable, os.path.join(HERE, 'skyfix.py'),
           celnav(fr['path']),
           '--center', '%.6f,%.6f' % (fr['lat'], fr['lon']),
           '--box', str(fr['box']), '--z', str(fr['z']),
           '--fov', str(fr['fov']), '--heading', str(fr['heading']),
           '--compass-sigma', str(sig), '--out', out] + flags.split()
    with open(out + '.log', 'w') as lg:
        subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=HERE,
                       env=dict(os.environ, OMP_NUM_THREADS='1'))
    return out


def psi_truth():
    """Heading offset that best explains the DEM at the true position."""
    f = os.path.join(OUT, 'psi_truth.json')
    if os.path.exists(f):
        return json.load(open(f))
    sys.path.insert(0, HERE)
    import skyfix as F
    import skyline as S
    import extract
    F.EXTRACTOR = 'ewasr'
    dem = os.path.expanduser('~/.horizonator/DEMs_COP30')
    res = {}
    for fr in frames():
        img = extract.load_image(celnav(fr['path']))
        cm = S.CMarcher(dem, (fr['lat'] - .6, fr['lat'] + .6),
                        (fr['lon'] - .8, fr['lon'] + .8), d_min=1000.0)
        el, r = cm.skyline(fr['lat'], fr['lon'], fr['z'], F.AZ)
        el = el + np.where(np.isfinite(r) & (r > 0),
                           9.0 / np.maximum(r, 1000.0), 0.0)
        el_obs, w, _ = F.observation(img, fr['fov'], fr['heading'], 0.0)
        if w.sum() <= 0:
            continue
        betas = np.linspace(-0.0101, 0.0101, 21)
        c, s, b = F.photo_cost(el_obs, w, np.nan_to_num(el),
                               np.arange(-300, 301, 2), betas)
        res[fr['tag']] = dict(psi=s * 0.1, beta_mrad=b * 1e3,
                              rms_mrad=math.sqrt(2 * c) * 1e3)
        print(fr['tag'], res[fr['tag']], flush=True)
    json.dump(res, open(f, 'w'), indent=1)
    return res


BANDS = ((0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 41))


def haze(a):
    """Residual photo - DEM at the truth, by DEM crest range. Far
    terrain the photo cannot see shows up as a band whose photo sits
    well BELOW the DEM (negative residual), and as a fit that improves
    when the ray-cast is cut short."""
    sys.path.insert(0, HERE)
    import skyfix as F
    import skyline as S
    import extract
    F.EXTRACTOR = 'ewasr'
    dem = os.path.expanduser('~/.horizonator/DEMs_COP30')
    betas = np.linspace(-0.0101, 0.0101, 21)
    out = {}
    for fr in frames():
        img = extract.load_image(celnav(fr['path']))
        el_obs, w, _ = F.observation(img, fr['fov'], fr['heading'], 0.0)
        if w.sum() <= 0:
            continue
        cm = S.CMarcher(dem, (fr['lat'] - .6, fr['lat'] + .6),
                        (fr['lon'] - .8, fr['lon'] + .8), d_min=1000.0)
        fits = {}
        for dmax in (15e3, 20e3, 40e3):
            cm.d_max = dmax
            el, r = cm.skyline(fr['lat'], fr['lon'], fr['z'], F.AZ)
            el = np.nan_to_num(el + np.where(
                np.isfinite(r) & (r > 0), 9.0 / np.maximum(r, 1000.0), 0))
            c, s, b = F.photo_cost(el_obs, w, el, np.arange(-300, 301, 2),
                                   betas)
            fits[int(dmax / 1e3)] = dict(rms=math.sqrt(2 * c) * 1e3,
                                         psi=s * 0.1, el=el, r=r, s=s, b=b)
        f40 = fits[40]
        eo = np.roll(el_obs, f40['s'])
        m = np.roll(w, f40['s']) > 0
        res = (eo[m] + f40['b'] - f40['el'][m]) * 1e3
        rr = np.nan_to_num(f40['r'][m]) / 1e3
        bands = []
        for lo, hi in BANDS:
            k = (rr >= lo) & (rr < hi)
            bands.append(dict(band=[lo, hi], frac=float(k.mean()),
                              res=float(np.median(res[k])) if k.any()
                              else None))
        out[fr['tag']] = dict(bands=bands, far_frac=float((rr >= 15).mean()),
                              rms={k: v['rms'] for k, v in fits.items()},
                              psi={k: v['psi'] for k, v in fits.items()})
        far = [x for x in bands if x['band'][0] >= 15 and x['frac'] > 0]
        print('%-28s far(>=15 km) %3.0f%%  res %s | rms 40/20/15 km '
              '%.1f/%.1f/%.1f mrad' % (
                  fr['tag'], 100 * out[fr['tag']]['far_frac'],
                  ' '.join('%d-%d:%+.0f' % (x['band'][0], x['band'][1],
                                            x['res']) for x in far) or '-',
                  fits[40]['rms'], fits[20]['rms'], fits[15]['rms']),
              flush=True)
    json.dump(out, open(os.path.join(OUT, 'haze.json'), 'w'), indent=1)
    haze_figure(out)
    v = [x for x in out.values() if x['far_frac'] > 0.05]
    print('\nframes with >5%% of columns crested beyond 15 km: %d of %d'
          % (len(v), len(out)))
    for lo, hi in BANDS:
        rs = [b['res'] for x in out.values() for b in x['bands']
              if b['band'] == [lo, hi] and b['res'] is not None
              and b['frac'] > 0.02]
        if rs:
            print('  crest %2d-%2d km: %2d frames, median of per-frame '
                  'median residual (photo - DEM) %+5.1f mrad, negative '
                  'on %d' % (lo, hi, len(rs), np.median(rs),
                             sum(r < 0 for r in rs)))
    d20 = [x['rms']['40'] - x['rms']['20'] if '40' in x['rms']
           else x['rms'][40] - x['rms'][20] for x in v]
    if d20:
        print('  cutting the ray-cast at 20 km changes the truth rms by '
              'a median %+.2f mrad on those frames (better on %d of %d)'
              % (-np.median(d20), sum(d > 0 for d in d20), len(d20)))


def haze_figure(out=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if out is None:
        out = json.load(open(os.path.join(OUT, 'haze.json')))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, (lo, hi) in enumerate(BANDS):
        rs = [b['res'] for x in out.values() for b in x['bands']
              if b['band'] == [lo, hi] and b['res'] is not None
              and b['frac'] > 0.02]
        xs = k + np.linspace(-0.25, 0.25, len(rs)) if len(rs) > 1 \
            else [k] * len(rs)
        ax.scatter(xs, np.clip(rs, -80, 40), s=14, color='#3a6ea5',
                   alpha=.7)
        if rs:
            ax.plot([k - .35, k + .35], [np.median(rs)] * 2, color='#c0392b',
                    lw=2.5)
    ax.axhline(0, color='0.4', lw=.8)
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels(['%d-%d' % (lo, min(hi, 40)) for lo, hi in BANDS])
    ax.set_xlabel('DEM crest range at the true position, km')
    ax.set_ylabel('photo - DEM, mrad (clipped to [-80, 40])')
    ax.set_title('Far terrain sits in the DEM skyline, not in the photo '
                 '(44 frames, red = median)', fontsize=10)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'haze.png'), dpi=110)


def run(a):
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(FRAMES))
    flags = meta['flags'].replace('~', os.path.expanduser('~'))
    jobs = [(fr, s) for s in SIGMAS for fr in frames()]
    with ThreadPoolExecutor(a.jobs) as ex:
        for k, out in enumerate(ex.map(lambda j: solve(j[0], j[1], flags),
                                       jobs), 1):
            ok = os.path.exists(out + '.json')
            print('%3d/%d %s %s' % (k, len(jobs), os.path.basename(out),
                                    '' if ok else 'NO RESULT'), flush=True)
    psi_truth()


def load_one(f):
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    return dict(err=math.hypot(j['dn_m'], j['de_m']), dn=j['dn_m'],
                de=j['de_m'], ok=j['status'] == 'ok',
                margin=j['basin_margin'],
                hoff=j['photos'][0]['heading_offset_deg'])


def report(a):
    pt = psi_truth()
    arms = ['hard6'] + ['soft%d' % s for s in SIGMAS]
    rows = []
    for fr in frames():
        r = dict(tag=fr['tag'], psi=pt.get(fr['tag'], {}).get('psi'))
        r['hard6'] = load_one(os.path.join(BASE, fr['tag'] + '_huber.json'))
        for s in SIGMAS:
            r['soft%d' % s] = load_one(
                os.path.join(OUT, '%s_soft%d.json' % (fr['tag'], s)))
        if all(r[k] for k in arms):
            rows.append(r)
    n = len(rows)
    print('frames with every arm: %d of %d\n' % (n, len(frames())))
    print('%-26s %6s | ' % ('frame', 'psi*')
          + ' | '.join('%-18s' % k for k in arms))
    for r in rows:
        cells = []
        for k in arms:
            x = r[k]
            cells.append('%6.0fm %s %+5.1f°  ' % (
                x['err'], 'ACC' if x['ok'] else ' - ', x['hoff']))
        print('%-26s %+6.1f | ' % (r['tag'][:26], r['psi'] or float('nan'))
              + ' | '.join(cells))
    summ = {}
    print('\n%-7s %7s %5s %5s | %4s %4s %5s | %s' % (
        'arm', 'median', '<300', '<500', 'acc', 'good', 'wrong',
        'wrong accepts (err m, psi* deg)'))
    for k in arms:
        e = np.array([r[k]['err'] for r in rows])
        acc = [r for r in rows if r[k]['ok']]
        wrong = [r for r in acc if r[k]['err'] >= BAD]
        summ[k] = dict(median=float(np.median(e)),
                       within300=int((e < GOOD).sum()),
                       within500=int((e < BAD).sum()),
                       accepted=len(acc),
                       good=sum(r[k]['err'] < GOOD for r in acc),
                       wrong=len(wrong),
                       wrong_tags=[r['tag'] for r in wrong])
        print('%-7s %6.0fm %5d %5d | %4d %4d %5d | %s' % (
            k, summ[k]['median'], summ[k]['within300'],
            summ[k]['within500'], summ[k]['accepted'], summ[k]['good'],
            summ[k]['wrong'],
            ', '.join('%s (%.0f, %+.1f)' % (r['tag'][3:], r[k]['err'],
                                            r['psi'] or float('nan'))
                      for r in wrong)))
    # does the compass error explain the failures?
    big = [r for r in rows if r['psi'] is not None and abs(r['psi']) > 6]
    print('\nframes whose compass is off by > 6 deg at the truth: %d '
          '(%s)' % (len(big), ', '.join('%s %+.1f' % (r['tag'][3:],
                                                      r['psi'])
                                        for r in big)))
    json.dump(dict(summary=summ, rows=rows),
              open(os.path.join(OUT, 'report.json'), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('--jobs', type=int, default=4)
    sp.add_parser('report')
    sp.add_parser('haze')
    a = ap.parse_args()
    dict(run=run, report=report, haze=haze)[a.cmd](a)
