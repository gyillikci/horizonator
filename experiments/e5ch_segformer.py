#!/usr/bin/env python3
"""E5ch: SegFormer (ADE20K) as the skyline front end, against eWaSR.

Three boundaries per frame, from the same photo:
  ewasr    eWaSR's sky/land boundary (the current front end);
  seg      SegFormer's sky boundary, same rule (first non-sky row below
           the top sky run), via segformer_bridge;
  seg_f    the faint-ridge step detector of E5cf run on SegFormer's map.

Each boundary is scored at the TRUTH the way E5cf scores it (DEM rms
after a wide heading/offset fit, and the far-column residual). The 4
frames of E5cf are the default; `all` runs the 44.

    python3 e5ch_segformer.py eval [TAG ... | all]
    python3 e5ch_segformer.py solve [TAG ... | all] [--jobs 4]
    python3 e5ch_segformer.py report [TAG ... | all]
"""

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from assets import celnav, need  # noqa: E402
import e5cf_faint_ridge as M  # noqa: E402

OUT = os.path.join(HERE, 'out', 'segformer')
FRAMES = os.path.join(HERE, 'data', 'campaign_frames.json')
FOUR = ['09_PF_new_bodrum9', '38_XKYU3498', '31_EWAC7374', '36_PQBC6867']
BAD, GOOD = 500.0, 300.0


def frames(tags):
    fs = json.load(open(FRAMES))['frames']
    if not tags:
        tags = FOUR
    if tags == ['all']:
        return fs
    return [f for f in fs if f['tag'] in tags]


def evaluate(a):
    import extract
    import skyfix as F
    import skyline as S
    from ewasr_bridge import EWasr
    from segformer_bridge import SegFormer
    os.makedirs(OUT, exist_ok=True)
    ew = EWasr(need('eWaSR'), need('ewasr_resnet18.pth'))
    sf = SegFormer()
    dem = os.path.expanduser('~/.horizonator/DEMs_COP30')
    res = {}
    for fr in frames(a.tags):
        img = extract.load_image(celnav(fr['path']))
        cls_e = ew.predict(img)
        cls_s = sf.predict(img)
        cm = S.CMarcher(dem, (fr['lat'] - .6, fr['lat'] + .6),
                        (fr['lon'] - .8, fr['lon'] + .8), d_min=1000.0)
        el, r = cm.skyline(fr['lat'], fr['lon'], fr['z'], F.AZ)
        el = np.nan_to_num(el + np.where(np.isfinite(r) & (r > 0),
                                         9.0 / np.maximum(r, 1000.0), 0))
        out = {}
        e_e = M.ewasr_boundary(cls_e)
        e_s = M.ewasr_boundary(cls_s)
        f_s, conf_s, _, info = M.detect(img, cls_s)
        bounds = {'ewasr': (np.nan_to_num(e_e, nan=0), np.isfinite(e_e) * 1.0),
                  'seg': (np.nan_to_num(e_s, nan=0), np.isfinite(e_s) * 1.0),
                  'seg_f': (np.nan_to_num(f_s, nan=0), conf_s)}
        for k, (rows, conf) in bounds.items():
            out[k] = M.truth_fit(F, S, fr, img, rows, conf, (el, r))
            np.savez(os.path.join(OUT, '%s_%s.npz' % (fr['tag'], k)),
                     rows=rows, conf=conf)
        out['sky_frac'] = dict(ewasr=float((cls_e == 2).mean()),
                               seg=float((cls_s == 2).mean()))
        res[fr['tag']] = out
        fmt = lambda q: ('%5.1f (far %+4.0f)' % (
            q['rms'], q['far_res'] or 0)) if q else '   -'
        print('%-26s ewasr %s | seg %s | seg+faint %s' % (
            fr['tag'], fmt(out['ewasr']), fmt(out['seg']),
            fmt(out['seg_f'])), flush=True)
    f = os.path.join(OUT, 'eval.json')
    allr = json.load(open(f)) if os.path.exists(f) else {}
    allr.update(res)
    json.dump(allr, open(f, 'w'), indent=1)


def solve_one(fr, k, arm, flags):
    out = os.path.join(OUT, '%s_%s_%s' % (fr['tag'], k, arm))
    if os.path.exists(out + '.json'):
        return out
    cmd = [sys.executable, os.path.join(HERE, 'skyfix.py'),
           celnav(fr['path']),
           '--center', '%.6f,%.6f' % (fr['lat'], fr['lon']),
           '--box', str(fr['box']), '--z', str(fr['z']),
           '--fov', str(fr['fov']), '--heading', str(fr['heading']),
           '--boundary-npy', os.path.join(OUT, '%s_%s.npz' % (fr['tag'], k)),
           '--out', out] + flags.split()
    if arm == 'soft5':
        cmd += ['--compass-sigma', '5']
    elif arm == 'ncc':
        # the literature matcher (E5bq): 1 - Pearson, hard +-6 window
        cmd += ['--cost', 'ncc']
    with open(out + '.log', 'w') as lg:
        subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=HERE,
                       env=dict(os.environ, OMP_NUM_THREADS='1'))
    return out


def solve(a):
    flags = json.load(open(FRAMES))['flags'].replace(
        '~', os.path.expanduser('~'))
    arms = ('hard6', 'soft5', 'ncc') if a.ncc else ('hard6', 'soft5')
    jobs = [(fr, k, arm) for arm in arms
            for k in ('seg', 'seg_f') for fr in frames(a.tags)
            if os.path.exists(os.path.join(OUT, '%s_%s.npz' % (fr['tag'], k)))]
    with ThreadPoolExecutor(a.jobs) as ex:
        for n, out in enumerate(ex.map(lambda j: solve_one(*j, flags), jobs),
                                1):
            print('%3d/%d %s %s' % (n, len(jobs), os.path.basename(out),
                                    '' if os.path.exists(out + '.json')
                                    else 'NO RESULT'), flush=True)


def load_one(f):
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    if j.get('dn_m') is None:
        return None
    return dict(err=math.hypot(j['dn_m'], j['de_m']), ok=j['status'] == 'ok',
                hoff=j['photos'][0]['heading_offset_deg'])


def report(a):
    arms = {'ewasr_hard6': ('baseline', '%s_huber.json'),
            'ewasr_soft5': ('compass', '%s_soft5.json'),
            'seg_hard6': ('segformer', '%s_seg_hard6.json'),
            'seg_soft5': ('segformer', '%s_seg_soft5.json'),
            'segf_hard6': ('segformer', '%s_seg_f_hard6.json'),
            'segf_soft5': ('segformer', '%s_seg_f_soft5.json')}
    rows = []
    for fr in frames(a.tags):
        r = dict(tag=fr['tag'])
        for k, (d, p) in arms.items():
            r[k] = load_one(os.path.join(HERE, 'out', d, p % fr['tag']))
        if all(r[k] for k in arms):
            rows.append(r)
    print('frames with every arm: %d' % len(rows))
    print('%-12s %7s %5s %5s | %4s %4s %5s' % (
        'arm', 'median', '<300', '<500', 'acc', 'good', 'wrong'))
    summ = {}
    for k in arms:
        e = np.array([r[k]['err'] for r in rows])
        gate = 10.0 if 'soft5' in k else 1e9
        acc = [r for r in rows if r[k]['ok'] and abs(r[k]['hoff']) <= gate]
        summ[k] = dict(median=float(np.median(e)),
                       within300=int((e < GOOD).sum()),
                       within500=int((e < BAD).sum()), accepted=len(acc),
                       good=sum(r[k]['err'] < GOOD for r in acc),
                       wrong=sum(r[k]['err'] >= BAD for r in acc),
                       wrong_tags=[r['tag'] for r in acc if r[k]['err'] >= BAD])
        print('%-12s %6.0fm %5d %5d | %4d %4d %5d  %s' % (
            k, summ[k]['median'], summ[k]['within300'],
            summ[k]['within500'], summ[k]['accepted'], summ[k]['good'],
            summ[k]['wrong'], ', '.join(t[3:] for t in summ[k]['wrong_tags'])))
    print('\nper frame (err m, A = accepted):')
    for r in rows:
        print('  %-24s ' % r['tag'][:24] + ' | '.join(
            '%5.0f%s' % (r[k]['err'], 'A' if r[k]['ok'] else '-')
            for k in arms))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(
        OUT, 'report_%d.json' % len(rows)), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    for n in ('eval', 'solve', 'report'):
        p = sp.add_parser(n)
        p.add_argument('tags', nargs='*')
        p.add_argument('--jobs', type=int, default=4)
        p.add_argument('--ncc', action='store_true',
                       help='solve: add the ncc matcher arm')
    a = ap.parse_args()
    dict(eval=evaluate, solve=solve, report=report)[a.cmd](a)
