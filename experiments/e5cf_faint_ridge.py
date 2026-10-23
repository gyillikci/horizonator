#!/usr/bin/env python3
"""E5cf: a faint-ridge detector on top of eWaSR.

E5cc showed that the far ridges missing from the photo skyline are in
the image: contrast 0.085-0.16 against the sky, which eWaSR calls sky.
The sky, though, is smooth in position and narrow in colour (3 PCs,
study 3.10). So:

  1. eWaSR's boundary e(x): the first non-sky row below the top sky run.
  2. Sky model: each linear-RGB channel as a low-order polynomial in
     image position (u, v; cubic in v for the horizon brightening),
     fitted robustly (clouds rejected) on eWaSR sky at least GAP px
     above e(x), where no faint ridge can hide.
  3. Above e(x), a pixel is terrain if its colour departs from the
     model by > K in the sky's own residual Mahalanobis metric (3x3)
     AND it is darker than the model. Clouds and glare are brighter.
  4. The boundary moves up from e(x) while the run of terrain pixels
     continues unbroken (gaps <= 2 px), at most SEARCH px; columns are
     then median-filtered so isolated runs cannot spike.

Scored at the TRUTH against the DEM (wide heading and offset fit), the
same way E5bz scores the residual: rms, and the photo-minus-DEM
residual of columns whose DEM crest is beyond 15 km.

    python3 e5cf_faint_ridge.py eval [TAG ...]   (default: the 4 frames)
    python3 e5cf_faint_ridge.py run  [TAG ...] [--jobs 4]
    python3 e5cf_faint_ridge.py report [TAG ...]
"""

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import median_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from assets import celnav, need  # noqa: E402

OUT = os.path.join(HERE, 'out', 'faint')
FRAMES = os.path.join(HERE, 'data', 'campaign_frames.json')
FOUR = ['09_PF_new_bodrum9', '38_XKYU3498', '31_EWAC7374', '36_PQBC6867']
GAP, SEARCH, BAND = 45, 80, 220      # px
K = 4.0
STEP_H, STEP_W, K_STEP = 5, 7, 6.0
CLOUD = 1.08         # above-step luminance over the column's sky median
BAD, GOOD = 500.0, 300.0


def frames(tags=None):
    fs = json.load(open(FRAMES))['frames']
    tags = tags or FOUR
    return [f for f in fs if tags == ['all'] or f['tag'] in tags]


def lin(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def ewasr_boundary(cls):
    """First non-sky row below the top sky run, per column (NaN: none)."""
    H, W = cls.shape
    e = np.full(W, np.nan)
    for x in range(W):
        sky = cls[:, x] == 2
        if not sky.any():
            continue
        s0 = int(np.argmax(sky))
        below = np.flatnonzero(~sky[s0:])
        if below.size and below[0] >= 3:
            e[x] = s0 + below[0]
    return e


def basis(u, v):
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, v ** 3,
                     u * v * v], -1)


def detect(img, cls):
    """Step detector (default): a faint ridge has a TOP EDGE.

    A smooth sky gradient, however strong near the horizon, has no
    step; terrain starts with one. Per column (colours boxed over STEP_W
    columns), D(y) = mean colour of rows y+1..y+STEP_H minus rows
    y-STEP_H..y-1. Its noise covariance is measured on eWaSR sky well
    above the boundary. The boundary moves up from e(x) to the TOPMOST
    row y within SEARCH px with a darker step of Mahalanobis > K_STEP
    whose band [y, e(x)) stays darker than the sky just above y."""
    H, W = cls.shape
    I = lin(img)
    e = ewasr_boundary(cls)
    ok = np.isfinite(e)
    from scipy.ndimage import uniform_filter1d
    Ib = uniform_filter1d(I, STEP_W, axis=1, mode='nearest')
    cs = np.cumsum(np.pad(Ib, ((1, 0), (0, 0), (0, 0))), axis=0)

    def mean_rows(a, b, x):            # rows [a, b) of column x
        return (cs[b, x] - cs[a, x]) / max(b - a, 1)
    wl = np.array([0.2126, 0.7152, 0.0722])
    # noise of D on sky well above the boundary
    Ds = []
    for x in np.flatnonzero(ok)[::5]:
        for y in range(int(max(e[x] - BAND, STEP_H)), int(e[x] - GAP), 4):
            if cls[y, x] == 2:
                Ds.append(mean_rows(y + 1, y + 1 + STEP_H, x)
                          - mean_rows(y - STEP_H, y, x))
    if len(Ds) < 50:
        return e, ok.astype(float), e, dict(moved=0)
    Ds = np.array(Ds)
    Ds = Ds[(np.abs(Ds - np.median(Ds, 0))
             < 4 * 1.4826 * np.median(np.abs(Ds - np.median(Ds, 0)), 0)
             + 1e-6).all(1)]
    Ci = np.linalg.inv(np.cov(Ds.T) + 1e-10 * np.eye(3))
    Lb = Ib @ wl
    new = e.copy()
    for x in np.flatnonzero(ok):
        y1 = int(e[x])
        # this column's own sky luminance, for the cloud test
        ref = Lb[int(max(e[x] - BAND, 0)):int(max(e[x] - GAP, 1)), x]
        ref = ref[cls[int(max(e[x] - BAND, 0)):int(max(e[x] - GAP, 1)),
                      x] == 2]
        # low quantile: clouds in the reference band are bright
        Lref = np.percentile(ref, 20) if ref.size else np.inf
        best = None
        for y in range(max(y1 - SEARCH, STEP_H + 1), y1 - 2):
            above = mean_rows(y - STEP_H, y, x)
            if above @ wl > CLOUD * Lref:
                continue                # a cloud's underside, not sky
            below = mean_rows(y + 1, min(y + 1 + STEP_H, y1), x)
            D = below - above
            if D @ wl >= 0 or D @ Ci @ D < K_STEP ** 2:
                continue
            band = mean_rows(y + 1, y1, x)
            if (band - above) @ wl > 0.5 * (D @ wl):
                continue                # band brightens back: not terrain
            # topmost qualifying step: place the edge where the step is
            # strongest within its run, not where it starts
            m_best, yb = -1.0, y
            for yy in range(y, min(y + 2 * STEP_H + 1, y1 - 2)):
                Dd = (mean_rows(yy + 1, min(yy + 1 + STEP_H, y1), x)
                      - mean_rows(yy - STEP_H, yy, x))
                if Dd @ wl >= 0:
                    break
                mm = Dd @ Ci @ Dd
                if mm > m_best:
                    m_best, yb = mm, yy
            best = yb
            break
        if best is not None and y1 - best >= 3:
            new[x] = best
    sm = median_filter(np.nan_to_num(new, nan=H), size=15, mode='nearest')
    new = np.where(ok, np.minimum(e, np.maximum(new, sm)), np.nan)
    moved = ok & (e - new >= 3)
    return new, ok.astype(float), e, dict(
        moved=int(moved.sum()),
        moved_px=float(np.median((e - new)[moved])) if moved.any() else 0.)


def detect_model(img, cls):
    """(rows, conf, e, info): sky-colour-model departure (first try;
    its extrapolated sky model mistakes horizon gradients for terrain)."""
    H, W = cls.shape
    I = lin(img)
    e = ewasr_boundary(cls)
    xs = np.arange(W)
    ok = np.isfinite(e)
    # sky samples: eWaSR sky, GAP..BAND px above its boundary
    ys_l, xs_l = [], []
    for x in xs[ok][::3]:
        y0, y1 = int(max(e[x] - BAND, 0)), int(e[x] - GAP)
        if y1 - y0 < 5:
            continue
        yy = np.arange(y0, y1, 3)
        yy = yy[cls[yy, x] == 2]
        ys_l.append(yy)
        xs_l.append(np.full(yy.size, x))
    if not ys_l:
        return e, ok.astype(float), e, dict(moved=0)
    ys, xx = np.concatenate(ys_l), np.concatenate(xs_l)
    u, v = xx / W, ys / H
    B = basis(u, v)
    Y = I[ys, xx]
    keep = np.ones(len(Y), bool)
    for _ in range(4):                       # robust: drop clouds
        coef = np.linalg.lstsq(B[keep], Y[keep], rcond=None)[0]
        res = Y - B @ coef
        mad = np.median(np.abs(res[keep]), axis=0) * 1.4826 + 1e-5
        keep = (np.abs(res) < 3 * mad).all(1)
    C = np.cov(res[keep].T) + 1e-9 * np.eye(3)
    Ci = np.linalg.inv(C)
    wl = np.array([0.2126, 0.7152, 0.0722])
    new = e.copy()
    for x in xs[ok]:
        y1 = int(e[x])
        y0 = max(y1 - SEARCH, 0)
        yy = np.arange(y0, y1)
        if yy.size == 0:
            continue
        uu, vv = np.full(yy.size, x / W), yy / H
        d = I[yy, x] - basis(uu, vv) @ coef
        m2 = np.einsum('ni,ij,nj->n', d, Ci, d)
        terr = (m2 > K * K) & (d @ wl < 0)
        # climb from e(x) while terrain continues (gaps <= 2 px)
        top, gap = y1, 0
        for y in range(y1 - 1, y0 - 1, -1):
            if terr[y - y0]:
                top, gap = y, 0
            else:
                gap += 1
                if gap > 2:
                    break
        if y1 - top >= 3:
            new[x] = top
    # no isolated spikes: a moved column must agree with its neighbours
    sm = median_filter(np.nan_to_num(new, nan=H), size=15, mode='nearest')
    new = np.where(ok, np.minimum(e, np.maximum(new, sm)), np.nan)
    conf = ok.astype(float)
    moved = ok & (e - new >= 3)
    return new, conf, e, dict(moved=int(moved.sum()),
                              moved_px=float(np.median((e - new)[moved]))
                              if moved.any() else 0.0)


def truth_fit(F, S, fr, img, rows, conf, cm_cache):
    """rms at the truth and the far-column residual for a boundary."""
    F.BOUNDARY_ROWS, F.BOUNDARY_CONF = rows, conf
    el_obs, w, d = F.observation(img, fr['fov'], fr['heading'], 0.0)
    F.BOUNDARY_ROWS = F.BOUNDARY_CONF = None
    if (w > 0).sum() < 50:
        return None
    el, r = cm_cache
    c, s, b = F.photo_cost(el_obs, w, el, np.arange(-300, 301, 2),
                           np.linspace(-0.03, 0.03, 61))
    eo, m = np.roll(el_obs, s), np.roll(w, s) > 0
    res = (eo[m] + b - el[m]) * 1e3
    far = np.nan_to_num(r[m]) > 15e3
    return dict(rms=math.sqrt(2 * c) * 1e3, psi=s * 0.1,
                far_frac=float(far.mean()),
                far_res=float(np.median(res[far])) if far.any() else None,
                cols=int(m.sum()))


def evaluate(a):
    import skyfix as F
    import skyline as S
    import extract
    from ewasr_bridge import EWasr
    os.makedirs(OUT, exist_ok=True)
    net = EWasr(need('eWaSR'), need('ewasr_resnet18.pth'))
    dem = os.path.expanduser('~/.horizonator/DEMs_COP30')
    res = {}
    for fr in frames(a.tags):
        img = extract.load_image(celnav(fr['path']))
        cls = net.predict(img)
        rows, conf, e, info = detect(img, cls)
        np.savez(os.path.join(OUT, fr['tag'] + '_faint.npz'), rows=rows,
                 conf=conf)
        cm = S.CMarcher(dem, (fr['lat'] - .6, fr['lat'] + .6),
                        (fr['lon'] - .8, fr['lon'] + .8), d_min=1000.0)
        el, r = cm.skyline(fr['lat'], fr['lon'], fr['z'], F.AZ)
        el = np.nan_to_num(el + np.where(np.isfinite(r) & (r > 0),
                                         9.0 / np.maximum(r, 1000.0), 0))
        F.EXTRACTOR = 'ewasr'
        out = dict(info=info)
        # skyfix's own default (eWaSR + the seam pre-check fallback)
        el_obs, w, _ = F.observation(img, fr['fov'], fr['heading'], 0.0)
        c, s, b = F.photo_cost(el_obs, w, el, np.arange(-300, 301, 2),
                               np.linspace(-0.03, 0.03, 61))
        out['default'] = dict(rms=math.sqrt(2 * c) * 1e3, psi=s * 0.1,
                              used=F.USED_EXTRACTOR)
        ok = np.isfinite(e)
        out['ewasr'] = truth_fit(F, S, fr, img, np.nan_to_num(e), ok * 1.0,
                                 (el, r))
        out['faint'] = truth_fit(F, S, fr, img, np.nan_to_num(rows),
                                 conf, (el, r))
        res[fr['tag']] = out
        fmt = lambda q: '%5.1f mrad (far %s)' % (
            q['rms'], '%+.0f' % q['far_res'] if q['far_res'] is not None
            else '-') if q else '   -'
        print('%-26s moved %4d cols (median %4.1f px) | rms at truth: '
              'default %5.1f (%s) | eWaSR raw %s | faint %s' % (
                  fr['tag'], info['moved'], info.get('moved_px', 0),
                  out['default']['rms'], out['default']['used'],
                  fmt(out['ewasr']), fmt(out['faint'])), flush=True)
        if fr['tag'] in FOUR:
            figure(fr, img, e, rows)
    f = os.path.join(OUT, 'eval.json')
    allr = json.load(open(f)) if os.path.exists(f) else {}
    allr.update(res)
    json.dump(allr, open(f, 'w'), indent=1)


def figure(fr, img, e, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(14, 4.2))
    lo, hi = np.nanpercentile(np.r_[e, rows], [1, 99])
    y0, y1 = int(max(lo - 70, 0)), int(min(hi + 50, H))
    crop = img[y0:y1]
    p1, p99 = np.percentile(crop, [1, 99])
    ax.imshow(np.clip((crop - p1) / (p99 - p1), 0, 1),
              extent=(0, W, y1, y0), aspect='auto')
    ax.plot(np.arange(W), e, color='#ff00ff', lw=1.2, label='eWaSR')
    ax.plot(np.arange(W), rows, color='#ffd23f', lw=1.2,
            label='faint-ridge detector')
    ax.set_title('%s (contrast stretched)' % fr['tag'], fontsize=10)
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fr['tag'] + '_faint.png'), dpi=100)
    plt.close(fig)


ARMS = dict(faint_hard6=[], faint_soft5=['--compass-sigma', '5'])


def solve(fr, arm, flags):
    out = os.path.join(OUT, '%s_%s' % (fr['tag'], arm))
    if os.path.exists(out + '.json'):
        return out
    cmd = [sys.executable, os.path.join(HERE, 'skyfix.py'),
           celnav(fr['path']),
           '--center', '%.6f,%.6f' % (fr['lat'], fr['lon']),
           '--box', str(fr['box']), '--z', str(fr['z']),
           '--fov', str(fr['fov']), '--heading', str(fr['heading']),
           '--boundary-npy', os.path.join(OUT, fr['tag'] + '_faint.npz'),
           '--out', out] + flags.split() + ARMS[arm]
    with open(out + '.log', 'w') as lg:
        subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=HERE,
                       env=dict(os.environ, OMP_NUM_THREADS='1'))
    return out


def run(a):
    flags = json.load(open(FRAMES))['flags'].replace(
        '~', os.path.expanduser('~'))
    jobs = [(fr, arm) for arm in ARMS for fr in frames(a.tags)
            if os.path.exists(os.path.join(OUT, fr['tag'] + '_faint.npz'))]
    with ThreadPoolExecutor(a.jobs) as ex:
        for k, out in enumerate(ex.map(lambda j: solve(*j, flags), jobs), 1):
            print('%3d/%d %s %s' % (k, len(jobs), os.path.basename(out),
                                    '' if os.path.exists(out + '.json')
                                    else 'NO RESULT'), flush=True)


def load_one(f):
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    return dict(err=math.hypot(j['dn_m'], j['de_m']), dn=j['dn_m'],
                de=j['de_m'], ok=j['status'] == 'ok',
                hoff=j['photos'][0]['heading_offset_deg'])


def report(a):
    src = {'hard6': ('baseline', '%s_huber.json'),
           'soft5': ('compass', '%s_soft5.json'),
           'faint_hard6': ('faint', '%s_faint_hard6.json'),
           'faint_soft5': ('faint', '%s_faint_soft5.json')}
    rows = []
    for fr in frames(a.tags):
        r = dict(tag=fr['tag'])
        for k, (d, p) in src.items():
            r[k] = load_one(os.path.join(HERE, 'out', d, p % fr['tag']))
        if all(r[k] for k in src):
            rows.append(r)
    print('%-26s | ' % 'frame' + ' | '.join('%-16s' % k for k in src))
    for r in rows:
        print('%-26s | ' % r['tag'][:26] + ' | '.join(
            '%6.0fm %s %+5.1f°' % (r[k]['err'], 'A' if r[k]['ok'] else '-',
                                   r[k]['hoff']) for k in src))
    summ = {}
    print('\n%-12s %7s %5s %5s | %4s %4s %5s   (soft arms: |psi| <= 2 '
          'sigma gate)' % ('arm', 'median', '<300', '<500', 'acc', 'good',
                           'wrong'))
    for k in src:
        e = np.array([r[k]['err'] for r in rows])
        gate = 10.0 if 'soft5' in k else 1e9
        acc = [r for r in rows if r[k]['ok'] and abs(r[k]['hoff']) <= gate]
        summ[k] = dict(median=float(np.median(e)),
                       within300=int((e < GOOD).sum()),
                       within500=int((e < BAD).sum()), accepted=len(acc),
                       good=sum(r[k]['err'] < GOOD for r in acc),
                       wrong=sum(r[k]['err'] >= BAD for r in acc),
                       wrong_tags=[r['tag'] for r in acc
                                   if r[k]['err'] >= BAD])
        print('%-12s %6.0fm %5d %5d | %4d %4d %5d  %s' % (
            k, summ[k]['median'], summ[k]['within300'],
            summ[k]['within500'], summ[k]['accepted'], summ[k]['good'],
            summ[k]['wrong'], ', '.join(t[3:] for t in summ[k]['wrong_tags'])))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(
        OUT, 'report_%d.json' % len(rows)), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    for n in ('eval', 'run', 'report'):
        p = sp.add_parser(n)
        p.add_argument('tags', nargs='*')
        p.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    dict(eval=evaluate, run=run, report=report)[a.cmd](a)
