#!/usr/bin/env python3
"""E5bv: better skyline extraction, judged by where the solver lands.

E5bu showed a perfect map barely helps the skyline alone: the limit is
the image side. E5bo compared front ends on three frames; this runs
them on all 44 field frames of the campaign and scores each by what
actually matters - the position the unchanged solver reaches from its
boundary - so pixel accuracy is never mistaken for usefulness.

Front ends, every one handed to skyfix through the same door
(--boundary-npy with rows + per-column confidence, then the same
plausibility screen, spike filter, DEM, search and gates):

  seam      extract.skyline_seam, the mountain seam detector
  ewasr     eWaSR maritime segmenter, first non-sky row below the sky
  sam_t     SAM 2.1-t prompted (E5bo), filled, top-connected sky
  fused_dp  NEW. The two segmenters disagree in different places
            (eWaSR rides cloud edges, SAM sometimes stops in a
            gradient sky, E5bo/E5bp), so their sky beliefs are
            averaged - eWaSR's soft sky probability and SAM's mask,
            blurred - and the boundary is chosen GLOBALLY: for every
            column and row the cost of "sky above, not-sky below" is
            the negative log-likelihood of the whole column under the
            fused probability, minus a bonus for a vertical image edge
            there, and dynamic programming (Lie et al. 2005; Ahmad et
            al. 2021 use the same multistage-graph idea on learned
            edges) finds the cheapest continuous path, with an L1
            price on row change between neighbouring columns and a
            fixed price for a genuine jump (a cliff, a near object).
            Per-column confidence is the sky/not-sky contrast across
            the chosen row, so a column where the fused belief is
            mush is down-weighted instead of trusted.

Reference: the current default pipeline (eWaSR with the seam fallback
pre-check), as solved in E5bq.

    python3 e5bv_extraction.py extract      # boundaries for every frame
    python3 e5bv_extraction.py solve [--jobs 4]
    python3 e5bv_extraction.py report
"""

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from assets import celnav  # noqa: E402

OUT = os.path.join(HERE, 'out', 'extract')
FRAMES = os.path.join(HERE, 'data', 'campaign_frames.json')
METHODS = ('seam', 'ewasr', 'sam_t', 'fused_dp', 'ewasr_w', 'fused_w')
BAD, GOOD = 500.0, 300.0


def frames():
    return json.load(open(FRAMES))['frames']


# ------------------------------------------------------------ fused DP
def dp_boundary(P, edge, alpha=0.4, jump=60.0, lam_edge=6.0, top=0.9,
                return_cost=False):
    """Globally optimal sky boundary through a soft sky map P (H x W,
    probability of sky). Returns rows (W,), confidence (W,)."""
    H, W = P.shape
    Hs = int(H * top)
    eps = 1e-3
    a = -np.log(np.clip(P[:Hs], eps, 1 - eps))          # cost: row is sky
    b = -np.log(np.clip(1 - P[:Hs], eps, 1 - eps))      # cost: not sky
    # boundary at r: rows < r are sky, rows >= r are not
    A = np.vstack([np.zeros((1, W)), np.cumsum(a, 0)])[:-1]
    Bc = np.cumsum(b[::-1], 0)[::-1]
    C = A + Bc - lam_edge * edge[:Hs]                   # (Hs, W)
    C = C - C.min(0, keepdims=True)
    acc = C[:, 0].copy()
    back = np.zeros((Hs, W), dtype=np.int32)
    r = np.arange(Hs)
    ar = alpha * r
    for x in range(1, W):
        # min over r' of acc[r'] + alpha |r - r'|, exactly, as two running
        # minima: forward  alpha*r + cummin(acc - alpha*r'), r' <= r
        #         backward -alpha*r + reverse cummin(acc + alpha*r'), r' >= r
        v1 = acc - ar
        c1 = np.minimum.accumulate(v1)
        i1 = np.maximum.accumulate(np.where(v1 == c1, r, 0))
        v2 = (acc + ar)[::-1]
        c2 = np.minimum.accumulate(v2)
        i2 = np.maximum.accumulate(np.where(v2 == c2, np.arange(Hs), 0))
        f1 = c1 + ar
        f2 = c2[::-1] - ar
        i2 = (Hs - 1 - i2)[::-1]
        f = np.minimum(f1, f2)
        arg = np.where(f1 <= f2, i1, i2)
        jb = int(np.argmin(acc))
        use_jump = acc[jb] + jump < f
        f = np.where(use_jump, acc[jb] + jump, f)
        arg = np.where(use_jump, jb, arg)
        back[:, x] = arg
        acc = f + C[:, x]
    rows = np.empty(W, dtype=int)
    rows[-1] = int(np.argmin(acc))
    for x in range(W - 1, 0, -1):
        rows[x - 1] = back[rows[x], x]
    # confidence: sky above vs not-sky below, 12 px either side
    conf = np.zeros(W)
    for x in range(W):
        y = rows[x]
        up = P[max(0, y - 12):y, x]
        dn = P[y:min(H, y + 12), x]
        if up.size >= 4 and dn.size >= 4:
            conf[x] = np.clip((up.mean() - dn.mean() - 0.3) / 0.5, 0, 1)
    if return_cost:
        return rows.astype(float), conf, C
    return rows.astype(float), conf


def column_sigma(C, rows, P, r_e, r_s, sky_any, nat=2.0, s0=3.0):
    """Per-column 1-sigma of the boundary row, in pixels, from three
    independent signs of doubt, and the matching weight.

      cost   the column's negative log-likelihood C(r) under the fused
             sky belief: the run of rows within `nat` nats of its best
             value around the chosen row. For a Gaussian likelihood that
             run is +-2 sigma, so sigma = run / 4. A boundary the image
             pins sharply has a narrow run; mush has a wide one.
      agree  where the two segmenters disagree (after their constant
             offset), and how far the chosen row is from their mean.
      contrast  sky above against not-sky below, 12 px either side:
             1.5 px at full contrast, growing as it fades (haze, cloud
             fringe, a ridge the same tone as the sky).

    weight = 1 / (1 + (sigma / s0)^2): nothing is dropped outright
    except columns with no sky at all; doubtful ones just count less."""
    Hs, W = C.shape
    H = P.shape[0]
    ok_s = np.isfinite(r_s)
    o_s = np.median((r_s - r_e)[ok_s]) if ok_s.any() else 0.0
    r_s2 = np.where(ok_s, r_s - o_s, r_e)
    sig = np.empty(W)
    for x in range(W):
        r = int(min(max(rows[x], 0), Hs - 1))
        lo, hi = max(0, r - 60), min(Hs, r + 61)
        seg = C[lo:hi, x]
        k = r - lo
        cmin = seg[max(0, k - 3):k + 4].min()
        okk = seg <= cmin + nat
        L = R = k
        while L > 0 and okk[L - 1]:
            L -= 1
        while R < seg.size - 1 and okk[R + 1]:
            R += 1
        s_cost = (R - L + 1) / 4.0
        mid = 0.5 * (r_e[x] + r_s2[x])
        s_agree = math.hypot(r_e[x] - r_s2[x], rows[x] - mid)
        up = P[max(0, r - 12):r, x]
        dn = P[r:min(H, r + 12), x]
        con = (up.mean() - dn.mean()) if up.size >= 4 and dn.size >= 4 else 0
        s_con = 1.5 / max(con, 0.05)
        sig[x] = math.sqrt(s_cost ** 2 + s_agree ** 2 + s_con ** 2)
    w = np.where(sky_any, 1.0 / (1.0 + (sig / s0) ** 2), 0.0)
    return sig, w


def extract_all(only=None):
    import extract
    import e5bo_segmenters as E
    from ewasr_bridge import EWasr
    from assets import need
    ew = EWasr(need('eWaSR'), need('ewasr_resnet18.pth'))
    os.makedirs(OUT, exist_ok=True)
    for fr in frames():
        tag = fr['tag']
        if only and tag not in only:
            continue
        if all(os.path.exists(os.path.join(OUT, '%s_%s.npz' % (tag, m)))
               for m in METHODS):
            continue
        img = extract.load_image(celnav(fr['path']))       # max 1600 wide
        H, W = img.shape[:2]
        img8 = (img * 255).astype(np.uint8)
        res = {}
        # seam
        r, c = extract.skyline_seam(img)
        res['seam'] = (np.asarray(r, float), np.asarray(c, float))
        # eWaSR: hard boundary as skyfix does, and the soft sky map
        pr = ew.predict_proba(img)
        sky = pr.argmax(0) == 2
        rows = np.full(W, H - 1, float)
        conf = np.zeros(W)
        has = sky.any(0)
        topr = np.where(has, sky.argmax(0), 0)
        for x in range(W):
            if has[x]:
                below = ~sky[topr[x]:, x]
                if below.any():
                    k = int(np.argmax(below))
                    if k >= 3:
                        rows[x], conf[x] = topr[x] + k, 1.0
        res['ewasr'] = (rows, conf)
        # SAM 2.1-t prompted
        m = E.sam_prompted(img8, 'sam2.1_t.pt')
        rs, cs = E.mask_to_boundary(m)
        if rs is None:
            rs, cs = np.full(W, H - 1.0), np.zeros(W)
        rs = np.where(np.isfinite(rs), rs, H - 1.0)
        res['sam_t'] = (np.asarray(rs, float), np.asarray(cs, float))
        # fused: average of the two soft beliefs, then global DP
        p_e = pr[2]
        p_s = ndimage.gaussian_filter(m.astype(float), 2.0)
        P = 0.5 * (p_e + p_s)
        g = ndimage.gaussian_filter(img.mean(2), 1.0)
        gy = np.abs(ndimage.sobel(g, axis=0))
        edge = gy / (np.percentile(gy, 99.5) + 1e-9)
        edge = np.clip(edge, 0, 1)
        res['fused_dp'] = dp_boundary(P, edge)
        # E5bw: the same boundaries, now with calibrated per-column
        # confidence instead of uniform (eWaSR) or clipped (fused) weights
        r_f, _, Cf = dp_boundary(P, edge, return_cost=True)
        r_e = res['ewasr'][0]
        r_s = np.where(np.asarray(cs) > 0, rs, np.nan)
        sky_any = sky.any(0)
        sig_e, w_e = column_sigma(Cf, r_e, P, r_e, r_s, sky_any & (res['ewasr'][1] > 0))
        sig_f, w_f = column_sigma(Cf, r_f, P, r_e, r_s, sky_any)
        res['ewasr_w'] = (r_e, w_e)
        res['fused_w'] = (r_f, w_f)
        extra = dict(ewasr_w=sig_e, fused_w=sig_f)
        for k, (r, c) in res.items():
            np.savez(os.path.join(OUT, '%s_%s.npz' % (tag, k)),
                     rows=r, conf=c, H=H, W=W,
                     sigma=extra.get(k, np.full(W, np.nan)))
        print('%-36s %s' % (tag, '  '.join(
            '%s %.0f%%' % (k, 100 * np.mean(np.asarray(c) > 0))
            for k, (r, c) in res.items())), flush=True)


# ------------------------------------------------------------ solving
def solve_one(fr, m, flags):
    out = os.path.join(OUT, 'solve_%s_%s' % (fr['tag'], m))
    if os.path.exists(out + '.json'):
        return out
    cmd = [sys.executable, os.path.join(HERE, 'skyfix.py'), celnav(fr['path']),
           '--center', '%.6f,%.6f' % (fr['lat'], fr['lon']),
           '--box', str(fr['box']), '--z', str(fr['z']),
           '--fov', str(fr['fov']), '--heading', str(fr['heading']),
           '--boundary-npy', os.path.join(OUT, '%s_%s.npz' % (fr['tag'], m)),
           '--out', out] + flags.split()
    with open(out + '.log', 'w') as lg:
        subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=HERE,
                       env=dict(os.environ, OMP_NUM_THREADS='1'))
    return out


def solve(a):
    meta = json.load(open(FRAMES))
    flags = meta['flags'].replace('~', os.path.expanduser('~'))
    jobs = [(fr, m) for fr in frames() for m in METHODS]
    with ThreadPoolExecutor(a.jobs) as ex:
        for k, out in enumerate(ex.map(lambda j: solve_one(j[0], j[1], flags),
                                       jobs), 1):
            print('%3d/%d %s %s' % (k, len(jobs), os.path.basename(out),
                                    '' if os.path.exists(out + '.json')
                                    else 'NO RESULT'), flush=True)


# ------------------------------------------------------------ report
def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ref = {r['tag']: r for r in json.load(open(os.path.join(
        HERE, 'out', 'baseline', 'report.json')))}
    res = {}
    for fr in frames():
        t = fr['tag']
        row = {}
        for m in METHODS:
            f = os.path.join(OUT, 'solve_%s_%s.json' % (t, m))
            if os.path.exists(f):
                j = json.load(open(f))
                row[m] = dict(err=math.hypot(j['dn_m'], j['de_m']),
                              ok=j['status'] == 'ok',
                              margin=j['basin_margin'])
        if t in ref:
            row['default'] = dict(err=ref[t]['huber']['err'],
                                  ok=ref[t]['huber']['ok'],
                                  margin=ref[t]['huber']['margin'])
        res[t] = row
    keys = ['default'] + list(METHODS)
    print('%-9s %4s | %7s %5s %5s %5s | %4s %5s %5s' % (
        'front end', 'n', 'median', '<300', '<500', '>=500', 'acc',
        'good', 'WRONG'))
    summ = {}
    for k in keys:
        rr = [r[k] for r in res.values() if k in r]
        if not rr:
            continue
        e = np.array([x['err'] for x in rr])
        acc = [x for x in rr if x['ok']]
        summ[k] = dict(n=len(rr), median=float(np.median(e)),
                       lt300=int((e < GOOD).sum()), lt500=int((e < BAD).sum()),
                       ge500=int((e >= BAD).sum()), accepted=len(acc),
                       acc_good=sum(x['err'] < GOOD for x in acc),
                       acc_wrong=sum(x['err'] >= BAD for x in acc))
        s = summ[k]
        print('%-9s %4d | %6.0fm %5d %5d %5d | %4d %5d %5d' % (
            k, s['n'], s['median'], s['lt300'], s['lt500'], s['ge500'],
            s['accepted'], s['acc_good'], s['acc_wrong']))
    # per-frame best: is any one front end right where the others fail?
    best = {k: 0 for k in keys}
    for r in res.values():
        if all(k in r for k in keys):
            best[min(keys, key=lambda k: r[k]['err'])] += 1
    print('closest to truth, frames won:', best)
    json.dump(dict(summary=summ, frames=res),
              open(os.path.join(OUT, 'report.json'), 'w'), indent=1)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, k in enumerate(keys):
        e = np.sort([r[k]['err'] for r in res.values() if k in r])
        ax.plot(e, np.arange(1, e.size + 1) / e.size, lw=2,
                label='%s (median %.0f m)' % (k, np.median(e)),
                color=['#8A8F98', '#B9770E', '#2E86C1', '#C0392B',
                       '#1E7F5C', '#5DADE2', '#7D3C98'][i % 7])
    for v in (GOOD, BAD):
        ax.axvline(v, color='#1A2732', lw=.7, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('position error of the solve (m)')
    ax.set_ylabel('fraction of the 44 field frames')
    ax.set_title('Same solver, same frames; only the skyline front end '
                 'differs', fontsize=10.5)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(alpha=.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'extraction.png'), dpi=110)
    print('wrote', os.path.join(OUT, 'extraction.png'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['extract', 'solve', 'report'])
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    {'extract': lambda: extract_all(), 'solve': lambda: solve(a),
     'report': report}[a.cmd]()
