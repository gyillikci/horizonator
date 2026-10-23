#!/usr/bin/env python3
"""E5cd: does the TEXTURE of the mountain body tell how far it is?

Idea (operator): between the waterline and the skyline, near terrain
shows rock, scrub and shadow - high spatial frequencies - while a far
ridge is smooth: haze lowers its contrast and the camera cannot
resolve its detail. The image's local frequency content would then be
a per-pixel range cue (a texture-gradient / aerial-perspective cue).

Truth for every body pixel comes from the DEM at the true position:
the pixel's ray (azimuth from the fitted heading, elevation from the
fitted offset) is marched outwards, and its range is the first point
whose terrain angle reaches the pixel's elevation. Body pixels are
eWaSR's land class between its sky and its water. Per pixel:

  tex   contrast-normalised fine texture: rms of (L - G1.5(L)) in a
        G4 window, over the G4 local mean (linear luminance L);
  lum   local mean luminance relative to the sky just above the
        column's skyline (aerial perspective: far terrain brightens
        towards the sky).

    python3 e5cd_body_texture.py
"""

import json
import math
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skyfix as F  # noqa: E402
import skyline as S  # noqa: E402
import extract  # noqa: E402
from assets import celnav, need  # noqa: E402

OUT = os.path.join(HERE, 'out', 'texture')
BANDS = ((1, 3), (3, 6), (6, 10), (10, 15), (15, 25), (25, 41))
REFF = 6371000.0 / (1 - S.K_REFRACTION)
XSTEP, YSTEP = 6, 2           # pixel subsampling


def lin(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def auc(neg, pos):
    """P(a pos sample scores higher than a neg one)."""
    x = np.r_[neg, pos]
    r = np.empty(x.size)
    r[np.argsort(x)] = np.arange(1, x.size + 1)
    return (r[neg.size:].sum() - pos.size * (pos.size + 1) / 2) \
        / (neg.size * pos.size)


def body_pixels(fr, net, dem, demdir):
    """The sampled body pixels of one frame with their TRUE range.

    Returns dict(img, cls, rms, x, y, r_km, top) or None: eWaSR land
    pixels between its sky and its water, every XSTEP-th column and
    YSTEP-th row; r_km from the DEM ray at the truth (fitted heading
    and offset); top[x] = first non-sky row of column x (or -1)."""
    img = extract.load_image(celnav(fr['path']))
    el_obs, w, d = F.observation(img, fr['fov'], fr['heading'], 0.0)
    if (w > 0).sum() < 200:
        return None
    cm = S.CMarcher(demdir, (fr['lat'] - .6, fr['lat'] + .6),
                    (fr['lon'] - .8, fr['lon'] + .8), d_min=1000.0)
    el, r = cm.skyline(fr['lat'], fr['lon'], fr['z'], F.AZ)
    el = np.nan_to_num(el + np.where(np.isfinite(r) & (r > 0),
                                     9.0 / np.maximum(r, 1000.0), 0))
    c, s, b = F.photo_cost(el_obs, w, el, np.arange(-300, 301, 2),
                           np.linspace(-0.03, 0.03, 61))
    H, W = img.shape[:2]
    f = (W / 2) / math.tan(math.radians(fr['fov']) / 2)
    cls = net.predict(img)
    # DEM ray profiles for the sampled columns
    xs = np.arange(XSTEP // 2, W, XSTEP)
    az = np.radians(fr['heading'] + np.degrees(
        np.arctan((xs - (W - 1) / 2) / f)) + s * 0.1)
    dd = np.arange(300.0, 40000.0, 30.0)
    mlat, mlon = S.meters_per_degree(fr['lat'])
    la = fr['lat'] + dd[None] * np.cos(az)[:, None] / mlat
    lo = fr['lon'] + dd[None] * np.sin(az)[:, None] / mlon
    h = dem.sample(la.ravel(), lo.ravel()).reshape(la.shape)
    ang = np.arctan((h - fr['z'] - dd[None] ** 2 / (2 * REFF)) / dd[None])
    run = np.maximum.accumulate(ang, axis=1)
    X, Y, R = [], [], []
    top = np.full(W, -1)
    for j, x in enumerate(xs):
        col = cls[:, x]
        sky = np.flatnonzero(col == 2)
        if sky.size < 5:
            continue
        top[x] = sky.max() + 1             # first non-sky row below sky
        water = np.flatnonzero((col == 1) & (np.arange(H) > top[x]))
        bot = water.min() if water.size else H
        ys = np.arange(top[x] + 3, bot - 3, YSTEP)
        ys = ys[cls[ys, x] == 0]
        if ys.size == 0:
            continue
        u = x - (W - 1) / 2
        e = np.arctan2((H - 1) / 2 - ys, math.hypot(u, f)) + b
        # first range whose running terrain angle reaches the pixel
        k = np.searchsorted(run[j], e)
        ok = k < dd.size
        X.extend([x] * int(ok.sum()))
        Y.extend(ys[ok].tolist())
        R.extend((dd[k[ok]] / 1e3).tolist())
    return dict(img=img, cls=cls, rms=math.sqrt(2 * c) * 1e3,
                x=np.array(X, int), y=np.array(Y, int), r_km=np.array(R),
                top=top)


def main():
    os.makedirs(OUT, exist_ok=True)
    from ewasr_bridge import EWasr
    net = EWasr(need('eWaSR'), need('ewasr_resnet18.pth'))
    F.EXTRACTOR = 'ewasr'
    demdir = os.path.expanduser('~/.horizonator/DEMs_COP30')
    dem = S.Dem(demdir)
    meta = json.load(open(os.path.join(HERE, 'data', 'campaign_frames.json')))
    P = dict(r=[], tex=[], lum=[], fit=[], tag=[])
    per = {}
    for fr in meta['frames']:
        bp = body_pixels(fr, net, dem, demdir)
        if bp is None or bp['x'].size == 0:
            continue
        img, rms = bp['img'], bp['rms']
        L = lin(img) @ np.array([0.2126, 0.7152, 0.0722])
        hp = L - gaussian_filter(L, 1.5)
        mu = gaussian_filter(L, 4)
        tex = np.sqrt(gaussian_filter(hp * hp, 4)) / np.maximum(mu, 1e-3)
        x, y, rr = bp['x'], bp['y'], bp['r_km']
        sky_col = np.zeros(img.shape[1])
        for c_ in np.unique(x):
            a = bp['top'][c_]
            sky_col[c_] = np.median(L[max(a - 14, 0):max(a - 4, 1), c_])
        sky_l = sky_col[x]
        tt = tex[y, x]
        n = x.size
        P['r'].extend(rr.tolist())
        P['tex'].extend(tt.tolist())
        P['lum'].extend((mu[y, x] / np.maximum(sky_l, 1e-3)).tolist())
        P['fit'].extend([rms < 8] * n)
        P['tag'].extend([fr['tag']] * n)
        if n > 50 and np.ptp(rr) > 2:
            rho = spearmanr(rr, tt).correlation
            per[fr['tag']] = dict(rms=rms, n=n, rho=float(rho),
                                  r_range=[float(rr.min()), float(rr.max())])
            print('%-28s rms %5.1f  %5d px  range %4.1f-%4.1f km  '
                  'Spearman(range, texture) %+.2f' % (
                      fr['tag'], rms, n, rr.min(), rr.max(), rho),
                  flush=True)
    A = {k: np.array(v) for k, v in P.items()}
    lines = []
    for name, sel in (('all frames', np.ones(A['r'].size, bool)),
                      ('DEM explains the photo (rms < 8)', A['fit'])):
        lines.append('\n%s: %d pixels from %d frames' % (
            name, sel.sum(), len(set(A['tag'][sel]))))
        lines.append('  %-9s %9s %9s %8s' % ('range km', 'texture',
                                             'lum/sky', 'pixels'))
        for lo_, hi_ in BANDS:
            m = sel & (A['r'] >= lo_) & (A['r'] < hi_)
            if m.sum() >= 50:
                lines.append('  %2d-%2d     %9.3f %9.2f %8d' % (
                    lo_, hi_, np.median(A['tex'][m]),
                    np.median(A['lum'][m]), m.sum()))
        lines.append('  pooled Spearman(range, texture) %+.2f, '
                     '(range, lum/sky) %+.2f' % (
                         spearmanr(A['r'][sel], A['tex'][sel]).correlation,
                         spearmanr(A['r'][sel], A['lum'][sel]).correlation))
        near, far = sel & (A['r'] < 6), sel & (A['r'] > 15)
        if near.any() and far.any():
            lines.append('  near (<6 km) vs far (>15 km): AUC texture '
                         '%.2f (low = far smoother), AUC lum/sky %.2f '
                         '(high = far brighter)' % (
                             auc(A['tex'][near], A['tex'][far]),
                             auc(A['lum'][near], A['lum'][far])))
    rh = [v['rho'] for v in per.values()]
    lines.append('\nper frame: Spearman(range, texture) median %+.2f, '
                 'negative on %d of %d' % (np.median(rh),
                                          sum(x < 0 for x in rh), len(rh)))
    np.savez_compressed(os.path.join(OUT, 'pixels.npz'), **A)
    lines += lofo(A)
    print('\n'.join(lines))
    json.dump(dict(frames=per, summary=lines),
              open(os.path.join(OUT, 'texture.json'), 'w'), indent=1)
    figure(A)


def logistic(X, y):
    """Class-balanced logistic regression (intercept first)."""
    from scipy.optimize import minimize
    sw = np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))
    Xb = np.c_[np.ones(len(X)), X]

    def nll(w):
        z = Xb @ w
        return np.sum(sw * (np.logaddexp(0, z) - y * z)) / len(y)

    def grad(w):
        p = 1 / (1 + np.exp(-(Xb @ w)))
        return Xb.T @ (sw * (p - y)) / len(y)
    return minimize(nll, np.zeros(Xb.shape[1]), jac=grad,
                    method='L-BFGS-B').x


def lofo(A):
    """Near (<6 km) vs far (>15 km) from texture + brightness, logistic
    regression trained on the other frames (leave one frame out), on
    the frames where the DEM explains the photo. Per-frame normalised
    variants remove exposure and in-camera sharpening."""
    sel = A['fit'] & ((A['r'] < 6) | (A['r'] > 15))
    tags = A['tag'][sel]
    y = (A['r'][sel] > 15).astype(int)
    lt, lm = np.log(np.maximum(A['tex'][sel], 1e-4)), A['lum'][sel]
    zt, zm = lt.copy(), lm.copy()
    for t in set(tags):
        m = tags == t
        zt[m] = (lt[m] - np.median(lt[m])) / (np.std(lt[m]) + 1e-6)
        zm[m] = (lm[m] - np.median(lm[m])) / (np.std(lm[m]) + 1e-6)
    out = ['\nleave-one-frame-out, near (<6 km) vs far (>15 km), frames '
           'with both: AUC']
    both = [t for t in set(tags) if y[tags == t].min() == 0
            and y[tags == t].max() == 1]
    for name, X in (('texture', lt[:, None]), ('brightness', lm[:, None]),
                    ('texture + brightness', np.c_[lt, lm]),
                    ('same, per-frame normalised', np.c_[zt, zm])):
        sc = np.full(y.size, np.nan)
        for t in both:
            te, tr = tags == t, tags != t
            wgt = logistic(X[tr], y[tr])
            sc[te] = X[te] @ wgt[1:] + wgt[0]
        a = [auc(sc[(tags == t) & (y == 0)], sc[(tags == t) & (y == 1)])
             for t in both]
        out.append('  %-28s median per-frame AUC %.2f over %d frames '
                   '(min %.2f)' % (name, np.median(a), len(a), min(a)))
    return out


def figure(A):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for sel, col, nm in ((np.ones(A['r'].size, bool), '#8b96a3',
                          'all frames'),
                         (A['fit'], '#c0392b', 'DEM explains the photo')):
        for i, key in enumerate(('tex', 'lum')):
            xs, q1, q2, q3 = [], [], [], []
            for lo_, hi_ in BANDS:
                m = sel & (A['r'] >= lo_) & (A['r'] < hi_)
                if m.sum() >= 50:
                    xs.append((lo_ + hi_) / 2)
                    a, b_, c = np.percentile(A[key][m], [25, 50, 75])
                    q1.append(a)
                    q2.append(b_)
                    q3.append(c)
            ax[i].plot(xs, q2, 'o-', color=col, label=nm + ' (median)')
            ax[i].fill_between(xs, q1, q3, color=col, alpha=.15)
    ax[0].set_ylabel('fine texture / local mean (contrast-normalised)')
    ax[0].set_yscale('log')
    ax[1].set_ylabel('terrain luminance / sky luminance')
    for a in ax:
        a.set_xlabel('true range of the pixel (DEM ray), km')
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    fig.suptitle('Mountain body between waterline and skyline: texture and '
                 'brightness against range (band = quartiles)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'texture.png'), dpi=110)


if __name__ == '__main__':
    main()
