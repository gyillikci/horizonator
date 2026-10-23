#!/usr/bin/env python3
"""E5cb: does a skyline's local frequency content tell how far it is?

Idea (operator): near ridges carry high-frequency detail; far ridges
keep only low frequencies, so the spectrum of the skyline could reveal
range, and with it which parts of the photo are hazy far terrain.

Geometry first: a feature of size L at range r subtends L/r, so far
terrain moves to HIGHER angular frequency but smaller amplitude. For
self-similar terrain (height ~ L^H) the angular spectrum keeps its
shape and scales as r^(H-1): range would show only in amplitude, not
in spectral shape. Haze and optics blur in angle, which can still
remove the far high frequencies from the PHOTO. So both skylines are
measured, at the truth, against the DEM crest range of each column:

  hf    rms of the skyline minus its own 2 deg running mean, in a 2 deg
        window (mrad): local high-frequency amplitude;
  ratio hf over the rms of the 2 deg running mean minus a 8 deg one:
        spectral SHAPE, independent of amplitude.

    python3 e5cb_spectral.py
"""

import json
import math
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skyfix as F  # noqa: E402
import skyline as S  # noqa: E402
import extract  # noqa: E402
from assets import celnav  # noqa: E402

OUT = os.path.join(HERE, 'out', 'spectral')
STEP = 0.1                      # deg per AZ bin
BANDS = ((0, 3), (3, 6), (6, 10), (10, 15), (15, 25), (25, 41))


def features(el, ok):
    """Local high-frequency amplitude and spectral-shape ratio (mrad)."""
    e = np.where(ok, el, np.nan)
    fill = np.interp(np.arange(e.size), np.flatnonzero(ok), e[ok])
    m2 = uniform_filter1d(fill, int(2 / STEP), mode='wrap')
    m8 = uniform_filter1d(fill, int(8 / STEP), mode='wrap')
    hp = (fill - m2) ** 2
    bp = (m2 - m8) ** 2
    # a running mean of squares can dip a hair below zero in floats
    sm = lambda x: np.maximum(uniform_filter1d(x, int(2 / STEP),
                                               mode='wrap'), 0.0)
    hf = np.sqrt(sm(hp)) * 1e3
    lf = np.sqrt(sm(bp)) * 1e3
    return hf, hf / np.maximum(lf, 1e-3)


def main():
    os.makedirs(OUT, exist_ok=True)
    F.EXTRACTOR = 'ewasr'
    meta = json.load(open(os.path.join(HERE, 'data', 'campaign_frames.json')))
    dem = os.path.expanduser('~/.horizonator/DEMs_COP30')
    betas = np.linspace(-0.03, 0.03, 61)
    pool = dict(r=[], dem_hf=[], dem_ratio=[], ph_hf=[], ph_ratio=[],
                fit=[])
    per = {}
    for fr in meta['frames']:
        img = extract.load_image(celnav(fr['path']))
        el_obs, w, _ = F.observation(img, fr['fov'], fr['heading'], 0.0)
        if (w > 0).sum() < 200:
            continue
        cm = S.CMarcher(dem, (fr['lat'] - .6, fr['lat'] + .6),
                        (fr['lon'] - .8, fr['lon'] + .8), d_min=1000.0)
        el, r = cm.skyline(fr['lat'], fr['lon'], fr['z'], F.AZ)
        el = np.nan_to_num(el + np.where(np.isfinite(r) & (r > 0),
                                         9.0 / np.maximum(r, 1000.0), 0))
        c, s, b = F.photo_cost(el_obs, w, el, np.arange(-300, 301, 2), betas)
        rms = math.sqrt(2 * c) * 1e3
        eo = np.roll(el_obs, s)
        m = np.roll(w, s) > 0
        # stay 2 deg inside the photo so the windows see only photo
        inner = m & (uniform_filter1d(m.astype(float), int(4 / STEP),
                                      mode='wrap') > 0.999)
        if inner.sum() < 100:
            continue
        d_hf, d_ratio = features(el, np.ones(el.size, bool))
        p_hf, p_ratio = features(eo, m)
        rr = np.nan_to_num(r) / 1e3
        k = inner & (rr > 0)
        rho = dict(dem_hf=spearmanr(rr[k], d_hf[k]).correlation,
                   dem_ratio=spearmanr(rr[k], d_ratio[k]).correlation,
                   ph_hf=spearmanr(rr[k], p_hf[k]).correlation,
                   ph_ratio=spearmanr(rr[k], p_ratio[k]).correlation)
        per[fr['tag']] = dict(rms=rms, n=int(k.sum()),
                              range_spread=float(np.ptp(rr[k])),
                              rho={a: float(v) for a, v in rho.items()})
        for a, v in (('r', rr), ('dem_hf', d_hf), ('dem_ratio', d_ratio),
                     ('ph_hf', p_hf), ('ph_ratio', p_ratio)):
            pool[a].extend(v[k][::5].tolist())        # ~0.5 deg apart
        pool['fit'].extend([rms < 8] * len(rr[k][::5]))
        print('%-28s rms %5.1f  range %4.1f-%4.1f km  rho(range, hf): '
              'DEM %+.2f photo %+.2f | rho(range, shape): DEM %+.2f '
              'photo %+.2f' % (fr['tag'], rms, rr[k].min(), rr[k].max(),
                               rho['dem_hf'], rho['ph_hf'],
                               rho['dem_ratio'], rho['ph_ratio']),
              flush=True)
    P = {a: np.array(v) for a, v in pool.items()}
    lines = []
    for name, sel in (('all frames', np.ones(P['r'].size, bool)),
                      ('DEM explains the photo (rms < 8)', P['fit'])):
        lines.append('\n%s: %d samples' % (name, sel.sum()))
        lines.append('  %-9s %8s %8s %8s %8s' % ('range km', 'DEM hf',
                                                 'DEM shp', 'photo hf',
                                                 'photo shp'))
        for lo, hi in BANDS:
            b = sel & (P['r'] >= lo) & (P['r'] < hi)
            if b.sum() < 20:
                continue
            lines.append('  %2d-%2d     %8.2f %8.2f %8.2f %8.2f   (n %d)' % (
                lo, hi, *[np.median(P[a][b]) for a in
                          ('dem_hf', 'dem_ratio', 'ph_hf', 'ph_ratio')],
                b.sum()))
        for a in ('dem_hf', 'dem_ratio', 'ph_hf', 'ph_ratio'):
            lines.append('  pooled Spearman(range, %s) %+.2f' % (
                a, spearmanr(P['r'][sel], P[a][sel]).correlation))
    # can the photo's spectrum CLASSIFY far (>15 km) from near (<6 km)?
    sel = P['fit']
    near, far = sel & (P['r'] < 6), sel & (P['r'] > 15)
    for a in ('ph_hf', 'ph_ratio'):
        x = np.r_[P[a][near], P[a][far]]
        y = np.r_[np.zeros(near.sum()), np.ones(far.sum())]
        if near.any() and far.any():
            o = np.argsort(x)
            ranks = np.empty(x.size)
            ranks[o] = np.arange(1, x.size + 1)
            auc = (ranks[y == 1].sum() - far.sum() * (far.sum() + 1) / 2) \
                / (near.sum() * far.sum())
            lines.append('  AUC far(>15 km) vs near(<6 km) from %s: %.2f '
                         '(0.5 = chance; <0.5 = far has LESS)' % (a, auc))
    print('\n'.join(lines))
    json.dump(dict(frames=per, summary=lines),
              open(os.path.join(OUT, 'spectral.json'), 'w'), indent=1)
    figure(P)


def figure(P):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sel = P['fit']
    for i, (a, lab) in enumerate((('hf', 'high-frequency amplitude, mrad'),
                                  ('ratio', 'spectral shape (hf / mid)'))):
        for src, col, nm in (('dem_', '#3a6ea5', 'DEM skyline'),
                             ('ph_', '#c0392b', 'photo skyline')):
            med = []
            for lo, hi in BANDS:
                b = sel & (P['r'] >= lo) & (P['r'] < hi)
                med.append(np.median(P[src + a][b]) if b.sum() >= 20
                           else np.nan)
            ax[i].plot([(lo + hi) / 2 for lo, hi in BANDS], med, 'o-',
                       color=col, label=nm)
        ax[i].set_xlabel('DEM crest range at the truth, km')
        ax[i].set_ylabel(lab)
        ax[i].set_yscale('log')
        ax[i].grid(alpha=.3)
        ax[i].legend()
    fig.suptitle('Skyline frequency content against range (frames where '
                 'the DEM explains the photo, medians)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'spectral.png'), dpi=110)


if __name__ == '__main__':
    main()
