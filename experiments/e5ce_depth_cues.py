#!/usr/bin/env python3
"""E5ce: which image cue tells the range of the mountain body best?

Same pixels and the same truth as E5cd (eWaSR land between its sky and
its water; true range from the DEM ray at the truth). Cues per pixel,
each from the literature in study section 3.10:

  tex     fine texture over local mean (E5cd; Make3D's texture energy)
  lum     local mean luminance over the sky's (aerial perspective)
  dcp     -ln t from the dark channel prior (He et al. 2009): t =
          1 - 0.95 min_patch min_c I_c / A_c, 15 px patch
  cap     value - saturation, the colour attenuation prior's depth
          proxy (Zhu et al. 2015)
  hline   -ln t from haze-lines (Berman et al. 2016): I - A in
          spherical coordinates, 1000 direction bins, t = r / r_max
  skysim  minus the Mahalanobis distance of the pixel's colour to the
          sky's own colour distribution (3-PC sky model)
  laws1   first principal component of 9 log Laws texture energies
          (Laws 1980; PCA across all frames)

A is the sky colour just above the skyline (median over the image).
Scored by the per-frame Spearman correlation with the true range, and
by a leave-one-frame-out linear regression of ln(range) on every cue:
how well does a model fitted on the OTHER frames rank this frame's
pixels by range?

    python3 e5ce_depth_cues.py
"""

import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter, convolve
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skyfix as F  # noqa: E402
import skyline as S  # noqa: E402
from assets import need  # noqa: E402
from e5cd_body_texture import body_pixels, lin  # noqa: E402

OUT = os.path.join(HERE, 'out', 'texture')
CUES = ['tex', 'lum', 'dcp', 'cap', 'hline', 'skysim', 'laws1']
# the sign physics predicts: + means the cue grows with range
EXPECT = dict(tex=-1, lum=+1, dcp=+1, cap=+1, hline=+1, skysim=+1)
CAP = 20000          # pixels per frame, so big frames do not dominate
L5 = np.array([1, 4, 6, 4, 1.])
E5 = np.array([-1, -2, 0, 2, 1.])
S5 = np.array([-1, 0, 2, 0, -1.])
R5 = np.array([1, -4, 6, -4, 1.])
LAWS = [(E5, E5), (S5, S5), (R5, R5), (E5, L5), (S5, L5), (R5, L5),
        (S5, E5), (R5, E5), (R5, S5)]


def sky_colour(img, cls, top):
    """Median colour of the sky just above the skyline, and the sky
    pixels' colour samples for the 3-PC model."""
    H, W = cls.shape
    pts = []
    for x in range(0, W, 4):
        t = top[x] if top[x] > 0 else None
        if t is None:
            sky = np.flatnonzero(cls[:, x] == 2)
            if sky.size < 5:
                continue
            t = sky.max() + 1
        pts.append(img[max(t - 30, 0):max(t - 4, 1), x])
    pts = np.concatenate(pts) if pts else img.reshape(-1, 3)[:1000]
    return np.median(pts, axis=0), pts


def features(img, cls, top):
    L = lin(img) @ np.array([0.2126, 0.7152, 0.0722])
    mu = gaussian_filter(L, 4)
    hp = L - gaussian_filter(L, 1.5)
    f = dict(tex=np.sqrt(gaussian_filter(hp * hp, 4)) / np.maximum(mu, 1e-3))
    A, skypts = sky_colour(img, cls, top)
    LA = float(lin(A) @ np.array([0.2126, 0.7152, 0.0722]))
    f['lum'] = mu / max(LA, 1e-3)
    dc = minimum_filter((img / np.maximum(A, 1e-3)).min(axis=2), size=15)
    f['dcp'] = -np.log(np.clip(1 - 0.95 * dc, 0.02, 1))
    mx, mn = img.max(axis=2), img.min(axis=2)
    sat = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0)
    f['cap'] = gaussian_filter(mx - sat, 2)
    # haze-lines: radius from the air-light along ~1000 colour directions
    D = img.reshape(-1, 3) - A
    rad = np.linalg.norm(D, axis=1) + 1e-6
    th = np.arccos(np.clip(D[:, 2] / rad, -1, 1))
    ph = np.arctan2(D[:, 1], D[:, 0])
    b = (np.floor(th / np.pi * 25).clip(0, 24) * 40
         + np.floor((ph + np.pi) / (2 * np.pi) * 40).clip(0, 39)).astype(int)
    rmax = np.zeros(1000)
    np.maximum.at(rmax, b, rad)
    t = np.clip(rad / np.maximum(rmax[b], 1e-6), 0.02, 1)
    f['hline'] = gaussian_filter(-np.log(t).reshape(L.shape), 2)
    # 3-PC sky colour model: Mahalanobis distance to the sky's colours
    mu_s = skypts.mean(axis=0)
    C = np.cov((skypts - mu_s).T) + 1e-6 * np.eye(3)
    Ci = np.linalg.inv(C)
    G = np.stack([gaussian_filter(img[..., c], 2) for c in range(3)], -1)
    Dm = G - mu_s
    f['skysim'] = -np.sqrt(np.einsum('...i,ij,...j->...', Dm, Ci, Dm))
    # Laws texture energies (log, contrast-normalised)
    laws = []
    for a, b_ in LAWS:
        k = np.outer(a, b_)
        e = np.abs(convolve(L, k, mode='nearest'))
        if not np.array_equal(a, b_):
            e = 0.5 * (e + np.abs(convolve(L, k.T, mode='nearest')))
        laws.append(np.log(gaussian_filter(e, 4) / np.maximum(mu, 1e-3)
                           + 1e-4))
    return f, np.stack(laws, -1)


def main():
    from ewasr_bridge import EWasr
    net = EWasr(need('eWaSR'), need('ewasr_resnet18.pth'))
    F.EXTRACTOR = 'ewasr'
    demdir = os.path.expanduser('~/.horizonator/DEMs_COP30')
    dem = S.Dem(demdir)
    meta = json.load(open(os.path.join(HERE, 'data', 'campaign_frames.json')))
    rng = np.random.default_rng(0)
    rows = []
    for fr in meta['frames']:
        bp = body_pixels(fr, net, dem, demdir)
        if bp is None or bp['x'].size < 200 or np.ptp(bp['r_km']) < 2:
            continue
        x, y, r = bp['x'], bp['y'], bp['r_km']
        if x.size > CAP:
            k = rng.choice(x.size, CAP, replace=False)
            x, y, r = x[k], y[k], r[k]
        f, laws = features(bp['img'], bp['cls'], bp['top'])
        row = dict(tag=fr['tag'], rms=bp['rms'], r=r,
                   laws=laws[y, x], **{c: f[c][y, x] for c in f})
        rows.append(row)
        print('%-28s rms %5.1f %6d px %4.1f-%4.1f km | %s' % (
            fr['tag'], bp['rms'], r.size, r.min(), r.max(),
            ' '.join('%s %+.2f' % (c, spearmanr(r, row[c]).correlation)
                     for c in f)), flush=True)
    # laws1: PCA over every frame's (standardised) log energies
    Lw = np.concatenate([q['laws'] for q in rows])
    m, sd = Lw.mean(0), Lw.std(0) + 1e-9
    _, _, Vt = np.linalg.svd((Lw - m) / sd, full_matrices=False)
    for q in rows:
        q['laws1'] = ((q['laws'] - m) / sd) @ Vt[0]
    print('laws PC1 loadings: %s' % np.round(Vt[0], 2))

    lines = []
    for name, sub in (('all frames', rows),
                      ('DEM explains the photo (rms < 8)',
                       [q for q in rows if q['rms'] < 8])):
        lines.append('\n%s: %d frames' % (name, len(sub)))
        lines.append('  %-8s %8s %10s' % ('cue', 'median', 'expected'))
        lines[-1] += '  per-frame Spearman(range, cue)'
        for c in CUES:
            rh = np.array([spearmanr(q['r'], q[c]).correlation for q in sub])
            e = EXPECT.get(c)
            lines.append('  %-8s %+8.2f %10s  right sign on %d of %d' % (
                c, np.median(rh), {1: '+', -1: '-', None: '?'}[e],
                int((np.sign(rh) == e).sum()) if e else
                int((rh > 0).sum()), len(rh)))
        # leave one frame out: ln r ~ all cues (+ the 9 Laws energies)
        for label, cols in (('lum alone', ['lum']),
                            ('tex + lum (E5cd)', ['tex', 'lum']),
                            ('haze cues (dcp, cap, hline, skysim)',
                             ['dcp', 'cap', 'hline', 'skysim']),
                            ('all cues', CUES)):
            sc = []
            for i, q in enumerate(sub):
                tr = [p for j, p in enumerate(sub) if j != i]
                Xtr = np.concatenate([np.c_[[p[c] for c in cols]].T
                                      for p in tr])
                ytr = np.concatenate([np.log(p['r']) for p in tr])
                mu_, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-9
                Z = np.c_[np.ones(len(Xtr)), (Xtr - mu_) / sd_]
                w = np.linalg.lstsq(Z, ytr, rcond=None)[0]
                Xte = np.c_[[q[c] for c in cols]].T
                pred = np.c_[np.ones(len(Xte)), (Xte - mu_) / sd_] @ w
                sc.append(spearmanr(q['r'], pred).correlation)
            sc = np.array(sc)
            lines.append('  leave-one-frame-out, %-36s median Spearman '
                         '%+.2f, positive on %d of %d' % (
                             label, np.median(sc), (sc > 0).sum(), sc.size))
    print('\n'.join(lines))
    json.dump(dict(summary=lines,
                   frames={q['tag']: {c: float(spearmanr(
                       q['r'], q[c]).correlation) for c in CUES}
                       for q in rows}),
              open(os.path.join(OUT, 'depth_cues.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
