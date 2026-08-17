#!/usr/bin/env python3
"""E4m: a genuinely diverse second detector family — does it decorrelate?

E4l ended on: four parameterizations of ONE detector share failure modes,
so ensembles cannot vote their way to truth; the oracle's headroom needs
a different detector FAMILY. This builds one — the shallow-learning
architecture of Ahmad et al. (IJCNN 2021): a 16x16 linear classifier on
normalized gray patches slid over the image (one correlation) giving a
dense score image, traced by a dynamic-programming seam. Their published
CH1-trained weights are license-restricted (non-commercial research
only), so the committable artifact here is OUR OWN training of the same
architecture: ridge-regressed patches from the CH1 masks of a training
half (even indices of the sorted 203), everything evaluated on the
held-out odd half. (Their weights, if present in the scratchpad clone,
are scored as a research reference only and never shipped.)

Stages:
  1. train the patch classifier on the even half   -> out/e4m_svm.npz
  2. extractor accuracy on the odd half vs the curated masks
  3. the E4l solve on the odd half with FIVE hypotheses (4 x continuation
     family + the seam family) -> cross-family agreement as a gate

Run:   python3 e4m_diverse.py [N_test]   (CSV to out/e4m_diverse.csv)
"""

import os
import sys
import glob
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
import skyline as S
import extract
from skyfix import basin_margin, fast_photo_cost
from e4e_gate_audit import ensure_tiles, CH1, DIR3, OUT, AZ, BOX

FSZ = 16
rng = np.random.default_rng(20260822)


def gray_norm(img_rgb):
    g = (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1]
         + 0.114 * img_rgb[..., 2]) * 255.0
    return (g - 128.0) / 128.0


def mask_rows(meta):
    H = int(open(meta).read().split('\n')[5])
    mask = np.asarray(Image.open(meta[:-8] + '-mask.png').convert('L')) > 127
    return np.where(mask.any(axis=0), mask.argmax(axis=0), H - 1)


def sample_patches(g, rows, n_pos=120, n_neg=240):
    H, W = g.shape
    h = FSZ // 2
    X, y = [], []
    cols = rng.choice(np.arange(h, W - h), min(n_pos, W - FSZ),
                      replace=False)
    for c in cols:
        r = int(rows[c])
        if h <= r < H - h:
            X.append(g[r - h + 1:r + h + 1, c - h + 1:c + h + 1].ravel())
            y.append(1.0)
    for _ in range(n_neg):
        c = rng.integers(h, W - h)
        off = rng.integers(10, 60) * rng.choice([-1, 1])
        r = int(np.clip(rows[c] + off, h, H - h - 1))
        X.append(g[r - h + 1:r + h + 1, c - h + 1:c + h + 1].ravel())
        y.append(-1.0)
    return X, y


def train(metas):
    X, y = [], []
    for meta in metas:
        img = np.asarray(Image.open(meta[:-4]).convert('RGB'),
                         dtype=np.float32) / 255.0
        xs, ys = sample_patches(gray_norm(img), mask_rows(meta))
        X += xs
        y += ys
    X = np.array(X)
    y = np.array(y)
    Xb = np.hstack([X, np.ones((len(X), 1))])
    lam = 1e-2 * len(X)
    w = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)
    pred = np.sign(Xb @ w)
    print(f'trained on {len(X)} patches from {len(metas)} photos; '
          f'train accuracy {100 * (pred == y).mean():.1f}%')
    return w


def seam_extract(img_rgb, w, max_step=20, lam=0.004):
    """Score image via one correlation, boundary via a DP seam."""
    g = gray_norm(img_rgb)
    K = w[:256].reshape(FSZ, FSZ)
    h = FSZ // 2
    gp = np.pad(g, h, mode='reflect')      # zero-padding would fabricate
    score = fftconvolve(gp, K[::-1, ::-1],  # huge border responses that
                        mode='same')[h:-h, h:-h] + w[256]  # capture seams
    # per-photo polarity: the learned template is bright-sky-over-dark-
    # terrain; snow ridges brighter than the sky flip its sign. Decide
    # from the image itself (top band = sky) rather than |score|, which
    # would also promote terrain-interior texture the template nulls
    if np.median(g[:g.shape[0] // 6]) < np.median(g[g.shape[0] // 2:]):
        score = -score
    cost = 1.0 - (score - score.min()) / (np.ptp(score) + 1e-9)
    # gentle topmost-edge prior: among comparable transitions the
    # skyline is the highest one, not the strongest terrain edge
    cost = cost + 0.30 * (np.arange(cost.shape[0], dtype=float)
                          / cost.shape[0])[:, None]
    H, W = cost.shape
    steps = np.arange(-max_step, max_step + 1)
    pen = lam * np.abs(steps)
    D = cost[:, 0].copy()
    back = np.zeros((H, W), dtype=np.int16)
    for c in range(1, W):
        M = np.full((steps.size, H), np.inf)
        for i, dr in enumerate(steps):
            src = np.roll(D, -dr)
            if dr > 0:
                src[-dr:] = np.inf
            elif dr < 0:
                src[:-dr] = np.inf
            M[i] = src + pen[i]
        ib = np.argmin(M, axis=0)
        D = M[ib, np.arange(H)] + cost[:, c]
        back[:, c] = steps[ib]
    r = int(np.argmin(D))
    rows = np.zeros(W)
    for c in range(W - 1, -1, -1):
        rows[c] = r
        r = int(np.clip(r + back[r, c], 0, H - 1))
    return rows


def to_grid(rows, f_px, W, H):
    u = np.arange(W) - (W - 1) / 2
    az_rel = np.degrees(np.arctan2(u, f_px))
    el_pt = np.arctan2((H - 1) / 2 - rows, np.hypot(u, f_px))
    el = np.zeros(AZ.size)
    wt = np.zeros(AZ.size)
    m = (AZ >= az_rel.min()) & (AZ <= az_rel.max())
    el[m] = np.interp(AZ[m], az_rel, el_pt)
    wt[m] = 1.0
    return el, wt


HYPS = [('default', dict()), ('loose', dict(nsigma=2.5, m_sustain=5)),
        ('strict', dict(nsigma=6.0, m_sustain=12)),
        ('deep', dict(search_frac=0.95))]
HEADING_SIGMA_DEG, PITCH_SIGMA_DEG = 1.0, 0.5


def audit_photo(meta, w_svm):
    v = open(meta).read().split('\n')
    f_px, lat_gt, lon_gt = float(v[0]), float(v[1]), float(v[2])
    W, H = int(v[4]), int(v[5])
    img = np.asarray(Image.open(meta[:-4]).convert('RGB'),
                     dtype=np.float32) / 255.0
    el_ref, wt_ref = to_grid(mask_rows(meta).astype(float), f_px, W, H)

    obs = []
    for hname, kw in HYPS:
        rows, _ = extract.skyline_seam(img, **kw)
        obs.append((hname,) + to_grid(rows, f_px, W, H))
    obs.append(('svm',) + to_grid(seam_extract(img, w_svm), f_px, W, H))

    ensure_tiles(lat_gt, lon_gt)
    cm = S.CMarcher(DIR3, (lat_gt - .7, lat_gt + .7),
                    (lon_gt - .9, lon_gt + .9), d_min=1000.)
    mlat, mlon = S.meters_per_degree(lat_gt)

    def z_at(la, lo):
        y = int(round((cm.lat_nw - la) / cm.dpp))
        x = int(round((lo - cm.lon_nw) / cm.dpp))
        return float(cm.mosaic[np.clip(y, 0, cm.mosaic.shape[0] - 1),
                               np.clip(x, 0, cm.mosaic.shape[1] - 1)]) + 2.

    def skyl(dn, de):
        la, lo = lat_gt + dn / mlat, lon_gt + de / mlon
        el, _ = cm.skyline(la, lo, z_at(la, lo), AZ)
        return el

    el_gt = skyl(0.0, 0.0)
    _, s_true, b_true = fast_photo_cost(
        el_ref, wt_ref, el_gt, range(-1800, 1800, 4),
        np.arange(-0.100, 0.1001, 0.010))
    s_c = s_true + int(round(rng.normal(0, HEADING_SIGMA_DEG) / 0.1))
    b_c = b_true + rng.normal(0, np.radians(PITCH_SIGMA_DEG))
    shifts = range(s_c - 20, s_c + 21, 2)
    betas = np.arange(b_c - 0.0175, b_c + 0.0176, 0.0035)

    step0 = 250.0
    g = np.arange(-BOX / 2, BOX / 2 + 1, step0)
    ccs = np.zeros((len(obs), g.size, g.size))
    for i, dn in enumerate(g):
        for j, de in enumerate(g):
            el = skyl(dn, de)
            for hh, (hname, eo, wt) in enumerate(obs):
                ccs[hh, i, j] = fast_photo_cost(eo, wt, el, shifts,
                                                betas)[0]
    out = []
    for hh, (hname, eo, wt) in enumerate(obs):
        cc = ccs[hh]
        i, j = np.unravel_index(np.argmin(cc), cc.shape)
        out.append(dict(h=hname, dn=float(g[i]), de=float(g[j]),
                        err=float(np.hypot(g[i], g[j])),
                        margin=float(basin_margin(cc, g,
                                                  min_sep=4 * step0))))
    return out


if __name__ == '__main__':
    metas = sorted(glob.glob(os.path.join(CH1, '*', '*.png.txt')))
    metas = [m for m in metas if os.path.exists(m[:-8] + '-mask.png')]
    train_m, test_m = metas[0::2], metas[1::2]
    print(f'{len(train_m)} train / {len(test_m)} test')
    w = train(train_m)
    np.savez(os.path.join(OUT, 'e4m_svm.npz'), w=w)

    # ---- extractor accuracy on the held-out half, vs the masks
    def acc(rows_fn, n=40):
        d = []
        for meta in test_m[:n]:
            img = np.asarray(Image.open(meta[:-4]).convert('RGB'),
                             dtype=np.float32) / 255.0
            r = rows_fn(img)
            d.append(np.median(np.abs(r - mask_rows(meta))))
        return float(np.median(d))
    print(f'median |row error| on {min(40, len(test_m))} test photos: '
          f'seam-family {acc(lambda im: seam_extract(im, w)):.1f} px, '
          f'continuation-family '
          f'{acc(lambda im: extract.skyline_seam(im)[0]):.1f} px',
          flush=True)

    if len(sys.argv) > 1:
        test_m = test_m[:int(sys.argv[1])]
    csvf = open(os.path.join(OUT, 'e4m_diverse.csv'), 'w', buffering=1)
    names = [h for h, _ in HYPS] + ['svm']
    csvf.write('photo,' + ','.join(f'err_{h},margin_{h},dn_{h},de_{h}'
                                   for h in names) + '\n')
    t0 = time.time()
    for n, meta in enumerate(test_m):
        name = os.path.basename(meta)[:-8]
        try:
            res = audit_photo(meta, w)
        except Exception as e:
            print(f'{name} ERROR: {e}', flush=True)
            continue
        csvf.write(name + ',' + ','.join(
            f"{r['err']:.0f},{r['margin']:.3f},{r['dn']:.0f},{r['de']:.0f}"
            for r in res) + '\n')
        print(f'[{n+1}/{len(test_m)} {(time.time()-t0)/60:.0f}min] {name}: '
              + '  '.join(f"{r['h']} {r['err']:.0f}" for r in res),
              flush=True)
    print('done -> out/e4m_diverse.csv')
