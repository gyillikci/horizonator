#!/usr/bin/env python3
"""E4s: Koschmieder haze ranging, assayed — conditional, do not rely.

Contrast decays with range through haze (Koschmieder), so per-azimuth
skyline contrast is in principle a RANGE channel — exactly what the
bearing-heavy skyline cost lacks. Two assays:

  synthetic  the E4c composites paint haze by true range, so the
             inversion is self-consistent: contrast recovers range
             with corr ~0.75, median |err| ~0.75 km over 4-14 km.
             Seductive — and a best case by construction.
  real       15 CH1 photos with GPS truth (true ranges from the
             marcher): only ~2/15 carry the signal (corr < -0.4), the
             genuinely hazy ones (fit visibility 6-23 km). Clear air
             has no gradient to read, and albedo variation (rock vs
             forest vs snow) swamps the contrast cue; Koschmieder's
             uniform-dark-object assumption rarely holds.

Verdict: a conditional auxiliary for hazy days, never a dependable
channel. Kept runnable so the synthetic best case cannot seduce later.

Run:   python3 e4s_haze_assay.py
"""

import os
import sys
import glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import skyline as S
import extract
from skyfix import AZ, fast_photo_cost
from e4e_gate_audit import CH1, DIR3, OUT


def synthetic_assay():
    img = extract.load_image(os.path.join(OUT, 'synth', 'strait2.jpg'))
    rows, conf = extract.skyline_seam(img)
    H, W, _ = img.shape
    cm = S.CMarcher(DIR3, (36.4, 37.6), (26.4, 28.0), d_min=1000.)
    el_t, r_t = cm.skyline(36.9622, 27.2384, 5.0, AZ)
    terr = np.array([0.42, 0.36, 0.28])   # the composite's terrain color
    est, cols = [], []
    for x in range(8, W - 8, 4):
        r0 = int(rows[x])
        if r0 < 14 or r0 > H - 14:
            continue
        below = img[r0 + 4:r0 + 12, x].mean(axis=0)
        above = img[max(r0 - 12, 0):r0 - 4, x].mean(axis=0)
        c = np.linalg.norm(below - above) / (np.linalg.norm(terr - above)
                                             + 1e-9)
        h = np.clip(1 - c, 0.0, 0.879)    # composite: haze = 0.88(1-e^-r/15k)
        est.append(-15000 * np.log(1 - h / 0.88))
        cols.append(x)
    est = np.array(est)
    f = (W / 2) / np.tan(np.radians(71.6) / 2)
    az = 25.0 + np.degrees(np.arctan2(np.array(cols) - (W - 1) / 2, f))
    rt = r_t[((az + 180) % 360 / 0.1).astype(int)]
    m = (rt > 500) & (rt < 39000) & np.isfinite(est)
    print(f'synthetic (self-consistent): n={m.sum()}  '
          f'corr {np.corrcoef(est[m], rt[m])[0, 1]:.2f}  '
          f'median |err| {np.median(np.abs(est[m] - rt[m])) / 1000:.2f} km')


def real_assay(n_photos=15):
    metas = sorted(glob.glob(os.path.join(CH1, '*', '*.png.txt')))
    metas = [m for m in metas
             if os.path.exists(m[:-8] + '-mask.png')][:n_photos]
    good = 0
    for meta in metas:
        v = open(meta).read().split('\n')
        f_px, lat, lon = float(v[0]), float(v[1]), float(v[2])
        W, H = int(v[4]), int(v[5])
        img = np.asarray(Image.open(meta[:-4]).convert('RGB'),
                         dtype=np.float32) / 255.
        mask = np.asarray(Image.open(meta[:-8] + '-mask.png')
                          .convert('L')) > 127
        rows = np.where(mask.any(axis=0), mask.argmax(axis=0), H - 1)
        cm = S.CMarcher(DIR3, (lat - .7, lat + .7), (lon - .9, lon + .9),
                        d_min=1000.)
        y = int(round((cm.lat_nw - lat) / cm.dpp))
        x = int(round((lon - cm.lon_nw) / cm.dpp))
        z = float(cm.mosaic[np.clip(y, 0, cm.mosaic.shape[0] - 1),
                            np.clip(x, 0, cm.mosaic.shape[1] - 1)]) + 2
        el, r = cm.skyline(lat, lon, z, AZ)
        u = np.arange(W) - (W - 1) / 2
        az_rel = np.degrees(np.arctan2(u, f_px))
        el_pt = np.arctan2((H - 1) / 2 - rows, np.hypot(u, f_px))
        eo = np.zeros(AZ.size)
        wt = np.zeros(AZ.size)
        mm = (AZ >= az_rel.min()) & (AZ <= az_rel.max())
        eo[mm] = np.interp(AZ[mm], az_rel, el_pt)
        wt[mm] = 1
        _, s, _ = fast_photo_cost(eo, wt, el, range(-1800, 1800, 2),
                                  np.arange(-.1, .101, .01))
        heading = -s * 0.1
        lncs, rs = [], []
        for xi in range(10, W - 10, 6):
            r0 = int(rows[xi])
            if r0 < 16 or r0 > H - 16:
                continue
            below = img[r0 + 4:r0 + 14, xi].mean(axis=0)
            above = img[max(r0 - 14, 0):r0 - 4, xi].mean(axis=0)
            c = np.linalg.norm(below - above)
            if c < 0.01:
                continue
            azx = heading + np.degrees(np.arctan2(xi - (W - 1) / 2, f_px))
            ri = r[int((azx + 180) % 360 / 0.1)]
            if 500 < ri < 39000:
                lncs.append(np.log(c))
                rs.append(ri)
        if len(rs) > 30:
            cc = np.corrcoef(lncs, rs)[0, 1]
            beta = -np.polyfit(rs, lncs, 1)[0]
            vis = 3.912 / max(beta, 1e-9) / 1000
            ok = cc < -0.4
            good += ok
            print(f'{os.path.basename(meta)[:20]:20s} n={len(rs):3d} '
                  f'corr={cc:+.2f}  visibility {min(vis, 999):4.0f} km'
                  f'{"  <-- signal" if ok else ""}')
    print(f'{good}/{len(metas)} real photos carry the range signal')


if __name__ == '__main__':
    synthetic_assay()
    real_assay()
