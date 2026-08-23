#!/usr/bin/env python3
"""E5w: recovering PeakFinder's DEM from the pixels it was burned into.

The byte-level answer to "how does PeakFinder embed its DEM data":
it does not. The exported JPEG carries standard EXIF (GPS, compass,
lens), two IPTC date blocks and the UserComment 'Labeled by
PeakFinder' — no XMP, no maker payload, no trailer. The rendered
skyline exists only as a 1-3 px dark stroke rasterized into the
image. So the only access road is optical: detect the stroke,
un-rasterize it back into a per-column curve, and then it can be
compared against anything — here, against OUR skyline from the same
DEM family, with the attitude fitted to the terrain silhouette
(E5s machinery: position is the EXIF GPS, not in question).

Stroke detection: the line is darker than its local background by
much more than image noise; a 9x9 median filter estimates the
background, and per column the topmost sustained dark run above the
water region is taken. Label pills and leader lines are dark too —
the same wide-median outlier rule as E5s drops them.

Run:  python3 e5w_pfline.py PF_ID [PF_ID ...]  (writes out/e5w/*)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

import extract
import skyline as S
import skyfix as SF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'e5w')
PHOTOS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      'celestial-navigation', 'peakfinder')
DEM = os.path.expanduser('~/.horizonator/DEMs_SRTM1_WM')
META = {  # id -> (lat, lon, heading, z)
    'PF_20260823_151556': (36.64087, 28.09550, 232.1, 6.0),
    'PF_20260823_151613': (36.64087, 28.09550, 218.6, 6.0),
    'PF_20260823_175647': (36.64935, 28.09263, 198.6, 4.0),
    'PF_20260823_175709': (36.64934, 28.09263, 191.4, 4.0),
}


def stroke_rows(img, seam_rows):
    """Per-column row of the burned-in PeakFinder stroke, NaN where
    absent. Looks in a band around the seam (the stroke rides near
    the true ridge) and above it (where the overlay floats when the
    app's pose is off)."""
    g = np.asarray(img, float).mean(axis=2)
    H, W = g.shape
    bg = median_filter(g, size=9)
    dark = bg - g                       # stroke: locally dark, thin
    rows = np.full(W, np.nan)
    for x in range(W):
        top = 8
        bot = min(int(seam_rows[x]) + 40, H - 1)
        if bot <= top + 8:
            continue
        seg = dark[top:bot, x]
        k = int(np.argmax(seg))
        if seg[k] > 0.06:
            rows[x] = top + k
    # drop label pills / leader lines: no terrain overlay jumps by a
    # quarter frame within a few columns
    ok = np.isfinite(rows)
    if ok.sum() > 60:
        idx = np.where(ok)[0]
        r = rows[idx]
        k = (idx.size // 4) | 1
        pad = np.pad(r, k // 2, mode='edge')
        med = np.median(np.lib.stride_tricks.sliding_window_view(pad, k),
                        -1)
        rows[idx[np.abs(r - med) > 0.04 * H]] = np.nan
    return rows


def run(pid, cm):
    lat, lon, heading, z = META[pid]
    img = extract.load_image(os.path.join(PHOTOS, pid + '.jpg'))
    H, W, _ = img.shape
    f_px = H * 24.0 / 36.0 * (H / 4032 if H > W else 1.0)
    f_px = H * 24.0 / 36.0
    seam, conf = extract.skyline_seam(img)
    pf = stroke_rows(img, seam)

    # attitude from the terrain silhouette at the fixed GPS position
    # (E5s): heading shift x pitch x roll grid
    fov = np.degrees(2 * np.arctan((W / 2) / f_px))
    el_syn_full, _ = cm.skyline(lat, lon, z, SF.AZ)
    el_syn_full = S.seahorizon_fill(el_syn_full, z)
    best = (np.inf,)
    for roll in np.arange(-3, 3.01, 0.5):
        el_obs, w, _ = SF.observation(img, fov, heading, roll, 0.0)
        c, s, b = SF.fast_photo_cost(
            el_obs, w, el_syn_full, range(-100, 101),
            betas=np.arange(-0.10, 0.3001, 0.002))
        if c < best[0]:
            best = (c, s, b, roll)
    c, sft, pitch, roll = best

    u = np.arange(W) - (W - 1) / 2
    az = (heading + sft * 0.1) + np.degrees(np.arctan2(u, f_px))
    el, _ = cm.skyline(lat, lon, z, az)
    el = S.seahorizon_fill(el, z)
    ours = (H - 1) / 2 - (np.tan(el - pitch) * np.hypot(u, f_px)
                          - np.radians(roll) * u)

    # PF stroke vs our DEM curve, in mrad, where both exist
    both = np.isfinite(pf)
    dpx = pf[both] - ours[both]
    print(f'{pid}: stroke recovered on {both.mean()*100:.0f}% of '
          f'columns; PF-line minus our DEM line: median '
          f'{np.median(dpx):+.0f} px = '
          f'{np.median(dpx)/f_px*1e3:+.1f} mrad '
          f'(spread {1.4826*np.median(np.abs(dpx-np.median(dpx))):.0f} px)')

    fig, ax = plt.subplots(figsize=(9, 16))
    ax.imshow(img)
    ax.plot(np.arange(W), pf, '.', color='#ff9f0a', ms=2,
            label='PeakFinder stroke, un-rasterized')
    ax.plot(np.arange(W), ours, color='#34c759', lw=1.6,
            label='our DEM skyline (attitude fitted to silhouette)')
    ax.plot(np.arange(W), seam, color='#ff3b30', lw=0.8, ls=':',
            label='extracted silhouette')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='lower left', fontsize=9)
    ax.set_title(f'{pid}: reading PeakFinder\'s DEM back out of the '
                 f'pixels', fontsize=10)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, pid + '.png')
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    print(p)


if __name__ == '__main__':
    ids = sys.argv[1:] or ['PF_20260823_151556']
    lat, lon = META[ids[0]][:2]
    cm = S.CMarcher(DEM, (lat - 0.6, lat + 0.6), (lon - 0.8, lon + 0.8))
    for pid in ids:
        run(pid, cm)
