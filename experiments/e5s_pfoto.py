#!/usr/bin/env python3
"""E5s: two PeakFinder-labeled frames from Kumlubük, fitted properly.

The photographs carry PeakFinder's own skyline overlay, visibly off
the real ridge. This script puts OUR skyline on the same pixels, two
ways per frame:

  AS-SHOT   the DEM rendered exactly as the EXIF claims the camera
            stood: GPS position, compass GPSImgDirection, camera
            assumed level (pitch = roll = 0, because the file
            carries no inclinometer data at all)
  FITTED    heading offset, pitch and roll co-estimated against the
            extracted silhouette at the SAME GPS position — the fix
            is not in question here, only the attitude

The gap between the two curves IS the answer to why such an overlay
cannot sit on the photograph from metadata alone: the compass and
the missing pitch/roll carry the misfit, not the terrain model.
The known-exact intrinsics make this clean: the app crops the 4:3
sensor to the screen aspect, so f_px = H * 24mm-equiv / 36 = 2688
regardless of the crop, and hFOV follows from the cropped width.

Run:  python3 e5s_pfoto.py IMG [IMG ...]   (writes out/e5s/*.png)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

import extract
import skyline as S
import skyfix as SF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'e5s')
DEM = os.path.expanduser('~/.horizonator/DEMs_SRTM1_WM')
Z = 3.0                      # eye height on the beach, m
ROLLS = np.arange(-4.0, 4.01, 0.5)
# no inclinometer data at all, and a beach snapshot can be aimed well
# above level (these two are: the sky owns half the frame) — search
# the pitch wide open
PITCHES = np.arange(-0.10, 0.3001, 0.002)    # rad
SHIFTS = range(-100, 101)                    # +-10 deg at 0.1 deg


def exif_geo(path):
    ex = Image.open(path).getexif()
    gps = ex.get_ifd(0x8825)
    def dms(v):
        return float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    lat = dms(gps[2]) * (1 if gps[1] == 'N' else -1)
    lon = dms(gps[4]) * (1 if gps[3] == 'E' else -1)
    heading = float(gps[17])
    return lat, lon, heading


def fit(img, f_px, heading, cm, lat, lon):
    """Joint (roll, pitch, heading-shift) search at the fixed GPS
    position, reusing the photo-cost machinery."""
    H, W, _ = img.shape
    fov = np.degrees(2 * np.arctan((W / 2) / f_px))
    el_syn, rng = cm.skyline(lat, lon, Z, SF.AZ)
    el_syn = S.seahorizon_fill(el_syn, Z)
    best = (np.inf,)
    for roll in ROLLS:
        el_obs, w, diag = SF.observation(img, fov, heading, roll, 0.0)
        # the PeakFinder overlay is IN the pixels: a white label pill
        # and its leader line stand tens of mrad above the ridge, and
        # the seam extractor climbs them. A terrain silhouette has no
        # 20-mrad upward plateau a quarter-frame wide, so columns far
        # above a wide rolling median are graphics, not terrain
        m = w > 0
        if m.sum() > 60:
            e = el_obs[m]
            k = (m.sum() // 4) | 1
            pad = np.pad(e, k // 2, mode='edge')
            med = np.median(
                np.lib.stride_tricks.sliding_window_view(pad, k), -1)
            wm = w.copy()
            wm[np.where(m)[0][e - med > 0.02]] = 0.0
        else:
            wm = w
        c, s, b = SF.fast_photo_cost(el_obs, wm, el_syn, SHIFTS,
                                     betas=PITCHES)
        if c < best[0]:
            best = (c, s, b, roll, diag)
    c, s, b, roll, diag = best
    return dict(cost=c, rms_mrad=float(np.sqrt(2 * c) * 1e3),
                dhead=s * 0.1, pitch=b, roll=roll, fov=fov,
                diag=diag, el_syn=el_syn, rng=rng)


def rows_of(el, f_px, H, u, pitch_rad, roll_deg):
    """Elevation angles -> image rows (inverse of observation, the
    e4y small-roll form)."""
    v = np.tan(el - pitch_rad) * np.hypot(u, f_px) \
        - np.radians(roll_deg) * u
    return (H - 1) / 2 - v


def render(path, tag, cm):
    lat, lon, heading = exif_geo(path)
    img = extract.load_image(path)
    H, W, _ = img.shape
    f_px = H * 24.0 / 36.0            # 24mm-equiv on the 36mm side
    r = fit(img, f_px, heading, cm, lat, lon)
    print(f"{tag}: heading {heading:.1f} -> offset {r['dhead']:+.1f} deg, "
          f"pitch {r['pitch']*1e3:+.1f} mrad, roll {r['roll']:+.1f} deg, "
          f"rms {r['rms_mrad']:.2f} mrad  (fov {r['fov']:.1f})")

    u = np.arange(W) - (W - 1) / 2
    fig, ax = plt.subplots(figsize=(9, 16))
    ax.imshow(img)
    for dhead, pitch, roll, col, lab in (
            (0.0, 0.0, 0.0, '#ff3b30',
             'as-shot: EXIF compass, camera assumed level'),
            (r['dhead'], r['pitch'], r['roll'], '#34c759',
             f"fitted: heading {r['dhead']:+.1f} deg, "
             f"pitch {r['pitch']*1e3:+.0f} mrad, "
             f"roll {r['roll']:+.1f} deg")):
        az = (heading + dhead) + np.degrees(np.arctan2(u, f_px))
        el, _ = cm.skyline(lat, lon, Z, az)
        el = S.seahorizon_fill(el, Z)
        rows = rows_of(el, f_px, H, u, pitch, roll)
        ax.plot(np.arange(W), rows, color=col, lw=2.0, alpha=0.9,
                label=lab)
    ex = r['diag']
    ax.plot(np.arange(W), ex['rows'], color='#ffd60a', lw=1.0,
            ls=':', alpha=0.9, label='extracted silhouette')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.set_title(f'{tag}   {lat:.5f}N {lon:.5f}E   compass '
                 f'{heading:.1f} deg', fontsize=10)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'{tag}.png')
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(p)
    return r


if __name__ == '__main__':
    paths = sys.argv[1:]
    lat0, lon0, _ = exif_geo(paths[0])
    cm = S.CMarcher(DEM, (lat0 - 0.6, lat0 + 0.6),
                    (lon0 - 0.8, lon0 + 0.8))
    for k, p in enumerate(paths):
        render(p, f'frame{k}', cm)
