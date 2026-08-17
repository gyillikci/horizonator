#!/usr/bin/env python3
"""E4y: depth layers on the photograph, and two extractors side by side.

Two things the field data asked for, in one picture per sighting.

DEPTH LAYERS. The matcher keeps only the topmost sky boundary, but the
DEM sees an island at 6 km in front of a coast at 40 km, and the near
layer is what carries position (E4y: a 500 m move swings the far coast
12 mrad and the island 81). skyline.visible_layers returns every
visible crest per azimuth; this draws them on the frame, so whether a
predicted layer lands on the island in the photograph is a matter of
looking rather than arguing.

EXTRACTORS. The mountain seam detector is one way to find the
boundary; the learned patch template + DP seam from E4m (trained on
CH1's even half, the approach Ahmad's skyline work uses) is another.
Both are drawn, so their disagreement is visible where it matters —
haze, ships, low coast.

Run:  python3 e4y_layers.py ID [ID ...]    (writes out/e4y/<id>.png)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extract
import skyline as S
import skyfix as SF
from e4m_diverse import seam_extract

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'e4y')
PHOTOS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      'celestial-navigation', 'theodolite')
INDEX = os.path.join(HERE, 'out', 'theodolite', 'index.json')
DEM = os.path.expanduser('~/.horizonator/DEMs_SRTM3')
COL = ['#34c759', '#ff9f0a', '#0a84ff']       # near -> far


def rows_of(el, f_px, H, u, pitch, roll):
    """Elevation angles -> image rows, inverting skyfix.observation."""
    v = np.tan(el) * np.hypot(u, f_px) - np.radians(roll) * u
    return (H - 1) / 2 - v


def render(sid, s, w_svm):
    e, a = s['exif'], s['attitude']
    img = extract.load_image(os.path.join(PHOTOS, s['raw']))
    H, W, _ = img.shape
    fov = a['fov_deg']
    f_px = (W / 2) / np.tan(np.radians(fov) / 2)
    z = max(e.get('alt_m') or 5.0, 2.0)
    pitch = a.get('pitch_deg') or 0.0
    roll = a.get('roll_deg') or 0.0

    rows_seam, _ = extract.skyline_seam(img)
    rows_svm = seam_extract(img, w_svm)

    dem = S.Dem(DEM)
    u = np.arange(W) - (W - 1) / 2
    az = a['heading_deg'] + np.degrees(np.arctan2(u, f_px))
    el_l, rng_l = S.visible_layers(dem, e['lat'], e['lon'], z, az)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10),
                             gridspec_kw=dict(height_ratios=[1.15, 1]))
    ax = axes[0]
    ax.imshow(img)
    ax.plot(np.arange(W), rows_seam, color='#ff3b30', lw=1.3,
            label='extracted: seam detector')
    ax.plot(np.arange(W), rows_svm, color='#bf5af2', lw=1.3, ls='-',
            alpha=0.9, label='extracted: learned template + DP seam')
    for k in range(el_l.shape[0]):
        if not np.isfinite(el_l[k]).any():
            continue
        r = rows_of(el_l[k] - np.radians(pitch), f_px, H, u, pitch, roll)
        med = np.nanmedian(rng_l[k]) / 1000.0
        ax.plot(np.arange(W), r, color=COL[k % len(COL)], lw=1.6,
                ls='--', alpha=0.95,
                label=f'DEM layer {k}: {med:.1f} km')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.set_title(f'{sid}   fov {fov:.1f} deg   heading '
                 f'{a["heading_deg"]:.1f} deg', fontsize=10)

    ax = axes[1]
    def prof(rows):
        v = (H - 1) / 2 - rows
        cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        ur, vr = u * cr - v * sr, u * sr + v * cr
        return (np.degrees(np.arctan2(ur, f_px)) + a['heading_deg'],
                np.arctan2(vr, np.hypot(ur, f_px))
                + np.radians(pitch))
    for rows, col, lab in ((rows_seam, '#ff3b30', 'seam detector'),
                           (rows_svm, '#bf5af2', 'learned template')):
        azp, elp = prof(rows)
        ax.plot(azp, elp * 1e3, color=col, lw=1.0, label=lab)
    for k in range(el_l.shape[0]):
        if not np.isfinite(el_l[k]).any():
            continue
        ax.plot(az, el_l[k] * 1e3, color=COL[k % len(COL)], lw=1.2,
                ls='--',
                label=f'DEM layer {k} ({np.nanmedian(rng_l[k])/1000:.1f} km)')
    ax.axhline(-S.horizon_dip_rad(z) * 1e3, color='#8e8e93', ls=':',
               lw=1, label='sea horizon')
    ax.set_xlabel('azimuth (deg true)')
    ax.set_ylabel('elevation (mrad)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'{sid}.png')
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    d = np.abs(rows_seam - rows_svm)
    print(f'{sid}: extractors differ by a median {np.median(d):.0f} px '
          f'(p90 {np.percentile(d, 90):.0f}); '
          f'{np.isfinite(el_l[:-1]).any(axis=0).mean()*100:.0f}% of '
          f'columns have a nearer layer than the sky boundary -> {p}')
    return p


if __name__ == '__main__':
    with open(INDEX) as fh:
        idx = {s['id']: s for s in json.load(fh)['sightings']}
    w = np.load(os.path.join(HERE, 'out', 'e4m_svm.npz'))['w']
    for sid in (sys.argv[1:] or ['OREJ1026', 'KWHC9160']):
        if sid in idx:
            render(sid, idx[sid], w)
        else:
            print(f'{sid}: not in the curated index')
