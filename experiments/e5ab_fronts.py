#!/usr/bin/env python3
"""E5ab companion: the segmenter and detector outputs that fed the
terrace trio pan, frame by frame.

Top row: eWaSR class overlay (red = obstacle/land, blue = water,
yellow tint = sky) on the full frame, with the extractor boundary
skyfix actually used (first non-sky below the first sky run — the
awning rule).
Bottom row: horizon-band crop with the eWaSR boundary and the seam
detector side by side, where their disagreement is the visible
difference between the two pan solutions (measured ~100 m apart).

Run:  python3 e5ab_fronts.py     (writes out/e5t/PAN_trio_fronts.png)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extract
import skyfix as SF

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      'celestial-navigation', 'peakfinder')
FRAMES = [('PF_20260823_151613', '218.6 deg'),
          ('PF_20260823_151556', '232.1 deg (awning)'),
          ('PF_20260824_122139', '297.0 deg (clean)')]


def main():
    SF.EXTRACTOR = 'ewasr'
    fig, axes = plt.subplots(2, 3, figsize=(16, 13),
                             gridspec_kw=dict(height_ratios=[1.5, 1]))
    for k, (pid, note) in enumerate(FRAMES):
        img = extract.load_image(os.path.join(PHOTOS, pid + '.jpg'))
        H, W, _ = img.shape
        rows_ew, conf_ew = SF.extract_boundary(img)
        cls = SF._EWASR.predict(img)
        seam, conf_s = extract.skyline_seam(img)

        ax = axes[0, k]
        ax.imshow(img)
        over = np.zeros((H, W, 4))
        over[cls == 0] = (1.0, 0.2, 0.2, 0.40)
        over[cls == 1] = (0.0, 0.5, 1.0, 0.35)
        over[cls == 2] = (1.0, 1.0, 0.2, 0.12)
        ax.imshow(over)
        r = np.where(conf_ew > 0, rows_ew, np.nan)
        ax.plot(np.arange(W), r, color='#1c1c1e', lw=1.6,
                label='eWaSR boundary (used by the pan)')
        ax.set_title(f'{pid}\n{note} — eWaSR classes', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='lower left', fontsize=8)

        ax = axes[1, k]
        band = np.where(np.isfinite(r), r, np.nanmedian(r))
        c = int(np.nanmedian(band))
        lo, hi = max(c - int(0.12 * H), 0), min(c + int(0.10 * H), H)
        ax.imshow(img[lo:hi])
        ax.plot(np.arange(W), r - lo, color='#ffd60a', lw=1.8,
                label='eWaSR')
        ax.plot(np.arange(W), seam - lo, color='#ff3b30', lw=1.2,
                ls='--', label='seam detector')
        d = np.abs(r - seam)
        ax.set_title(f'boundary band — median |eWaSR-seam| '
                     f'{np.nanmedian(d):.0f} px', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylim(hi - lo, 0)
        ax.legend(loc='lower left', fontsize=8)

    fig.suptitle('the fronts that fed the terrace trio pan', fontsize=12)
    fig.tight_layout()
    p = os.path.join(HERE, 'out', 'e5t', 'PAN_trio_fronts.png')
    fig.savefig(p, dpi=100)
    print(p)


if __name__ == '__main__':
    main()
