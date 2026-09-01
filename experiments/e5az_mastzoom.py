#!/usr/bin/env python3
"""E5az: which masts actually reach the silhouette, and by how much.

The spike filter looks for one- and two-column excursions. A moored
vessel is wider than that, so this measures the boundary against a
WIDE-window trend (terrain varies slowly; rigging does not) and draws
the mast regions zoomed, because the argument is settled by looking.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extract
import skyfix

photo, out = sys.argv[1], sys.argv[2]
img = extract.load_image(photo)
H, W = img.shape[:2]
skyfix.EXTRACTOR = 'ewasr'
skyfix.MASK_COLS[:] = []
rows, conf = skyfix.extract_boundary(img)
r = np.asarray(rows, float)

k = 201 | 1
pad = np.pad(r, k // 2, mode='edge')
trend = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=-1)
dev = trend - r                       # + = boundary ABOVE the trend
sc = 1.4826 * np.median(np.abs(dev - np.median(dev)))
thr = max(4.0 * sc, 0.008 * H)
hit = dev > thr
xs = np.where(hit)[0]
runs = np.split(xs, np.where(np.diff(xs) > 4)[0] + 1) if xs.size else []
runs = [g for g in runs if len(g) >= 5]

f_px = (W / 2) / np.tan(np.radians(73.74) / 2)
print(f'trend penceresi {k}, eşik {thr:.1f} px')
for g in runs:
    a, b = int(g[0]), int(g[-1])
    pk = float(dev[a:b + 1].max())
    print(f'  sütun {a}-{b} ({len(g)} sütun): trendin {pk:.0f} px üstünde '
          f'= {pk / f_px * 1e3:.1f} mrad')

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
ax = fig.add_subplot(gs[0, :])
ax.imshow(img)
ax.plot(np.arange(W), r, '-', lw=1.4, color='#39ff14', label='extracted boundary')
ax.plot(np.arange(W), trend, '--', lw=1.2, color='#ffd400',
        label=f'{k}-column terrain trend')
for g in runs:
    ax.axvspan(g[0], g[-1], color='#ff2d55', alpha=0.25)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
ax.legend(loc='lower left', fontsize=9, framealpha=0.85)
ax.set_title('boundary vs slow terrain trend — red = rises above it', fontsize=11)

zones = [(760, 900, 'centre sloop'), (1400, 1545, 'right gulet'),
         (0, 140, 'left marina')]
for i, (a, b, nm) in enumerate(zones):
    az_ = fig.add_subplot(gs[1, i])
    top = int(max(0, r[a:b].min() - 45)); bot = int(min(H, r[a:b].max() + 45))
    az_.imshow(img[top:bot, a:b])
    az_.plot(np.arange(b - a), r[a:b] - top, '-', lw=2.0, color='#39ff14')
    az_.plot(np.arange(b - a), trend[a:b] - top, '--', lw=1.4, color='#ffd400')
    over = float((trend[a:b] - r[a:b]).max())
    az_.set_title(f'{nm}: {over:+.0f} px above trend '
                  f'({over / f_px * 1e3:+.1f} mrad)', fontsize=10)
    az_.axis('off')
fig.tight_layout()
fig.savefig(out, dpi=115)
print('->', out)
