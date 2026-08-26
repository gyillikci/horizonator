#!/usr/bin/env python3
"""E5ar: does the mast filter actually catch the masts?

skyfix.observation() drops one- and two-column spikes that stand far
outside the local angular trend — rigging, lamp posts, masts, things
no DEM carries. This measures the filter on a frame full of moored
sailboats and draws what it removed, because a filter nobody has
watched work is a filter nobody knows works.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extract
import skyfix

photo = sys.argv[1]
out = sys.argv[2]
fov = float(sys.argv[3]) if len(sys.argv) > 3 else 73.74

img = extract.load_image(photo)
H, W = img.shape[:2]
f_px = (W / 2) / np.tan(np.radians(fov) / 2)
skyfix.EXTRACTOR = 'ewasr'
rows, conf = skyfix.extract_boundary(img)

u = np.arange(W) - (W - 1) / 2.0
v = (H - 1) / 2.0 - np.asarray(rows, float)
el = np.arctan2(v, np.hypot(u, f_px))

# the filter, exactly as observation() applies it
k = max(5, el.size // 200) | 1
pad = np.pad(el, k // 2, mode='edge')
med = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=-1)
dev = el - med
scale = 1.4826 * np.median(np.abs(dev)) + 1e-6
thr = max(6.0 * scale, 3e-3)
spike = np.abs(dev) > thr
acts = bool(spike.any() and spike.mean() < 0.25)

had = np.asarray(conf) > 0
print(f'window k={k}, robust scale {scale*1e3:.2f} mrad, threshold {thr*1e3:.2f} mrad')
print(f'spike columns: {int(spike.sum())} of {W} ({spike.mean():.1%}) '
      f'-> filter {"ACTS" if acts else "STANDS DOWN (>25% guard)"}')
print(f'  of those confident: {int((spike & had).sum())}')
print(f'  tallest excursion: {np.max(np.abs(dev))*1e3:.1f} mrad '
      f'({np.max(np.abs(dev))*f_px:.0f} px)')

fig, ax = plt.subplots(figsize=(15, 15 * H / W + 1.2))
ax.imshow(img)
x = np.arange(W)
keep = had & ~(spike if acts else np.zeros(W, bool))
ax.plot(x[keep], np.asarray(rows, float)[keep], '.', ms=2.0, color='#39ff14',
        label=f'terrain silhouette — used ({int(keep.sum())} cols)')
sp = had & spike
ax.plot(x[sp], np.asarray(rows, float)[sp], '.', ms=4.5, color='#ff2d55',
        label=f'masts and rigging — dropped ({int(sp.sum())} cols)')
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis('off')
ax.legend(loc='lower left', fontsize=9, markerscale=4, framealpha=0.85)
ax.set_title(f'mast filter: threshold {thr*1e3:.1f} mrad above the local trend, '
             f'{int(sp.sum())} columns removed', fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=115)
print('->', out)
