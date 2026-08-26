#!/usr/bin/env python3
"""E5ap visual: what --across-water keeps and what it discards.

Left of the argument: a crest seen ACROSS the bay has water below it
in its own column. The near land the observer stands on runs land all
the way to the bottom of the frame. eWaSR already labels all three, so
the test costs nothing and — unlike --dmin — asks the IMAGE rather than
the DEM, needing no prior on the position being solved for.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import extract
import skyfix

photo = sys.argv[1]
out = sys.argv[2]
img = extract.load_image(photo)
H, W = img.shape[:2]

skyfix.EXTRACTOR = 'ewasr'
skyfix.ACROSS_WATER = False
rows_b, conf_b = skyfix.extract_boundary(img)
skyfix.ACROSS_WATER = True
rows_a, conf_a = skyfix.extract_boundary(img)
cls = skyfix._EWASR.predict(img)

fig, axes = plt.subplots(2, 1, figsize=(15, 2 * 15 * H / W + 1.4))
axes[0].imshow(img)
axes[0].imshow(cls, alpha=0.32, interpolation='nearest',
               cmap=ListedColormap(['#d94f2b', '#2f7fd9', '#f2e34a']))
axes[0].set_title('eWaSR segmentation — land (red) / water (blue) / sky (yellow)',
                  fontsize=11)

axes[1].imshow(img)
x = np.arange(W)
kept = np.asarray(conf_a) > 0
had = np.asarray(conf_b) > 0
drop = had & ~kept
axes[1].plot(x[kept], np.asarray(rows_a, float)[kept], '.', ms=2.2,
             color='#39ff14', label=f'across water — used ({int(kept.sum())} cols)')
axes[1].plot(x[drop], np.asarray(rows_b, float)[drop], '.', ms=2.2,
             color='#ff2d55', label=f'near land — discarded ({int(drop.sum())} cols)')
axes[1].set_title('--across-water: only the hills standing above the water '
                  'reach the cost', fontsize=11)
for ax in axes:
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
axes[1].legend(loc='lower left', fontsize=9, markerscale=5, framealpha=0.85)
fig.tight_layout()
fig.savefig(out, dpi=115)
print(f'kept {int(kept.sum())}/{W}, discarded {int(drop.sum())} -> {out}')
