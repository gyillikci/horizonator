#!/usr/bin/env python3
"""E5bu figure: 30 m map (E5bt) against a perfect map (E5BT_MAP=perfect),
median error per scenario and variant, online and smoothed."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
a = json.load(open(os.path.join(HERE, 'out', 'tracker', 'summary.json')))
b = json.load(open(os.path.join(HERE, 'out', 'tracker_perfect', 'summary.json')))
scens = list(b)
lab = {'A': 'A skyline', 'D': 'D skyline + 2 trackers', 'E': 'E 2 trackers only'}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, key, ttl in ((axs[0], 'online_median', 'online (what the navigator sees)'),
                     (axs[1], 'smooth_median', 'smoothed (whole track, end of run)')):
    x = np.arange(len(scens))
    w = 0.13
    for i, v in enumerate('ADE'):
        for j, (src, alpha, hatch) in enumerate(((a, 0.45, '//'), (b, 1.0, ''))):
            vals = [src[s][v][key] for s in scens]
            ax.bar(x + (2 * i + j - 2.5) * w, vals, w, alpha=alpha, hatch=hatch,
                   color=['#2E86C1', '#7D3C98', '#D68910'][i],
                   label='%s, %s' % (lab[v], ['30 m map', 'perfect map'][j]))
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in scens], fontsize=9)
    ax.set_title('median position error, ' + ttl, fontsize=10.5)
    ax.grid(alpha=.3, which='both', axis='y')
axs[0].set_ylabel('m')
axs[1].legend(fontsize=7.5, ncol=1, loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'out', 'tracker_perfect', 'map_compare.png'), dpi=110)
print('wrote out/tracker_perfect/map_compare.png')
