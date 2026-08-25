#!/usr/bin/env python3
"""E5al: the crest-deficit sweep, re-run on repaired code.

E5ae calibrated --crest-dh 9 before the level-chain work. E5ak showed
that work had silently regressed the terrace frame, so the sweep is
repeated here on clean code to ask whether the optimum moved.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FR = [('north beach', 36.64935, 28.09263, 200.6, '#2a6fb0'),
      ('terrace', 36.64095, 28.09548, 297.0, '#c0563a'),
      ('Akyaka clean', 37.05190, 28.32292, 196.5, '#4a8c5c')]
DHS = [0, 3, 6, 9, 12, 15]
TAGS = ['NB', 'TW', 'AK']

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
means, alongs = [], []
for ax, key, lab in zip(axes, ('tot', 'along', 'across'),
                        ('total error (m)', 'along-sight (m)',
                         'across-sight (m)')):
    for (name, la, lo, hd, col), tag in zip(FR, TAGS):
        ys = []
        for dh in DHS:
            j = json.load(open(f'out/e5al/{tag}_dh{dh}.json'))
            dn = (j['lat']-la)*111132.
            de = (j['lon']-lo)*111320.*np.cos(np.radians(la))
            h = np.radians(hd)
            ys.append({'tot': np.hypot(dn, de),
                       'along': dn*np.cos(h)+de*np.sin(h),
                       'across': -dn*np.sin(h)+de*np.cos(h)}[key])
        ax.plot(DHS, ys, 'o-', color=col, lw=1.8, ms=5, label=name)
    ax.axvline(9, color='#888', ls='--', lw=1, zorder=0)
    ax.set_xlabel('--crest-dh (m)')
    ax.set_ylabel(lab)
    ax.grid(alpha=0.25)
    if key == 'tot':
        ax.legend(fontsize=9)
    if key == 'across':
        ax.axhline(0, color='#bbb', lw=0.8, zorder=0)
axes[0].set_title('total: minimum still at dh = 9')
axes[1].set_title('along-sight: the crest-deficit bias, falling with dh')
axes[2].set_title('across-sight: what breaks the terrace beyond dh = 9')
fig.suptitle('E5al — crest-deficit sweep on repaired code '
             '(dashed line: the shipped dh = 9)', fontsize=11)
fig.tight_layout()
fig.savefig('out/e5al/dh_curve.png', dpi=115)
print('-> out/e5al/dh_curve.png')
