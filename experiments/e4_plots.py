#!/usr/bin/env python3
"""Figure for E4 (run after e4_real.py): observed (hand-digitized) vs
predicted skylines, and the FOV-position coupling over the cost surface."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')

d = np.load(os.path.join(OUT, 'e4_result.npz'))
R = json.load(open(os.path.join(OUT, 'e4_result.json')))
az, w = d['az'], d['w']
sh = int(round(d['shift_deg'] / 0.1))
m = w > 0

fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2),
                        gridspec_kw=dict(width_ratios=[1.35, 1]))
fig.suptitle('E4 — real photo, Gulf of Akbük, 25 km² box '
             '(hand-digitized skyline, GPS used only to center the box)',
             fontsize=12)

# (a) skyline overlay in the observed sector
ax = axs[0]
eo = np.roll(d['el_obs'], sh)     # apply the co-estimated heading offset
mm = np.roll(m, sh)
ax.plot(az[mm], eo[mm] * 1e3, color='#111111', lw=2,
        label='observed (hand-digitized, shifted %.1f°)' % d['shift_deg'])
ax.plot(az[mm], d['el_est'][mm] * 1e3, color='#0072b2', lw=1.4,
        label='predicted at estimate')
ax.plot(az[mm], d['el_gps'][mm] * 1e3, color='#d55e00', lw=1.4, ls='--',
        label='predicted at GPS position')
ax.set_xlabel('azimuth (deg true)')
ax.set_ylabel('skyline elevation (mrad)')
ax.set_title('(a) observed vs predicted skyline (d < 1 km masked)',
             fontsize=10)
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25, lw=0.5)

# (b) cost surface + the FOV-position slide
ax = axs[1]
g = d['g'] / 1e3
c = np.sqrt(2 * d['cc']) * 1e3
im = ax.pcolormesh(g, g, c, cmap='Blues', shading='nearest')
fovs = sorted(R['fov_estimates'], key=float)
xs = [R['fov_estimates'][f]['de'] / 1e3 for f in fovs]
ys = [R['fov_estimates'][f]['dn'] / 1e3 for f in fovs]
ax.plot(xs, ys, '-', color='#111111', lw=1)
for f, x, y in zip(fovs, xs, ys):
    ax.plot(x, y, 'o', color='#0072b2', ms=7)
    ax.annotate(f'{f}°', (x, y), textcoords='offset points',
                xytext=(6, 4), fontsize=8, color='#111111')
ax.plot(0, 0, 'X', color='#d55e00', ms=13, mec='white', mew=0.8,
        label='GPS truth')
ax.set_xlabel('east offset (km)')
ax.set_ylabel('north offset (km)')
ax.set_title('(b) estimate vs assumed FOV (~130 m/deg slide)', fontsize=10)
ax.legend(frameon=True, framealpha=0.85, edgecolor='none', loc='upper left', fontsize=9)
ax.set_aspect('equal')
cb = fig.colorbar(im, ax=ax, shrink=0.9)
cb.set_label('RMS skyline residual (mrad)')

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'e4_real.png'), dpi=110, bbox_inches='tight')
print('wrote', os.path.join(OUT, 'e4_real.png'))
