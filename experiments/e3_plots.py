#!/usr/bin/env python3
"""Figure for E3 (run after e3_scale.py): the L0 cost map over the 100 km
box, per-trial errors vs visible-land fraction, and the timing budget with
a CM5 projection."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skyline as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
CM5_FACTOR = 2.3   # measured-Xeon-core to Cortex-A76 slowdown assumption

with open(os.path.join(OUT, 'e3_results.json')) as f:
    R = json.load(f)
d = np.load(os.path.join(OUT, 'e3_l0map.npz'))
cand, c0, gt0 = d['cand'], d['c0'], d['gt0']

LAT_C, LON_C = 36.60, 26.90
mlat, mlon = S.meters_per_degree(LAT_C)

fig = plt.figure(figsize=(13.5, 6.2))
fig.suptitle('E3 — 100 km × 100 km sea-masked hierarchical search '
             '(Dodecanese / SE Aegean, z=5 m)', fontsize=12)
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.45, wspace=0.22)

# (a) L0 cost map with land mask
ax = fig.add_subplot(gs[:, 0])
dem = S.Dem(os.path.expanduser(os.environ.get('HORIZONATOR_DEMS',
                                              '~/.horizonator/DEMs_SRTM3')))
gk = np.linspace(-50e3, 50e3, 401)
LA = LAT_C + gk[:, None] / mlat + 0 * gk[None, :]
LO = LON_C + 0 * gk[:, None] + gk[None, :] / mlon
land = dem.sample(LA, LO) > 0
ax.imshow(np.where(land, 1.0, np.nan), extent=[-50, 50, -50, 50],
          origin='lower', cmap='Greys', vmin=0, vmax=1.6, zorder=1)
sc = ax.scatter(cand[:, 1] / 1e3, cand[:, 0] / 1e3,
                c=np.sqrt(2 * c0) * 1e3, s=7, marker='s',
                cmap='Blues', zorder=2)
ax.plot(gt0[1] / 1e3, gt0[0] / 1e3, 'X', color='#d55e00', ms=12,
        mec='white', mew=0.8, label='ground truth (trial 0)', zorder=3)
ax.set_xlabel('east (km)')
ax.set_ylabel('north (km)')
ax.set_title('(a) L0 cost over the box, 2 km grid (land gray)', fontsize=10)
ax.legend(frameon=False, loc='upper left', fontsize=9)
ax.set_aspect('equal')
cb = fig.colorbar(sc, ax=ax, shrink=0.9)
cb.set_label('RMS skyline residual (mrad)')

# (b) error vs visible land fraction
ax = fig.add_subplot(gs[0, 1])
err = np.array([t['err'] for t in R['trials']])
lf = np.array([t['land_frac'] for t in R['trials']]) * 100
ax.plot(lf, err, 'o', color='#0072b2', ms=7)
ax.set_xlabel('land in view (% of azimuths)')
ax.set_ylabel('position error (m)')
ax.set_title(f'(b) error vs visibility — {sum(err<100)}/{len(err)} trials '
             f'< 100 m, CEP50 {np.percentile(err,50):.0f} m', fontsize=10)
ax.set_ylim(0, max(err) * 1.15)
ax.grid(alpha=0.25, lw=0.5)

# (c) timing budget, this machine and CM5-projected
ax = fig.add_subplot(gs[1, 1])
tL0 = np.mean([t['t_L0'] for t in R['trials']])
trf = np.mean([t['t_refine'] for t in R['trials']])
rows = [('4-core x86\n(measured)', tL0, trf),
        ('CM5 4×A76\n(projected ×%.1f)' % CM5_FACTOR,
         tL0 * CM5_FACTOR, trf * CM5_FACTOR)]
y = np.arange(len(rows))
ax.barh(y, [r[1] for r in rows], height=0.55, color='#0072b2',
        label='L0: 2336 coarse candidates')
ax.barh(y, [r[2] for r in rows], left=[r[1] for r in rows], height=0.55,
        color='#d55e00', label='refine: 534 evaluations')
for yi, r in zip(y, rows):
    ax.text(r[1] + r[2] + 0.5, yi, f'{r[1]+r[2]:.0f} s', va='center',
            fontsize=10)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('time per fix (s)')
ax.set_xlim(0, (tL0 + trf) * CM5_FACTOR * 1.28)
ax.set_title('(c) time per 100 km-box fix, native marcher', fontsize=10)
ax.legend(frameon=False, fontsize=8, loc='lower right')
ax.grid(alpha=0.25, lw=0.5, axis='x')

fig.savefig(os.path.join(OUT, 'e3_scale.png'), dpi=110,
            bbox_inches='tight')
print('wrote', os.path.join(OUT, 'e3_scale.png'))
