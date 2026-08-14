#!/usr/bin/env python3
"""Figures for E1 (run after e1_closed_loop.py). One figure per site:

  (a) the 360 deg skyline at the box center
  (b) the cost surface over the 1km box, full panorama
  (c) the cost surface with only the 90 deg sector (limited FOV)
  (d) localization error scatter for all configs

The cost surfaces are shown as sqrt(2*cost): the RMS skyline residual in
mrad, a physical quantity.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')

# Okabe-Ito, colorblind-safe; fixed assignment per config, marker too (identity
# is never color-alone)
CONFIGS = [('clean/360', '#0072b2', 'o'),
           ('clean/90',  '#d55e00', 's'),
           ('noise/360', '#009e73', '^'),
           ('bias/360',  '#cc79a7', 'D')]

with open(os.path.join(OUT, 'e1_results.json')) as f:
    results = json.load(f)

for name, res in results.items():
    d = np.load(os.path.join(OUT, f'e1_{name}.npz'))
    idx, az = d['idx'], d['az']
    x = idx * 25.0  # lattice pitch, m
    gt = d['gt']

    fig = plt.figure(figsize=(13, 9.5))
    fig.suptitle(f'E1 closed loop — site {name} '
                 f'({res["site"]["lat"]}, {res["site"]["lon"]}), z=5 m, '
                 f'1 km × 1 km box, 25 m lattice', fontsize=12)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.75, 1.25],
                          hspace=0.3, wspace=0.25)

    # (a) skyline at the box center
    ax = fig.add_subplot(gs[0, :])
    el = d['el_center'] * 1e3
    r = d['r_center']
    ax.fill_between(az, el, el.min() - 0.5, color='#c8d6e5', lw=0)
    ax.plot(az, el, color='#0072b2', lw=1.2)
    ax.set_xlim(-180, 180)
    ax.set_xlabel('azimuth (deg, 0 = N)')
    ax.set_ylabel('skyline elevation (mrad)')
    # land = skyline meaningfully above the sea-horizon dip (the renderer
    # legitimately returns the sea surface itself in open-water azimuths)
    dip = np.sqrt(2 * 5.0 / (6371000.0 / (1 - 0.13)))
    land = d['el_center'] > (-dip + 0.5e-3)
    rl = r[land]
    ax.set_title(f'(a) skyline at box center — land in {land.mean()*100:.0f}% '
                 f'of azimuths, mean land range '
                 f'{np.mean(rl)/1e3:.1f} km', fontsize=10)
    sc = res['site']['sector_center_deg']
    sc = (sc + 180) % 360 - 180
    for lo, hi in [(sc - 45, sc + 45),
                   (sc - 45 + 360, sc + 45 + 360),
                   (sc - 45 - 360, sc + 45 - 360)]:
        if hi > -180 and lo < 180:  # handle wrap at +-180
            ax.axvspan(max(lo, -180), min(hi, 180),
                       color='#d55e00', alpha=0.12, lw=0)
    ax.text(np.clip(sc, -160, 160), ax.get_ylim()[1], '90° sector',
            color='#a04000', ha='center', va='top', fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)

    # (b),(c) cost surfaces
    for k, (key, ttl) in enumerate([('csurf', '(b) cost surface, 360° panorama'),
                                    ('csurf90', '(c) cost surface, 90° sector')]):
        ax = fig.add_subplot(gs[1, k])
        c = np.sqrt(2.0 * d[key]) * 1e3  # RMS residual, mrad
        im = ax.pcolormesh(x, x, c, cmap='Blues', shading='nearest')
        i0, j0 = np.unravel_index(np.argmin(d[key]), d[key].shape)
        ax.plot(gt[0, 1], gt[0, 0], 'X', color='#111111', ms=11,
                label='ground truth')
        ax.plot(x[j0], x[i0], 'o', mfc='none', mec='#d55e00', mew=2.2, ms=12,
                label='cost minimum')
        ax.set_xlabel('east offset (m)')
        ax.set_ylabel('north offset (m)')
        ax.set_title(ttl, fontsize=10)
        ax.set_aspect('equal')
        if k == 0:
            ax.legend(frameon=False, loc='upper left', fontsize=9)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label('RMS skyline residual (mrad)')

    # (d) error scatter, its own figure
    fig2, ax = plt.subplots(figsize=(6.5, 6.2))
    for key, col, mk in CONFIGS:
        rows = np.array(res['res'][key])
        de = rows[:, 3] - rows[:, 1]
        dn = rows[:, 2] - rows[:, 0]
        cep = res['stats'][key]['cep50']
        ax.plot(de, dn, mk, color=col, ms=6, ls='none',
                label=f'{key}  (CEP50 {cep:.0f} m)')
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(cep * np.cos(th), cep * np.sin(th), color=col, lw=1.0,
                alpha=0.6)
    ax.axhline(0, color='#999999', lw=0.6)
    ax.axvline(0, color='#999999', lw=0.6)
    ax.set_xlabel('east error (m)')
    ax.set_ylabel('north error (m)')
    ax.set_title(f'site {name}: localization error, {len(gt)} trials/config',
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25, lw=0.5)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, f'e1_{name}_errors.png'), dpi=110)

    fig.savefig(os.path.join(OUT, f'e1_{name}.png'), dpi=110)
    print('wrote', os.path.join(OUT, f'e1_{name}.png'), 'and _errors.png')
