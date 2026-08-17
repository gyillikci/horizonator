#!/usr/bin/env python3
"""E4x overview: the field set on one page.

Four panels, all measured on the 83 curated Theodolite pairs:

  1  fix error against the range of the terrain that forms the
     silhouette (the E4x finding: near subjects fail, and the knee is
     around 3 km, which is where the range gate now sits)
  2  fix error by field of view, the band comparison
  3  what the gates do — error against basin margin, with the accept
     threshold drawn, showing that the margin does not separate
  4  the uncertainty estimators against actual error: the covariance
     sigma and the jackknife spread

Run:  python3 e4x_overview.py   (writes out/e4x/overview.png)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'e4x')
FG, ACC, BAD, GRID = '#1c1c1e', '#0a84ff', '#ff3b30', '#d1d1d6'


def load():
    p = os.path.join(HERE, 'out', 'e4v_results_all.json')
    with open(p) as f:
        rows = json.load(f)
    rng = {}
    pr = os.path.join(HERE, 'out', 'e4x_ranges.json')
    if os.path.exists(pr):
        with open(pr) as f:
            rng = json.load(f)
    for r in rows:
        r['range_km'] = rng.get(r['id'])
    return rows


def main():
    rows = load()
    err = np.array([r['err_m'] for r in rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0][0]
    v = [(r['range_km'], r['err_m']) for r in rows if r.get('range_km')]
    if v:
        x, y = np.array([a for a, _ in v]), np.array([b for _, b in v])
        ax.scatter(x, y, c=np.where(y < 500, ACC, BAD), s=26,
                   edgecolor='none', alpha=0.85)
        ax.axvline(3.0, color=FG, ls='--', lw=1,
                   label='range gate, 3 km')
        ax.axhline(500, color=GRID, lw=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('range of the terrain forming the silhouette (km)')
        ax.set_ylabel('fix error (m)')
        ax.legend(fontsize=8)
    ax.set_title('near subjects fail', fontsize=10)

    ax = axes[0][1]
    bands = {}
    for r in rows:
        bands.setdefault(round(r['fov'], 1), []).append(r['err_m'])
    ks = sorted(bands)
    ax.bar(range(len(ks)), [np.median(bands[k]) for k in ks],
           color=[ACC if np.median(bands[k]) < 1000 else BAD
                  for k in ks])
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f'{k:.0f}\nn={len(bands[k])}' for k in ks],
                       fontsize=8)
    ax.set_xlabel('field of view (deg)')
    ax.set_ylabel('median fix error (m)')
    ax.set_title('41 deg is the best band', fontsize=10)

    ax = axes[1][0]
    mg = np.array([r['margin'] for r in rows])
    ax.scatter(mg, err, c=np.where(err < 500, ACC, BAD), s=26,
               edgecolor='none', alpha=0.85)
    ax.axvline(0.15, color=FG, ls='--', lw=1, label='accept gate 0.15')
    ax.axhline(500, color=GRID, lw=1)
    ax.set_xscale('symlog', linthresh=0.1)
    ax.set_yscale('log')
    ax.set_xlabel('basin margin')
    ax.set_ylabel('fix error (m)')
    ax.legend(fontsize=8)
    ax.set_title('the margin does not separate good from bad',
                 fontsize=10)

    ax = axes[1][1]
    sg = np.array([np.hypot(r['sigma_n'], r['sigma_e']) for r in rows])
    ax.scatter(sg, err, s=26, color=BAD, edgecolor='none', alpha=0.8,
               label='covariance sigma')
    jp = os.path.join(HERE, 'out', 'e4w_jack.json')
    if os.path.exists(jp):
        with open(jp) as f:
            jk = json.load(f)
        ax.scatter([x[3] for x in jk], [x[1] for x in jk], s=40,
                   color=ACC, edgecolor='none', label='jackknife spread')
    lim = [10, 10000]
    ax.plot(lim, lim, color=FG, lw=1, ls='--', label='honest (1:1)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel('claimed uncertainty (m)')
    ax.set_ylabel('actual error (m)')
    ax.legend(fontsize=8)
    ax.set_title('what the instrument claims vs what it delivers',
                 fontsize=10)

    for a in axes.ravel():
        a.grid(alpha=0.25)
    fig.suptitle(f'Theodolite field set — {len(rows)} solved sightings',
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'overview.png')
    fig.savefig(p, dpi=115)
    print(p)


if __name__ == '__main__':
    main()
