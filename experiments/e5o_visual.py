#!/usr/bin/env python3
"""E5o visual: what the dual-DEM consensus statistic sees.

Left: dem_split vs actual error over the six-frame A/B batch, with
the basin margin beside each point — the picture of why the split is
the missing error predictor (SRYK4301 has a CONFIDENT margin of 1.19
and a 3 km error; the split of 5.3 km is the only statistic in the
toolkit that flags it).

Right: the per-azimuth disagreement the statistic is built from —
SRTM1 vs GLO-30 skylines from the true position for the flagged
frame (SRYK4301) and a clean one (KWHC9160).

Run:  python3 e5o_visual.py            (writes out/e5o/summary.png)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import skyline as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'e5o')
DEM_A = os.path.expanduser('~/.horizonator/DEMs_SRTM1')
DEM_B = os.path.expanduser('~/.horizonator/DEMs_GLO30_1')

# the six-frame A/B batch (e5o_run.log)
ROWS = [  # id, plain err, cons err, split, margin
    ('MYQR7719', 396, 396, 148, 1.45),
    ('KWHC9160', 436, 404, 101, 1.83),
    ('PQBC6867', 690, 1371, 735, 0.59),
    ('EWAC7374', 1181, 1153, 229, 0.25),
    ('SRYK4301', 3066, 3272, 5286, 1.19),
    ('APST5638', 162, 545, 136, 0.52),
]
FRAMES = {  # id -> (lat, lon, fov, heading, z)
    'SRYK4301': (38.80562, 26.97368, 10.3, 346.9, 4.1),
    'KWHC9160': (40.90622, 29.13945, 10.3, 244.3, 2.5),
}


def skyline_pair(lat, lon, z, az):
    # d_min=1000 matches the solver's default, so the panels show the
    # disagreement the solver actually integrates over
    a = S.CMarcher(DEM_A, (lat - 0.6, lat + 0.6), (lon - 0.8, lon + 0.8),
                   d_min=1000.0)
    b = S.CMarcher(DEM_B, (lat - 0.6, lat + 0.6), (lon - 0.8, lon + 0.8),
                   d_min=1000.0)
    ea, _ = a.skyline(lat, lon, z, az)
    eb, _ = b.skyline(lat, lon, z, az)
    return ea, eb


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6),
                             gridspec_kw=dict(width_ratios=[1.1, 1, 1]))

    ax = axes[0]
    split = np.array([r[3] for r in ROWS], float)
    err = np.array([r[1] for r in ROWS], float)
    ax.scatter(split, err, s=70, c='#0a84ff', zorder=3)
    for sid, e, _, sp, mg in ROWS:
        ax.annotate(f'{sid}\nmargin {mg:.2f}', (sp, e), fontsize=7.5,
                    textcoords='offset points', xytext=(7, -3))
    lim = [70, 8000]
    ax.plot(lim, lim, color='#8e8e93', ls=':', lw=1,
            label='split = error')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*lim)
    ax.set_ylim(100, 5000)
    ax.set_xlabel('dem_split (m) — SRTM1 vs GLO-30 fix separation')
    ax.set_ylabel('actual error of the SRTM1 fix (m)')
    r = np.argsort(np.argsort(split))
    q = np.argsort(np.argsort(err))
    ax.set_title(f'the split predicts the error '
                 f'(rank corr {np.corrcoef(r, q)[0, 1]:+.2f}; '
                 f'margin: -0.14)', fontsize=10)
    ax.grid(alpha=0.25, which='both')
    ax.legend(fontsize=8, loc='upper left')

    # SRYK4301's disagreement traces to phantom terrain in the SRTM1
    # tile: raw-SRTM sea noise of 13-20 m up to ~2 km offshore north
    # of the site, which GLO-30 (water-masked) correctly reports as 0
    for ax, sid, note in ((axes[1], 'SRYK4301',
                           'flagged: split 5286 m, error 3066 m'),
                          (axes[2], 'KWHC9160',
                           'clean: split 101 m, error 436 m')):
        lat, lon, fov, hd, z = FRAMES[sid]
        az = hd + np.linspace(-fov / 2, fov / 2, 400)
        ea, eb = skyline_pair(lat, lon, z, az)
        ax.plot(az, ea * 1e3, color='#ff3b30', lw=1.2, label='SRTM1')
        ax.plot(az, eb * 1e3, color='#0a84ff', lw=1.2, label='GLO-30')
        ax.fill_between(az, ea * 1e3, eb * 1e3, color='#ff9f0a',
                        alpha=0.35, label='disagreement')
        d = np.abs(ea - eb)
        ax.set_title(f'{sid} — {note}\nmedian |Δel| '
                     f'{np.median(d)*1e3:.1f} mrad, p90 '
                     f'{np.percentile(d, 90)*1e3:.1f}', fontsize=9)
        ax.set_xlabel('azimuth (deg true)')
        ax.set_ylabel('skyline elevation (mrad)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'summary.png')
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    print(p)


if __name__ == '__main__':
    main()
