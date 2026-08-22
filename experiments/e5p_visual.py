#!/usr/bin/env python3
"""E5p visual: what the landcover correction does to the field fixes.

Left: per-frame error under raw SRTM1 and the three corrected
variants — water mask only, water mask + tree -8 m, + tree -15 m.
The picture of the verdict: the water mask never hurts, the constant
canopy subtraction is noise.

Right: the SRYK4301 sector, SRTM1 raw vs water-masked vs GLO-30 —
the phantom offshore plateau (raw-SRTM sea noise) is erased by the
mask, the two families now agree... and the fix does not move.
That frame's failure is deeper than the water noise.

Run:  python3 e5p_visual.py            (writes out/e5p/summary.png)
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
OUT = os.path.join(HERE, 'out', 'e5p')

FRAMES = ['MYQR7719', 'KWHC9160', 'PQBC6867',
          'EWAC7374', 'SRYK4301', 'APST5638']
ERR = {  # id -> (raw SRTM1 [e5o], wm, lc8, lc15)   meters
    'MYQR7719': (396, 390, 410, 560),
    'KWHC9160': (436, 429, 463, 678),
    'PQBC6867': (690, 613, 805, 365),
    'EWAC7374': (1181, 1181, 1081, 1081),
    'SRYK4301': (3066, 3066, 3060, 3060),
    'APST5638': (162, 153, 469, 541),
}
VARIANTS = ['raw SRTM1', 'water mask', 'wm + tree -8 m',
            'wm + tree -15 m']
VCOL = ['#8e8e93', '#0a84ff', '#34c759', '#ff9f0a']
SRYK = (38.80562, 26.97368, 10.3, 346.9, 4.1)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8),
                             gridspec_kw=dict(width_ratios=[1.25, 1]))

    ax = axes[0]
    x = np.arange(len(FRAMES))
    wbar = 0.2
    for k, (v, col) in enumerate(zip(VARIANTS, VCOL)):
        e = [ERR[f][k] for f in FRAMES]
        ax.bar(x + (k - 1.5) * wbar, e, wbar, color=col, label=v)
    med = [np.median([ERR[f][k] for f in FRAMES]) for k in range(4)]
    ax.set_xticks(x)
    ax.set_xticklabels(FRAMES, fontsize=8, rotation=20)
    ax.set_ylabel('error (m)')
    ax.set_yscale('log')
    ax.set_title('medians: ' + '  '.join(
        f'{v.split("+")[-1].strip()} {m:.0f} m'
        for v, m in zip(VARIANTS, med)), fontsize=9)
    ax.grid(alpha=0.25, axis='y', which='both')
    ax.legend(fontsize=8)

    ax = axes[1]
    lat, lon, fov, hd, z = SRYK
    az = hd + np.linspace(-fov / 2, fov / 2, 400)
    for store, col, lab in (
            ('DEMs_SRTM1', '#ff3b30', 'SRTM1 raw'),
            ('DEMs_SRTM1_WM', '#0a84ff', 'SRTM1 water-masked'),
            ('DEMs_GLO30_1', '#34c759', 'GLO-30')):
        cm = S.CMarcher(os.path.expanduser('~/.horizonator/' + store),
                        (lat - 0.6, lat + 0.6), (lon - 0.8, lon + 0.8),
                        d_min=1000.0)
        el, _ = cm.skyline(lat, lon, z, az)
        ax.plot(az, el * 1e3, color=col, lw=1.3, label=lab,
                ls='--' if 'GLO' in lab else '-',
                alpha=0.85 if 'raw' in lab else 1.0)
    ax.set_title('SRYK4301 sector: the phantom offshore plateau is\n'
                 'erased, the families agree — and the fix stays 3 km '
                 'off', fontsize=9)
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
