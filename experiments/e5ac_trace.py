#!/usr/bin/env python3
"""E5ac: the pan solver's convergence path, drawn on the map.

Reads a skyfix --out JSON carrying search_trace (every improvement
of the running best during the coarse scan, then each refinement
stage winner), and draws each iteration as a numbered dot over a
coastline backdrop, with the GPS truth in its own color.

Run:  python3 e5ac_trace.py out/e5t/PAN_trace.json 36.64095,28.09548
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import skyline as S

HERE = os.path.dirname(os.path.abspath(__file__))
DEM = os.path.expanduser('~/.horizonator/DEMs_SRTM1_WM')


def main():
    jp = sys.argv[1] if len(sys.argv) > 1 else 'out/e5t/PAN_trace.json'
    la0, lo0 = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2
                else '36.64095,28.09548').split(',')]
    with open(os.path.join(HERE, jp) if not os.path.isabs(jp) else jp) as f:
        j = json.load(f)
    tr = j['search_trace']
    mlat = 111132.0
    mlon = 111320.0 * np.cos(np.radians(la0))

    half = 3600.0
    la = np.linspace(la0 - half / mlat, la0 + half / mlat, 340)
    lo = np.linspace(lo0 - half / mlon, lo0 + half / mlon, 340)
    dem = S.Dem(DEM)
    land = np.array([[dem.sample(a, o) > 0.5 for o in lo] for a in la])

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.contourf(lo, la, land, levels=[0.5, 1.5], colors=['#d9cfa8'])
    ax.contour(lo, la, land, levels=[0.5], colors=['#8e8e93'],
               linewidths=0.7)

    xs = [lo0 + t['de'] / mlon for t in tr]
    ys = [la0 + t['dn'] / mlat for t in tr]
    ax.plot(xs, ys, '-', color='#0a84ff', lw=1.2, alpha=0.6, zorder=3)
    ax.scatter(xs, ys, c=np.arange(len(tr)), cmap='viridis', s=90,
               zorder=4, edgecolors='white', linewidths=0.8)
    for k, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(str(k + 1), (x, y), fontsize=9, fontweight='bold',
                    color='#1c1c1e', textcoords='offset points',
                    xytext=(8, 6))
    ax.plot(xs[-1], ys[-1], 'x', color='#ff3b30', ms=16, mew=3.5,
            zorder=5, label=f'kestirim (it{len(tr)})')
    ax.plot(lo0, la0, '*', color='#34c759', ms=24,
            markeredgecolor='#1c1c1e', zorder=5,
            label='gerçek konum (GPS)')

    dn = (j['lat'] - la0) * mlat
    de = (j['lon'] - lo0) * mlon
    ax.set_title(f'Temiz üçlü pan — arama yörüngesi: {len(tr)} '
                 f'iterasyon\nkestirim hatası: Δlat {dn:+.0f} m, '
                 f'Δlon {de:+.0f} m  (margin '
                 f'{j["basin_margin"]:.2f}, rms {j["rms_mrad"]:.1f} '
                 f'mrad, karar: '
                 f'{"kabul" if j["fix_ok"] else "ret"})', fontsize=11)
    ax.set_xlabel('boylam')
    ax.set_ylabel('enlem')
    ax.set_aspect(1 / np.cos(np.radians(la0)))
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.2)
    out = os.path.join(HERE, 'out', 'e5t', 'PAN_trace_map.png')
    fig.tight_layout()
    fig.savefig(out, dpi=115)
    print(out)


if __name__ == '__main__':
    main()
