#!/usr/bin/env python3
"""Diagnostic figures for E4h: per-photo cost landscape over the 5 km box
(GPS truth vs solved fix) and observed-vs-predicted skyline overlays.
Writes out/e4h_viz/{A,B,C,D,joint}.png."""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from e4h_bafa2 import (observation, masked_cost, align, PHOTOS,
                       WATERLINES, EL_WLS, AZ, LAT_GT, LON_GT, Z, BOX,
                       DMIN, DIR3, OUT)
import skyline as S

VIZ = os.path.join(OUT, 'e4h_viz')
os.makedirs(VIZ, exist_ok=True)

SURF = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
BLUE = '#2a78d6'      # observed
ORANGE = '#eb6834'    # predicted at fix / fix marker
SEQ = LinearSegmentedColormap.from_list('seq_blue', [
    '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95',
    '#0d366b'])

cm = S.CMarcher(DIR3, (LAT_GT - .6, LAT_GT + .6),
                (LON_GT - .8, LON_GT + .8), d_min=DMIN)
mlat, mlon = S.meters_per_degree(LAT_GT)
SHIFTS = range(-100, 101, 2)
RES = json.load(open(os.path.join(OUT, 'e4h_results.json')))


def skyl(dn, de):
    el, _ = cm.skyline(LAT_GT + dn / mlat, LON_GT + de / mlon, Z, AZ)
    return el


def coarse_map(obs_list):
    g = np.arange(-BOX / 2, BOX / 2 + 1, 250.0)
    cc = np.array([[sum(masked_cost(skyl(dn, de), eo, w, SHIFTS)
                        for eo, w in obs_list) for de in g] for dn in g])
    return g, cc


def figure(key, obs_list, res, title):
    g, cc = coarse_map(obs_list)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), width_ratios=[1.0, 1.6])
    fig.patch.set_facecolor(SURF)
    for a in (ax, ax2):
        a.set_facecolor(SURF)
        for s in a.spines.values():
            s.set_color(INK2), s.set_linewidth(0.6)
        a.tick_params(colors=INK2, labelsize=8)

    q = cc.max() - cc            # match quality: darker = better
    km = g / 1000.0
    im = ax.imshow(q, origin='lower', cmap=SEQ,
                   extent=[km[0], km[-1], km[0], km[-1]], aspect='equal')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('match quality (max cost − cost)', fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    ax.plot(0, 0, marker='x', ms=10, mew=2.2, color=INK, zorder=5)
    ax.annotate('GPS', (0, 0), textcoords='offset points', xytext=(6, 6),
                fontsize=8, color=INK)
    de0, dn0 = res['de_m'] / 1000, res['dn_m'] / 1000
    ax.plot(de0, dn0, marker='o', ms=9, mfc=ORANGE, mec='white', mew=2,
            zorder=6)
    ax.annotate('fix', (de0, dn0), textcoords='offset points',
                xytext=(6, -12), fontsize=8, color=INK)
    ax.set_xlabel('east (km)', fontsize=9, color=INK)
    ax.set_ylabel('north (km)', fontsize=9, color=INK)
    ax.set_title('cost landscape, 5 km box', fontsize=10, color=INK)

    el_fix = skyl(res['dn_m'], res['de_m'])
    el_gt = skyl(0.0, 0.0)
    offs = []
    for eo, w in obs_list:
        _, s, b = align(el_fix, eo, w, SHIFTS)   # what the solver matched
        offs.append(s * 0.1)
        eo_a, w_a = np.roll(eo, s) + b, np.roll(w, s)
        m = w_a > 0
        a = AZ[m]
        o = np.argsort(a)
        ax2.plot(a[o], eo_a[m][o] * 1e3, color=BLUE, lw=1.8, zorder=4)
        ax2.plot(a[o], el_fix[m][o] * 1e3, color=ORANGE, lw=1.4, zorder=3)
        ax2.plot(a[o], el_gt[m][o] * 1e3, color=INK2, lw=1.0, ls='--',
                 zorder=2)
    ax2.plot([], [], color=BLUE, lw=1.8,
             label='observed (digitized, aligned: '
                   + '/'.join(f'{o:+.1f}°' for o in offs) + ')')
    ax2.plot([], [], color=ORANGE, lw=1.4, label='predicted at fix')
    ax2.plot([], [], color=INK2, lw=1.0, ls='--', label='predicted at GPS')
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax2.grid(alpha=0.25, lw=0.5)
    ax2.set_xlabel('azimuth (deg true)', fontsize=9, color=INK)
    ax2.set_ylabel('elevation (mrad)', fontsize=9, color=INK)
    ax2.set_title('skyline: observation vs DEM prediction', fontsize=10,
                  color=INK)
    fig.suptitle(title, fontsize=11, color=INK, y=1.00)
    fig.tight_layout()
    p = os.path.join(VIZ, key + '.png')
    fig.savefig(p, dpi=120, facecolor=SURF, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p, flush=True)


if __name__ == '__main__':
    obs = {n: observation(pts, f, h, p, r, waterline=WATERLINES[n],
                          el_wl=EL_WLS[n])
           for n, pts, h, p, r, f in PHOTOS}
    meta = {n: (h, f) for n, _, h, p, r, f in PHOTOS}
    for n in obs:
        h, f = meta[n]
        r = RES[n]
        figure(n, [obs[n]], r,
               f"photo {n}  ·  heading {h:.0f}° true, FOV {f:.1f}°  ·  "
               f"err {r['err_m']:.0f} m, rms {r['rms_mrad']:.1f} mrad, "
               f"margin {r['margin']:.2f}")
    r = RES['A+B+C+D']
    figure('joint', list(obs.values()), r,
           f"joint A+B+C+D  ·  err {r['err_m']:.0f} m, "
           f"rms {r['rms_mrad']:.1f} mrad, margin {r['margin']:.2f}")
