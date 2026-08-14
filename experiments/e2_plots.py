#!/usr/bin/env python3
"""Figure for E2 (run after e2_ablations.py): CEP50 (thick) and CEP95 (thin)
localization error vs each nuisance axis, for both sites."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')

SITECOL = {'A-strait': '#0072b2', 'B-offshore': '#d55e00'}  # Okabe-Ito
MEAN_LAND_RANGE = {'A-strait': 14.5e3, 'B-offshore': 23.8e3}

with open(os.path.join(OUT, 'e2_results.json')) as f:
    R2 = json.load(f)
with open(os.path.join(OUT, 'e1_results.json')) as f:
    R1 = json.load(f)
base = {s: R1[s]['stats']['clean/360'] for s in R2}  # zero-nuisance baseline

fig, axs = plt.subplots(2, 3, figsize=(14, 8.5))
fig.suptitle('E2 ablations — localization error vs nuisance, 1 km box, '
             'z=5 m, 20 trials/point (CEP50 thick, CEP95 thin)', fontsize=12)

def series(ax, site, xs, key, x0=None, ls='-', label=None):
    c = SITECOL[site]
    y50 = [R2[site][key][k]['cep50'] for k in xs]
    y95 = [R2[site][key][k]['cep95'] for k in xs]
    xv = [float(k) for k in xs]
    if x0 is not None:  # prepend the zero-nuisance baseline
        xv = [x0] + xv
        y50 = [base[site]['cep50']] + y50
        y95 = [base[site]['cep95']] + y95
    ax.plot(xv, y50, ls, color=c, lw=2, marker='o', ms=5,
            label=label or site)
    ax.plot(xv, y95, ls, color=c, lw=0.9, alpha=0.55, marker='o', ms=3)
    return xv

# (a) skyline noise
ax = axs[0, 0]
for s in R2:
    series(ax, s, ['0.5', '1', '2', '4'], 'noise', x0=0.0)
ax.set_xlabel('skyline noise sigma (mrad)')
ax.set_ylabel('position error (m)')
ax.set_title('(a) random skyline noise', fontsize=10)
ax.legend(frameon=False, fontsize=9)

# (b) heading bias, naive vs co-estimated
ax = axs[0, 1]
bx = ['0.05', '0.1', '0.2', '0.5', '1', '2']
for s in R2:
    series(ax, s, bx, 'bias', label=f'{s} naive')
    series(ax, s, bx, 'bias_azfit', ls='--', label=f'{s} az co-est.')
    e = np.array([float(k) for k in bx])
    ax.plot(e, MEAN_LAND_RANGE[s] * np.radians(e), ':', color=SITECOL[s],
            lw=1, alpha=0.8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('heading bias (deg)')
ax.set_ylabel('position error (m)')
ax.set_title('(b) heading bias — dotted: predicted d·eps', fontsize=10)
ax.legend(frameon=False, fontsize=8)

# (c) observer height / tide mismatch
ax = axs[0, 2]
for s in R2:
    xs = ['-2', '-1', '+1', '+2']
    xv = [float(k) for k in xs]
    y50 = [R2[s]['dz'][k]['cep50'] for k in xs]
    y95 = [R2[s]['dz'][k]['cep95'] for k in xs]
    xv = xv[:2] + [0.0] + xv[2:]
    y50 = y50[:2] + [base[s]['cep50']] + y50[2:]
    y95 = y95[:2] + [base[s]['cep95']] + y95[2:]
    ax.plot(xv, y50, '-', color=SITECOL[s], lw=2, marker='o', ms=5, label=s)
    ax.plot(xv, y95, '-', color=SITECOL[s], lw=0.9, alpha=0.55, marker='o', ms=3)
ax.set_xlabel('height/tide mismatch dz (m)')
ax.set_ylabel('position error (m)')
ax.set_title('(c) observer-height error', fontsize=10)
ax.legend(frameon=False, fontsize=9)

# (d) FOV
ax = axs[1, 0]
for s in R2:
    series(ax, s, ['40', '90', '180', '360'], 'fov')
ax.set_xlabel('azimuth FOV used (deg)')
ax.set_ylabel('position error (m)')
ax.set_title('(d) field of view', fontsize=10)
ax.legend(frameon=False, fontsize=9)

# (e) cloud truncation
ax = axs[1, 1]
for s in R2:
    series(ax, s, ['10', '25', '50'], 'cloud', x0=0.0)
ax.set_xlabel('top % of land skyline truncated by cloud')
ax.set_ylabel('position error (m)')
ax.set_title('(e) cloud base truncation', fontsize=10)
ax.legend(frameon=False, fontsize=9)

# (f) refraction mismatch + DEM cross-test, grouped bars
ax = axs[1, 2]
cats = ["k'=0.10", "k'=0.16", "k'=0.20", '1" obs\nvs 3" map']
w = 0.38
for off, s in zip((-w / 2, w / 2), R2):
    vals = [R2[s]['refraction']['0.1']['cep50'],
            R2[s]['refraction']['0.16']['cep50'],
            R2[s]['refraction']['0.2']['cep50'],
            R2[s]['dem']['SRTM1obs']['cep50']]
    v95 = [R2[s]['refraction']['0.1']['cep95'],
           R2[s]['refraction']['0.16']['cep95'],
           R2[s]['refraction']['0.2']['cep95'],
           R2[s]['dem']['SRTM1obs']['cep95']]
    x = np.arange(len(cats)) + off
    ax.bar(x, vals, w * 0.92, color=SITECOL[s], label=s)
    ax.plot(x, v95, 'o', color=SITECOL[s], ms=4, mfc='none')
    ax.axhline(base[s]['cep50'], color=SITECOL[s], lw=0.9, ls=':', alpha=0.8)
ax.set_xticks(np.arange(len(cats)))
ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel('position error (m)')
ax.set_title('(f) refraction mismatch & DEM source\n'
             '(bars CEP50, rings CEP95, dotted baseline)', fontsize=10)
ax.legend(frameon=False, fontsize=9)

for ax in axs.flat:
    ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'e2_ablations.png'), dpi=110)
print('wrote', os.path.join(OUT, 'e2_ablations.png'))
