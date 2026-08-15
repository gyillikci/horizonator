#!/usr/bin/env python3
"""Storyboard of the full skyline-matching pipeline, stage by stage, on a
real photo (CH1 alpine benchmark, the ~50 m success case): input image ->
boundary evidence -> extracted skyline -> angular domain -> DEM ->
candidate matching -> cost surface -> fix + NMEA. Writes
out/viz_pipeline.png.

Run:   python3 viz_pipeline.py     (needs the CH1 set + DEM tiles; no GL)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import skyline as S
import extract

CH1 = '/home/user/celestial-navigation/CH1/cvg'
NAME = '2011-10-04_14.26.13_01024'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
DIR3 = os.path.expanduser('~/.horizonator/DEMs_SRTM3')
AZ = np.arange(-180.0, 180.0, 0.1) + 0.05
BETAS = np.arange(-0.100, 0.1001, 0.010)
BOX = 5000.0

# ---------------- metadata + input
v = open(os.path.join(CH1, NAME + '.png.txt')).read().split('\n')
f_px, lat_gt, lon_gt = float(v[0]), float(v[1]), float(v[2])
W, H = int(v[4]), int(v[5])
img = extract.load_image(os.path.join(CH1, NAME + '.png'))

# ---------------- stage 2: boundary evidence (extractor internals)
Hs = int(img.shape[0] * 0.85)
c = np.cumsum(np.vstack([np.zeros((1, img.shape[1], 3)), img[:Hs]]), axis=0)
rr = np.arange(8, Hs)
near = (c[rr - 1] - c[rr - 4]) / 3.0
far = (c[rr - 4] - c[rr - 7]) / 3.0
pred = np.full((Hs, img.shape[1], 3), np.nan)
pred[8:] = 2.0 * near - far
dev = np.linalg.norm(img[:Hs] - pred, axis=2)
dev[:8] = 0.0

# ---------------- stage 3: the seam
rows, conf = extract.skyline_seam(img)

# ---------------- stage 4: angular domain (pinhole)
u = np.arange(img.shape[1]) - (img.shape[1] - 1) / 2
scale = img.shape[1] / W
vv = (img.shape[0] - 1) / 2 - rows
az_rel = np.degrees(np.arctan2(u / scale, f_px))
el_pt = np.arctan2(vv / scale, np.hypot(u / scale, f_px))
el_obs = np.full(AZ.size, 0.0)
wt = np.zeros(AZ.size)
m = (AZ >= az_rel.min()) & (AZ <= az_rel.max())
el_obs[m] = np.interp(AZ[m], az_rel, el_pt)
wt[m] = np.interp(AZ[m], az_rel, conf)
wt = wt / (wt[m].max() + 1e-9)

# ---------------- stages 5-7: DEM, matching, cost surface
cm = S.CMarcher(DIR3, (lat_gt - 0.7, lat_gt + 0.7),
                (lon_gt - 0.9, lon_gt + 0.9), d_min=1000.0)
mlat, mlon = S.meters_per_degree(lat_gt)


def z_at(lat, lon):
    y = int(round((cm.lat_nw - lat) / cm.dpp))
    x = int(round((lon - cm.lon_nw) / cm.dpp))
    return float(cm.mosaic[np.clip(y, 0, cm.mosaic.shape[0] - 1),
                           np.clip(x, 0, cm.mosaic.shape[1] - 1)]) + 2.0


def pcost(el_syn, shifts):
    best = (np.inf, 0, 0.0)
    for s in shifts:
        eo = np.roll(el_obs, s)
        ww = np.roll(wt, s)
        mm = ww > 0
        r = el_syn[mm] - eo[mm]
        wm = ww[mm]
        rb = np.abs(r[None, :] - BETAS[:, None])
        h = np.where(rb <= 3e-3, 0.5 * rb * rb, 3e-3 * (rb - 1.5e-3))
        cvec = (h * wm[None, :]).sum(1) / wm.sum()
        i = int(np.argmin(cvec))
        if cvec[i] < best[0]:
            best = (float(cvec[i]), s, float(BETAS[i]))
    return best


def C(dn, de, shifts):
    la, lo = lat_gt + dn / mlat, lon_gt + de / mlon
    el, _ = cm.skyline(la, lo, z_at(la, lo), AZ)
    return pcost(el, shifts), el


print('computing coarse cost surface (full heading search)...', flush=True)
shifts_all = range(-1800, 1800, 4)
g = np.arange(-BOX / 2, BOX / 2 + 1, 250.0)
cc = np.array([[C(dn, de, shifts_all)[0][0] for de in g] for dn in g])
i, j = np.unravel_index(np.argmin(cc), cc.shape)
dn0, de0 = g[i], g[j]
(c0, s0, b0), _ = C(dn0, de0, shifts_all)
near_s = range(s0 - 30, s0 + 31, 2)
for step in (50.0, 12.5):
    best = (np.inf, dn0, de0)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            cbest, _ = C(dn0 + di * step, de0 + dj * step, near_s)
            if cbest[0] < best[0]:
                best = (cbest[0], dn0 + di * step, de0 + dj * step)
    _, dn0, de0 = best
(cb, sb, bb), el_best = C(dn0, de0, near_s)
(_, _, _), el_far = C(dn0 + 2000.0, de0 + 2000.0, near_s)
err = float(np.hypot(dn0, de0))
print(f'fix: dn {dn0:+.0f} de {de0:+.0f} -> err {err:.0f} m, '
      f'heading offset {sb*0.1:.1f} deg', flush=True)

# ---------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

fig = plt.figure(figsize=(15, 15))
fig.suptitle(f'The skyline-matching pipeline, stage by stage — real photo '
             f'{NAME}.png (CH1 alpine benchmark)', fontsize=13, y=0.99)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.42, wspace=0.18,
                       height_ratios=[1, 1, 0.9, 1])

ax = fig.add_subplot(gs[0, 0])
ax.imshow(img)
ax.axis('off')
ax.set_title('1 · input photograph\n(calibrated focal '
             f'{f_px:.0f} px → FOV {np.degrees(2*np.arctan(W/2/f_px)):.1f}°)',
             fontsize=10)

ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(dev, cmap='Blues', vmax=np.percentile(dev, 99.5))
ax.plot(np.arange(img.shape[1]), rows, color='#d55e00', lw=1.4)
ax.axis('off')
ax.set_title('2 · boundary evidence: deviation from local sky\n'
             'continuation, and the first-sustained-crossing seam (orange)',
             fontsize=10)

ax = fig.add_subplot(gs[1, 0])
ax.imshow(img)
ax.plot(np.arange(img.shape[1]), rows, color='#d55e00', lw=1.6)
ax.axis('off')
ax.set_title('3 · extracted skyline over the photo', fontsize=10)

ax = fig.add_subplot(gs[1, 1])
ax.plot(az_rel, el_pt * 1e3, color='#0072b2', lw=1.6)
ax.set_xlabel('azimuth relative to camera axis (deg)')
ax.set_ylabel('elevation (mrad)')
ax.grid(alpha=0.25, lw=0.5)
ax.set_title('4 · pinhole mapping: pixels → angles\n'
             'θ(az): the 1-D matching representation', fontsize=10)

ax = fig.add_subplot(gs[2, 0])
sub = cm.mosaic[::6, ::6]
ax.imshow(sub, cmap='Greys', origin='upper',
          extent=[cm.lon_nw, cm.lon_nw + cm.mosaic.shape[1] * cm.dpp,
                  cm.lat_nw - cm.mosaic.shape[0] * cm.dpp, cm.lat_nw])
bx = BOX / 2 / mlon
by = BOX / 2 / mlat
ax.add_patch(plt.Rectangle((lon_gt - bx, lat_gt - by), 2 * bx, 2 * by,
                           fill=False, edgecolor='#d55e00', lw=1.6))
ax.set_title('5 · the DEM mosaic and the 5 km search box', fontsize=10)
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')

ax = fig.add_subplot(gs[2, 1])
mm = np.roll(wt, sb) > 0
ax.plot(AZ[mm], (np.roll(el_obs, sb)[mm] + bb) * 1e3, color='#111111',
        lw=2.0, label='observed (aligned)')
ax.plot(AZ[mm], el_best[mm] * 1e3, color='#0072b2', lw=1.4,
        label=f'synthetic at fix ({err:.0f} m from GT)')
ax.plot(AZ[mm], el_far[mm] * 1e3, color='#d55e00', lw=1.2, ls='--',
        label='synthetic 2.8 km away')
ax.set_xlabel('azimuth (deg true)')
ax.set_ylabel('elevation (mrad)')
ax.legend(frameon=False, fontsize=8)
ax.grid(alpha=0.25, lw=0.5)
ax.set_title('6 · matching: observed vs DEM-synthesized skylines',
             fontsize=10)

ax = fig.add_subplot(gs[3, 0])
im = ax.pcolormesh(g / 1e3, g / 1e3, np.sqrt(2 * cc) * 1e3, cmap='Blues',
                   shading='nearest')
ax.plot(0, 0, 'X', color='#d55e00', ms=12, mec='white', mew=0.8,
        label='GPS ground truth')
ax.plot(de0 / 1e3, dn0 / 1e3, 'o', mfc='none', mec='#009e73', mew=2.2,
        ms=13, label='cost minimum (the fix)')
ax.set_xlabel('east offset (km)')
ax.set_ylabel('north offset (km)')
ax.set_aspect('equal')
ax.legend(frameon=True, framealpha=0.85, edgecolor='none', fontsize=8,
          loc='upper left')
cb2 = fig.colorbar(im, ax=ax, shrink=0.85)
cb2.set_label('RMS skyline residual (mrad)')
ax.set_title('7 · match cost over the search box', fontsize=10)

ax = fig.add_subplot(gs[3, 1])
ax.axis('off')
lat_e, lon_e = lat_gt + dn0 / mlat, lon_gt + de0 / mlon
d = int(abs(lat_e))
mmin = (abs(lat_e) - d) * 60
dl = int(abs(lon_e))
mlo = (abs(lon_e) - dl) * 60
body = (f'GPGGA,102613.00,{d:02d}{mmin:07.4f},N,{dl:03d}{mlo:07.4f},E,'
        f'1,08,1.2,{z_at(lat_e, lon_e):.1f},M,0.0,M,,')
cs = 0
for ch in body:
    cs ^= ord(ch)
ax.text(0.02, 0.97, '8 · output', fontsize=11, weight='bold',
        va='top', transform=ax.transAxes)
ax.text(0.02, 0.05, (
    f'fix:      {lat_e:.6f}°N  {lon_e:.6f}°E\n'
    f'truth:    {lat_gt:.6f}°N  {lon_gt:.6f}°E\n'
    f'error:    {err:.0f} m   (5 km box, heading searched 360°)\n'
    f'residual: {np.sqrt(2*cb)*1e3:.1f} mrad\n'
    f'heading:  {(sb*0.1) % 360:.1f}° offset from grid zero\n\n'
    f'NMEA out (→ chartplotter):\n'
    f'${body}*{cs:02X}\n\n'
    f'…and as a GTSAM SkylineFactor into the\n'
    f'iSAM2 graph when underway (skynav.py)'),
    fontsize=9, family='monospace', va='bottom', transform=ax.transAxes)

fig.savefig(os.path.join(OUT, 'viz_pipeline.png'), dpi=110,
            bbox_inches='tight')
print('wrote', os.path.join(OUT, 'viz_pipeline.png'))
