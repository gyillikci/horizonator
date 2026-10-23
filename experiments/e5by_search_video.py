#!/usr/bin/env python3
"""E5by: a video of the skyline fix searching the topography.

Replays skyfix's own search on one real photograph - here the Bosphorus
frame with the operator's hand-drawn skyline and the E5bi attitude
(roll 1.75, pitch 1.45, +-12 deg heading window), the 54 m fix - and
records every candidate it evaluates: the 21 x 21 coarse grid over the
5 km box, then the two 5 x 5 refinements (50 m, 12.5 m). Each frame:

  TOPOGRAPHY  hillshaded GLO-30 around the box, the sea, the box, the
              cells evaluated so far coloured by match cost, the current
              candidate with its camera view cone (at the heading the
              solver chose there), the running best, the EXIF truth;
              the refinement stages zoom in;
  PHOTO       the photograph with the operator's skyline (yellow) and
              the skyline the DEM predicts FROM THE CURRENT CANDIDATE
              (cyan), projected back through the camera with that
              candidate's heading offset, beta and the frame's roll;
  COST        every evaluation's cost and the running best.

The search is reproduced with skyfix's own functions and parameters;
the script checks that its final fix equals the solver's.

    python3 e5by_search_video.py
"""

import math
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.patches import Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skyfix as F  # noqa: E402
import skyline as S  # noqa: E402
import extract  # noqa: E402
from assets import celnav  # noqa: E402

CFG = dict(photo='peakfinder/PF_new_bosphorus1.jpg', lat=41.12590,
           lon=29.07076, z=7.0, heading=47.10, fov=73.7, roll=1.75,
           pitch=1.45, window=12.0, box=5000.0,
           boundary=os.path.join(HERE, 'data', 'bosphorus1_hand_skyline.npy'),
           dem='~/.horizonator/DEMs_COP30', name='bosphorus')
OUT = os.path.join(HERE, 'out', 'search_video')
BG, PANEL, GRID, TXT, MUTED = '#0d1117', '#151b23', '#2a3440', '#d9dee4', '#8b96a3'


def run_search(c):
    """skyfix's search, every evaluation recorded."""
    F.BOUNDARY_ROWS = np.load(c['boundary'])
    F.BOUNDARY_CONF = None
    img = extract.load_image(celnav(c['photo']))
    el_obs, w, diag = F.observation(img, c['fov'], c['heading'], c['roll'],
                                    c['pitch'])
    half = max(0.010, 2.0 * np.radians(0.29))
    betas = np.arange(-half, half * 1.001, half / 5)
    hw = int(round(c['window'] / 0.1))
    shifts = np.arange(-hw, hw + 1, 2)
    mlat, mlon = S.meters_per_degree(c['lat'])
    cm = S.CMarcher(os.path.expanduser(c['dem']), (c['lat'] - .6, c['lat'] + .6),
                    (c['lon'] - .8, c['lon'] + .8), d_min=1000.0)
    need = np.abs(((F.AZ - c['heading'] + 180) % 360) - 180) <= c['fov'] / 2 + c['window'] + 2
    evals = []

    def C(dn, de, stage):
        el, _ = cm.skyline(c['lat'] + dn / mlat, c['lon'] + de / mlon, c['z'],
                           F.AZ)
        cost, s, b = F.fast_photo_cost(el_obs, w, el, shifts, betas)
        evals.append(dict(dn=float(dn), de=float(de), cost=float(cost),
                          shift=int(s), beta=float(b), stage=stage,
                          el=el[need].astype(np.float32)))
        return cost

    step0 = max(c['box'] / 20, 100.0)
    g = np.arange(-c['box'] / 2, c['box'] / 2 + 1, step0)
    cc = np.array([[C(dn, de, 'coarse') for de in g] for dn in g])
    i, j = np.unravel_index(np.argmin(cc), cc.shape)
    dn0, de0 = g[i], g[j]
    for step in (step0 / 5, step0 / 20):
        best = (np.inf, dn0, de0)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                cst = C(dn0 + di * step, de0 + dj * step, 'refine %.1f m' % step)
                if cst < best[0]:
                    best = (cst, dn0 + di * step, de0 + dj * step)
        _, dn0, de0 = best
    return img, diag, evals, need, (dn0, de0), mlat, mlon


def hillshade(c, mlat, mlon, half=6500.0, step=30.0):
    dem = S.Dem(os.path.expanduser(c['dem']))
    xs = np.arange(-half, half + 1, step)
    E, N = np.meshgrid(xs, xs[::-1])
    h = dem.sample(c['lat'] + N / mlat, c['lon'] + E / mlon).astype(float)
    ls = LightSource(azdeg=315, altdeg=40)
    land = plt.get_cmap('gist_earth')(Normalize(-150, 400)(np.maximum(h, 1)))
    rgb = ls.shade_rgb(land[..., :3], h, vert_exag=2.5, blend_mode='soft',
                       dx=step, dy=step)
    sea = h <= 0.5
    rgb[sea] = np.array([0.10, 0.25, 0.38])
    return rgb, (-half, half, -half, half)


def project(diag, el_syn_need, AZn, shift, beta, c, H, W):
    """Candidate skyline back into image rows, per column."""
    f = (W / 2) / math.tan(math.radians(c['fov']) / 2)
    az_rel = diag['az_rel']
    az = c['heading'] + az_rel + shift * 0.1
    es = np.interp((az - AZn[0]) % 360 + AZn[0], AZn, el_syn_need)
    ur = f * np.tan(np.radians(az_rel))
    vr = np.tan(es - beta - math.radians(c['pitch'])) * np.hypot(ur, f)
    cr, sr = math.cos(math.radians(c['roll'])), math.sin(math.radians(c['roll']))
    u = ur * cr + vr * sr
    v = -ur * sr + vr * cr
    return u + (W - 1) / 2, (H - 1) / 2 - v


def main():
    c = CFG
    os.makedirs(OUT, exist_ok=True)
    fdir = os.path.join(OUT, 'frames_' + c['name'])
    os.makedirs(fdir, exist_ok=True)
    img, diag, ev, need, fix, mlat, mlon = run_search(c)
    print('evaluations: %d, fix dn %+.1f de %+.1f m (%.0f m)'
          % (len(ev), fix[0], fix[1], math.hypot(*fix)), flush=True)
    rgb, ext = hillshade(c, mlat, mlon)
    H, W = img.shape[:2]
    AZn = F.AZ[need]
    cost = np.array([e['cost'] for e in ev])
    lc = np.log10(cost)
    norm = Normalize(lc.min(), np.percentile(lc, 95))
    cmap = plt.get_cmap('magma_r')
    obs_rows = diag['rows']

    # schedule: intro hold, coarse in batches, hold, refinements one by one
    sched = [('intro', 0)] * 12
    ncoarse = sum(e['stage'] == 'coarse' for e in ev)
    for k in range(6, ncoarse + 1, 6):
        sched.append(('eval', k))
    if sched[-1][1] != ncoarse:
        sched.append(('eval', ncoarse))
    sched += [('eval', ncoarse)] * 10
    for k in range(ncoarse + 1, len(ev) + 1):
        sched.append(('eval', k))
    sched += [('final', len(ev))] * 24

    run_best = np.minimum.accumulate(cost)
    for fi, (mode, k) in enumerate(sched):
        k = max(k, 1)
        cur = ev[k - 1]
        ib = int(np.argmin(cost[:k]))
        best = ev[ib]
        stage = cur['stage']
        fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
        gs = fig.add_gridspec(2, 2, left=0.045, right=0.985, top=0.92,
                              bottom=0.07, wspace=0.08, hspace=0.25,
                              width_ratios=[1.0, 1.05], height_ratios=[1.25, 0.8])
        fig.text(0.03, 0.955, 'SKYLINE FIX · SEARCHING THE TOPOGRAPHY',
                 color=TXT, fontsize=15, weight='bold', family='monospace')
        lbl = {'intro': 'photo and search box',
               'final': 'fix found'}.get(mode, stage)
        fig.text(0.985, 0.955, '%s   ·   evaluation %d / %d' % (
            lbl, 0 if mode == 'intro' else k, len(ev)), color=TXT,
            fontsize=12, ha='right', family='monospace')

        # ---------------- topography
        ax = fig.add_subplot(gs[:, 0])
        ax.imshow(rgb, extent=ext, origin='upper', zorder=0)
        b2 = c['box'] / 2
        ax.add_patch(Rectangle((-b2, -b2), 2 * b2, 2 * b2, fill=False,
                               ec='#ffffff', lw=1.2, ls='--', zorder=2))
        if mode != 'intro':
            for e in ev[:k]:
                sz = 250.0 if e['stage'] == 'coarse' else (
                    50.0 if '50' in e['stage'] else 12.5)
                ax.add_patch(Rectangle((e['de'] - sz / 2, e['dn'] - sz / 2),
                                       sz, sz, fc=cmap(norm(math.log10(e['cost']))),
                                       ec='none', zorder=3,
                                       alpha=0.82 if (stage == 'coarse' and mode != 'final')
                                       or e['stage'] != 'coarse' else 0.35))
            # the current candidate's camera view cone
            hd = math.radians(c['heading'] + cur['shift'] * 0.1)
            L = 4200.0
            a1, a2 = hd - math.radians(c['fov'] / 2), hd + math.radians(c['fov'] / 2)
            cone = [(cur['de'], cur['dn']),
                    (cur['de'] + L * math.sin(a1), cur['dn'] + L * math.cos(a1)),
                    (cur['de'] + L * math.sin(a2), cur['dn'] + L * math.cos(a2))]
            ax.add_patch(Polygon(cone, closed=True, fc='#45d6ff', alpha=0.10,
                                 ec='#45d6ff', lw=1, zorder=4))
            ax.scatter([cur['de']], [cur['dn']], s=60, marker='s', fc='none',
                       ec='#45d6ff', lw=2, zorder=6, label='current candidate')
            ax.scatter([best['de']], [best['dn']], s=110, marker='*',
                       color='#ffd23f', ec='#000000', lw=0.6, zorder=7,
                       label='best so far')
        ax.scatter([0], [0], s=130, marker='+', color='#ff5a4f', lw=2.5,
                   zorder=8, label='GPS truth (EXIF)')
        # view: whole box during the coarse stage, zoom during refinement
        if mode == 'intro' or stage == 'coarse':
            vx, vy, vh = 0.0, 0.0, 3600.0
        else:
            zoom = 600.0 if '50' in stage else 180.0
            if mode == 'final':
                zoom = 180.0
            vx, vy, vh = best['de'], best['dn'], zoom
        ax.set_xlim(vx - vh, vx + vh)
        ax.set_ylim(vy - vh, vy + vh)
        ax.set_aspect('equal')
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel('east of the box centre (m)', color=MUTED, fontsize=9)
        ax.set_ylabel('north (m)', color=MUTED, fontsize=9)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title('TOPOGRAPHY · GLO-30 hillshade · cells coloured by match '
                     'cost (bright = good match)', color=TXT, fontsize=10, loc='left',
                     family='monospace')
        ax.legend(loc='lower left', fontsize=8.5, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TXT)
        if mode == 'final':
            err = math.hypot(best['dn'], best['de'])
            ax.text(0.98, 0.97, 'FIX  Δn %+.0f m  Δe %+.0f m  ->  %.0f m from truth'
                    % (best['dn'], best['de'], err), transform=ax.transAxes,
                    ha='right', va='top', color='#0d1117', fontsize=11,
                    weight='bold', bbox=dict(fc='#ffd23f', ec='none', pad=4),
                    zorder=9)

        # ---------------- photo with the candidate's skyline
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(img)
        ok = diag['conf'] > 0
        ax.plot(np.arange(W)[ok], obs_rows[ok], color='#ffd23f', lw=1.6,
                label='skyline in the photo (operator)')
        if mode != 'intro':
            show = best if mode == 'final' else cur
            xs, ys = project(diag, show['el'], AZn, show['shift'], show['beta'],
                             c, H, W)
            ax.plot(xs, ys, color='#45d6ff', lw=1.5,
                    label='DEM skyline from the %s' % (
                        'fix' if mode == 'final' else 'candidate'))
        ax.set_xlim(0, W)
        ax.set_ylim(H * 0.62, H * 0.30)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title('PHOTO · Bosphorus · roll 1.75° · candidate heading %+.1f° '
                     '· β %+.1f mrad' % ((show['shift'] * 0.1, show['beta'] * 1e3)
                                         if mode != 'intro' else (0, 0)),
                     color=TXT, fontsize=10, loc='left', family='monospace')
        ax.legend(loc='lower left', fontsize=8.5, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TXT)

        # ---------------- cost
        ax = fig.add_subplot(gs[1, 1])
        ax.set_facecolor(PANEL)
        if mode != 'intro':
            ax.scatter(np.arange(1, k + 1), cost[:k], s=6,
                       c=[cmap(norm(x)) for x in lc[:k]], zorder=2)
            ax.plot(np.arange(1, k + 1), run_best[:k], color='#ffd23f', lw=1.8,
                    label='best so far')
            ax.axvline(ncoarse + 0.5, color=GRID, lw=1)
        ax.set_xlim(0, len(ev) + 1)
        ax.set_yscale('log')
        ax.set_ylim(cost.min() * 0.7, cost.max() * 1.3)
        ax.set_xlabel('evaluation  ·  441 coarse (250 m), then 25 + 25 '
                      'refinement (50 m, 12.5 m)', color=MUTED, fontsize=9)
        ax.set_ylabel('match cost (Huber)', color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, lw=0.5, which='both')
        for sp in ax.spines.values():
            sp.set_color(GRID)
        if mode != 'intro':
            ax.legend(loc='upper right', fontsize=8.5, facecolor=PANEL,
                      edgecolor=GRID, labelcolor=TXT)
        fig.savefig(os.path.join(fdir, 'f%04d.png' % fi), facecolor=BG)
        plt.close(fig)
        if fi % 30 == 0:
            print('frame %d / %d' % (fi + 1, len(sched)), flush=True)
    mp4 = os.path.join(OUT, 'search_%s.mp4' % c['name'])
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '12',
                    '-i', os.path.join(fdir, 'f%04d.png'), '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-crf', '20', mp4], check=True)
    print('wrote', mp4)


if __name__ == '__main__':
    main()
