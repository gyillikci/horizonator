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

    python3 e5by_search_video.py [bosphorus|bodrum9|bodrum9_soft]
    python3 e5by_search_video.py 38_XKYU3498 [5]

Any tag of data/campaign_frames.json runs with that frame's E5bm
settings; a second argument is the soft compass sigma in degrees.

bodrum9_soft is the same search under E5bz's soft compass prior
(sigma 5 deg, search +-20 deg) instead of the hard +-6 deg window; the
heading panel then shows the prior's penalty and the raw photo cost
apart, and the last frame carries the solver's verdict.
"""

import json
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

CFGS = {
    # E5bi's 54 m configuration: hand skyline, attitude given, +-12 deg
    'bosphorus': dict(
        photo='peakfinder/PF_new_bosphorus1.jpg', lat=41.12590,
        lon=29.07076, z=7.0, heading=47.10, fov=73.7, roll=1.75,
        pitch=1.45, window=12.0, box=5000.0, crest_dh=0.0,
        boundary=os.path.join(HERE, 'data', 'bosphorus1_hand_skyline.npy'),
        extractor='seam', dem='~/.horizonator/DEMs_COP30', name='bosphorus',
        label='Bosphorus · hand skyline · roll 1.75°'),
    # E5bm's uniform configuration: eWaSR, prior attitude, +-6 deg -
    # the frame whose phone compass was ~13 deg out (E5bf)
    'bodrum9': dict(
        photo='peakfinder/PF_new_bodrum9.jpg', lat=37.01992, lon=27.44426,
        z=9.0, heading=167.093, fov=73.7, roll=0.0, pitch=0.0, window=6.0,
        box=6000.0, crest_dh=9.0, boundary=None, extractor='ewasr',
        dem='~/.horizonator/DEMs_COP30', name='bodrum9',
        label='Bodrum 9 · eWaSR skyline · prior attitude',
        verdict=os.path.join(HERE, 'out', 'baseline',
                             '09_PF_new_bodrum9_huber.json')),
}
# E5bz: the same frame, soft compass prior instead of the hard window
CFGS['bodrum9_soft'] = dict(
    CFGS['bodrum9'], name='bodrum9_soft', compass_sigma=5.0,
    label='Bodrum 9 · eWaSR · soft compass σ 5°',
    verdict=os.path.join(HERE, 'out', 'compass',
                         '09_PF_new_bodrum9_soft5.json'))


def frame_cfg(tag, sigma=None):
    """A campaign frame (data/campaign_frames.json) with its E5bm
    flags: eWaSR, prior attitude, crest 9 m, +-6 deg or a soft prior."""
    fr = next(f for f in json.load(open(os.path.join(
        HERE, 'data', 'campaign_frames.json')))['frames'] if f['tag'] == tag)
    name = tag + ('_soft%g' % sigma if sigma else '')
    c = dict(photo=fr['path'], lat=fr['lat'], lon=fr['lon'], z=fr['z'],
             heading=fr['heading'], fov=fr['fov'], roll=0.0, pitch=0.0,
             window=6.0, box=fr['box'], crest_dh=9.0, boundary=None,
             extractor='ewasr', dem='~/.horizonator/DEMs_COP30', name=name,
             label='%s · eWaSR · %s' % (tag[3:], 'soft compass σ %g°' % sigma
                                         if sigma else 'compass ±6°'),
             verdict=os.path.join(HERE, 'out', 'compass', '%s_soft%g.json'
                                  % (tag, sigma)) if sigma else
             os.path.join(HERE, 'out', 'baseline', tag + '_huber.json'))
    if sigma:
        c['compass_sigma'] = float(sigma)
    return c


WIDE = 30.0          # the heading panel shows the cost far beyond the window
OUT = os.path.join(HERE, 'out', 'search_video')
BG, PANEL, GRID, TXT, MUTED = '#0d1117', '#151b23', '#2a3440', '#d9dee4', '#8b96a3'


def run_search(c):
    """skyfix's search, every evaluation recorded."""
    F.BOUNDARY_ROWS = np.load(c['boundary']) if c['boundary'] else None
    F.BOUNDARY_CONF = None
    F.EXTRACTOR = c['extractor']
    img = extract.load_image(celnav(c['photo']))
    el_obs, w, diag = F.observation(img, c['fov'], c['heading'], c['roll'],
                                    c['pitch'])
    if F.USED_EXTRACTOR and F.USED_EXTRACTOR != c['extractor']:
        # say what the skyline really came from
        c['label'] = c['label'].replace('eWaSR', '%s (eWaSR fell back)'
                                        % F.USED_EXTRACTOR)
        c['extractor'] = F.USED_EXTRACTOR
    half = max(0.010, 2.0 * np.radians(0.29))
    betas = np.arange(-half, half * 1.001, half / 5)
    if c.get('compass_sigma'):
        # skyfix --compass-sigma, defaults: sigma_m 3 mrad, n_eff 20
        F.COMPASS_K = 0.5 * (0.1 / c['compass_sigma']) ** 2 * (3e-3) ** 2 / 20.0
        c['window'] = min(max(c['window'], 4 * c['compass_sigma']), 180.0)
    hw = int(round(c['window'] / 0.1))
    shifts = np.arange(-hw, hw + 1, 2)
    mlat, mlon = S.meters_per_degree(c['lat'])
    cm = S.CMarcher(os.path.expanduser(c['dem']), (c['lat'] - .6, c['lat'] + .6),
                    (c['lon'] - .8, c['lon'] + .8), d_min=1000.0)
    need = np.abs(((F.AZ - c['heading'] + 180) % 360) - 180) <= c['fov'] / 2 + WIDE + 2
    evals = []

    def C(dn, de, stage):
        el, r = cm.skyline(c['lat'] + dn / mlat, c['lon'] + de / mlon, c['z'],
                           F.AZ)
        if c['crest_dh']:
            el = el + np.where(np.isfinite(r) & (r > 0),
                               c['crest_dh'] / np.maximum(r, 1000.0), 0.0)
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
    # the heading the photo really wants: the best offset at the TRUE
    # position over the wide range - where the compass should have been
    el_t, r_t = cm.skyline(c['lat'], c['lon'], c['z'], F.AZ)
    if c['crest_dh']:
        el_t = el_t + np.where(np.isfinite(r_t) & (r_t > 0),
                               c['crest_dh'] / np.maximum(r_t, 1000.0), 0.0)
    lags = np.arange(-int(WIDE / 0.1), int(WIDE / 0.1) + 1, 2)
    # the photo's own preference, without the prior's penalty
    ct = F.photo_cost_curve(el_obs, w, el_t, lags, betas) \
        - F.COMPASS_K * lags.astype(float) ** 2
    ctx = dict(el_obs=el_obs, w=w, betas=betas, lags=lags,
               psi_truth=float(lags[int(np.argmin(ct))] * 0.1))
    return img, diag, evals, need, (dn0, de0), mlat, mlon, ctx


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


def main(name='bosphorus', sigma=None):
    c = CFGS[name] if name in CFGS else frame_cfg(name, sigma)
    os.makedirs(OUT, exist_ok=True)
    fdir = os.path.join(OUT, 'frames_' + c['name'])
    os.makedirs(fdir, exist_ok=True)
    img, diag, ev, need, fix, mlat, mlon, ctx = run_search(c)
    print('best heading offset at the true position: %+.1f deg (window '
          '+-%g)' % (ctx['psi_truth'], c['window']), flush=True)
    curves = {}

    def heading_curve(i):
        """cost against heading offset at evaluation i, beta minimised,
        far beyond the window the solver was allowed"""
        if i not in curves:
            full = np.zeros(F.AZ.size)
            full[need] = np.nan_to_num(ev[i]['el'])
            curves[i] = F.photo_cost_curve(ctx['el_obs'], ctx['w'], full,
                                           ctx['lags'], ctx['betas'])
        return curves[i]
    clamp = lambda e: abs(e['shift'] * 0.1) >= c['window'] - 0.15
    soft = bool(c.get('compass_sigma'))
    verdict = json.load(open(c['verdict'])) if c.get('verdict') \
        and os.path.exists(c['verdict']) else None
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
        gs = fig.add_gridspec(3, 2, left=0.045, right=0.985, top=0.92,
                              bottom=0.07, wspace=0.08, hspace=0.42,
                              width_ratios=[1.0, 1.05],
                              height_ratios=[1.0, 0.85, 0.75])
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
            # where the heading the solver chose sits ON the window edge
            cl = [e for e in ev[:k] if clamp(e)]
            if cl:
                ax.scatter([e['de'] for e in cl], [e['dn'] for e in cl], s=7,
                           color='#ff5a4f', zorder=4,
                           label='heading clamped at ±%g°' % c['window'])
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
            vx, vy, vh = best['de'], best['dn'], zoom
            if mode == 'final':
                # frame both the fix and the truth, whatever their distance
                d = math.hypot(best['de'], best['dn'])
                vx, vy = best['de'] / 2, best['dn'] / 2
                vh = max(180.0, 0.75 * d)
                ax.plot([0, best['de']], [0, best['dn']], color='#ffd23f',
                        lw=1.5, ls=':', zorder=8)
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
            if verdict:
                acc = verdict['status'] == 'ok'
                why = [r.split(':')[0] for r in verdict.get('reasons', [])]
                ax.text(0.98, 0.90, 'SOLVER: %s%s' % (
                    'ACCEPTED' if acc else 'REJECTED (inconclusive)',
                    '' if acc else '\n' + '\n'.join('· ' + w for w in why)),
                    transform=ax.transAxes, ha='right', va='top',
                    color='#ffffff', fontsize=10, weight='bold',
                    family='monospace', zorder=9,
                    # an accepted wrong fix is the failure; a rejection
                    # of a wrong fix is the integrity working
                    bbox=dict(fc=('#c0392b' if err >= 500 else '#1f7a4d')
                              if acc else '#2f5f8a', ec='none', pad=4))

        # ---------------- photo with the candidate's skyline
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(img)
        ok = diag['conf'] > 0
        ax.plot(np.arange(W)[ok], obs_rows[ok], color='#ffd23f', lw=1.6,
                label='skyline in the photo (%s)' % (
                    'operator' if c['boundary'] else c['extractor']))
        if mode != 'intro':
            show = best if mode == 'final' else cur
            xs, ys = project(diag, show['el'], AZn, show['shift'], show['beta'],
                             c, H, W)
            ax.plot(xs, ys, color='#45d6ff', lw=1.5,
                    label='DEM skyline from the %s' % (
                        'fix' if mode == 'final' else 'candidate'))
        ax.set_xlim(0, W)
        r0, r1 = np.percentile(obs_rows[ok], [2, 98])
        ax.set_ylim(min(H, r1 + 0.16 * H), max(0, r0 - 0.16 * H))
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title('PHOTO · %s · heading %+.1f° · β %+.1f mrad' % (
            (c['label'],) + ((show['shift'] * 0.1, show['beta'] * 1e3)
                             if mode != 'intro' else (0, 0))),
                     color=TXT, fontsize=10, loc='left', family='monospace')
        ax.legend(loc='lower left', fontsize=8.5, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TXT)

        # ---------------- heading search at the shown candidate
        ax = fig.add_subplot(gs[1, 1])
        ax.set_facecolor(PANEL)
        if mode != 'intro':
            ish = ib if mode == 'final' else k - 1
            hc = heading_curve(ish)
            x = ctx['lags'] * 0.1
            pen = F.COMPASS_K * ctx['lags'].astype(float) ** 2
            if soft:
                ax.axvspan(-c['window'], c['window'], color='#45d6ff',
                           alpha=0.06, label='searched ±%g°' % c['window'])
                ax.plot(x, hc - pen, color='#8b96a3', lw=1.0, ls='--',
                        label='photo cost alone')
                ax.plot(x, hc.min() + pen, color='#c38bff', lw=1.0, ls=':',
                        label='compass penalty (σ %g°)' % c['compass_sigma'])
            else:
                ax.axvspan(-c['window'], c['window'], color='#45d6ff',
                           alpha=0.10, label='allowed: compass ±%g°'
                           % c['window'])
            ax.plot(x, hc, color='#d9dee4', lw=1.3,
                    label='photo cost + penalty' if soft else None)
            ins = np.abs(x) <= c['window'] + 1e-9
            jc = int(np.argmin(np.where(ins, hc, np.inf)))
            jg = int(np.argmin(hc))
            ax.scatter([x[jc]], [hc[jc]], s=55, color='#45d6ff', zorder=5,
                       label='chosen %+.1f°' % x[jc])
            if jg != jc:
                ax.scatter([x[jg]], [hc[jg]], s=55, facecolors='none',
                           edgecolors='#ffffff', lw=1.8, zorder=5,
                           label='best %s %+.1f°' % (
                               'beyond the search' if soft else
                               'without the bound', x[jg]))
            ax.axvline(ctx['psi_truth'], color='#ff5a4f', lw=1.2, ls='--',
                       label='best at the true position %+.1f°' % ctx['psi_truth'])
            if clamp(ev[ish]):
                ax.text(0.02, 0.92, 'CLAMPED: heading on the window edge',
                        transform=ax.transAxes, ha='left', color='#ff5a4f',
                        fontsize=9.5, weight='bold')
            lo_ = np.nanmin(hc - pen) if soft else np.nanmin(hc)
            hi_ = np.nanpercentile(hc, 90)
            ax.set_ylim(lo_ - 0.05 * (hi_ - lo_), hi_ + 0.25 * (hi_ - lo_))
            ax.legend(loc='upper right', fontsize=7.5, facecolor=PANEL,
                      edgecolor=GRID, labelcolor=TXT, ncol=2)
        ax.set_xlim(-WIDE, WIDE)
        ax.set_xlabel('heading offset from the compass (deg)  ·  at this '
                      'candidate, β minimised', color=MUTED, fontsize=9)
        ax.set_ylabel('cost', color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, lw=0.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title('HEADING SEARCH AT THIS CANDIDATE', color=TXT,
                     fontsize=10, loc='left', family='monospace')

        # ---------------- cost
        ax = fig.add_subplot(gs[2, 1])
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
        st0 = c['box'] / 20
        ax.set_xlabel('evaluation  ·  %d coarse (%g m), then 25 + 25 '
                      'refinement (%g m, %g m)' % (ncoarse, st0, st0 / 5,
                                                    st0 / 20),
                      color=MUTED, fontsize=9)
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
    main(sys.argv[1] if len(sys.argv) > 1 else 'bosphorus',
         float(sys.argv[2]) if len(sys.argv) > 2 else None)
