#!/usr/bin/env python3
"""E5bx: a replay video of one processed navigation scenario.

Reads a traced E5bt run (E5BT_TRACE=1 writes one record per keyframe)
and renders, per keyframe, a replay-tool style frame:

  CAMERA     a synthetic view from the TRUE pose (wide 90 deg camera,
             vertical exaggerated so the thin skyline band is visible):
             terrain silhouette ray-cast from the truth DEM and shaded
             by range, the sea below the horizon, the tracker windows
             (250 px on a 1920 px sensor) on the hills they hold, with
             the identity the system inferred - green when it is the
             right hill, red when wrong, amber while still unknown - and
             the skyline the navigator PREDICTS from its estimated pose
             (cyan, dashed): the gap between the two lines is the
             navigation error, seen through the camera;
  MAP        truth, free INS, the estimate with its 2-sigma ellipse,
             skyline fixes, hills, and the tracker rays;
  TIMELINE   position error over the whole run, events, cursor;
  TELEMETRY  error components, yaw error, tracks, identification.

    E5BT_TRACE=1 python3 -c "import e5bt_tracker as E; E.simulate('drone_low_nominal_D')"
    python3 e5bx_video.py out/tracker_trace/drone_low_nominal_D.json
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
from matplotlib.patches import Ellipse, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skyline as S  # noqa: E402

LAT0, LON0 = 36.95, 27.25
FOV, VEX = 90.0, 3.0              # camera field of view; vertical exaggeration
EL_LO, EL_HI = -6.0, 5.0          # elevation band shown, deg
K_REFR, R_E = 0.13, 6371000.0
BG, PANEL, GRID = '#0d1117', '#151b23', '#2a3440'
TXT, MUTED = '#d9dee4', '#8b96a3'
C_TRUE, C_EST, C_FREE, C_FIX = '#f5f5f5', '#38a6f4', '#8b96a3', '#f0a030'
C_OK, C_BAD, C_UNK, C_PRED = '#3ad07a', '#ff5a4f', '#f5b83d', '#45d6ff'


def main(path):
    r = json.load(open(path))
    tr = r['trace']
    name = os.path.basename(path)[:-5]
    out_dir = os.path.join(os.path.dirname(path), 'frames_' + name)
    os.makedirs(out_dir, exist_ok=True)
    mlat, mlon = S.meters_per_degree(LAT0)
    dem = lambda d: os.path.expanduser('~/.horizonator/' + d)
    cm_true = S.CMarcher(dem('DEMs_SRTM1'), (36.4, 37.4), (26.6, 27.9),
                         d_min=100.0)
    cm_map = S.CMarcher(dem('DEMs_COP30'), (36.4, 37.4), (26.6, 27.9),
                        d_min=100.0)
    to_ll = lambda p: (LAT0 + p[1] / mlat, LON0 + p[0] / mlon)
    hills = r['hills']
    T = np.array([s['truth'] for s in tr])
    E = np.array([s['est'] for s in tr])
    Fr = np.array([s['free'] for s in tr])
    t = np.array([s['t'] for s in tr]) / 60
    err = np.hypot(*(E - T).T)
    ferr = np.hypot(*(Fr - T).T)
    fixes = [(s['t'] / 60, np.hypot(*(np.array(s['fix']) - np.array(s['truth']))))
             for s in tr if s['fix']]
    # events: identification and release of a hill
    events, prev = [], {}
    for s in tr:
        cur = {d['j']: d['id'] for d in s['tracks']}
        for j, i in cur.items():
            if prev.get(j, 'none') is None and i is not None:
                events.append((s['t'] / 60, 'id', i == j))
            if j not in prev:
                events.append((s['t'] / 60, 'acq', None))
        for j in prev:
            if j not in cur:
                events.append((s['t'] / 60, 'rel', None))
        prev = cur
    az_rel = np.arange(-FOV / 2, FOV / 2 + 1e-9, 0.1)
    title = {'drone': 'drone 300 m', 'vessel': 'vessel 5 m'}.get(
        name.split('_')[0], name.split('_')[0])
    grade = name.split('_')[1]
    var = name.split('_')[-1]
    vdesc = {'D': 'INS + skyline + wide camera, 2 hill trackers',
             'E': 'INS + wide camera, 2 hill trackers',
             'B': 'INS + skyline + gimbal hill tracker',
             'A': 'INS + skyline'}.get(var, var)

    for n, s in enumerate(tr):
        fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
        gs = fig.add_gridspec(3, 3, left=0.02, right=0.985, top=0.93,
                              bottom=0.05, hspace=0.32, wspace=0.12,
                              height_ratios=[1.15, 1.15, 0.75],
                              width_ratios=[1.55, 1.0, 0.62])
        fig.text(0.02, 0.965, 'SKYFIX REPLAY', color=TXT, fontsize=15,
                 weight='bold', family='monospace')
        fig.text(0.155, 0.965, '%s  ·  %s IMU  ·  %s' % (title, grade, vdesc),
                 color=MUTED, fontsize=11.5)
        fig.text(0.985, 0.965, 't = %5.1f / %.1f min    keyframe %d / %d' % (
            s['t'] / 60, t[-1], n + 1, len(tr)), color=TXT, fontsize=11.5,
            ha='right', family='monospace')

        # ---------------- camera
        ax = fig.add_subplot(gs[0:2, 0])
        ax.set_facecolor('#6f9fd4')
        p = np.array(s['truth'])
        la, lo = to_ll(p)
        yaw_t = math.degrees(s['yaw_true'])
        el, rg = cm_true.skyline(la, lo, s['z'], (yaw_t + az_rel) % 360)
        dip = -math.degrees(math.sqrt(2 * s['z'] / (R_E / (1 - K_REFR))))
        eld = np.where(np.isfinite(el), np.degrees(el), dip)
        y = lambda e: e * VEX
        # sky gradient
        sky = np.linspace(0, 1, 64)[:, None]
        ax.imshow(np.dstack([0.42 + 0.35 * sky, 0.62 + 0.25 * sky,
                             0.85 + 0.1 * sky]) * np.ones((1, 2, 1)),
                  extent=[-FOV / 2, FOV / 2, y(EL_HI), y(EL_LO)],
                  aspect='auto', origin='upper', zorder=0)
        # sea below the horizon dip, terrain above it, shaded by range
        ax.fill_between([-FOV / 2, FOV / 2], y(EL_LO), y(dip),
                        color='#1f4d6e', zorder=1)
        rr = np.where(np.isfinite(rg), rg, 4e4)
        shade = np.clip(np.log10(np.maximum(rr, 500)) - 2.7, 0, 2) / 2
        for i in range(0, az_rel.size - 1, 3):
            if eld[i] > dip + 0.01:
                c = (0.20 + 0.45 * shade[i], 0.26 + 0.42 * shade[i],
                     0.20 + 0.55 * shade[i])
                ax.fill_between(az_rel[i:i + 4], y(dip), y(eld[i:i + 4]),
                                color=c, lw=0, zorder=2)
        ax.plot(az_rel, y(eld), color=C_TRUE, lw=1.3, zorder=4,
                label='true skyline')
        # predicted skyline from the ESTIMATED pose, in the true camera
        pe = np.array(s['est'])
        lae, loe = to_ll(pe)
        yaw_e = math.degrees(s['yaw_est'])
        el2, _ = cm_map.skyline(lae, loe, s['z'], (yaw_e + az_rel) % 360)
        el2d = np.where(np.isfinite(el2), np.degrees(el2), dip)
        # the navigator thinks the camera points at yaw_e; on the true
        # image its prediction lands shifted by the yaw error
        shift = yaw_e - yaw_t
        ax.plot(az_rel - shift, y(el2d), color=C_PRED, lw=1.4, ls='--',
                zorder=5, label='skyline predicted from the estimate')
        # tracker windows
        box_w = FOV * 250 / 1920
        for d in s['tracks']:
            h = hills[d['j']]
            dv = np.array(h['true']) - p
            rng = math.hypot(*dv)
            a = (math.degrees(math.atan2(dv[0], dv[1])) - yaw_t + 540) % 360 - 180
            e = math.degrees(math.atan2(h['h'] - s['z'], rng)
                             - rng * (1 - K_REFR) / (2 * R_E))
            col = C_UNK if d['id'] is None else (C_OK if d['id'] == d['j']
                                                 else C_BAD)
            ax.add_patch(Rectangle((a - box_w / 2, y(e) - box_w * 0.5),
                                   box_w, box_w, fill=False, ec=col, lw=2.2,
                                   zorder=6))
            lab = ('?  identifying' if d['id'] is None else
                   'hill #%d%s' % (d['id'], '' if d['id'] == d['j']
                                   else '  WRONG'))
            ax.text(a - box_w / 2, y(e) + box_w * 0.55, lab + '  %.1f km' % (
                rng / 1000), color=col, fontsize=9.5, weight='bold',
                zorder=7, family='monospace')
        if s['photo']:
            ax.text(FOV / 2 - 1, y(EL_HI) - 1.2, 'SKYLINE PHOTO', ha='right',
                    va='top', color='#0d1117', fontsize=10, weight='bold',
                    bbox=dict(fc=C_FIX, ec='none', pad=3), zorder=8)
        ax.set_xlim(-FOV / 2, FOV / 2)
        ax.set_ylim(y(EL_LO), y(EL_HI))
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_xticklabels(['-40°', '-20°', 'bow', '+20°', '+40°'],
                           color=MUTED, fontsize=8)
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title('FRONT CAMERA  ·  90° wide  ·  vertical ×%g' % VEX,
                     color=TXT, fontsize=10.5, loc='left', family='monospace')
        ax.legend(loc='lower left', fontsize=8.5, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TXT)

        # ---------------- map
        ax = fig.add_subplot(gs[0:2, 1])
        ax.set_facecolor(PANEL)
        half = 4500
        c0 = T[n]
        ax.plot(*T[:n + 1].T, color=C_TRUE, lw=2, label='truth')
        ax.plot(*T[n:].T, color=C_TRUE, lw=0.8, alpha=0.25)
        ax.plot(*Fr[:n + 1].T, color=C_FREE, lw=1.2, ls='--', label='free INS')
        ax.plot(*E[:n + 1].T, color=C_EST, lw=1.8, label='estimate')
        fx = np.array([q['fix'] for q in tr[:n + 1] if q['fix']])
        if fx.size:
            ax.scatter(*fx.T, s=14, color=C_FIX, zorder=4, label='skyline fix')
        H = np.array([h['true'] for h in hills])
        m = np.all(np.abs(H - c0) < half * 1.6, axis=1)
        ax.scatter(*H[m].T, marker='^', s=34, color='#a58b5a', zorder=3,
                   label='hills')
        for d in s['tracks']:
            if d['id'] is None:
                continue
            he = s['hills_est'].get(str(d['id']))
            if he:
                col = C_OK if d['id'] == d['j'] else C_BAD
                ax.plot([pe[0], he[0]], [pe[1], he[1]], color=col, lw=1.1,
                        alpha=0.85)
        if s['cov']:
            C = np.array(s['cov'])
            w_, v_ = np.linalg.eigh(C)
            ang = math.degrees(math.atan2(v_[1, 1], v_[0, 1]))
            ax.add_patch(Ellipse(pe, 4 * math.sqrt(max(w_[1], 1)),
                                 4 * math.sqrt(max(w_[0], 1)), angle=ang,
                                 fc=C_EST, alpha=0.18, ec=C_EST, lw=1))
        ax.scatter(*T[n], s=60, color=C_TRUE, zorder=6, marker='o')
        ax.scatter(*pe, s=60, color=C_EST, zorder=7, marker='x', lw=2.5)
        ax.set_xlim(c0[0] - half, c0[0] + half)
        ax.set_ylim(c0[1] - half, c0[1] + half)
        ax.set_aspect('equal')
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, lw=0.6)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title("BIRD'S-EYE  ·  9 km window  ·  2σ ellipse", color=TXT,
                     fontsize=10.5, loc='left', family='monospace')
        ax.legend(loc='upper left', fontsize=8, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TXT, ncol=2)

        # ---------------- telemetry
        ax = fig.add_subplot(gs[0:2, 2])
        ax.set_facecolor(PANEL)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        dE, dN = (pe - T[n])
        sig2 = (2 * math.sqrt(max(np.linalg.eigvalsh(np.array(s['cov'])).max(),
                                  0)) if s['cov'] else float('nan'))
        yerr = (s['yaw_est'] - s['yaw_true'] + math.pi) % (2 * math.pi) - math.pi
        nid = sum(1 for d in s['tracks'] if d['id'] is not None)
        lines = [
            ('POSITION ERROR', '%6.0f m' % err[n], C_EST),
            ('  Δ north', '%+6.0f m' % dN, TXT),
            ('  Δ east', '%+6.0f m' % dE, TXT),
            ('  2σ (major)', '%6.0f m' % sig2, TXT),
            ('FREE INS ERROR', '%6.0f m' % ferr[n], C_FREE),
            ('YAW ERROR', '%+6.2f °' % math.degrees(yerr), TXT),
            ('ALTITUDE', '%6.0f m' % s['z'], TXT),
            ('', '', TXT),
            ('TRACKERS', '%d held' % len(s['tracks']), TXT),
            ('  identified', '%d' % nid, C_OK if nid else MUTED),
            ('  unknown', '%d' % (len(s['tracks']) - nid),
             C_UNK if len(s['tracks']) - nid else MUTED),
            ('HILLS MAPPED', '%d' % len(s['hills_est']), TXT),
            ('SKYLINE FIXES', '%d' % sum(1 for q in tr[:n + 1] if q['fix']),
             C_FIX),
        ]
        ax.text(0.06, 0.97, 'TELEMETRY', color=MUTED, fontsize=10,
                transform=ax.transAxes, va='top', family='monospace')
        for i, (k_, v, c) in enumerate(lines):
            yy = 0.90 - i * 0.066
            ax.text(0.06, yy, k_, color=MUTED if k_.startswith('  ') else TXT,
                    fontsize=10, transform=ax.transAxes, family='monospace')
            ax.text(0.95, yy, v, color=c, fontsize=10.5, ha='right',
                    transform=ax.transAxes, family='monospace', weight='bold')

        # ---------------- timeline
        ax = fig.add_subplot(gs[2, :])
        ax.set_facecolor(PANEL)
        ax.plot(t, ferr, color=C_FREE, lw=1, ls='--', label='free INS')
        ax.plot(t[:n + 1], err[:n + 1], color=C_EST, lw=2, label='estimate')
        ax.plot(t[n:], err[n:], color=C_EST, lw=1, alpha=0.25)
        if fixes:
            ft, fe = np.array(fixes).T
            mm = ft <= t[n] + 1e-9
            ax.scatter(ft[mm], fe[mm], s=12, color=C_FIX, zorder=4,
                       label='skyline fix error')
        for et, kind, ok in events:
            if et > t[n] + 1e-9:
                continue
            col = {'id': C_OK if ok else C_BAD, 'acq': C_UNK,
                   'rel': MUTED}[kind]
            ax.axvline(et, color=col, lw=0.8, alpha=0.7, ymax=0.12)
        ax.axvline(t[n], color='#ffffff', lw=1.2)
        ax.set_yscale('log')
        ax.set_ylim(3, max(3e4, ferr.max() * 1.2))
        ax.set_xlim(0, t[-1])
        ax.set_xlabel('time (min)  ·  ticks: amber acquired, green identified, '
                      'grey released', color=MUTED, fontsize=9)
        ax.set_ylabel('error (m)', color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, lw=0.5, which='both')
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.legend(loc='upper left', fontsize=8.5, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TXT, ncol=3)
        fig.savefig(os.path.join(out_dir, 'f%04d.png' % n), facecolor=BG)
        plt.close(fig)
        if n % 20 == 0:
            print('frame %d / %d' % (n + 1, len(tr)), flush=True)
    mp4 = os.path.join(os.path.dirname(path), name + '.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '10',
                    '-i', os.path.join(out_dir, 'f%04d.png'), '-c:v',
                    'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', mp4],
                   check=True)
    print('wrote', mp4)


if __name__ == '__main__':
    main(sys.argv[1])
