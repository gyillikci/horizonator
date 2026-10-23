#!/usr/bin/env python3
"""E5cc: the day's visibility from the photo's own contrast.

Haze erases far ridges (E5bz, E5cb). How far one can see is a property
of the day, so a fixed range cut fails clear days (E5ca). Aerial
perspective gives it from the image (Koschmieder): a ridge at range r
shows against the horizon sky with apparent contrast

    C(r) = C0 exp(-3.912 r / V)

where V is the meteorological visibility (2% threshold). Per column:

  contrast  c = (L_sky - L_terrain) / L_sky, linear luminance, medians
            of the 10 px bands 4-14 px above and below the photo's own
            boundary;
  range     the DEM crest range r at the PRIOR position (box centre),
            only on columns where the photo skyline matches the DEM
            within 3 mrad after a wide heading/offset fit: terrain that
            is demonstrably visible.

ln c = ln C0 - (3.912 / V) r is fitted robustly (Theil-Sen over 1 km
range bins). The farthest terrain a segmenter can still separate from
sky has c = EPS, so

    r_vis = V ln(C0 / EPS) / 3.912      (EPS = 0.05, fixed a priori)

and skyfix runs with --dmax r_vis. Without range leverage (visible
ridges spanning < 4 km) or without decay, V is unknown: no cut.

Caveat: here the prior position IS the truth (EXIF box centre), so the
ranges are exact; `estimate` also repeats it 2 km off to show how much
that matters.

    python3 e5cc_visibility.py estimate [TAG ...]
    python3 e5cc_visibility.py run [TAG ...] [--jobs 4]
    python3 e5cc_visibility.py report [TAG ...]
"""

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import theilslopes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from assets import celnav  # noqa: E402

OUT = os.path.join(HERE, 'out', 'visibility')
FRAMES = os.path.join(HERE, 'data', 'campaign_frames.json')
EPS = 0.05          # contrast below which the segmenter calls it sky
RMIN, RMAX = 5.0, 40.0
SPAN = 4.0          # km of visible-range spread needed to fit a slope
FOUR = ['09_PF_new_bodrum9', '38_XKYU3498', '31_EWAC7374', '36_PQBC6867']
BAD, GOOD = 500.0, 300.0


def frames(tags=None):
    fs = json.load(open(FRAMES))['frames']
    return [f for f in fs if not tags or f['tag'] in tags]


def lin(v):
    """sRGB [0,1] -> linear light."""
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def column_contrast(img, rows, conf):
    """Weber contrast of the terrain against the sky right above it."""
    L = lin(img) @ np.array([0.2126, 0.7152, 0.0722])
    H, W = L.shape
    c = np.full(W, np.nan)
    for x in range(W):
        r = int(rows[x])
        if conf[x] <= 0 or r < 16 or r > H - 16:
            continue
        sky = np.median(L[r - 14:r - 4, x])
        ter = np.median(L[r + 4:r + 14, x])
        if sky > 1e-3:
            c[x] = (sky - ter) / sky
    return c


def estimate_frame(fr, dn=0.0, de=0.0, F=None, S=None, img=None, obs=None):
    """V and r_vis for one frame, ranges from the prior shifted (dn, de)."""
    dem = os.path.expanduser('~/.horizonator/DEMs_COP30')
    el_obs, w, d = obs
    mlat, mlon = S.meters_per_degree(fr['lat'])
    lat, lon = fr['lat'] + dn / mlat, fr['lon'] + de / mlon
    cm = S.CMarcher(dem, (lat - .6, lat + .6), (lon - .8, lon + .8),
                    d_min=1000.0)
    el, r = cm.skyline(lat, lon, fr['z'], F.AZ)
    el = np.nan_to_num(el + np.where(np.isfinite(r) & (r > 0),
                                     9.0 / np.maximum(r, 1000.0), 0))
    c, s, b = F.photo_cost(el_obs, w, el, np.arange(-200, 201, 2),
                           np.linspace(-0.03, 0.03, 61))
    con = column_contrast(img, d['rows'], d['conf'])
    # each image column -> its azimuth bin under the fitted heading
    az = (fr['heading'] + d['az_rel'] + s * 0.1 + 180) % 360 - 180
    k = np.clip(np.searchsorted(F.AZ, az), 0, F.AZ.size - 1)
    res = (d['el_pt'] + b - el[k]) * 1e3
    rr = np.nan_to_num(r[k]) / 1e3
    vis = np.isfinite(con) & (con > 0.005) & (np.abs(res) < 3.0) & (rr > 0)
    hidden = np.isfinite(res) & (res < -8.0) & (rr > 0) & (d['conf'] > 0)
    out = dict(psi=s * 0.1, beta_mrad=b * 1e3, n_vis=int(vis.sum()),
               hidden_cols=int(hidden.sum()),
               hidden_r_km=float(np.median(rr[hidden])) if hidden.any()
               else None)
    if vis.sum() < 30:
        out.update(V_km=None, r_vis_km=RMAX, why='too few visible columns')
        return out, None
    edges = np.arange(0, 41, 1.0)
    xs, ys, ns = [], [], []
    for lo in edges[:-1]:
        m = vis & (rr >= lo) & (rr < lo + 1)
        if m.sum() >= 5:
            xs.append(lo + 0.5)
            ys.append(math.log(np.median(con[m])))
            ns.append(int(m.sum()))
    out.update(bins=dict(r=xs, ln_c=ys, n=ns),
               vis_span_km=float(max(xs) - min(xs)) if xs else 0.0)
    if len(xs) < 2 or max(xs) - min(xs) < SPAN:
        out.update(V_km=None, r_vis_km=RMAX,
                   why='visible ridges span < %g km: no range leverage'
                   % SPAN)
        return out, (rr, con, vis, hidden)
    slope, icpt, lo_s, hi_s = theilslopes(ys, xs)
    C0 = min(math.exp(icpt), 1.0)
    out.update(slope=float(slope), slope_ci=[float(lo_s), float(hi_s)],
               C0=C0)
    if slope >= 0 or hi_s >= 0:
        out.update(V_km=None, r_vis_km=RMAX,
                   why='no significant contrast decay with range')
        return out, (rr, con, vis, hidden)
    V = 3.912 / -slope
    r_vis = V * math.log(C0 / EPS) / 3.912
    # never cut below the farthest terrain we SEE: it is visible
    r_vis = float(np.clip(max(r_vis, rr[vis].max()), RMIN, RMAX))
    out.update(V_km=float(V), r_vis_km=r_vis, why='fitted')
    return out, (rr, con, vis, hidden)


def estimate(a):
    import skyfix as F
    import skyline as S
    import extract
    os.makedirs(OUT, exist_ok=True)
    F.EXTRACTOR = 'ewasr'
    f = os.path.join(OUT, 'visibility.json')
    allv = json.load(open(f)) if os.path.exists(f) else {}
    plots = {}
    for fr in frames(a.tags):
        img = extract.load_image(celnav(fr['path']))
        obs = F.observation(img, fr['fov'], fr['heading'], 0.0)
        if obs[1].sum() <= 0:
            continue
        v, pts = estimate_frame(fr, F=F, S=S, img=img, obs=obs)
        # how much the prior's error matters: same, 2 km north / east
        v['shifted'] = {}
        for key, (dn, de) in (('2km_N', (2000, 0)), ('2km_E', (0, 2000))):
            vs, _ = estimate_frame(fr, dn, de, F=F, S=S, img=img, obs=obs)
            v['shifted'][key] = dict(V_km=vs['V_km'],
                                     r_vis_km=vs['r_vis_km'])
        allv[fr['tag']] = v
        plots[fr['tag']] = (v, pts)
        print('%-28s V %s  r_vis %4.1f km  (%s; %d visible cols, %d hidden '
              'at %s km) | 2 km off: %s' % (
                  fr['tag'], '%5.1f km' % v['V_km'] if v['V_km'] else
                  '   ?    ', v['r_vis_km'], v['why'], v['n_vis'],
                  v['hidden_cols'],
                  '%.0f' % v['hidden_r_km'] if v['hidden_r_km'] else '-',
                  ', '.join('%.1f' % x['r_vis_km']
                            for x in v['shifted'].values())), flush=True)
    json.dump(allv, open(f, 'w'), indent=1)
    figure({k: p for k, p in plots.items() if k in FOUR})


def figure(plots):
    if not plots:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(plots), figsize=(4.2 * len(plots), 4),
                            squeeze=False)
    for ax, (tag, (v, pts)) in zip(axs[0], plots.items()):
        if pts is not None:
            rr, con, vis, hidden = pts
            ax.scatter(rr[vis], np.log(np.clip(con[vis], 1e-3, None)),
                       s=3, color='#3a6ea5', alpha=.35,
                       label='visible terrain (matches DEM)')
            if hidden.any():
                ax.scatter(rr[hidden], np.full(hidden.sum(), math.log(EPS)),
                           s=4, marker='|', color='#c0392b',
                           label='DEM ridge missing from photo')
        if 'bins' in v and v['bins']['r']:
            ax.plot(v['bins']['r'], v['bins']['ln_c'], 'ko', ms=4,
                    label='1 km bin medians')
        x = np.linspace(0, 40, 50)
        if v.get('V_km'):
            ax.plot(x, math.log(v['C0']) - 3.912 / v['V_km'] * x, 'k-',
                    lw=1.2, label='Koschmieder fit, V %.0f km'
                    % v['V_km'])
        ax.axhline(math.log(EPS), color='#c0392b', lw=.8, ls=':')
        ax.axvline(v['r_vis_km'], color='#2e8b57', lw=1.2, ls='--',
                   label='r_vis %.1f km' % v['r_vis_km'])
        ax.set_xlim(0, 41)
        ax.set_ylim(math.log(0.01), 0.2)
        ax.set_xlabel('DEM crest range, km')
        ax.set_ylabel('ln contrast against the sky')
        ax.set_title(tag[3:], fontsize=10)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'koschmieder_4.png'), dpi=110)


def solve(fr, r_vis, arm, flags):
    out = os.path.join(OUT, '%s_%s' % (fr['tag'], arm))
    if os.path.exists(out + '.json'):
        return out
    cmd = [sys.executable, os.path.join(HERE, 'skyfix.py'),
           celnav(fr['path']),
           '--center', '%.6f,%.6f' % (fr['lat'], fr['lon']),
           '--box', str(fr['box']), '--z', str(fr['z']),
           '--fov', str(fr['fov']), '--heading', str(fr['heading']),
           '--dmax', '%.0f' % (r_vis * 1000), '--out', out] + flags.split()
    if arm == 'vis_soft5':
        cmd += ['--compass-sigma', '5']
    with open(out + '.log', 'w') as lg:
        subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=HERE,
                       env=dict(os.environ, OMP_NUM_THREADS='1'))
    return out


def run(a):
    vis = json.load(open(os.path.join(OUT, 'visibility.json')))
    flags = json.load(open(FRAMES))['flags'].replace(
        '~', os.path.expanduser('~'))
    jobs = [(fr, vis[fr['tag']]['r_vis_km'], arm)
            for arm in ('vis_hard6', 'vis_soft5')
            for fr in frames(a.tags) if fr['tag'] in vis]
    with ThreadPoolExecutor(a.jobs) as ex:
        for k, out in enumerate(ex.map(lambda j: solve(*j, flags), jobs), 1):
            print('%3d/%d %s %s' % (k, len(jobs), os.path.basename(out),
                                    '' if os.path.exists(out + '.json')
                                    else 'NO RESULT'), flush=True)


def load_one(f):
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    return dict(err=math.hypot(j['dn_m'], j['de_m']), dn=j['dn_m'],
                de=j['de_m'], ok=j['status'] == 'ok',
                hoff=j['photos'][0]['heading_offset_deg'])


ARMS = {
    'hard6': ('baseline', '%s_huber.json'),
    'soft5': ('compass', '%s_soft5.json'),
    'd15': ('dmax', '%s_d15.json'),
    'vis_hard6': ('visibility', '%s_vis_hard6.json'),
    'vis_soft5': ('visibility', '%s_vis_soft5.json'),
}


def report(a):
    vis = json.load(open(os.path.join(OUT, 'visibility.json')))
    rows = []
    for fr in frames(a.tags):
        r = dict(tag=fr['tag'], r_vis=vis.get(fr['tag'], {}).get('r_vis_km'))
        for arm, (d, pat) in ARMS.items():
            r[arm] = load_one(os.path.join(HERE, 'out', d, pat % fr['tag']))
        if all(r[k] for k in ARMS):
            rows.append(r)
    print('frames with every arm: %d\n' % len(rows))
    print('%-24s %5s | ' % ('frame', 'r_vis')
          + ' | '.join('%-14s' % k for k in ARMS))
    for r in rows:
        print('%-24s %5.1f | ' % (r['tag'][:24], r['r_vis'] or 40)
              + ' | '.join('%6.0fm %s %+5.1f' % (
                  r[k]['err'], 'A' if r[k]['ok'] else '-', r[k]['hoff'])
                  for k in ARMS))
    summ = {}
    print('\n%-10s %7s %5s %5s | %4s %4s %5s' % (
        'arm', 'median', '<300', '<500', 'acc', 'good', 'wrong'))
    for k in ARMS:
        e = np.array([r[k]['err'] for r in rows])
        gate = 10.0 if 'soft5' in k else 1e9   # |psi| <= 2 sigma
        acc = [r for r in rows if r[k]['ok'] and abs(r[k]['hoff']) <= gate]
        summ[k] = dict(median=float(np.median(e)),
                       within300=int((e < GOOD).sum()),
                       within500=int((e < BAD).sum()), accepted=len(acc),
                       good=sum(r[k]['err'] < GOOD for r in acc),
                       wrong=sum(r[k]['err'] >= BAD for r in acc),
                       wrong_tags=[r['tag'] for r in acc
                                   if r[k]['err'] >= BAD])
        print('%-10s %6.0fm %5d %5d | %4d %4d %5d  %s' % (
            k + ('+gate' if 'soft5' in k else ''), summ[k]['median'],
            summ[k]['within300'], summ[k]['within500'], summ[k]['accepted'],
            summ[k]['good'], summ[k]['wrong'],
            ', '.join(t[3:] for t in summ[k]['wrong_tags'])))
    json.dump(dict(summary=summ, rows=rows),
              open(os.path.join(OUT, 'report%s.json' % (
                  '_' + str(len(rows)) if a.tags else '')), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    for n in ('estimate', 'run', 'report'):
        p = sp.add_parser(n)
        p.add_argument('tags', nargs='*')
        p.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    dict(estimate=estimate, run=run, report=report)[a.cmd](a)
