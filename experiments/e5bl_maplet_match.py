#!/usr/bin/env python3
"""E5bl: GIANT-style 2D template matching on a coastal skyline.

GIANT's Surface Feature Navigation never matches curve points. It cuts
a small 2D patch -- a maplet -- around each catalogued feature, renders
it at the a priori pose, and finds it in the image by normalised cross
correlation. A correlation peak is localised in BOTH image directions,
which is why the tangential sliding that invalidates point-to-point PnP
over a silhouette never arises there (E5bk).

This runs that idea on our data, with one change forced by the domain
gap: GIANT's templates are shaded surface patches and ours cannot be
(no albedo, hazy sunset silhouettes -- E5be), so the patches here are
cut from the BOUNDARY RASTER rather than from intensity. The shape is
what both sides genuinely share.

The output is the argument itself. At a summit the correlation surface
is a tight blob; along a smooth flank it is an elongated trough -- the
sliding, drawn from measurement rather than asserted.
"""

import json
import os

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extract
import skyline as S

P = '/home/user/celestial-navigation/peakfinder/'

# two frames, chosen for their relief: the Bosphorus shore is nearly
# flat (the whole silhouette spans ~49 px of a 1200 px frame), the
# Bodrum island has a real crest. The contrast is the point.
FRAMES = {
    'bosphorus': dict(photo=P + 'PF_new_bosphorus1.jpg',
                      hand='out/bos/user_skyline.npy',
                      fix='out/bos/BOS_roll_pitch.json',
                      lat=41.12590, lon=29.07076, z=7.0, fov=73.7,
                      roll=1.75, label='Bogaz - alcak kabartma'),
    'bodrum': dict(photo=P + 'PF_new_bodrum9.jpg',
                   hand=None,              # seam extraction
                   fix='out/bd9/BD9_h180_cop.json',
                   lat=37.01992, lon=27.44426, z=9.0, fov=73.7,
                   roll=0.0, label='Bodrum / Kara Ada - gercek sirt'),
}
TPL = 81          # template edge, px
SEARCH = 70       # half-width of the search window, px
N_FEAT = 5


def boundary_raster(rows, shape, sigma=2.0):
    """Sky above the curve, not-sky below, softened at the edge.

    A FILLED mask, not a thin line. Normalised cross correlation on a
    one-pixel curve is degenerate -- most of the patch is background, so
    two near-empty patches score 1.0 and the peak lands wherever the
    window runs out. A filled region is what GIANT actually correlates
    (a rendered surface patch has area), and it is stable.
    """
    H, W = shape
    rr = np.arange(H)[:, None]
    img = (rr > np.rint(rows)[None, :]).astype(np.float32)
    k = int(6 * sigma) | 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def curvature_extrema(el, n, min_sep):
    """Columns where the profile bends hardest -- summits and notches."""
    s = cv2.GaussianBlur(el.astype(np.float32).reshape(-1, 1), (1, 31), 9).ravel()
    curv = np.abs(np.gradient(np.gradient(s)))
    picked = []
    order = np.argsort(curv)[::-1]
    for c in order:
        if c < TPL // 2 + SEARCH or c > el.size - TPL // 2 - SEARCH:
            continue
        if all(abs(c - p) >= min_sep for p in picked):
            picked.append(int(c))
        if len(picked) == n:
            break
    return sorted(picked), curv


def run(cfg):
    img = extract.load_image(cfg['photo'])
    H, W = img.shape[:2]
    FOV = cfg['fov']; LAT = cfg['lat']; LON = cfg['lon']; Z = cfg['z']
    f = (W / 2) / np.tan(np.radians(FOV) / 2)
    u = np.arange(W) - (W - 1) / 2.0

    # observed boundary: the operator's digitisation where we have one,
    # otherwise the seam detector
    if cfg['hand']:
        hand = np.load(cfg['hand'])
        rows_obs = np.interp(np.linspace(0, hand.size - 1, W),
                             np.arange(hand.size), hand) * (W / hand.size)
    else:
        rows_obs = np.asarray(extract.skyline_seam(img)[0], float)
    el_obs = (np.arctan2(H / 2 - rows_obs, f)
              + u * np.tan(np.radians(cfg['roll'])) / f)
    rows_obs = H / 2 - f * np.tan(el_obs)

    # predicted boundary: the DEM at the accepted fix
    j = json.load(open(cfg['fix']))
    p = j['photos'][0]
    hdg = p['heading_deg'] + p['heading_offset_deg']
    mlat, mlon = S.meters_per_degree(LAT)
    cm = S.CMarcher(os.path.expanduser('~/.horizonator/DEMs_COP30'),
                    (LAT - .6, LAT + .6), (LON - .6, LON + .6), d_min=300.)
    az = hdg + np.linspace(-FOV / 2, FOV / 2, W)
    el_syn, rng = cm.skyline(LAT + j['dn_m'] / mlat, LON + j['de_m'] / mlon,
                             Z, az)
    el_syn = el_syn + np.where(np.isfinite(rng) & (rng > 0),
                               9.0 / np.maximum(rng, 1000.), 0.0)
    beta = np.median(el_syn - el_obs)
    rows_syn = H / 2 - f * np.tan(el_syn - beta)

    obs_r = boundary_raster(rows_obs, (H, W))
    syn_r = boundary_raster(rows_syn, (H, W))

    cols, curv = curvature_extrema(el_syn, N_FEAT, min_sep=150)
    # one deliberately flat control point, to show what sliding looks like
    flat = int(np.argsort(curv[TPL // 2 + SEARCH: W - TPL // 2 - SEARCH])[0]
               + TPL // 2 + SEARCH)

    results = []
    for c, kind in [(x, 'summit') for x in cols] + [(flat, 'flat')]:
        r0 = int(round(rows_syn[c]))
        h = TPL // 2
        if r0 - h - SEARCH < 0 or r0 + h + SEARCH >= H:
            continue
        tpl = syn_r[r0 - h:r0 + h + 1, c - h:c + h + 1]
        win = obs_r[r0 - h - SEARCH:r0 + h + SEARCH + 1,
                    c - h - SEARCH:c + h + SEARCH + 1]
        if tpl.std() < 0.05:          # a uniform patch correlates with anything
            continue
        surf = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        # TM_CCOEFF_NORMED divides by the window patch's standard
        # deviation, so wherever the template slides fully into the
        # uniform sky above the curve or the uniform ground below it,
        # the denominator collapses and OpenCV returns 1.0. Those are
        # not matches; suppress every position whose window patch is
        # flat before taking the peak.
        m1 = cv2.boxFilter(win, cv2.CV_32F, (TPL, TPL), normalize=True)
        m2 = cv2.boxFilter(win * win, cv2.CV_32F, (TPL, TPL), normalize=True)
        sd = np.sqrt(np.maximum(m2 - m1 * m1, 0))[TPL // 2:TPL // 2 + surf.shape[0],
                                                  TPL // 2:TPL // 2 + surf.shape[1]]
        surf = np.where(sd > 0.05, surf, -1.0).astype(np.float32)
        _, peak, _, loc = cv2.minMaxLoc(surf)
        dx, dy = loc[0] - SEARCH, loc[1] - SEARCH

        def width(prof, i):
            """How far the score stays within 10% of the peak."""
            thr = peak - 0.10 * (peak - prof.min())
            k = i
            while k > 0 and prof[k] > thr:
                k -= 1
            m = i
            while m < prof.size - 1 and prof[m] > thr:
                m += 1
            return m - k

        wx = int(width(surf[loc[1]], loc[0]))
        wy = int(width(surf[:, loc[0]], loc[1]))
        results.append(dict(col=c, kind=kind, peak=float(peak), dx=dx, dy=dy,
                            wx=wx, wy=wy, surf=surf, row=r0,
                            truth=int(round(rows_obs[c] - r0)),
                            aniso=wx / max(wy, 1)))
    return dict(img=img, rows_obs=rows_obs, rows_syn=rows_syn,
                results=results, H=H, W=W, label=cfg['label'],
                err=float(np.hypot(j['dn_m'], j['de_m'])))


def draw(frames, out='out/bos/BOS_maplet.png'):
    nrow = len(frames)
    ncol = max(len(fr['results']) for fr in frames)
    fig = plt.figure(figsize=(15.5, nrow * 6.6))
    gs = fig.add_gridspec(nrow * 2, ncol, height_ratios=[2.1, 1.35] * nrow,
                          hspace=0.42, wspace=0.28)
    for k, fr in enumerate(frames):
        ax = fig.add_subplot(gs[2 * k, :])
        ax.imshow(fr['img'])
        x = np.arange(fr['W'])
        ax.plot(x, fr['rows_obs'], '-', lw=1.7, color='#E9973F',
                label='gozlenen siluet')
        ax.plot(x, fr['rows_syn'], '-', lw=1.4, color='#2BA3C7',
                label='DEM, kabul edilen poz')
        for r in fr['results']:
            col = '#2A6E51' if r['kind'] == 'summit' else '#AE342A'
            ax.add_patch(plt.Rectangle((r['col'] - TPL // 2, r['row'] - TPL // 2),
                                       TPL, TPL, fill=False, ec=col, lw=1.9))
            ax.plot([r['col']], [r['row']], 'o', ms=5, color=col)
        lo = min(fr['rows_syn'].min(), fr['rows_obs'].min())
        hi = max(fr['rows_syn'].max(), fr['rows_obs'].max())
        pad = max(150, TPL)
        ax.set_xlim(0, fr['W'])
        ax.set_ylim(min(hi + pad, fr['H']), max(lo - pad, 0))
        ax.axis('off')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_title('%s   ·   fiks hatasi %.0f m   ·   yesil: egrilik '
                     'ekstremumu, kirmizi: duz yamac kontrolu'
                     % (fr['label'], fr['err']), fontsize=12)
        for i, r in enumerate(fr['results']):
            a = fig.add_subplot(gs[2 * k + 1, i])
            a.imshow(r['surf'], cmap='magma',
                     extent=[-SEARCH, SEARCH, SEARCH, -SEARCH])
            a.plot([r['dx']], [r['dy']], '+', ms=12, mew=2.2, color='#7FE3FF')
            a.plot([0], [r['truth']], 'x', ms=9, mew=2.0, color='#39FF14')
            col = '#2A6E51' if r['kind'] == 'summit' else '#AE342A'
            a.set_title('col %d · %s\npik %.3f · genislik %d x %d px'
                        % (r['col'], r['kind'], r['peak'], r['wx'], r['wy']),
                        fontsize=9, color=col)
            a.set_xticks([-50, 0, 50]); a.set_yticks([-50, 0, 50])
            a.tick_params(labelsize=7)
            if i == 0:
                a.set_ylabel('korelasyon yuzeyi', fontsize=9)
    fig.savefig(out, dpi=104, bbox_inches='tight')
    return out


def main():
    frames = []
    for key in ('bodrum', 'bosphorus'):
        fr = run(FRAMES[key])
        frames.append(fr)
        print('\n%s  (fiks hatasi %.0f m)' % (fr['label'], fr['err']))
        print('%-6s %-8s %7s %6s %6s %7s %8s %8s %7s'
              % ('col', 'kind', 'peak', 'dx', 'dy', 'truth', 'width_x',
                 'width_y', 'aniso'))
        for r in fr['results']:
            print('%-6d %-8s %7.3f %6d %6d %7d %8d %8d %7.1f'
                  % (r['col'], r['kind'], r['peak'], r['dx'], r['dy'],
                     r['truth'], r['wx'], r['wy'], r['aniso']))
    out = draw(frames)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
