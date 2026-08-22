#!/usr/bin/env python3
"""E5b: fix the position from the FOREGROUND silhouette alone.

Everything measured so far says the position information is in the
near layer and the matcher was using the far one. A 500 m move swings
a 40 km coast by 12 mrad and a 6 km island by 81 (E4y), so this drops
the far layer entirely and matches only what stands in front.

Image side: near terrain is dark and saturated, far terrain is pale
and blue — atmospheric perspective is the depth cue, so a 'nearness'
score (darkness plus saturation) segments the foreground, and its
upper boundary within the band above the sea horizon is the
foreground silhouette.

DEM side: skyline.visible_layers gives every visible crest; the
foreground profile is the highest crest whose range falls inside a
chosen band, so the far coast is never synthesised at all.

Run:  python3 e5b_foreground.py ID [--rmin KM] [--rmax KM]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import skyline as S, skyfix as SF, extract

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      'celestial-navigation', 'theodolite')
FRAC = 0.40
BETA_MAX = 0.004
USE_SAM = False
EDGES = False
DEM = os.path.expanduser('~/.horizonator/DEMs_SRTM1')


def nearness(img):
    """Dark and saturated = near; pale and blue = far."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.max(img, axis=2)
    mn = np.min(img, axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    return (1.0 - mx) + sat - 0.6 * np.clip(b - r, 0, 1)


def foreground_rows(img, hz_rows, frac=0.40, min_run=3):
    """Topmost row per column, above the sea horizon, where the scene
    turns 'near'. Columns with no near material return NaN."""
    n = nearness(img)
    H, W = n.shape
    lo, hi = np.percentile(n, [20, 98])
    thr = lo + frac * (hi - lo)
    out = np.full(W, np.nan)
    for x in range(W):
        top = 0
        bot = int(np.clip(hz_rows[x], 1, H - 1))
        col = n[top:bot, x] > thr
        if col.sum() < min_run:
            continue
        # first row starting a sustained near run
        c = np.convolve(col.astype(float), np.ones(min_run), 'valid')
        k = np.argmax(c >= min_run)
        if c[k] >= min_run:
            out[x] = top + k
    return out


def dem_foreground(dem, lat, lon, z, az, rmin, rmax):
    el, rng = S.visible_layers(dem, lat, lon, z, az, n_layers=4)
    out = np.full(az.size, np.nan)
    for j in range(az.size):
        m = np.isfinite(el[:, j]) & (rng[:, j] >= rmin) & (rng[:, j] <= rmax)
        if m.any():
            out[j] = np.nanmax(el[m, j])
    return out


def run(sid, rmin_km, rmax_km, box=6000.0, step0=250.0):
    idx = {x['id']: x for x in json.load(open(os.path.join(
        HERE, 'out', 'theodolite', 'index.json')))['sightings']}
    s = idx[sid]; e, a = s['exif'], s['attitude']
    lat, lon = e['lat'], e['lon']
    z = max(e.get('alt_m') or 5.0, 2.0)
    img = extract.load_image(os.path.join(PHOTOS, s['raw']))
    H, W, _ = img.shape
    f_px = (W / 2) / np.tan(np.radians(a['fov_deg']) / 2)
    dip = S.horizon_dip_rad(z)
    lvl = extract.sea_horizon_attitude_radon(img, f_px, dip)
    pitch = lvl['pitch_deg'] if lvl else (a.get('pitch_deg') or 0.0)
    roll = lvl['roll_deg'] if lvl else (a.get('roll_deg') or 0.0)
    u = np.arange(W) - (W - 1) / 2
    vv = (np.tan(-dip - np.radians(pitch)) * np.hypot(u, f_px)
          - np.radians(roll) * u)
    hz = (H - 1) / 2 - vv

    if USE_SAM:
        import json as _j
        jp = os.path.join(HERE, 'out', 'e5k', f'{sid}.json')
        with open(jp) as fh:
            saved = _j.load(fh)['rows']
        rows = np.array([np.nan if v is None else v for v in saved])
        # e5k contours are at the photo's native load width; e5b works
        # at the same extract.load_image width, so no rescale needed
        print(f'{sid}: using the MobileSAM contour '
              f'({100*np.isfinite(rows).mean():.0f}% of columns)')
    else:
        rows = foreground_rows(img, hz, frac=FRAC)
    ok = np.isfinite(rows)
    print(f'{sid}: foreground found in {100*ok.mean():.0f}% of columns')
    if ok.sum() < 30:
        print('  too little foreground'); return None
    v = (H - 1) / 2 - rows[ok]
    uu = u[ok]
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    ur, vr = uu * cr - v * sr, uu * sr + v * cr
    az_obs = a['heading_deg'] + np.degrees(np.arctan2(ur, f_px))
    el_obs = np.arctan2(vr, np.hypot(ur, f_px)) + np.radians(pitch)
    o = np.argsort(az_obs); az_obs, el_obs = az_obs[o], el_obs[o]

    if EDGES:
        # height-free observables: the island's angular WIDTH pins the
        # range (width ~ size/distance, immune to the ~7 mrad DEM
        # height bias E4x measured) and its centre bearing pins the
        # direction, with the compass shift clamped to the prior
        okc = np.isfinite(rows)
        span = np.where(okc)[0]
        if span.size < 20:
            print('  too little contour for edges'); return None
        u_all = np.arange(W) - (W - 1) / 2
        azL = a['heading_deg'] + np.degrees(np.arctan2(
            u_all[span[0]], f_px))
        azR = a['heading_deg'] + np.degrees(np.arctan2(
            u_all[span[-1]], f_px))
        w_obs = azR - azL
        c_obs = 0.5 * (azR + azL)
        dem = S.Dem(DEM)
        # the SAME window for model and photo: the DEM run must be
        # truncated by the frame exactly as the observation is, or a
        # feature extending past the frame edge inflates the predicted
        # width (measured: 10.1 deg at truth vs 7.36 observed, with
        # the DEM run reaching 184 deg against a frame ending at 181)
        half = a['fov_deg'] / 2
        az = np.linspace(a['heading_deg'] - half,
                         a['heading_deg'] + half, 500)
        mlat, mlon = S.meters_per_degree(lat)
        g = np.arange(-box / 2, box / 2 + 1, step0)
        cc = np.full((g.size, g.size), np.inf)
        for i, dn in enumerate(g):
            for j, de in enumerate(g):
                fg = dem_foreground(dem, lat + dn / mlat,
                                    lon + de / mlon, z,
                                    az, rmin_km * 1e3, rmax_km * 1e3)
                m = np.isfinite(fg)
                if m.sum() < 5:
                    continue
                idx = np.where(m)[0]
                # longest contiguous run = the island
                brk = np.where(np.diff(idx) > 3)[0]
                runs = np.split(idx, brk + 1)
                r0 = max(runs, key=len)
                wL, wR = az[r0[0]], az[r0[-1]]
                w_pred = wR - wL
                c_pred = 0.5 * (wR + wL)
                dc = np.clip(c_obs - c_pred, -6.0, 6.0)  # compass prior
                cc[i, j] = ((w_obs - w_pred) ** 2
                            + (c_obs - c_pred - dc) ** 2
                            + 0.05 * dc ** 2)
        i, j = np.unravel_index(np.argmin(cc), cc.shape)
        err = float(np.hypot(g[i], g[j]))
        margin = SF.basin_margin(cc, g, min_sep=4 * step0)
        print(f'  EDGE fix: err {err:.0f} m   width_obs {w_obs:.2f} deg'
              f'   cost {cc[i, j]:.3f}   margin {margin:.2f}')
        return dict(id=sid, err_m=err, margin=margin, mode='edges')

    dem = S.Dem(DEM)
    az = np.linspace(az_obs[0] - 8, az_obs[-1] + 8, 400)
    mlat, mlon = S.meters_per_degree(lat)
    g = np.arange(-box / 2, box / 2 + 1, step0)
    shifts = np.arange(-6.0, 6.01, 0.25)
    betas = (np.arange(-BETA_MAX, BETA_MAX * 1.001, max(BETA_MAX / 8, 1e-9))
             if BETA_MAX > 0 else np.array([0.0]))
    cc = np.full((g.size, g.size), np.inf)
    for i, dn in enumerate(g):
        for j, de in enumerate(g):
            fg = dem_foreground(dem, lat + dn / mlat, lon + de / mlon,
                                z, az, rmin_km * 1e3, rmax_km * 1e3)
            m = np.isfinite(fg)
            if m.sum() < 20:
                continue
            best = np.inf
            for sh in shifts:
                pred = np.interp(az_obs + sh, az[m], fg[m],
                                 left=np.nan, right=np.nan)
                k = np.isfinite(pred)
                if k.sum() < 20:
                    continue
                r = el_obs[k] - pred[k]
                for b in betas:
                    rb = np.abs(r - b)
                    h = np.where(rb <= 3e-3, .5 * rb * rb,
                                 3e-3 * (rb - 1.5e-3))
                    best = min(best, float(h.mean()))
            cc[i, j] = best
    i, j = np.unravel_index(np.argmin(cc), cc.shape)
    err = float(np.hypot(g[i], g[j]))
    margin = SF.basin_margin(cc, g, min_sep=4 * step0)
    print(f'  foreground-only fix: err {err:.0f} m   rms '
          f'{np.sqrt(2*cc[i,j])*1e3:.2f} mrad   margin {margin:.2f}   '
          f'(band {rmin_km:.0f}-{rmax_km:.0f} km)')
    return dict(id=sid, err_m=err, margin=margin,
                rms=float(np.sqrt(2 * cc[i, j]) * 1e3),
                band=[rmin_km, rmax_km])


if __name__ == '__main__':
    argv = sys.argv[1:]
    args, skip = [], False
    for k, x in enumerate(argv):
        if skip:
            skip = False
            continue
        if x.startswith('--'):
            skip = x not in ('--sam', '--edges')
            continue
        args.append(x)
    gv = lambda f, d: (float(sys.argv[sys.argv.index(f) + 1])
                       if f in sys.argv else d)
    globals()['FRAC'] = gv('--frac', 0.40)
    globals()['BETA_MAX'] = gv('--beta', 4.0) * 1e-3
    globals()['USE_SAM'] = '--sam' in sys.argv
    globals()['EDGES'] = '--edges' in sys.argv
    out = [run(s, gv('--rmin', 2.0), gv('--rmax', 15.0)) for s in args]
    with open(os.path.join(HERE, 'out', 'e5b_foreground.json'), 'w') as f:
        json.dump([x for x in out if x], f, indent=1)
