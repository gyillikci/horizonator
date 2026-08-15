#!/usr/bin/env python3
"""E4d: REAL-photo validation on the CH1 benchmark (Baatz/Saurer, ECCV'12
"Large Scale Visual Geo-Localization of Images in Mountainous Terrain") —
the dataset the user pushed to celestial-navigation:main under CH1/.

Per photo, CH1 provides the calibrated focal length (pixels), ground-truth
lat/lon, and a hand-made sky mask. This is a harder regime than the
study's maritime scenario: terrestrial observers standing on alpine
relief, UNKNOWN heading (full-circle azimuth search), unknown pitch/roll
(elevation offset window widened to +-100 mrad — affordable here because
high-relief skylines carry strong shape information). Observer height
follows the DEM per candidate (+2 m).

Each photo is solved twice over a 5 km box centered on the ground truth:
with the automatic extractor (extract.skyline_seam) and with the
ground-truth sky mask — separating extraction error from matching error.

Run:   python3 e4d_ch1.py [N_PHOTOS]     (default 12; no GL needed)
"""

import os
import sys
import json
import glob
import gzip
import time
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import skyline as S
import extract

CH1 = os.path.expanduser('/home/user/celestial-navigation/CH1')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
DIR1 = os.path.expanduser('~/.horizonator/DEMs_SRTM1')
DIR3 = os.path.expanduser('~/.horizonator/DEMs_SRTM3')

BOX = 5000.0
AZ = np.arange(-180.0, 180.0, 0.1) + 0.05
BETAS = np.arange(-0.300, 0.3001, 0.015)  # handheld pitch reaches +-15 deg
N_PHOTOS = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def ensure_tiles(lat, lon, margin_lat=0.7, margin_lon=0.9):
    import math
    for la in range(math.floor(lat - margin_lat), math.floor(lat + margin_lat) + 1):
        for lo in range(math.floor(lon - margin_lon), math.floor(lon + margin_lon) + 1):
            t = f"N{la:02d}E{lo:03d}"
            p3 = os.path.join(DIR3, t + '.hgt')
            if os.path.exists(p3):
                continue
            p1 = os.path.join(DIR1, t + '.hgt')
            if not os.path.exists(p1):
                url = f'https://s3.amazonaws.com/elevation-tiles-prod/skadi/{t[:3]}/{t}.hgt.gz'
                print('  fetching', t, flush=True)
                with urllib.request.urlopen(url) as r:
                    open(p1, 'wb').write(gzip.decompress(r.read()))
            a = np.fromfile(p1, dtype='>i2').reshape(3601, 3601)
            a[::3, ::3].astype('>i2').tofile(p3)


def obs_from_rows(rows, conf, W, H, f_px):
    """pixel rows of the skyline -> (el_obs, weights) on the global azimuth
    grid, camera heading placed at azimuth 0 (heading is searched)."""
    u = np.arange(len(rows)) * (W / len(rows)) - (W - 1) / 2
    v = (H - 1) / 2 - rows * (W / len(rows))   # rows are in resized px
    az_rel = np.degrees(np.arctan2(u, f_px))
    el_pt = np.arctan2(v, np.hypot(u, f_px))
    el = np.full(AZ.size, np.nan)
    wt = np.zeros(AZ.size)
    m = (AZ >= az_rel.min()) & (AZ <= az_rel.max())
    el[m] = np.interp(AZ[m], az_rel, el_pt)
    wt[m] = np.interp(AZ[m], az_rel, conf)
    wt = wt / (wt[m].max() + 1e-9)
    return np.where(np.isfinite(el), el, 0.0), wt


def pcost(el_obs, w, el_syn, shifts):
    best = (np.inf, 0, 0.0)
    wsum0 = None
    for s in shifts:
        eo = np.roll(el_obs, s)
        ww = np.roll(w, s)
        m = ww > 0
        r = el_syn[m] - eo[m]
        wm = ww[m]
        rb = np.abs(r[None, :] - BETAS[:, None])
        h = np.where(rb <= 3e-3, 0.5 * rb * rb, 3e-3 * (rb - 1.5e-3))
        c = (h * wm[None, :]).sum(1) / wm.sum()
        i = int(np.argmin(c))
        if c[i] < best[0]:
            best = (float(c[i]), s, float(BETAS[i]))
    return best


metas = sorted(glob.glob(os.path.join(CH1, 'cvg', '*.png.txt')))
metas = [m for m in metas
         if os.path.exists(m[:-8] + '-mask.png')][::4][:N_PHOTOS]
print(f'{len(metas)} photos selected')

results = []
for meta in metas:
    v = open(meta).read().split('\n')
    f_px, lat_gt, lon_gt = float(v[0]), float(v[1]), float(v[2])
    W, H = int(v[4]), int(v[5])
    name = v[6]
    img_path = meta[:-4]
    fov = np.degrees(2 * np.arctan((W / 2) / f_px))
    print(f'{name}: fov {fov:.1f} deg, gt ({lat_gt:.4f},{lon_gt:.4f})',
          flush=True)
    ensure_tiles(lat_gt, lon_gt)

    cm = S.CMarcher(DIR3, (lat_gt - 0.7, lat_gt + 0.7),
                    (lon_gt - 0.9, lon_gt + 0.9), d_min=1000.0)
    mlat, mlon = S.meters_per_degree(lat_gt)

    def z_at(lat, lon):
        y = int(round((cm.lat_nw - lat) / cm.dpp))
        x = int(round((lon - cm.lon_nw) / cm.dpp))
        return float(cm.mosaic[np.clip(y, 0, cm.mosaic.shape[0] - 1),
                               np.clip(x, 0, cm.mosaic.shape[1] - 1)]) + 2.0

    # observation A: automatic extraction; observation B: ground-truth mask
    img = extract.load_image(img_path)
    scale = img.shape[1] / W
    rows_a, conf_a = extract.skyline_seam(img)
    obs_a = obs_from_rows(rows_a / scale, conf_a, W, H, f_px)

    mask = np.asarray(Image.open(meta[:-8] + '-mask.png').convert('L'))
    terr = mask > 127
    rows_b = np.where(terr.any(axis=0), terr.argmax(axis=0), H - 1)
    obs_b = obs_from_rows(rows_b.astype(float) * (img.shape[1] / W) / scale,
                          np.ones(W), W, H, f_px)

    t0 = time.time()
    shifts_all = range(-1800, 1800, 4)

    def solve(el_obs, w):
        def C(dn, de, shifts):
            la = lat_gt + dn / mlat
            lo = lon_gt + de / mlon
            el, _ = cm.skyline(la, lo, z_at(la, lo), AZ)
            return pcost(el_obs, w, el, shifts)
        g = np.arange(-BOX / 2, BOX / 2 + 1, 250.0)
        cc = [[C(dn, de, shifts_all)[0] for de in g] for dn in g]
        cc = np.array(cc)
        i, j = np.unravel_index(np.argmin(cc), cc.shape)
        dn0, de0 = g[i], g[j]
        _, s0, _ = C(dn0, de0, shifts_all)
        near = range(s0 - 30, s0 + 31, 2)
        for step in (50.0, 12.5):
            best = (np.inf, dn0, de0)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    c = C(dn0 + di * step, de0 + dj * step, near)[0]
                    if c < best[0]:
                        best = (c, dn0 + di * step, de0 + dj * step)
            _, dn0, de0 = best
        return best[0], dn0, de0

    ca, dna, dea = solve(*obs_a)
    cb, dnb, deb = solve(*obs_b)
    err_a = float(np.hypot(dna, dea))
    err_b = float(np.hypot(dnb, deb))
    results.append(dict(name=name, fov=fov, err_extract=err_a,
                        err_mask=err_b,
                        rms_extract=float(np.sqrt(2 * ca) * 1e3),
                        rms_mask=float(np.sqrt(2 * cb) * 1e3)))
    print(f'  extractor: err {err_a:7.1f} m (rms {np.sqrt(2*ca)*1e3:.1f}) | '
          f'GT mask: err {err_b:7.1f} m (rms {np.sqrt(2*cb)*1e3:.1f})  '
          f'{time.time()-t0:.0f}s', flush=True)

ea = np.array([r['err_extract'] for r in results])
eb = np.array([r['err_mask'] for r in results])
print(f'\nextractor: median {np.median(ea):.0f} m, <200m: {(ea<200).sum()}/{len(ea)}')
print(f'GT mask:   median {np.median(eb):.0f} m, <200m: {(eb<200).sum()}/{len(eb)}')
with open(os.path.join(OUT, 'e4d_results.json'), 'w') as f:
    json.dump(results, f, indent=1)
print('wrote', os.path.join(OUT, 'e4d_results.json'))
