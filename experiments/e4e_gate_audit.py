#!/usr/bin/env python3
"""E4e: audit the inconclusive gates against the FULL CH1 set (203 photos).

For every CH1 photo (ground-truth masks as the observation, isolating
matching from extraction), solve the coarse 5 km landscape with a full
heading search and score the four skyfix trust gates: basin margin,
boundary railing, residual, skyline relief. Each solve is classified:

  TRUE-ACCEPT    gates pass,  error < 500 m   (genuine convergence)
  FALSE-ACCEPT   gates pass,  error >= 500 m  (overfit slipped through)
  CAUGHT         gates fail,  error >= 500 m  (correctly inconclusive)
  OVER-CAUTIOUS  gates fail,  error < 500 m   (good fix rejected)

The FALSE-ACCEPT rate is the headline number: how often a wrong position
would be reported as a valid fix in the hardest (attitude-free) regime.

Run:   python3 e4e_gate_audit.py     (~3 h for 203 photos; CSV written
incrementally to out/e4e_audit.csv)
"""

import os
import sys
import glob
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import skyline as S
from skyfix import basin_margin


def ensure_tiles(lat, lon, margin_lat=0.7, margin_lon=0.9):
    import math
    import gzip
    import urllib.request
    d1 = os.path.expanduser('~/.horizonator/DEMs_SRTM1')
    os.makedirs(d1, exist_ok=True)
    for la in range(math.floor(lat - margin_lat),
                    math.floor(lat + margin_lat) + 1):
        for lo in range(math.floor(lon - margin_lon),
                        math.floor(lon + margin_lon) + 1):
            t = f"N{la:02d}E{lo:03d}"
            p3 = os.path.join(DIR3, t + '.hgt')
            if os.path.exists(p3):
                continue
            p1 = os.path.join(d1, t + '.hgt')
            if not os.path.exists(p1):
                url = ('https://s3.amazonaws.com/elevation-tiles-prod/'
                       f'skadi/{t[:3]}/{t}.hgt.gz')
                print('  fetching', t, flush=True)
                with urllib.request.urlopen(url) as r:
                    open(p1, 'wb').write(gzip.decompress(r.read()))
            a = np.fromfile(p1, dtype='>i2').reshape(3601, 3601)
            a[::3, ::3].astype('>i2').tofile(p3)

CH1 = '/home/user/celestial-navigation/CH1'
DIR3 = os.path.expanduser('~/.horizonator/DEMs_SRTM3')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
AZ = np.arange(-180., 180., 0.1) + 0.05
BETAS = np.arange(-0.100, 0.1001, 0.010)
BOX = 5000.0
SHIFT_STEP = 8          # 0.8 deg heading sampling in the audit


def audit_photo(meta):
    v = open(meta).read().split('\n')
    f_px, lat_gt, lon_gt = float(v[0]), float(v[1]), float(v[2])
    W, H = int(v[4]), int(v[5])
    mask = np.asarray(Image.open(meta[:-8] + '-mask.png').convert('L')) > 127
    rows = np.where(mask.any(axis=0), mask.argmax(axis=0), H - 1).astype(float)
    u = np.arange(W) - (W - 1) / 2
    az_rel = np.degrees(np.arctan2(u, f_px))
    el_pt = np.arctan2((H - 1) / 2 - rows, np.hypot(u, f_px))
    el_obs = np.zeros(AZ.size)
    wt = np.zeros(AZ.size)
    m = (AZ >= az_rel.min()) & (AZ <= az_rel.max())
    el_obs[m] = np.interp(AZ[m], az_rel, el_pt)
    wt[m] = 1.0
    relief = float(np.std(el_obs[wt > 0]) * 1e3)

    ensure_tiles(lat_gt, lon_gt)
    cm = S.CMarcher(DIR3, (lat_gt - .7, lat_gt + .7),
                    (lon_gt - .9, lon_gt + .9), d_min=1000.)
    mlat, mlon = S.meters_per_degree(lat_gt)

    def z_at(la, lo):
        y = int(round((cm.lat_nw - la) / cm.dpp))
        x = int(round((lo - cm.lon_nw) / cm.dpp))
        return float(cm.mosaic[np.clip(y, 0, cm.mosaic.shape[0] - 1),
                               np.clip(x, 0, cm.mosaic.shape[1] - 1)]) + 2.

    def C(dn, de):
        la, lo = lat_gt + dn / mlat, lon_gt + de / mlon
        el, _ = cm.skyline(la, lo, z_at(la, lo), AZ)
        best = np.inf
        for s in range(-1800, 1800, SHIFT_STEP):
            eo = np.roll(el_obs, s)
            ww = np.roll(wt, s)
            mm = ww > 0
            r = el[mm] - eo[mm]
            rb = np.abs(r[None, :] - BETAS[:, None])
            h = np.where(rb <= 3e-3, .5 * rb * rb, 3e-3 * (rb - 1.5e-3))
            cval = h.mean(1).min()
            if cval < best:
                best = cval
        return best

    step0 = 250.0
    g = np.arange(-BOX / 2, BOX / 2 + 1, step0)
    cc = np.array([[C(dn, de) for de in g] for dn in g])
    i, j = np.unravel_index(np.argmin(cc), cc.shape)
    dn0, de0 = g[i], g[j]
    err = float(np.hypot(dn0, de0))
    margin = basin_margin(cc, g, min_sep=4 * step0)
    boundary = max(abs(dn0), abs(de0)) >= BOX / 2 - step0
    rms = float(np.sqrt(2 * cc[i, j]) * 1e3)
    ok = (margin >= 0.15) and (not boundary) and (rms <= 12.0) \
        and (relief >= 1.5)
    correct = err < 500
    verdict = ('TRUE-ACCEPT' if correct else 'FALSE-ACCEPT') if ok else \
              ('OVER-CAUTIOUS' if correct else 'CAUGHT')
    return err, margin, boundary, rms, relief, verdict


if __name__ == '__main__':
    metas = sorted(glob.glob(os.path.join(CH1, '*', '*.png.txt')))
    metas = [m for m in metas if os.path.exists(m[:-8] + '-mask.png')]
    if len(sys.argv) > 1:
        metas = metas[:int(sys.argv[1])]
    print(f'{len(metas)} photos')
    csv = open(os.path.join(OUT, 'e4e_audit.csv'), 'w', buffering=1)
    csv.write('photo,err_m,margin,boundary,rms_mrad,relief_mrad,verdict\n')
    counts = {}
    t0 = time.time()
    for n, meta in enumerate(metas):
        name = os.path.basename(meta)[:-8]
        try:
            err, margin, boundary, rms, relief, verdict = audit_photo(meta)
        except Exception as e:
            print(f'{name} ERROR: {e}', flush=True)
            continue
        counts[verdict] = counts.get(verdict, 0) + 1
        csv.write(f'{name},{err:.0f},{margin:.3f},{int(boundary)},'
                  f'{rms:.1f},{relief:.1f},{verdict}\n')
        print(f'[{n+1}/{len(metas)} {(time.time()-t0)/60:.0f}min] '
              f'{name}: err {err:6.0f} m  {verdict}', flush=True)
    print('\nconfusion:', counts)
    total_ok = counts.get('TRUE-ACCEPT', 0) + counts.get('FALSE-ACCEPT', 0)
    if total_ok:
        print(f'false-accept rate among accepted fixes: '
              f'{counts.get("FALSE-ACCEPT", 0)}/{total_ok}')
