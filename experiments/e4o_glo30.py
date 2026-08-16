#!/usr/bin/env python3
"""E4o: Copernicus GLO-30 as an independent DEM family.

Everything so far synthesizes and solves against the same SRTM tiles,
so DEM systematics cancel. Real deployment does not get that favor:
the world is the "rendering DEM" and whatever the instrument carries is
the solving DEM. This measures the cross-family error two ways:

  1. skyline-to-skyline: the same viewpoints marched over SRTM3 and
     over GLO-30 (Copernicus DSM, TanDEM-X era) -> the per-azimuth
     elevation-angle discrepancy, i.e. a MEASURED sigma_DEM for the
     --sigma-dem weighting and the beta window (previously guessed at
     1.5 mrad).
  2. cross-DEM solves: the E4c composites (SRTM-rendered world) solved
     with --dem pointed at GLO-30 -> position error against the
     same-DEM baseline.

GLO-30 tiles are fetched from the AWS open-data bucket and converted to
SRTM3-shaped .hgt (1201x1201 3" big-endian) under
~/.horizonator/DEMs_GLO30. Writes out/e4o_results.json.
"""

import os
import sys
import json
import subprocess
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import skyline as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))
DIRG = os.path.expanduser('~/.horizonator/DEMs_GLO30')
AZ = np.arange(-180.0, 180.0, 0.1) + 0.05

SITES = [  # E4c sea cases + spread coastal viewpoints, z a few meters
    ('strait1', 36.9500, 27.2500, 5.0),
    ('strait2', 36.9622, 27.2384, 5.0),
    ('offshore', 36.6050, 26.8590, 5.0),
    ('gulf_e', 36.9800, 27.4500, 5.0),
    ('kos_s', 36.8300, 27.1500, 5.0),
    ('open_w', 36.7000, 26.9500, 5.0),
]


def fetch_glo30(lat_lo=36, lat_hi=37, lon_lo=26, lon_hi=27):
    os.makedirs(DIRG, exist_ok=True)
    for la in range(lat_lo, lat_hi + 1):
        for lo in range(lon_lo, lon_hi + 1):
            name = f'N{la:02d}E{lo:03d}'
            dst = os.path.join(DIRG, name + '.hgt')
            if os.path.exists(dst):
                continue
            url = ('https://copernicus-dem-30m.s3.amazonaws.com/'
                   f'Copernicus_DSM_COG_10_N{la:02d}_00_E{lo:03d}_00_DEM/'
                   f'Copernicus_DSM_COG_10_N{la:02d}_00_E{lo:03d}_00_DEM.tif')
            tmp = dst + '.tif'
            print('fetching', name, flush=True)
            urllib.request.urlretrieve(url, tmp)
            a = np.asarray(Image.open(tmp), dtype=np.float32)  # 3600^2, 1"
            d = a[::3, ::3]                                    # 1200^2, 3"
            h = np.zeros((1201, 1201), dtype=np.float32)
            h[:1200, :1200] = d
            h[1200, :1200] = d[-1]                             # edge dup:
            h[:1200, 1200] = d[:, -1]                          # <=1 px seam
            h[1200, 1200] = d[-1, -1]
            np.round(h).astype('>i2').tofile(dst)
            os.remove(tmp)


if __name__ == '__main__':
    fetch_glo30()
    cm_s = S.CMarcher(DIR3, (36.4, 37.4), (26.4, 27.9), d_min=1000.)
    cm_g = S.CMarcher(DIRG, (36.4, 37.4), (26.4, 27.9), d_min=1000.)

    out = {'skyline_diff': {}}
    alld = []
    for name, lat, lon, z in SITES:
        el_s, _ = cm_s.skyline(lat, lon, z, AZ)
        el_g, _ = cm_g.skyline(lat, lon, z, AZ)
        # compare where either DEM sees terrain above the sea horizon
        dip = S.horizon_dip_rad(z)
        m = (el_s > -dip + 1e-4) | (el_g > -dip + 1e-4)
        d = (el_g - el_s)[m] * 1e3
        alld.append(d)
        out['skyline_diff'][name] = dict(
            n_az=int(m.sum()), median_mrad=float(np.median(d)),
            rms_mrad=float(np.sqrt(np.mean(d ** 2))),
            p90_abs_mrad=float(np.percentile(np.abs(d), 90)))
        print(f'{name:9s}: terrain az {m.sum():4d}  bias '
              f'{np.median(d):+5.2f} mrad  rms {np.sqrt(np.mean(d**2)):5.2f}'
              f'  p90|d| {np.percentile(np.abs(d), 90):5.2f}', flush=True)
    alld = np.concatenate(alld)
    out['sigma_dem_mrad'] = dict(
        rms=float(np.sqrt(np.mean(alld ** 2))),
        median_abs=float(np.median(np.abs(alld))),
        p90_abs=float(np.percentile(np.abs(alld), 90)))
    print(f"\nmeasured cross-DEM skyline error: rms "
          f"{out['sigma_dem_mrad']['rms']:.2f} mrad, median|d| "
          f"{out['sigma_dem_mrad']['median_abs']:.2f}, p90|d| "
          f"{out['sigma_dem_mrad']['p90_abs']:.2f}")

    # ---- cross-DEM solves: SRTM-rendered composites vs GLO-30 solver
    CASES = [('strait1', 36.9500, 27.2500, 0.5, 0.0, 180.0),
             ('strait2', 36.9622, 27.2384, -0.5, 0.5, 25.0),
             ('offshore', 36.6050, 26.8590, 0.0, 0.0, 60.0)]
    rng = np.random.default_rng(20260824)
    out['cross_solve'] = {}
    for name, lat, lon, pitch, roll, heading in CASES:
        mlat, mlon = S.meters_per_degree(lat)
        off = rng.uniform(-1200, 1200, 2)
        center = f'{lat + off[0] / mlat:.6f},{lon + off[1] / mlon:.6f}'
        row = {}
        for dem, tag in ((DIR3, 'same'), (DIRG, 'cross')):
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, 'skyfix.py'),
                 os.path.join(OUT, 'synth', name + '.jpg'),
                 '--center', center, '--box', '5000',
                 '--pitch', str(pitch), '--roll', str(roll),
                 '--dmin', '1000', '--dem', dem],
                capture_output=True, text=True)
            t = p.stdout
            j = json.loads(t[t.index('{'):t.rindex('}') + 1])
            err = float(np.hypot((j['lat'] - lat) * mlat,
                                 (j['lon'] - lon) * mlon))
            row[tag] = dict(err_m=err, status=j['status'],
                            rms_mrad=j['rms_mrad'],
                            margin=j['basin_margin'])
        out['cross_solve'][name] = row
        print(f"{name:9s}: same-DEM {row['same']['err_m']:5.0f} m "
              f"({row['same']['status']})  ->  GLO-30 "
              f"{row['cross']['err_m']:5.0f} m ({row['cross']['status']}, "
              f"rms {row['cross']['rms_mrad']:.1f} mrad)", flush=True)
    with open(os.path.join(OUT, 'e4o_results.json'), 'w') as f:
        json.dump(out, f, indent=1)
