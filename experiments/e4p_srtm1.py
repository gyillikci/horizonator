#!/usr/bin/env python3
"""E4p: SRTM1 at full resolution — the free accuracy upgrade.

The pipeline has always downloaded 1-arcsecond SRTM tiles from the AWS
skadi bucket and DECIMATED them to 3" before use. The mosaic loader
handles 3601-square tiles natively, so pointing the marcher at the
undecimated tiles is a directory switch (CMarcher now also scales its
march step to the DEM posting: 90 m at 3", 40 m at 1"). This measures
what the decimation has been costing, three ways:

  A  skyline-to-skyline: SRTM3 vs SRTM1 at the E4o coastal viewpoints
     — the elevation-angle detail 3" was throwing away, plus timing.
  B  the E4c sea composites solved with --dem DEMs_SRTM1.
  C  a 20-photo CH1 subset (E4f instrumented regime) solved on both.

Writes out/e4p_results.json.
Run:   python3 e4p_srtm1.py
"""

import os
import sys
import glob
import gzip
import json
import math
import subprocess
import time
import urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import skyline as S
from skyfix import basin_margin, fast_photo_cost
from e4e_gate_audit import CH1, OUT, AZ, BOX

HERE = os.path.dirname(os.path.abspath(__file__))
DIR3 = os.path.expanduser('~/.horizonator/DEMs_SRTM3')
DIR1 = os.path.expanduser('~/.horizonator/DEMs_SRTM1')


def ensure_tiles1(lat, lon, mlat=0.7, mlon=0.9):
    os.makedirs(DIR1, exist_ok=True)
    for la in range(math.floor(lat - mlat), math.floor(lat + mlat) + 1):
        for lo in range(math.floor(lon - mlon),
                        math.floor(lon + mlon) + 1):
            t = f'N{la:02d}E{lo:03d}'
            p = os.path.join(DIR1, t + '.hgt')
            if os.path.exists(p):
                continue
            url = ('https://s3.amazonaws.com/elevation-tiles-prod/'
                   f'skadi/{t[:3]}/{t}.hgt.gz')
            print('  fetching SRTM1', t, flush=True)
            with urllib.request.urlopen(url) as r:
                open(p, 'wb').write(gzip.decompress(r.read()))


SITES = [('strait1', 36.9500, 27.2500, 5.0),
         ('strait2', 36.9622, 27.2384, 5.0),
         ('offshore', 36.6050, 26.8590, 5.0),
         ('gulf_e', 36.9800, 27.4500, 5.0),
         ('kos_s', 36.8300, 27.1500, 5.0),
         ('open_w', 36.7000, 26.9500, 5.0)]

if __name__ == '__main__':
    out = {}
    ensure_tiles1(36.9, 27.1)
    cm3 = S.CMarcher(DIR3, (36.4, 37.4), (26.4, 27.9), d_min=1000.)
    cm1 = S.CMarcher(DIR1, (36.4, 37.4), (26.4, 27.9), d_min=1000.)
    print(f'd_step: SRTM3 {cm3.d_step:.0f} m, SRTM1 {cm1.d_step:.0f} m')

    # ---- A: skyline detail + timing
    diffs, t3, t1 = [], [], []
    out['sites'] = {}
    for name, lat, lon, z in SITES:
        t0 = time.time()
        el3, _ = cm3.skyline(lat, lon, z, AZ)
        t3.append(time.time() - t0)
        t0 = time.time()
        el1, _ = cm1.skyline(lat, lon, z, AZ)
        t1.append(time.time() - t0)
        dip = S.horizon_dip_rad(z)
        m = (el3 > -dip + 1e-4) | (el1 > -dip + 1e-4)
        d = (el1 - el3)[m] * 1e3
        diffs.append(d)
        out['sites'][name] = dict(rms_mrad=float(np.sqrt(np.mean(d**2))),
                                  p90_abs=float(np.percentile(np.abs(d),
                                                              90)))
        print(f'{name:9s}: rms {out["sites"][name]["rms_mrad"]:5.2f} mrad'
              f'  p90|d| {out["sites"][name]["p90_abs"]:5.2f}', flush=True)
    alld = np.concatenate(diffs)
    out['detail_gain'] = dict(
        rms_mrad=float(np.sqrt(np.mean(alld**2))),
        median_abs=float(np.median(np.abs(alld))),
        ms_skyline_srtm3=float(np.mean(t3) * 1e3),
        ms_skyline_srtm1=float(np.mean(t1) * 1e3))
    print(f"\nA: decimation was hiding {out['detail_gain']['rms_mrad']:.2f}"
          f" mrad rms of skyline detail; skyline "
          f"{out['detail_gain']['ms_skyline_srtm3']:.0f} -> "
          f"{out['detail_gain']['ms_skyline_srtm1']:.0f} ms\n", flush=True)

    # ---- B: E4c sea composites on both DEMs
    CASES = [('strait1', 36.9500, 27.2500, 0.5, 0.0),
             ('strait2', 36.9622, 27.2384, -0.5, 0.5),
             ('offshore', 36.6050, 26.8590, 0.0, 0.0)]
    rng = np.random.default_rng(20260825)
    out['e4c'] = {}
    for name, lat, lon, pitch, roll in CASES:
        mla, mlo = S.meters_per_degree(lat)
        off = rng.uniform(-1200, 1200, 2)
        center = f'{lat + off[0] / mla:.6f},{lon + off[1] / mlo:.6f}'
        row = {}
        for dem, tag in ((DIR3, 'srtm3'), (DIR1, 'srtm1')):
            t0 = time.time()
            p = subprocess.run(
                [sys.executable, os.path.join(HERE, 'skyfix.py'),
                 os.path.join(OUT, 'synth', name + '.jpg'),
                 '--center', center, '--box', '5000',
                 '--pitch', str(pitch), '--roll', str(roll),
                 '--dmin', '1000', '--dem', dem],
                capture_output=True, text=True)
            t = p.stdout
            j = json.loads(t[t.index('{'):t.rindex('}') + 1])
            err = float(np.hypot((j['lat'] - lat) * mla,
                                 (j['lon'] - lon) * mlo))
            row[tag] = dict(err_m=err, rms_mrad=j['rms_mrad'],
                            margin=j['basin_margin'],
                            t_s=time.time() - t0)
        out['e4c'][name] = row
        print(f"B {name:9s}: SRTM3 {row['srtm3']['err_m']:5.0f} m "
              f"({row['srtm3']['t_s']:.0f}s)  ->  SRTM1 "
              f"{row['srtm1']['err_m']:5.0f} m ({row['srtm1']['t_s']:.0f}s,"
              f" rms {row['srtm1']['rms_mrad']:.1f} mrad)", flush=True)

    # ---- C: CH1 subset, instrumented regime, both DEMs
    metas = sorted(glob.glob(os.path.join(CH1, '*', '*.png.txt')))
    metas = [m for m in metas if os.path.exists(m[:-8] + '-mask.png')][:20]
    rng = np.random.default_rng(20260819)
    out['ch1'] = {}
    print('', flush=True)
    for meta in metas:
        v = open(meta).read().split('\n')
        f_px, lat_gt, lon_gt = float(v[0]), float(v[1]), float(v[2])
        W, H = int(v[4]), int(v[5])
        mask = np.asarray(Image.open(meta[:-8] + '-mask.png')
                          .convert('L')) > 127
        rows = np.where(mask.any(axis=0), mask.argmax(axis=0),
                        H - 1).astype(float)
        u = np.arange(W) - (W - 1) / 2
        az_rel = np.degrees(np.arctan2(u, f_px))
        el_pt = np.arctan2((H - 1) / 2 - rows, np.hypot(u, f_px))
        el_obs = np.zeros(AZ.size)
        wt = np.zeros(AZ.size)
        mm = (AZ >= az_rel.min()) & (AZ <= az_rel.max())
        el_obs[mm] = np.interp(AZ[mm], az_rel, el_pt)
        wt[mm] = 1.0
        ensure_tiles1(lat_gt, lon_gt)
        name = os.path.basename(meta)[:-8]
        row = {}
        for dem, tag in ((DIR3, 'srtm3'), (DIR1, 'srtm1')):
            cm = S.CMarcher(dem, (lat_gt - .7, lat_gt + .7),
                            (lon_gt - .9, lon_gt + .9), d_min=1000.)
            mla, mlo = S.meters_per_degree(lat_gt)

            def z_at(la, lo):
                y = int(round((cm.lat_nw - la) / cm.dpp))
                x = int(round((lo - cm.lon_nw) / cm.dpp))
                return float(cm.mosaic[
                    np.clip(y, 0, cm.mosaic.shape[0] - 1),
                    np.clip(x, 0, cm.mosaic.shape[1] - 1)]) + 2.

            def skyl(dn, de):
                la, lo = lat_gt + dn / mla, lon_gt + de / mlo
                el, _ = cm.skyline(la, lo, z_at(la, lo), AZ)
                return el

            el_gt = skyl(0.0, 0.0)
            _, s_true, b_true = fast_photo_cost(
                el_obs, wt, el_gt, range(-1800, 1800, 4),
                np.arange(-0.100, 0.1001, 0.010))
            s_c = s_true + int(round(rng.normal(0, 1.0) / 0.1))
            b_c = b_true + rng.normal(0, np.radians(0.5))
            shifts = range(s_c - 20, s_c + 21, 2)
            betas = np.arange(b_c - 0.0175, b_c + 0.0176, 0.0035)
            g = np.arange(-BOX / 2, BOX / 2 + 1, 250.0)
            cc = np.array([[fast_photo_cost(el_obs, wt, skyl(dn, de),
                                            shifts, betas)[0]
                            for de in g] for dn in g])
            i, j = np.unravel_index(np.argmin(cc), cc.shape)
            row[tag] = dict(err=float(np.hypot(g[i], g[j])),
                            margin=float(basin_margin(cc, g,
                                                      min_sep=1000.0)))
        out['ch1'][name] = row
        print(f"C {name[:24]:24s}: SRTM3 {row['srtm3']['err']:5.0f} m "
              f"(margin {row['srtm3']['margin']:5.2f})  SRTM1 "
              f"{row['srtm1']['err']:5.0f} m "
              f"(margin {row['srtm1']['margin']:5.2f})", flush=True)

    e3 = np.array([r['srtm3']['err'] for r in out['ch1'].values()])
    e1 = np.array([r['srtm1']['err'] for r in out['ch1'].values()])
    m3 = np.array([r['srtm3']['margin'] for r in out['ch1'].values()])
    m1 = np.array([r['srtm1']['margin'] for r in out['ch1'].values()])
    out['ch1_summary'] = dict(
        median3=float(np.median(e3)), median1=float(np.median(e1)),
        within500_3=int((e3 < 500).sum()), within500_1=int((e1 < 500).sum()),
        med_margin3=float(np.median(m3)), med_margin1=float(np.median(m1)))
    print(f"\nC summary (n=20): median {np.median(e3):.0f} -> "
          f"{np.median(e1):.0f} m; <500 m {(e3 < 500).sum()} -> "
          f"{(e1 < 500).sum()}; median margin {np.median(m3):.2f} -> "
          f"{np.median(m1):.2f}")
    with open(os.path.join(OUT, 'e4p_results.json'), 'w') as f:
        json.dump(out, f, indent=1)
