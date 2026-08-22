#!/usr/bin/env python3
"""E5r: per-peak height residuals — the vertical-angle check as a
diagnostic, not a re-solve.

After the matcher has placed a frame (E4v fixes), every matched crest
carries one more number the fix did not isolate: the difference
between the peak's apparent angular height and the DEM's prediction,
converted to METERS via the crest's range. The profile cost already
consumed these pixels, so this is not new position information — it
is a decomposition of the residual into the two things a skyline can
be wrong about:

  a GLOBAL offset (weighted median residual) — pitch-prior error,
  refraction, or a uniform DEM bias; inseparable in one frame (E5n)
  PER-PEAK height residuals after that offset — how much taller or
  shorter each crest stands in the photograph than in the model,
  in meters, at its own range

Aggregated over every solved field frame, the per-peak numbers are a
field-measured map of DEM-vs-world height error (is the canopy bias
real? how big? which site?), and the per-frame spread is a candidate
error predictor to hold against dem_split (E5o).

Run:  python3 e5r_peaks.py     (writes out/e5r/*.png, peaks.json)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import extract
import skyline as S
import skyfix as SF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'e5r')
PHOTOS = os.environ.get(
    'THEODOLITE_DIR',
    os.path.join(os.path.dirname(os.path.dirname(HERE)),
                 'celestial-navigation', 'theodolite'))
INDEX = os.path.join(HERE, 'out', 'theodolite', 'index.json')
RESULTS = os.path.join(HERE, 'out', 'e4v_results_mask_all.json')
DEM = os.path.expanduser('~/.horizonator/DEMs_SRTM1_WM')
RE_EFF = 6371e3 / (1 - 0.13)             # standard refraction k=0.13

_marchers = {}


def marcher(lat, lon):
    key = (round(lat, 1), round(lon, 1))
    if key not in _marchers:
        _marchers[key] = S.CMarcher(DEM, (lat - 0.6, lat + 0.6),
                                    (lon - 0.8, lon + 0.8))
    return _marchers[key]


def frame_peaks(s, r):
    """One solved frame -> list of per-peak height residuals."""
    e, a = s['exif'], s['attitude']
    img = extract.load_image(os.path.join(PHOTOS, s['raw']))
    z = max(e.get('alt_m') or 5.0, 2.0)
    dip = S.horizon_dip_rad(z)
    heading = r['heading'] + (r.get('heading_offset') or 0.0)
    el_obs, w, _ = SF.observation(img, r['fov'], heading,
                                  r['level_roll'], r['level_pitch'])
    m = w > 0.2
    if m.sum() < 30:
        return None
    mlat = 111132.0
    mlon = 111320.0 * np.cos(np.radians(e['lat']))
    lat_f = e['lat'] + r['dn'] / mlat
    lon_f = e['lon'] + r['de'] / mlon
    cm = marcher(e['lat'], e['lon'])
    az = SF.AZ[m]
    el_syn, rng = cm.skyline(lat_f, lon_f, z, az)
    good = np.isfinite(el_syn) & (el_syn > -dip + 1.5e-3) & \
        np.isfinite(el_obs[m])
    if good.sum() < 20:
        return None
    res = el_obs[m] - el_syn
    beta = float(np.median(res[good]))     # global offset, mrad-scale
    if abs(beta) > 0.1:
        # a tenth of a radian of frame-wide offset is not attitude or
        # DEM bias, it is a garbage extraction — refuse the frame
        return None

    # a frame that crosses +-180 deg true lands in AZ as disjoint
    # segments; peaks are found per contiguous segment so the seam
    # never fabricates a crest
    brk = np.where(np.diff(az) > 0.3)[0] + 1
    segs = np.split(np.arange(az.size), brk)
    pk, prom = [], []
    for sg in segs:
        if sg.size < 8:
            continue
        p_, pr_ = find_peaks(np.where(good[sg], el_syn[sg], -1.0),
                             prominence=2e-3, distance=10)
        pk += list(sg[0] + p_)
        prom += list(pr_['prominences'])
    props = dict(prominences=np.array(prom))
    seg_of = np.zeros(az.size, int)
    for k, sg in enumerate(segs):
        seg_of[sg] = k
    out = []
    for p in pk:
        sl = slice(max(p - 3, 0), p + 4)
        if not good[sl].all() or len(set(seg_of[sl])) > 1:
            continue
        R = float(rng[p])
        dh = float((np.mean(res[sl]) - beta) * R)
        H = float((el_syn[p] + dip) * R + R * R / (2 * RE_EFF))
        out.append(dict(az=float(az[p]), R_km=R / 1000.0, H_m=H,
                        dh_m=dh, prom_mrad=float(
                            props['prominences'][list(pk).index(p)] * 1e3)))
    if not out:
        return None
    return dict(id=r['id'], err_m=r['err_m'], margin=r['margin'],
                fov=r['fov'], site=r['site'], beta_mrad=beta * 1e3,
                z=z, peaks=out,
                # kept for the gallery
                _az=az, _obs=el_obs[m], _syn=el_syn, _good=good,
                _dip=dip)


def gallery(frames):
    picks = sorted(frames, key=lambda f: f['err_m'])
    idx = np.linspace(0, len(picks) - 1, 6).astype(int)
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    for ax, f in zip(axes.ravel(), [picks[i] for i in idx]):
        az, obs, syn, good = f['_az'], f['_obs'], f['_syn'], f['_good']
        obs = np.where(good, obs, np.nan).copy()
        syn = np.where(good, syn, np.nan).copy()
        gap = np.where(np.diff(az) > 0.3)[0] + 1   # break at az seams
        obs[gap] = np.nan
        syn2 = syn.copy()
        syn2[gap] = np.nan
        ax.plot(az, obs * 1e3, color='#ff3b30',
                lw=1.1, label='photograph')
        ax.plot(az, (syn2 + f['beta_mrad'] / 1e3) * 1e3,
                color='#0a84ff', lw=1.1,
                label='DEM at the fix (+offset)')
        ax.axhline(-f['_dip'] * 1e3, color='#8e8e93', ls=':', lw=0.8)
        for p in f['peaks']:
            i = np.argmin(np.abs(az - p['az']))
            ax.annotate(f"{p['dh_m']:+.0f} m",
                        (p['az'], (syn[i] + f['beta_mrad'] / 1e3) * 1e3),
                        fontsize=8, color='#1c1c1e', fontweight='bold',
                        textcoords='offset points', xytext=(-8, 10))
            ax.plot(p['az'], syn[i] * 1e3 + f['beta_mrad'], 'v',
                    color='#ff9f0a', ms=6)
        ax.set_title(f"{f['id']}  err {f['err_m']:.0f} m  offset "
                     f"{f['beta_mrad']:+.1f} mrad  "
                     f"{len(f['peaks'])} peaks", fontsize=9)
        ax.set_xlabel('azimuth (deg true)')
        ax.set_ylabel('elevation (mrad)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    p = os.path.join(OUT, 'gallery.png')
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(p)


def summary(frames):
    all_dh = np.array([p['dh_m'] for f in frames for p in f['peaks']])
    med = np.array([np.median([p['dh_m'] for p in f['peaks']])
                    for f in frames])
    mad = np.array([np.median(np.abs(np.array(
        [p['dh_m'] for p in f['peaks']]) - m))
        for f, m in zip(frames, med)])
    err = np.array([f['err_m'] for f in frames])
    mg = np.array([f['margin'] for f in frames])

    def rank(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return np.corrcoef(ra, rb)[0, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = axes[0]
    ax.hist(np.clip(all_dh, -80, 80), bins=40, color='#0a84ff',
            alpha=0.85)
    ax.axvline(np.median(all_dh), color='#ff3b30', lw=1.5,
               label=f'median {np.median(all_dh):+.1f} m')
    ax.set_xlabel('peak height residual, photo - DEM (m)')
    ax.set_ylabel('peaks')
    ax.set_title(f'{all_dh.size} matched crests on {len(frames)} '
                 f'frames', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    for ax, x, name in ((axes[1], np.abs(med), '|median dh| per frame'),
                        (axes[2], mad, 'peak-residual spread (MAD)')):
        ax.scatter(x, err, s=30, c='#0a84ff', alpha=0.8)
        ax.set_xlabel(name + ' (m)')
        ax.set_ylabel('fix error (m)')
        ax.set_yscale('log')
        ax.set_title(f'rank corr with error: {rank(x, err):+.2f} '
                     f'(margin: {rank(-mg, err):+.2f})', fontsize=10)
        ax.grid(alpha=0.25, which='both')

    fig.tight_layout()
    p = os.path.join(OUT, 'summary.png')
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(p)
    print(f'\nall peaks: median {np.median(all_dh):+.1f} m, '
          f'IQR {np.percentile(all_dh, 25):+.1f}..'
          f'{np.percentile(all_dh, 75):+.1f} m')
    for site in sorted({f['site'] for f in frames}):
        d = np.array([p['dh_m'] for f in frames if f['site'] == site
                      for p in f['peaks']])
        if d.size >= 5:
            print(f'  site {site}: {d.size:3d} peaks, median '
                  f'{np.median(d):+6.1f} m')
    print(f'per-frame |median dh| vs err rank corr '
          f'{rank(np.abs(med), err):+.2f}; MAD vs err '
          f'{rank(mad, err):+.2f}; margin vs err {rank(-mg, err):+.2f}')


if __name__ == '__main__':
    with open(INDEX) as fh:
        idx = {s['id']: s for s in json.load(fh)['sightings']}
    with open(RESULTS) as fh:
        rows = json.load(fh)
    frames = []
    for r in rows:
        if r['id'] not in idx:
            continue
        try:
            f = frame_peaks(idx[r['id']], r)
        except Exception as ex:
            print(f"  {r['id']}: {ex}")
            continue
        if f:
            frames.append(f)
            print(f"{f['id']}: {len(f['peaks'])} peaks, offset "
                  f"{f['beta_mrad']:+.1f} mrad, dh "
                  f"{[round(p['dh_m']) for p in f['peaks']]}",
                  flush=True)
    print(f'\n{len(frames)} frames usable of {len(rows)} solved')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'peaks.json'), 'w') as fh:
        json.dump([{k: v for k, v in f.items()
                    if not k.startswith('_')} for f in frames], fh,
                  indent=1)
    gallery(frames)
    summary(frames)
