#!/usr/bin/env python3
"""E5bd: subject-range pre-screen and truth diagnosis for one frame.

The E5ba false accept showed that the four shipped gates all stay
silent on a near-field scene while the DEM has essentially no shape
agreement at the TRUE position. The two quantities that do separate
such a frame are computed here, as a diagnostic — not as a gate:

  pre-screen (needs no photograph): ray-cast the DEM horizon over the
  frame's azimuth span and report the subject-range distribution, the
  fraction of the horizon inside --near, and the relief.

  truth diagnosis (needs the photograph and the true position):
  constant offset, residual rms after removing it, and the Pearson
  correlation between the DEM and photo elevation profiles. Healthy
  frames in this campaign run 0.90-0.995; BD9 ran +0.097.

usage: e5bd_prescreen.py PHOTO LAT LON Z HEADING FOV [--dem D] [--near M]
       (pass PHOTO as '-' to run the pre-screen alone)
"""

import argparse
import os

import numpy as np

import skyline as S


def prescreen(cm, lat, lon, z, heading, fov, near_m=3000.0, az_step=0.1):
    """Ray-cast the DEM horizon across the frame and summarise it."""
    half = fov / 2.0
    az = (heading + np.arange(-half, half + 1e-9, az_step)) % 360.0
    el, rng = cm.skyline(lat, lon, z, az)   # skyline() takes DEGREES
    ok = np.isfinite(rng) & (rng > 0) & np.isfinite(el)
    if ok.sum() < 10:
        raise SystemExit('the DEM returns no horizon here')
    r = rng[ok]
    return {
        'az': az, 'el': el, 'rng': rng, 'ok': ok,
        'p10': np.percentile(r, 10) / 1e3,
        'p50': np.percentile(r, 50) / 1e3,
        'p90': np.percentile(r, 90) / 1e3,
        'near_frac': float((r < near_m).mean()),
        'relief_mrad': float((el[ok].max() - el[ok].min()) * 1e3),
    }


def diagnose(photo, pre, fov):
    """Compare the photo's elevation profile with the DEM's, at truth."""
    import extract
    import skyfix

    img = extract.load_image(photo)
    H, W = img.shape[:2]
    rows, conf = skyfix.extract_boundary(img)
    rows, conf = np.asarray(rows, float), np.asarray(conf, float)
    f_px = (W / 2) / np.tan(np.radians(fov) / 2)
    el_obs = np.arctan2((H / 2) - rows, f_px)        # +up, radians

    # resample the DEM profile onto the photo's columns: column 0 is the
    # LEFT edge, i.e. heading - fov/2, matching prescreen's az grid
    x = np.arange(W)
    az_col = np.linspace(0.0, len(pre['az']) - 1.0, W)
    el_syn = np.interp(az_col, np.arange(len(pre['az'])), pre['el'])
    good = (conf > 0) & np.isfinite(el_syn)
    if good.sum() < 50:
        raise SystemExit('too few usable columns')
    d = el_syn[good] - el_obs[good]
    beta = float(np.median(d))
    res = d - beta
    a, b = el_syn[good], el_obs[good]
    corr = float(np.corrcoef(a - a.mean(), b - b.mean())[0, 1])
    return {'beta_mrad': beta * 1e3,
            'rms_mrad': float(np.sqrt(np.mean(res ** 2)) * 1e3),
            'corr': corr, 'n_cols': int(good.sum()), 'W': W}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('photo')
    ap.add_argument('lat', type=float)
    ap.add_argument('lon', type=float)
    ap.add_argument('z', type=float)
    ap.add_argument('heading', type=float)
    ap.add_argument('fov', type=float)
    ap.add_argument('--dem', default='~/.horizonator/DEMs_SRTM1_WM')
    ap.add_argument('--dmin', type=float, default=1000.0)
    ap.add_argument('--near', type=float, default=3000.0)
    ap.add_argument('--crest-dh', type=float, default=0.0)
    a = ap.parse_args()

    cm = S.CMarcher(os.path.expanduser(a.dem),
                    (a.lat - 0.6, a.lat + 0.6), (a.lon - 0.6, a.lon + 0.6),
                    d_min=a.dmin)
    pre = prescreen(cm, a.lat, a.lon, a.z, a.heading, a.fov, a.near)
    if a.crest_dh:
        r = pre['rng']
        pre['el'] = pre['el'] + np.where(np.isfinite(r) & (r > 0),
                                         a.crest_dh / np.maximum(r, a.dmin),
                                         0.0)
    print('pre-screen at the given position (no photograph needed)')
    print('  subject range  p10 %.1f / p50 %.1f / p90 %.1f km'
          % (pre['p10'], pre['p50'], pre['p90']))
    print('  inside %.0f m: %.0f%% of the horizon'
          % (a.near, 100 * pre['near_frac']))
    print('  relief across the frame: %.1f mrad' % pre['relief_mrad'])
    if pre['p50'] < 5.0 or pre['near_frac'] > 0.15:
        print('  -> near-field regime (E5am/E5ba): weakly constrained')

    if a.photo != '-':
        d = diagnose(a.photo, pre, a.fov)
        print('truth diagnosis (%d/%d columns)' % (d['n_cols'], d['W']))
        print('  constant offset      %+.1f mrad' % d['beta_mrad'])
        print('  residual rms         %.1f mrad' % d['rms_mrad'])
        print('  shape correlation    %+.3f' % d['corr'])
        if d['corr'] < 0.5:
            print('  -> the DEM does not describe this scene')


if __name__ == '__main__':
    main()
