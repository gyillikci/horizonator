#!/usr/bin/env python3
"""E5bm: re-sweep the heading on every frame in the campaign.

The solver searches heading only within --heading-window (default +-6
deg) of the compass prior. E5bf found a phone compass 13 deg wrong on
the Bodrum frame, which put the truth outside that window and produced
a confident 750 m false accept; E5bh found the same phone 0.9 deg out
on the Bosphorus. Compass error is episodic, so the window may have
quietly wrecked frames nobody looked at twice. This sweeps +-30 deg on
all of them and says which.

Per frame, at the EXIF position:
  * the boundary is extracted once (seam detector; eWaSR needs torch);
  * each column gets its EXACT bearing, az = atan2(u', f) and
    el = atan2(v', hypot(u', f)) after rotating the image plane by the
    candidate roll -- the same projection skyfix.observation uses. The
    earlier diagnostic scripts used a linear column-to-azimuth map,
    which is ~2 deg off at mid-frame for a 74 deg lens; symmetric, so
    it does not bias a heading, but it flattens the peak;
  * the DEM horizon is ray-cast ONCE over prior +- (fov/2 + 32) deg on
    a 0.05 deg grid, and every (heading, roll) candidate is an
    interpolation into it, so the sweep costs one ray-cast per frame;
  * score = robust (Huber) RMS of DEM minus photo elevation after
    removing only the median offset, so pitch is free and SCALE IS NOT.

The first version scored with Pearson correlation and a +-3 deg roll.
That was wrong, and the early rows showed it: best headings landing
exactly on the +-30 deg sweep edge with roll pinned at +-3. Pearson is
invariant to scale, and a free roll adds a linear ramp across the
columns, so together they let any sloped profile match any other sloped
profile -- the score rewarded overall trend, not shape. Elevation
angles are physical units at a known focal length; nothing should be
allowed to rescale them.

A sweep is only as good as its extraction: a cloud-riding boundary
(E5bh) has a "best heading" that means nothing. Every result reports
whether its optimum sits on the sweep edge, and the ratio of the best
score to the score at the compass prior.

    python3 e5bm_heading_resweep.py FRAMES.json [--dem DIR] [--out DIR]
"""

import argparse
import json
import os
import sys

import numpy as np

import extract
import skyline as S

WINDOW = 6.0          # the solver's default --heading-window
SPAN = 30.0           # how far this sweep looks
STEP = 0.25
ROLLS = np.arange(-2.0, 2.01, 0.5)
HUBER = 3e-3          # rad; the solver's own robust radius


def column_bearings(rows, H, W, fov, roll):
    """Exact (azimuth offset, elevation) per column, as skyfix does it."""
    f = (W / 2) / np.tan(np.radians(fov) / 2)
    u = np.arange(W) - (W - 1) / 2
    v = (H - 1) / 2 - rows
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    ur = u * cr - v * sr
    vr = u * sr + v * cr
    return np.degrees(np.arctan2(ur, f)), np.arctan2(vr, np.hypot(ur, f))


def sweep(fr, dem_dir):
    img = extract.load_image(fr['path'])
    H, W = img.shape[:2]
    rows, conf = extract.skyline_seam(img)
    rows = np.asarray(rows, float)
    ok = np.asarray(conf, float) > 0
    if ok.sum() < 100:
        return dict(status='no boundary')

    prior = float(fr['exif_heading'] if fr.get('exif_heading') is not None
                  else fr['heading'])
    fov = float(fr['fov'])
    z = float(fr['z'] if fr.get('z') is not None else 5.0)
    half = fov / 2 + SPAN + 2.0
    grid = prior + np.arange(-half, half + 1e-9, 0.05)
    cm = S.CMarcher(os.path.expanduser(dem_dir),
                    (fr['lat'] - .6, fr['lat'] + .6),
                    (fr['lon'] - .6, fr['lon'] + .6), d_min=1000.0)
    el_g, r_g = cm.skyline(fr['lat'], fr['lon'], z, grid)
    good_g = np.isfinite(el_g)
    if good_g.mean() < 0.5:
        return dict(status='no terrain in DEM')
    el_g = np.where(good_g, el_g, np.nanmin(el_g[good_g]))

    heads = prior + np.arange(-SPAN, SPAN + 1e-9, STEP)
    score = np.full((ROLLS.size, heads.size), np.inf)
    for i, roll in enumerate(ROLLS):
        az_rel, el_obs = column_bearings(rows, H, W, fov, roll)
        for k, h in enumerate(heads):
            d = (np.interp(h + az_rel, grid, el_g) - el_obs)[ok]
            d = d - np.median(d)
            a = np.abs(d)
            hub = np.where(a <= HUBER, 0.5 * a * a, HUBER * (a - 0.5 * HUBER))
            score[i, k] = float(np.sqrt(2 * hub.mean()))
    i_r, k = np.unravel_index(int(np.argmin(score)), score.shape)
    s_best = float(score[i_r, k])
    h_best = float(heads[k])
    line = score[i_r]
    if 0 < k < line.size - 1:
        y0, y1, y2 = line[k - 1], line[k], line[k + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            h_best = heads[k] + 0.5 * STEP * (y0 - y2) / den
    k0 = int(np.argmin(np.abs(heads - prior)))
    s_prior = float(score[:, k0].min())
    delta = float(((h_best - prior + 180) % 360) - 180)
    edge = bool(k in (0, heads.size - 1) or i_r in (0, ROLLS.size - 1))
    return dict(status='ok', prior=prior, best=h_best, delta=delta,
                roll=float(ROLLS[i_r]), rms_best_mrad=s_best * 1e3,
                rms_prior_mrad=s_prior * 1e3,
                gain=s_prior / max(s_best, 1e-9), edge=edge,
                outside=bool(abs(delta) > WINDOW), cols=int(ok.sum()), W=W,
                rng_p50_km=float(np.nanpercentile(r_g[good_g], 50) / 1e3))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames')
    ap.add_argument('--dem', default='~/.horizonator/DEMs_COP30')
    ap.add_argument('--out', default='out/resweep')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frames = json.load(open(a.frames))
    res = []
    for n, fr in enumerate(frames, 1):
        try:
            r = sweep(fr, a.dem)
        except Exception as e:                         # keep the batch going
            r = dict(status='error: %s' % str(e)[:80])
        r.update(file=fr['file'], lat=fr['lat'], lon=fr['lon'],
                 path=fr['path'], fov=fr['fov'], z=fr.get('z'))
        res.append(r)
        if r['status'] == 'ok':
            print('%2d/%d %-34s prior %6.1f best %6.1f  d %+6.1f  rms %5.1f '
                  '(prior %5.1f) mrad roll %+.1f %s%s'
                  % (n, len(frames), fr['file'][:34], r['prior'], r['best'],
                     r['delta'], r['rms_best_mrad'], r['rms_prior_mrad'],
                     r['roll'], 'OUTSIDE ' if r['outside'] else '',
                     'EDGE' if r['edge'] else ''), flush=True)
        else:
            print('%2d/%d %-34s %s' % (n, len(frames), fr['file'][:34],
                                       r['status']), flush=True)
    json.dump(res, open(os.path.join(a.out, 'resweep.json'), 'w'), indent=1)
    print('\nwrote', os.path.join(a.out, 'resweep.json'))


if __name__ == '__main__':
    sys.exit(main())
