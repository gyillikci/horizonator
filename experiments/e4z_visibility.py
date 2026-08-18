#!/usr/bin/env python3
"""E4z: synthesise the skyline only as far as the air is clear.

The most repeated disagreement in the field data is the DEM holding a
ridge up where the photograph shows nothing: at APST5638 it keeps
11-13 mrad across half the frame over an observed 2-6, and at SRYK4301
the same. The model has no atmosphere, so it treats a 40 km ridge in
summer haze as visible. The camera does not.

This sweeps a visibility limit: the marcher's d_max is the range past
which terrain is simply not synthesised, so the silhouette becomes the
highest crest the air actually shows. One nuisance parameter for the
whole frame — deliberately not a per-azimuth choice, which would let
the cost pick a layer wherever it liked and buy a lower residual at a
wrong place (the lesson from widening the heading window).

Run:  python3 e4z_visibility.py [ID ...]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import skyline as S, skyfix as SF, extract

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      'celestial-navigation', 'theodolite')
DEM = os.path.expanduser('~/.horizonator/DEMs_SRTM3')
VIS_KM = [8.0, 15.0, 25.0, 40.0, 60.0]


def solve(s, v_km, step0=250.0, box=6000.0):
    e, a = s['exif'], s['attitude']
    img = extract.load_image(os.path.join(PHOTOS, s['raw']))
    z = max(e.get('alt_m') or 5.0, 2.0)
    f_px = (img.shape[1] / 2) / np.tan(np.radians(a['fov_deg']) / 2)
    dip = S.horizon_dip_rad(z)
    lvl = extract.sea_horizon_attitude_radon(img, f_px, dip)
    pitch = lvl['pitch_deg'] if lvl else (a.get('pitch_deg') or 0.0)
    roll = lvl['roll_deg'] if lvl else (a.get('roll_deg') or 0.0)
    H, W, _ = img.shape
    u = np.arange(W) - (W - 1) / 2
    vv = (np.tan(-dip - np.radians(pitch)) * np.hypot(u, f_px)
          - np.radians(roll) * u)
    hz = (H - 1) / 2 - vv                      # sea-horizon rows
    el_obs, w, _ = SF.observation(img, a['fov_deg'], a['heading_deg'],
                                  roll, pitch, horizon_rows=hz)
    if w.sum() < 1e-6:
        return None
    betas = SF.BETAS_TIGHT if lvl else SF.BETAS
    cm = S.CMarcher(DEM, (e['lat'] - .6, e['lat'] + .6),
                    (e['lon'] - .8, e['lon'] + .8), d_min=1000.)
    cm.d_max = v_km * 1000.0
    mlat, mlon = S.meters_per_degree(e['lat'])
    g = np.arange(-box / 2, box / 2 + 1, step0)
    shifts = np.arange(-60, 61, 2)
    cc = np.empty((g.size, g.size))
    for i, dn in enumerate(g):
        for j, de in enumerate(g):
            el, _ = cm.skyline(e['lat'] + dn / mlat, e['lon'] + de / mlon,
                               z, SF.AZ)
            cc[i, j] = SF.fast_photo_cost(el_obs, w, el, shifts, betas)[0]
    i, j = np.unravel_index(np.argmin(cc), cc.shape)
    return dict(v_km=v_km, err_m=float(np.hypot(g[i], g[j])),
                rms_mrad=float(np.sqrt(2 * cc[i, j]) * 1e3),
                margin=SF.basin_margin(cc, g, min_sep=4 * step0))


if __name__ == '__main__':
    idx = {x['id']: x for x in json.load(
        open(os.path.join(HERE, 'out', 'theodolite', 'index.json')))['sightings']}
    ids = sys.argv[1:] or ['APST5638', 'SRYK4301', 'CDKT5817', 'KWHC9160']
    out = {}
    for sid in ids:
        if sid not in idx:
            continue
        print(f'{sid}:', flush=True)
        rows = []
        for v in VIS_KM:
            r = solve(idx[sid], v)
            if r is None:
                print(f'  {v:5.0f} km  no usable observation'); continue
            rows.append(r)
            print(f'  {v:5.0f} km   err {r["err_m"]:6.0f} m   '
                  f'rms {r["rms_mrad"]:5.2f}   margin {r["margin"]:5.2f}',
                  flush=True)
        out[sid] = rows
    with open(os.path.join(HERE, 'out', 'e4z_visibility.json'), 'w') as f:
        json.dump(out, f, indent=1)
