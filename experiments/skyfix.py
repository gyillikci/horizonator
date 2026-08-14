#!/usr/bin/env python3
"""skyfix: estimate camera position from one photograph's skyline.

    python3 skyfix.py IMAGE --center LAT,LON [options]

The search box is centered on --center (e.g. a dead-reckoning position or,
for validation, the photo's GPS — used only to place the box). Camera FOV,
heading prior and altitude come from EXIF when present, overridable:

  --center LAT,LON   box center (required; or --center-exif)
  --box M            box size in meters (default 5000)
  --fov DEG          horizontal FOV (default: EXIF FocalLengthIn35mmFilm)
  --heading DEG      heading prior of the image center, true (default:
                     EXIF GPSImgDirection; else a full-circle search)
  --z M              camera height above sea level (default: EXIF
                     GPSAltitude, else 10)
  --dem DIR          .hgt directory (default ~/.horizonator/DEMs_SRTM3)
  --roll DEG         camera roll (default 0)
  --pitch DEG        camera pitch prior, positive up (default 0). Getting
                     this right to ~0.5 deg matters: the residual elevation
                     offset co-estimated per candidate is only +-10 mrad,
                     deliberately tight -- a wide offset window discards the
                     absolute-elevation information that pins range
  --dmin M           near-field mask (default 1000 m). Needed for real
                     land-based observers, where the DEM smears the
                     observer's own bluff into blocking terrain; harmless
                     at sea. Reduce toward ~150 only if the camera's
                     near-field is genuinely open water
  --out PREFIX       write PREFIX.json and PREFIX.png diagnostics

Co-estimated per candidate: azimuth offset (+-6 deg around the prior, or
full circle if no prior) and a residual elevation offset (+-10 mrad).
Position: coarse-to-fine over the box. Output includes a
Laplace-approximation covariance.
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S
import extract

AZ = np.arange(-180.0, 180.0, 0.1) + 0.05
BETAS = np.arange(-0.010, 0.0101, 0.002)  # residual after the pitch prior


def observation(img, fov_deg, heading, roll_deg, pitch_deg=0.0):
    """Extract the skyline and map it to the global azimuth grid.
    Returns (el_obs, weights, diag) with diag holding extraction results."""
    rows, conf = extract.skyline_seam(img)
    H, W, _ = img.shape
    f = (W / 2) / np.tan(np.radians(fov_deg) / 2)
    u = np.arange(W) - (W - 1) / 2
    v = (H - 1) / 2 - rows
    cr, sr = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    ur = u * cr - v * sr
    vr = u * sr + v * cr
    az_rel = np.degrees(np.arctan2(ur, f))
    el_pt = np.arctan2(vr, np.hypot(ur, f)) + np.radians(pitch_deg)

    el = np.full(AZ.size, np.nan)
    wt = np.zeros(AZ.size)
    rel = (AZ - heading + 180.0) % 360.0 - 180.0
    m = (rel >= az_rel.min()) & (rel <= az_rel.max())
    order = np.argsort(az_rel)
    el[m] = np.interp(rel[m], az_rel[order], el_pt[order])
    wt[m] = np.interp(rel[m], az_rel[order], conf[order])
    wt = wt / (wt[m].max() + 1e-9)
    return (np.where(np.isfinite(el), el, 0.0), wt,
            dict(rows=rows, conf=conf, az_rel=az_rel, el_pt=el_pt))


def photo_cost(el_obs, w, el_syn, shifts):
    best = (np.inf, 0, 0.0)
    for s in shifts:
        eo = np.roll(el_obs, s)
        ww = np.roll(w, s)
        m = ww > 0
        r = el_syn[m] - eo[m]
        wm = ww[m]
        for b in BETAS:
            rb = np.abs(r - b)
            h = np.where(rb <= 3e-3, 0.5 * rb * rb, 3e-3 * (rb - 1.5e-3))
            c = float(np.sum(h * wm) / np.sum(wm))
            if c < best[0]:
                best = (c, s, b)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--center')
    ap.add_argument('--center-exif', action='store_true')
    ap.add_argument('--box', type=float, default=5000.0)
    ap.add_argument('--fov', type=float)
    ap.add_argument('--heading', type=float)
    ap.add_argument('--z', type=float)
    ap.add_argument('--roll', type=float, default=0.0)
    ap.add_argument('--pitch', type=float, default=0.0)
    ap.add_argument('--dem', default=os.path.expanduser(
        os.environ.get('HORIZONATOR_DEMS', '~/.horizonator/DEMs_SRTM3')))
    ap.add_argument('--dmin', type=float, default=1000.0)
    ap.add_argument('--out')
    args = ap.parse_args()

    ex = extract.read_exif(args.image)
    fov = args.fov or ex.get('fov_deg')
    if not fov:
        sys.exit('no FOV: none in EXIF, pass --fov')
    heading = args.heading if args.heading is not None \
        else ex.get('heading_deg')
    z = args.z if args.z is not None else ex.get('alt_m', 10.0)
    if args.center:
        lat_c, lon_c = (float(x) for x in args.center.split(','))
    elif args.center_exif and 'lat' in ex:
        lat_c, lon_c = ex['lat'], ex['lon']
    else:
        sys.exit('no box center: pass --center LAT,LON or --center-exif')
    print(f'fov {fov:.1f} deg, heading '
          f'{"none (full search)" if heading is None else heading}, '
          f'z {z:.1f} m, box {args.box:.0f} m at ({lat_c:.5f},{lon_c:.5f})')

    img = extract.load_image(args.image)
    el_obs, w, diag = observation(img, fov, heading if heading is not None
                                  else 0.0, args.roll, args.pitch)
    shifts = range(-60, 61, 2) if heading is not None \
        else range(-1800, 1800, 4)

    mlat, mlon = S.meters_per_degree(lat_c)
    lat_span = (lat_c - 0.6, lat_c + 0.6)
    lon_span = (lon_c - 0.8, lon_c + 0.8)
    cm = S.CMarcher(args.dem, lat_span, lon_span, d_min=args.dmin)

    def C(dn, de):
        el, _ = cm.skyline(lat_c + dn / mlat, lon_c + de / mlon, z, AZ)
        return photo_cost(el_obs, w, el, shifts)[0]

    step0 = max(args.box / 20, 100.0)
    g = np.arange(-args.box / 2, args.box / 2 + 1, step0)
    cc = np.array([[C(dn, de) for de in g] for dn in g])
    i, j = np.unravel_index(np.argmin(cc), cc.shape)
    dn0, de0 = g[i], g[j]
    for step in (step0 / 5, step0 / 20):
        best = (np.inf, dn0, de0)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                c = C(dn0 + di * step, de0 + dj * step)
                if c < best[0]:
                    best = (c, dn0 + di * step, de0 + dj * step)
        _, dn0, de0 = best

    # Laplace covariance from a local quadratic fit of the cost
    h = step0 / 10
    c0 = C(dn0, de0)
    cnn = (C(dn0 + h, de0) - 2 * c0 + C(dn0 - h, de0)) / h ** 2
    cee = (C(dn0, de0 + h) - 2 * c0 + C(dn0, de0 - h)) / h ** 2
    cne = (C(dn0 + h, de0 + h) - C(dn0 + h, de0 - h)
           - C(dn0 - h, de0 + h) + C(dn0 - h, de0 - h)) / (4 * h ** 2)
    Hm = np.array([[cnn, cne], [cne, cee]])
    try:
        cov = 2 * c0 * np.linalg.inv(Hm)  # scaled: residual-level heuristic
        sig = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        sig = [np.nan, np.nan]

    lat_e = lat_c + dn0 / mlat
    lon_e = lon_c + de0 / mlon
    el_syn, _ = cm.skyline(lat_e, lon_e, z, AZ)
    cbest, sbest, bbest = photo_cost(el_obs, w, el_syn, shifts)
    result = dict(lat=lat_e, lon=lon_e,
                  dn_m=dn0, de_m=de0,
                  sigma_n_m=float(sig[0]), sigma_e_m=float(sig[1]),
                  cost=cbest, rms_mrad=float(np.sqrt(2 * cbest) * 1e3),
                  heading_offset_deg=sbest * 0.1,
                  el_offset_mrad=bbest * 1e3,
                  fov_deg=fov, z_m=z)
    print(json.dumps(result, indent=1))

    if args.out:
        with open(args.out + '.json', 'w') as fjs:
            json.dump(result, fjs, indent=1)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1, figsize=(11, 7))
        ax = axs[0]
        ax.imshow(img)
        ax.plot(np.arange(img.shape[1]), diag['rows'], color='#d55e00',
                lw=1.2)
        ax.set_title('extracted skyline', fontsize=10)
        ax.axis('off')
        ax = axs[1]
        mm = np.roll(w, sbest) > 0
        ax.plot(AZ[mm], (np.roll(el_obs, sbest)[mm] + bbest) * 1e3,
                color='#111111', lw=1.8, label='observed')
        ax.plot(AZ[mm], el_syn[mm] * 1e3, color='#0072b2', lw=1.2,
                label='predicted at fix')
        ax.set_xlabel('azimuth (deg true)')
        ax.set_ylabel('elevation (mrad)')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        fig.tight_layout()
        fig.savefig(args.out + '.png', dpi=110)
        print('wrote', args.out + '.json/.png')


if __name__ == '__main__':
    main()
