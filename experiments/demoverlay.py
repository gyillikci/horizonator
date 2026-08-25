#!/usr/bin/env python3
"""Draw the DEM's own horizon back onto the photograph.

The solver compares observation and model in elevation-vs-azimuth
space; this projects the model the other way — through the camera
model at the declared position and attitude — so the DEM lands on the
pixels it claims to explain. Every crest the terrain shows along each
ray is drawn, not just the topmost one (skyline.visible_layers), so a
layered coastal scene reads as the layers it actually has: the near
headland, the mid ridge, the far coast.

  python3 demoverlay.py SOLVE.json PHOTO.jpg OUT.png [--truth LAT,LON]

Colour encodes RANGE, which is what makes the picture diagnostic: a
near layer swings hard with a small position error while a far coast
barely moves, so a mismatch that is large on warm colours and small on
cool ones is a position error, and one that is uniform across colours
is an attitude or focal-length error.
"""
import sys
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import extract
import skyline as S


def project(az_deg, el_rad, heading, pitch_deg, roll_deg, f_px, W, H):
    """Camera model of skyfix.observation(), inverted: world (az, el)
    -> pixel (col, row). Returns NaN outside the frame."""
    rel = np.radians((np.asarray(az_deg) - heading + 180.0) % 360.0 - 180.0)
    el_cam = np.asarray(el_rad) - np.radians(pitch_deg)
    ur = f_px * np.tan(rel)
    vr = np.hypot(ur, f_px) * np.tan(el_cam)
    cr, sr = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    u = ur * cr + vr * sr          # inverse of the solver's rotation
    v = -ur * sr + vr * cr
    col = u + (W - 1) / 2.0
    row = (H - 1) / 2.0 - v
    off = (np.abs(rel) > np.pi / 2) | ~np.isfinite(col) | ~np.isfinite(row)
    col = np.where(off, np.nan, col)
    row = np.where(off, np.nan, row)
    return col, row


def draw(ax, col, row, rng, cmap, lw, label, vmin, vmax):
    pts = np.stack([col, row], axis=1).reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    good = np.isfinite(seg).all(axis=(1, 2))
    # a segment spanning a jump in range is a silhouette edge, not a
    # crest: drop it rather than draw a line across the sky
    jump = np.abs(np.diff(col)) > 0.05 * (np.nanmax(col) - np.nanmin(col))
    good &= ~jump
    if not good.any():
        return None
    lc = LineCollection(seg[good], cmap=cmap, linewidths=lw,
                        norm=plt.Normalize(vmin, vmax))
    lc.set_array(np.asarray(rng)[:-1][good] / 1000.0)
    lc.set_label(label)
    ax.add_collection(lc)
    return lc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('solve')
    ap.add_argument('photo')
    ap.add_argument('out')
    ap.add_argument('--truth', default=None,
                    help='LAT,LON — also draw the DEM horizon from the '
                         'true position, for comparison')
    ap.add_argument('--dem', default='~/.horizonator/DEMs_SRTM1_WM')
    ap.add_argument('--dmin', type=float, default=1000.0,
                    help='near-field mask, MUST match the solve: skyfix '
                         'defaults to 1000 m (150 m under --dmin-soft). '
                         'Get this wrong and the marcher calls the near '
                         'shoreline the skyline — at 150 m a 15 m bank '
                         'sits 5 degrees up, which is exactly the wrong '
                         'answer this tool first drew.')
    ap.add_argument('--crest-dh', type=float, default=9.0)
    ap.add_argument('--at-truth', action='store_true',
                    help='draw the DEM horizon FROM the true position as '
                         'the main (range-coloured) curve — what the DEM '
                         'claims the camera should have seen from where it '
                         'actually stood')
    ap.add_argument('--show-extract', action='store_true',
                    help='also draw the image-side boundary the solver '
                         'extracted, so model and observation are compared '
                         'on the pixels rather than in angle space')
    args = ap.parse_args()

    j = json.load(open(args.solve))
    p = j['photos'][0]
    img = extract.load_image(args.photo)
    H, W = img.shape[:2]
    f_px = (W / 2) / np.tan(np.radians(p['fov_deg']) / 2)
    # the solver's own corrections: heading trade and elevation offset
    heading = p['heading_deg'] + p.get('heading_offset_deg', 0.0)
    beta = p.get('el_offset_mrad', 0.0) * 1e-3
    z = j['z_m']

    # only the azimuths the frame can show: marching the full circle
    # at this resolution costs minutes and draws nothing
    span = np.degrees(np.arctan((W / 2) / f_px)) + 3.0
    az = (heading + np.arange(-span, span, 0.04) + 180.0) % 360.0 - 180.0
    import os
    demdir = os.path.expanduser(args.dem)
    # the SOLVER's own synthesis, not a second opinion: same marcher,
    # same d_min, same refraction. A different marcher would draw a
    # different DEM than the one the fix was measured against — the
    # first version of this tool used visible_layers with n_layers=1,
    # which returns the NEAREST crest rather than the sky boundary,
    # and painted near-field junk 5 degrees above the ridge.
    def sky(la, lo):
        cm = S.CMarcher(demdir, (la - 0.6, la + 0.6), (lo - 0.8, lo + 0.8),
                        d_min=args.dmin)
        return cm.skyline(la, lo, z, az)

    fig, ax = plt.subplots(figsize=(16, 16 * H / W + 1.4))
    ax.imshow(img)

    lat, lon = j['lat'], j['lon']
    if args.at_truth and args.truth:
        lat, lon = [float(x) for x in args.truth.split(',')]
    el, rng = sky(lat, lon)
    el = el + np.where(np.isfinite(rng) & (rng > 0),
                       args.crest_dh / np.maximum(rng, args.dmin), 0.0)
    vmin = float(np.nanpercentile(rng, 3)) / 1000.0
    vmax = float(np.nanpercentile(rng, 97)) / 1000.0
    c, r = project(az, el + beta, heading, p['pitch_deg'],
                   p['roll_deg'], f_px, W, H)
    lc = draw(ax, c, r, rng, 'plasma', 2.4,
              'DEM horizon at the fix', vmin, vmax)

    if args.show_extract:
        import skyfix
        skyfix.EXTRACTOR = 'ewasr'
        rows_e, conf_e = skyfix.extract_boundary(img)
        xs = np.arange(W)[np.asarray(conf_e) > 0]
        ys = np.asarray(rows_e, float)[np.asarray(conf_e) > 0]
        ax.plot(xs, ys, '.', ms=1.6, color='#39ff14',
                label='extracted boundary (what the camera saw)')

    if args.truth and not args.at_truth:
        tla, tlo = [float(x) for x in args.truth.split(',')]
        elt, rngt = sky(tla, tlo)
        elt = elt + np.where(np.isfinite(rngt) & (rngt > 0),
                             args.crest_dh / np.maximum(rngt, args.dmin), 0.0)
        c, r = project(az, elt + beta, heading, p['pitch_deg'],
                       p['roll_deg'], f_px, W, H)
        ax.plot(c, r, '--', color='#00ffe0', lw=1.5,
                label='DEM horizon from TRUE position')

    if lc is not None:
        cb = fig.colorbar(lc, ax=ax, fraction=0.026, pad=0.01)
        cb.set_label('range to the crest (km)')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.85)
    ttl = (f"DEM overlay — {'TRUE position' if args.at_truth else 'fix'} "
           f"{lat:.5f}, {lon:.5f}  "
           f"(heading {heading:.1f}, pitch {p['pitch_deg']:+.2f}, "
           f"roll {p['roll_deg']:+.2f}, rms {j['rms_mrad']:.1f} mrad)")
    ax.set_title(ttl, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=115)
    print('->', args.out)


if __name__ == '__main__':
    main()
