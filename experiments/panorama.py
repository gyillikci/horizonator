#!/usr/bin/env python3
"""Render a colorized 360-deg scene panorama from the horizonator.

    xvfb-run -a python3 panorama.py LAT LON [Z] [-o OUT.png]

Equirectangular, 0.05 deg/px (7200x800 before the 2x downscale), azimuth
ruler along the bottom (center = North). Geometry comes from the
curvature+refraction-patched GL renderer; coloring is the E4c scene
model (sky gradient, range-dependent haze, sea/terrain classification by
hit height) plus a slope-based hillshade, an elevation tint and a
sea-horizon brightness gradient. Z defaults to the DEM height + 3 m.
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

import numpy as np
from PIL import Image, ImageDraw
import skyline as S

W, H = 7200, 800            # az -180..180 (north-centered), el +20..-20


def render(lat, lon, z, dir_dems):
    gl = S.GlSkyline(lat, lon, width=W, height=H, render_radius_m=45000.,
                     dir_dems=dir_dems)
    rng = gl.h.render(-180.0, 180.0, lat=lat, lon=lon, z=z, zfar=40000.,
                      return_image=False)
    del gl
    valid = rng > 0
    el = np.linspace(20, -20, H, endpoint=False)[:, None] - 20.0 / H

    sky_top = np.array([0.35, 0.55, 0.90])
    sky_low = np.array([0.72, 0.82, 0.95])
    terr = np.array([0.42, 0.36, 0.28])
    rock = np.array([0.55, 0.50, 0.42])
    sea = np.array([0.10, 0.22, 0.42])

    nz = np.random.default_rng(0)
    img = np.zeros((H, W, 3))
    t = np.clip((el + 20) / 40, 0, 1)[..., None]
    img[:] = sky_low[None, None] * (1 - t) + sky_top[None, None] * t
    r = np.where(valid, rng, 0.0)
    haze = 0.92 * np.clip(1 - np.exp(-r / 9000.0), 0, 1)[..., None]
    hit_h = z + r * np.tan(np.radians(el)) + r * r / (2.0 * S.REFF)
    is_sea = (hit_h < 12.0) & (r > 500)
    t2 = np.clip(hit_h / 350.0, 0, 1)[..., None]
    terrc = terr[None, None] * (1 - t2) + rock[None, None] * t2
    gx = np.gradient(np.where(valid, hit_h, 0.0), axis=1)
    slope = gx / np.maximum(r * np.radians(0.05), 1.0)
    amb = 0.80 + 0.20 * np.clip((el + 20) / 25.0, 0, 1)
    shade = ((1.0 - 0.45 * np.tanh(slope * 2.0)) * amb)[..., None]
    seac = sea[None, None] + 0.35 * np.clip(1 + el / 10.0, 0, 1)[..., None] ** 2 \
        * np.array([0.9, 0.95, 1.0])[None, None]
    ground = np.where(is_sea[..., None], seac, terrc * shade)
    tex = np.where(is_sea[..., None], 0.015, 0.035) \
        * nz.standard_normal((H, W, 1))
    scene = (ground + tex) * (1 - haze) + img * haze
    img = np.where(valid[..., None], scene, img)
    below = (~valid) & (el < -0.5)
    img = np.where(below[..., None],
                   sea[None, None] + 0.04 * nz.standard_normal((H, W, 1)),
                   img)
    img += 0.008 * nz.standard_normal(img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def with_ruler(img):
    strip = np.full((46, W, 3), 250, dtype=np.uint8)
    im = Image.fromarray(np.vstack([img, strip]))
    d = ImageDraw.Draw(im)
    for a in range(0, 360, 10):
        x = int(((a + 180.0) % 360.0) / 360.0 * W)
        major = a % 30 == 0
        d.line([(x, H), (x, H + (18 if major else 9))], fill=(60, 60, 60),
               width=2 if major else 1)
        if major:
            lbl = {0: 'N 0', 90: 'E 90', 180: 'S 180',
                   270: 'W 270'}.get(a, str(a))
            d.text((x + 4, H + 20), lbl, fill=(30, 30, 30))
    return im.resize((W // 2, (H + 46) // 2), Image.LANCZOS)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('lat', type=float)
    ap.add_argument('lon', type=float)
    ap.add_argument('z', type=float, nargs='?')
    ap.add_argument('-o', '--out', default='pano.png')
    ap.add_argument('--dem', default=os.path.expanduser(
        os.environ.get('HORIZONATOR_DEMS', '~/.horizonator/DEMs_SRTM3')))
    args = ap.parse_args()
    z = args.z
    if z is None:
        cm = S.CMarcher(args.dem, (args.lat - .1, args.lat + .1),
                        (args.lon - .1, args.lon + .1))
        y = int(round((cm.lat_nw - args.lat) / cm.dpp))
        x = int(round((args.lon - cm.lon_nw) / cm.dpp))
        z = float(max(cm.mosaic[y, x], 0.0)) + 3.0
        print(f'z from DEM: {z:.1f} m')
    with_ruler(render(args.lat, args.lon, z, args.dem)).save(args.out)
    print('wrote', args.out)
