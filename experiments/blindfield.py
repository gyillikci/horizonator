#!/usr/bin/env python3
"""Blind field-trial harness: the human-in-the-loop version of the
sealed-centre protocol (blindbox, E5ae-era).

The uploaded frames carry the GPS truth twice — burned into the
pixels as a coordinate pill and in the EXIF GPS IFD — so a blind
trial cannot rest on a promise not to look. This script is the only
process that touches the original:

  prepare   reads the EXIF once, SEALS lat/lon (+altitude) into
            sealed.json, and emits for the solver side only:
              redacted.jpg   pixels with the bottom strip blacked
                             out (the pill lives there) and ALL
                             EXIF stripped
              meta.json      heading (GPSImgDirection), FOV from
                             FocalLengthIn35mmFilm + aspect,
                             timestamp — no coordinates
  score     after the fix is declared, opens the seal and reports
            dlat/dlon against the truth

The solve step centres its (wide) box on a locality the PHOTOGRAPHER
names in words — the lost navigator's "somewhere off Kumlubük" —
never on the sealed coordinates.

Run:
  python3 blindfield.py prepare UPLOAD.jpg --dir out/blindfield/t1
  ... solve on redacted.jpg with meta.json + a 20 km box ...
  python3 blindfield.py score --dir out/blindfield/t1 --fix LAT,LON
"""

import os
import sys
import json
import argparse

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare', 'score'])
    ap.add_argument('photo', nargs='?')
    ap.add_argument('--dir', required=True)
    ap.add_argument('--fix', default=None)
    ap.add_argument('--strip-frac', type=float, default=0.10,
                    help='bottom fraction of the frame to black out')
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    if args.cmd == 'prepare':
        im = Image.open(args.photo)
        ex = im.getexif()
        sub = ex.get_ifd(0x8769)
        gps = ex.get_ifd(0x8825)

        def dms(v):
            return float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        lat = dms(gps[2]) * (1 if gps.get(1, 'N') == 'N' else -1)
        lon = dms(gps[4]) * (1 if gps.get(3, 'E') == 'E' else -1)
        alt = gps.get(6)
        with open(os.path.join(args.dir, 'sealed.json'), 'w') as f:
            json.dump(dict(lat=lat, lon=lon,
                           alt=float(alt) if alt else None), f)

        heading = float(gps[17]) if 17 in gps else None
        w, h = im.size
        f35 = sub.get(41989)
        fov = None
        if f35:
            # f35 is defined on the 36 mm side = the long axis
            f_px = max(w, h) * float(f35) / 36.0
            fov = float(np.degrees(2 * np.arctan((w / 2) / f_px)))
        with open(os.path.join(args.dir, 'meta.json'), 'w') as f:
            json.dump(dict(heading_deg=heading, fov_deg=fov,
                           width=w, height=h,
                           time=sub.get(36867)), f)

        a = np.asarray(im.convert('RGB')).copy()
        a[int(h * (1 - args.strip_frac)):, :, :] = 0
        out = os.path.join(args.dir, 'redacted.jpg')
        Image.fromarray(a).save(out, quality=92)   # no EXIF carried
        print(f'sealed: lat/lon/alt -> {args.dir}/sealed.json')
        print(f'solver side: {out} + meta.json '
              f'(heading {heading}, fov {fov and round(fov, 1)})')
        return

    with open(os.path.join(args.dir, 'sealed.json')) as f:
        seal = json.load(f)
    la, lo = [float(x) for x in args.fix.split(',')]
    mlat = 111132.0
    mlon = 111320.0 * np.cos(np.radians(seal['lat']))
    dn = (la - seal['lat']) * mlat
    de = (lo - seal['lon']) * mlon
    print(f"truth: {seal['lat']:.5f}, {seal['lon']:.5f} "
          f"(alt {seal['alt']})")
    print(f'declared fix error: dlat {dn:+.0f} m, dlon {de:+.0f} m '
          f'(total {np.hypot(dn, de):.0f} m)')


if __name__ == '__main__':
    main()
