#!/usr/bin/env python3
"""Capture attitude vs final attitude, side by side.

CAPTURE is what the photograph arrived with: EXIF heading, and pitch
and roll where the app recorded them (Theodolite does; the stock
camera does not). FINAL is what the fit actually settled on — the
levelled or supplied pitch and roll, the heading after the solver's
azimuth trade, and the elevation offset beta, which acts as a pitch
correction and is therefore folded into an EFFECTIVE pitch.

Reading the pair is the fastest diagnostic in the instrument: a large
capture-to-final pitch change means the attitude was never known; an
effective pitch sitting exactly on the beta band edge (+-2, +-5 or
+-10 mrad depending on the anchor) means the fit was CLAMPED and
wanted to go further.

  python3 attitude_report.py SOLVE.json [PHOTO.jpg] [--truth-pitch D]
"""
import os
import sys
import json

import numpy as np

MRAD = 180.0 / np.pi * 1e-3          # deg per mrad


def exif_attitude(path):
    from PIL import Image
    ex = Image.open(path).getexif()
    sub, gps = ex.get_ifd(0x8769), ex.get_ifd(0x8825)
    out = {}
    if gps:
        def dms(v):
            return float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        if 2 in gps:
            out['lat'] = dms(gps[2]) * (1 if gps.get(1, 'N') == 'N' else -1)
        if 4 in gps:
            out['lon'] = dms(gps[4]) * (1 if gps.get(3, 'E') == 'E' else -1)
        if 17 in gps:
            out['heading'] = float(gps[17])
        if 6 in gps:
            a = float(gps[6])
            out['alt'] = a if a > 0 else None
    # Theodolite writes the inclinometer into ImageDescription as
    # "vert_angle_deg=... / horiz_angle_deg=..", and the same numbers
    # again as XML in the MakerNote. vert = pitch, horiz = roll.
    import re
    blob = ''
    for v in (ex.get(270), sub.get(37500), ex.get(37500)):
        if isinstance(v, bytes):
            v = v.decode('utf8', 'ignore')
        if isinstance(v, str):
            blob += ' ' + v
    for key, tag in (('pitch', 'vert_angle_deg'), ('roll', 'horiz_angle_deg')):
        m = re.search(tag + r'[=>]\s*(-?\d+(?:\.\d+)?)', blob)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r'azimuth_deg[=>]\s*(-?\d+(?:\.\d+)?)', blob)
    if m:
        out['az_acc'] = float(m.group(1))
    out['software'] = sub.get(305) or ex.get(305)
    return out


def band_for(source):
    """The beta half-width the solver granted, in mrad."""
    return {'waterline': 5.0, 'radon': 2.0, 'sea-horizon': 2.0}.get(source, 10.0)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    truth_pitch = None
    for i, a in enumerate(sys.argv):
        if a == '--truth-pitch':
            truth_pitch = float(sys.argv[i + 1])
    j = json.load(open(args[0]))
    photo = args[1] if len(args) > 1 else None
    p = j['photos'][0]

    cap = exif_attitude(photo) if photo and os.path.exists(photo) else {}
    hd_fin = p['heading_deg'] + p.get('heading_offset_deg', 0.0)
    beta = p.get('el_offset_mrad', 0.0)
    pitch_eff = p['pitch_deg'] + beta * MRAD
    band = band_for(p.get('attitude_source'))
    clamped = abs(abs(beta) - band) < 0.15

    def f(x, w=8, s='%+.2f'):
        return ('%*s' % (w, s % x)) if x is not None else '%*s' % (w, '—')

    print(f"{'':10s} {'heading':>10s} {'pitch':>8s} {'roll':>8s}   kaynak")
    print(f"{'ÇEKİM':10s} {f(cap.get('heading'), 10, '%.1f')} "
          f"{f(cap.get('pitch'))} {f(cap.get('roll'))}   "
          f"{'EXIF (' + str(cap.get('software') or 'yalnız yön') + ')' if cap else 'EXIF okunmadı'}")
    print(f"{'ÇÖZÜM':10s} {hd_fin:10.1f} {p['pitch_deg']:+8.2f} "
          f"{p['roll_deg']:+8.2f}   {p.get('attitude_source')}")
    print(f"{'ETKİN':10s} {'':10s} {pitch_eff:+8.2f} {'':8s}   "
          f"pitch + beta({beta:+.1f} mrad)"
          f"{'  << BANT KENARI ±%.0f, SIKIŞIK' % band if clamped else ''}")
    print()
    print(f"  yön kayması   : {p.get('heading_offset_deg', 0.0):+.1f}°")
    print(f"  beta (yükseklik ofseti): {beta:+.1f} mrad "
          f"(izin verilen ±{band:.0f})")
    if cap.get('az_acc') is not None:
        print(f"  pusula doğruluğu (çekim): ±{cap['az_acc']:.0f}°")
    if cap.get('pitch') is not None:
        print(f"  çekim→çözüm pitch farkı: "
              f"{p['pitch_deg'] - cap['pitch']:+.2f}°")
    if truth_pitch is not None:
        print(f"  ölçülen gerçek pitch: {truth_pitch:+.2f}°  "
              f"(etkinden fark {pitch_eff - truth_pitch:+.2f}°)")


if __name__ == '__main__':
    main()
