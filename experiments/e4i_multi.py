#!/usr/bin/env python3
"""E4i: multi-photo skyfix — the telephoto-pan procedure, validated.

From ONE position (the E4c strait2 sea case), synthesize an instrumented
photo set: three 8x-telephoto frames (f35 192 mm, 10.3 deg FOV) panned
across the terrain plus one 1x wide frame (f35 24 mm, 71.6 deg), each
with correct EXIF (focal, GPS direction, altitude). Then solve with the
multi-photo skyfix CLI:

    wide      the single wide frame        (the old way)
    pan       the three telephoto frames   (the E4h field procedure)
    all       wide + pan, precision-weighted

Run headlessly:   xvfb-run -a python3 e4i_multi.py
"""

import os
import sys
import json
import subprocess
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

import numpy as np
import piexif
from PIL import Image
import skyline as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
IMGDIR = os.path.join(OUT, 'multi')
os.makedirs(IMGDIR, exist_ok=True)
DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))

W, H = 2016, 1512
LAT, LON, Z = 36.9622, 27.2384, 5.0     # E4c strait2, at sea
PITCH, ROLL = -0.5, 0.5
BOX = 5000.0
FRAMES = [                               # (name, f35_mm, heading_deg)
    ('tele_a', 192, 355.0),
    ('tele_b', 192, 25.0),
    ('tele_c', 192, 55.0),
    ('wide', 24, 25.0),
]
rng = np.random.default_rng(20260819)


def synth_photo(name, f35, heading):
    fov = np.degrees(2 * np.arctan(0.8 * 21.63 / f35))
    gl = S.GlSkyline(LAT, LON, width=7200, height=800,
                     render_radius_m=45000., dir_dems=DIR3)
    rngimg = gl.h.render(heading - 180.0, heading + 180.0, lat=LAT,
                         lon=LON, z=Z, zfar=40000., return_image=False)
    del gl
    valid = rngimg > 0

    f = (W / 2) / np.tan(np.radians(fov) / 2)
    u = (np.arange(W) - (W - 1) / 2)[None, :]
    v = ((H - 1) / 2 - np.arange(H))[:, None]
    cr, sr = np.cos(np.radians(ROLL)), np.sin(np.radians(ROLL))
    ur = u * cr - v * sr
    vr = u * sr + v * cr
    az = (heading + np.degrees(np.arctan2(ur, f))) % 360.0
    el = np.degrees(np.arctan2(vr, np.hypot(ur, f))) + PITCH

    ix = np.clip((((az - heading + 180.0) % 360.0) / 0.05).astype(int),
                 0, 7199)
    iy = np.clip(((20.0 - el) / 0.05).astype(int), 0, 799)
    r = rngimg[iy, ix]
    isv = valid[iy, ix]

    sky_top = np.array([0.35, 0.55, 0.90])
    sky_low = np.array([0.72, 0.82, 0.95])
    terr = np.array([0.42, 0.36, 0.28])
    sea = np.array([0.10, 0.22, 0.42])
    img = np.zeros((H, W, 3))
    t = np.clip((el + 20) / 40, 0, 1)[..., None]
    img[:] = sky_low[None, None] * (1 - t) + sky_top[None, None] * t
    haze = 0.88 * np.clip(1 - np.exp(-np.maximum(r, 0) / 15000.0),
                          0, 1)[..., None]
    hit_h = Z + r * np.tan(np.radians(el)) + r * r / (2.0 * S.REFF)
    ground = np.where((hit_h < 12.0)[..., None] & (r > 500)[..., None],
                      sea[None, None], terr[None, None])
    tex = 0.06 * rng.standard_normal((H, W, 1))
    scene = (ground + tex) * (1 - haze) + img * haze
    img = np.where(isv[..., None], scene, img)
    below = (~isv) & (el < -2.0)
    img = np.where(below[..., None], sea[None, None]
                   + 0.04 * rng.standard_normal((H, W, 1)), img)
    img += 0.008 * rng.standard_normal(img.shape)
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    path = os.path.join(IMGDIR, name + '.jpg')
    Image.fromarray(img).save(path, quality=92)

    def dms(x):
        d = int(x)
        m = int((x - d) * 60)
        s = (x - d - m / 60) * 3600
        return ((d, 1), (m, 1), (int(s * 100), 100))
    exif = {
        'Exif': {piexif.ExifIFD.FocalLengthIn35mmFilm: f35},
        'GPS': {piexif.GPSIFD.GPSLatitude: dms(LAT),
                piexif.GPSIFD.GPSLatitudeRef: b'N',
                piexif.GPSIFD.GPSLongitude: dms(LON),
                piexif.GPSIFD.GPSLongitudeRef: b'E',
                piexif.GPSIFD.GPSAltitude: (int(Z * 10), 10),
                piexif.GPSIFD.GPSImgDirection: (int(heading * 10), 10),
                piexif.GPSIFD.GPSImgDirectionRef: b'T'}}
    piexif.insert(piexif.dump(exif), path)
    return path


if __name__ == '__main__':
    paths = {n: synth_photo(n, f35, h) for n, f35, h in FRAMES}
    print('frames synthesized', flush=True)
    mlat, mlon = S.meters_per_degree(LAT)
    off = rng.uniform(-1500, 1500, 2)
    center = f'{LAT + off[0] / mlat:.6f},{LON + off[1] / mlon:.6f}'

    SETS = [('wide', [paths['wide']]),
            ('pan', [paths['tele_a'], paths['tele_b'], paths['tele_c']]),
            ('all', [paths['tele_a'], paths['tele_b'], paths['tele_c'],
                     paths['wide']])]
    results = {}
    for name, imgs in SETS:
        t0 = time.time()
        p = subprocess.run(
            [sys.executable, os.path.join(HERE, 'skyfix.py'), *imgs,
             '--center', center, '--box', str(BOX),
             '--pitch', str(PITCH), '--roll', str(ROLL),
             '--dmin', '1000',
             '--out', os.path.join(IMGDIR, name)],
            capture_output=True, text=True)
        dt = time.time() - t0
        if p.returncode not in (0, 2):
            print(name, 'FAILED:', p.stderr[-400:])
            continue
        R = json.load(open(os.path.join(IMGDIR, name + '.json')))
        err = np.hypot((R['lat'] - LAT) * mlat, (R['lon'] - LON) * mlon)
        results[name] = dict(err_m=float(err), rms_mrad=R['rms_mrad'],
                             sigma=[R['sigma_n_m'], R['sigma_e_m']],
                             margin=R['basin_margin'],
                             status=R['status'], t_s=dt)
        print(f"{name:5s}: err {err:6.1f} m  "
              f"sig ({R['sigma_n_m']:.0f},{R['sigma_e_m']:.0f}) m  "
              f"margin {R['basin_margin']:.2f}  {R['status']}  {dt:.0f}s",
              flush=True)
    with open(os.path.join(OUT, 'e4i_results.json'), 'w') as f:
        json.dump(results, f, indent=1)
