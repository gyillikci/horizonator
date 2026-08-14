#!/usr/bin/env python3
"""E4c: end-to-end validation of the automatic pipeline (extract.py +
skyfix.py) on photo-realistic composite JPEGs with real EXIF.

Scenes are composited from GL-rendered ground truth (the independent
renderer): sky gradient above the skyline, range-dependent haze and
texture below it, sea coloring, camera pitch/roll, sensor noise. EXIF
(FocalLengthIn35mmFilm, GPSImgDirection, GPSAltitude, GPS position) is
written with piexif, so the CLI's EXIF path is exercised too. The search
box is centered on the ground truth plus a random offset (a dead-reckoning
prior), never on the truth itself.

Run headlessly:   xvfb-run -a python3 e4c_synth.py
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
IMGDIR = os.path.join(OUT, 'synth')
os.makedirs(IMGDIR, exist_ok=True)
DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))

W, H = 2016, 1512
F35 = 24          # -> 71.6 deg horizontal FOV
FOV = np.degrees(2 * np.arctan(0.8 * 21.63 / F35))
BOX = 5000.0

# ground truth: (lat, lon, z, heading, pitch_deg, roll_deg, dmin)
# All cases use the real-world recommended near-field mask (dmin 1 km).
# The 'bafa' land case fails with it (its view contains genuine terrain at
# 100-800 m that the solver then cannot predict) and recovers to ~20 m with
# dmin=150 in this DEM-consistent synthetic loop -- but with real photos a
# low dmin re-opens the observer's-own-bluff problem (study doc E4). Soft
# near-field weighting is the open work item for land-based observers
CASES = [
    ('strait1',  36.9500, 27.2500,  5.0, 180.0,  0.5,  0.0, 1000.),
    ('strait2',  36.9622, 27.2384,  5.0,  25.0, -0.5,  0.5, 1000.),
    ('strait3',  36.9411, 27.2661,  5.0, 300.0,  1.0, -0.5, 1000.),
    ('offshore', 36.6050, 26.8590,  5.0,  60.0,  0.0,  0.0, 1000.),
    ('bafa',     37.4768, 27.4142, 16.0,  37.0,  1.5,  0.5, 1000.),
    ('bafa2',    37.4700, 27.3900, 12.0, 352.0,  0.0, -0.5, 1000.),
]

rng = np.random.default_rng(20260817)


def synth_photo(name, lat, lon, z, heading, pitch, roll):
    # render the panorama CENTERED on the camera heading: the +-180 render
    # seam otherwise lands mid-image for south-facing cameras
    gl = S.GlSkyline(lat, lon, width=7200, height=800,
                     render_radius_m=45000., dir_dems=DIR3)
    rngimg = gl.h.render(heading - 180.0, heading + 180.0, lat=lat, lon=lon,
                         z=z, zfar=40000., return_image=False)
    del gl
    valid = rngimg > 0

    f = (W / 2) / np.tan(np.radians(FOV) / 2)
    u = (np.arange(W) - (W - 1) / 2)[None, :]
    v = ((H - 1) / 2 - np.arange(H))[:, None]
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    ur = u * cr - v * sr
    vr = u * sr + v * cr
    az = (heading + np.degrees(np.arctan2(ur, f))) % 360.0
    el = np.degrees(np.arctan2(vr, np.hypot(ur, f))) + pitch

    # sample the equirectangular range render (0.05 deg/px, el in [-20,20])
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
    # haze asymptotes to 94% of the local sky color: distant ridges stay
    # faintly visible, as in real hazy conditions
    haze = 0.88 * np.clip(1 - np.exp(-np.maximum(r, 0) / 15000.0),
                          0, 1)[..., None]
    # classify sea by the hit point's height above sea level (the naive
    # el-threshold misclassifies the strip just below the sea horizon)
    hit_h = z + r * np.tan(np.radians(el)) + r * r / (2.0 * S.REFF)
    ground = np.where((hit_h < 12.0)[..., None] & (r > 500)[..., None],
                      sea[None, None], terr[None, None])
    tex = 0.06 * rng.standard_normal((H, W, 1))
    scene = (ground + tex) * (1 - haze) + img * haze
    img = np.where(isv[..., None], scene, img)
    # rays below -2 deg that missed everything (closer than the near clip
    # plane): sea, not sky
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
        'Exif': {piexif.ExifIFD.FocalLengthIn35mmFilm: F35},
        'GPS': {piexif.GPSIFD.GPSLatitude: dms(lat),
                piexif.GPSIFD.GPSLatitudeRef: b'N',
                piexif.GPSIFD.GPSLongitude: dms(lon),
                piexif.GPSIFD.GPSLongitudeRef: b'E',
                piexif.GPSIFD.GPSAltitude: (int(z * 10), 10),
                piexif.GPSIFD.GPSImgDirection: (int(heading * 10), 10),
                piexif.GPSIFD.GPSImgDirectionRef: b'T'}}
    piexif.insert(piexif.dump(exif), path)
    return path


results = []
for name, lat, lon, z, heading, pitch, roll, dmin in CASES:
    path = synth_photo(name, lat, lon, z, heading, pitch, roll)
    mlat, mlon = S.meters_per_degree(lat)
    off = rng.uniform(-1500, 1500, 2)
    center = f'{lat + off[0] / mlat:.6f},{lon + off[1] / mlon:.6f}'
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, os.path.join(HERE, 'skyfix.py'), path,
         '--center', center, '--box', str(BOX),
         '--pitch', str(pitch), '--roll', str(roll),
         '--dmin', str(dmin),
         '--out', os.path.join(IMGDIR, name)],
        capture_output=True, text=True)
    dt = time.time() - t0
    if out.returncode:
        print(name, 'FAILED:', out.stderr[-500:])
        continue
    R = json.load(open(os.path.join(IMGDIR, name + '.json')))
    err = np.hypot((R['lat'] - lat) * mlat, (R['lon'] - lon) * mlon)
    results.append(dict(name=name, err_m=float(err), t_s=dt,
                        rms_mrad=R['rms_mrad'],
                        sigma=[R['sigma_n_m'], R['sigma_e_m']],
                        heading_off=R['heading_offset_deg'],
                        el_off_mrad=R['el_offset_mrad']))
    print(f'{name:9s} err {err:6.1f} m  rms {R["rms_mrad"]:.2f} mrad  '
          f'sig ({R["sigma_n_m"]:.0f},{R["sigma_e_m"]:.0f}) m  '
          f'hdg_off {R["heading_offset_deg"]:+.1f}  {dt:.0f}s', flush=True)

errs = np.array([r['err_m'] for r in results])
print(f'\n{len(results)}/{len(CASES)} ok, median {np.median(errs):.1f} m, '
      f'max {errs.max():.1f} m')
with open(os.path.join(OUT, 'e4c_results.json'), 'w') as f:
    json.dump(results, f, indent=1)
