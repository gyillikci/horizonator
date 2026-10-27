#!/usr/bin/env python3
"""Camera poses of the 44 campaign frames at their truth, for driver.mjs.

Position: the frame's truth (lat, lon); height: its EXIF altitude (above mean
sea level) plus the EGM96 geoid separation N, since the tiles are
WGS84-ellipsoidal. Field of view and image size: those the solver uses.

Attitude: the EXIF compass is off by up to ~15 deg on some frames, so the
heading and pitch come from fitting the frame's SegFormer skyline (E5ch's
seg boundary) to the COP30 skyline at the truth, as e5cf_faint_ridge's
truth_fit does; roll from the frame's sea line where it was levelled, else 0.
This only has to point the render at the right terrain: compare.py fits the
offsets between render and photo again.

    python3 make_poses.py OUT.json [TAG ...]
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
import extract  # noqa: E402
import skyfix as F  # noqa: E402
import skyline as S  # noqa: E402
from assets import celnav  # noqa: E402

FRAMES = os.path.join(EXP, 'data', 'campaign_frames.json')
SEG = os.path.join(EXP, 'out', 'segformer')
LEVEL = os.path.join(EXP, 'out', 'seg_level')
DEM = os.path.expanduser('~/.horizonator/DEMs_COP30')


def geoid_n(lat, lon):
    import pyproj
    pyproj.network.set_network_enabled(True)
    t = pyproj.Transformer.from_crs('EPSG:4326+5773', 'EPSG:4979', always_xy=True)
    n = t.transform(lon, lat, 0.0)[2]
    if not math.isfinite(n) or abs(n) > 150:
        raise RuntimeError('no geoid separation at %.5f %.5f' % (lat, lon))
    return float(n)


def level_roll(tag, img, fov):
    """Roll from the frame's sea line, as skyfix --level-npy would fit it."""
    p = os.path.join(LEVEL, tag + '_sealine.npz')
    if not os.path.exists(p):
        return 0.0, False
    src = np.load(p)['rows'].astype(float)
    W = img.shape[1]
    rows = np.interp(np.linspace(0, src.size - 1, W), np.arange(src.size), src) * (W / src.size)
    f_px = (W / 2) / np.tan(np.radians(fov) / 2)
    lv = extract.sea_horizon_attitude(np.nan_to_num(rows), np.isfinite(rows) * 1.0, img.shape, f_px,
                                      S.horizon_dip_rad(2.0))
    return (float(lv['roll_deg']), True) if lv else (0.0, False)


def main():
    out = sys.argv[1]
    tags = sys.argv[2:]
    frames = json.load(open(FRAMES))['frames']
    if tags:
        frames = [f for f in frames if f['tag'] in tags]
    poses = []
    for fr in frames:
        img = extract.load_image(celnav(fr['path']))
        H, W = img.shape[:2]
        roll, levelled = level_roll(fr['tag'], img, fr['fov'])
        b = np.load(os.path.join(SEG, fr['tag'] + '_seg.npz'))
        F.BOUNDARY_ROWS, F.BOUNDARY_CONF = b['rows'], b['conf']
        el_obs, w, _ = F.observation(img, fr['fov'], fr['heading'], roll)
        cm = S.CMarcher(DEM, (fr['lat'] - .6, fr['lat'] + .6), (fr['lon'] - .8, fr['lon'] + .8), d_min=1000.0)
        el, r = cm.skyline(fr['lat'], fr['lon'], fr['z'], F.AZ)
        el = np.nan_to_num(el + np.where(np.isfinite(r) & (r > 0), 9.0 / np.maximum(r, 1000.0), 0.0))
        c, s, beta = F.photo_cost(el_obs, w, el, np.arange(-300, 301, 2), np.linspace(-0.14, 0.14, 281))
        n = geoid_n(fr['lat'], fr['lon'])
        poses.append(dict(
            tag=fr['tag'], lat=fr['lat'], lon=fr['lon'], z_msl=fr['z'], geoid_n=n,
            h_ellipsoid=fr['z'] + n, heading_deg=(fr['heading'] + 0.1 * s) % 360.0,
            pitch_deg=math.degrees(beta), roll_deg=roll, hfov_deg=fr['fov'], width=int(W), height=int(H),
            attitude_source=dict(heading_offset_deg=0.1 * s, fit_rms_mrad=math.sqrt(2 * c) * 1e3,
                                 roll_from_sea_line=levelled),
            sse=16, timeout_s=420))
        print('%-28s %4dx%-4d N %5.1f  heading %+6.1f  pitch %+5.2f  roll %+5.2f  fit %.1f mrad'
              % (fr['tag'], W, H, n, 0.1 * s, math.degrees(beta), roll, math.sqrt(2 * c) * 1e3), flush=True)
    json.dump(poses, open(out, 'w'), indent=1)
    print('wrote', out, len(poses), 'poses')


if __name__ == '__main__':
    main()
