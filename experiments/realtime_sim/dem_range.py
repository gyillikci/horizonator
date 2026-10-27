#!/usr/bin/env python3
"""Where does COP30 stop describing the world? Per column of each render: the exact skyline
(Google Photorealistic 3D Tiles, the render's segmentation pass) against COP30's at the same
pose (same exact camera, straight rays), binned by the range of the ridge COP30 puts there.

    python3 dem_range.py RENDER_DIR
"""
import glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP); sys.path.insert(0, os.path.join(EXP, 'cesium_render'))
import skyfix as F  # noqa: E402
import skyline as S  # noqa: E402
from compare import seg_lines, project, DEM  # noqa: E402

d = sys.argv[1]
edges = [0, 1000, 2000, 3000, 5000, 8000, 12000, 20000, 30000, 60000]
acc = {i: [] for i in range(len(edges) - 1)}
sgn = {i: [] for i in range(len(edges) - 1)}
cms = {}
for f in sorted(glob.glob(os.path.join(d, 'cruise_*_seg.png'))):
    t = os.path.basename(f)[:-8]
    p = json.load(open(os.path.join(d, t + '.json')))['pose']
    W, H = p['width'], p['height']
    exact, _, _ = seg_lines(f)
    key = (round(p['lat'], 1), round(p['lon'], 1))
    if key not in cms:
        cms[key] = S.CMarcher(DEM, (p['lat'] - .6, p['lat'] + .6), (p['lon'] - .8, p['lon'] + .8), d_min=50.0,
                              d_max=60000.0, refraction_k=0.0)
    el, rng = cms[key].skyline(p['lat'], p['lon'], p['z_msl'], F.AZ)
    rows = project(F.AZ, el, p, W, H)
    # range per column: the azimuth each column looks along (centre-row approximation)
    fpx = (W / 2) / math.tan(math.radians(p['hfov_deg']) / 2)
    az_col = p['heading_deg'] + np.degrees(np.arctan((np.arange(W) - (W - 1) / 2) / fpx))
    r_col = np.interp((az_col + 180) % 360 - 180, F.AZ, rng)
    m = np.isfinite(exact) & np.isfinite(rows) & np.isfinite(r_col)
    s_ = (rows[m] - exact[m]) / fpx * 1e3              # + = COP30's ridge lower in the frame
    e = np.abs(s_)
    rb = np.digitize(r_col[m], edges) - 1
    for i in acc:
        acc[i].append(e[rb == i]); sgn[i].append(s_[rb == i])
print('%s: |render - COP30| skyline, by the range of the COP30 ridge in that column' % d)
print('  %-12s %8s %9s %9s %12s' % ('range', 'columns', '|d| p50', 'p90 mrad', 'signed p50'))
for i, v in acc.items():
    x = np.concatenate(v) if v else np.array([])
    if x.size > 100:
        print('  %5.1f-%-5.1f km %8d %9.2f %9.2f %+12.2f' % (edges[i] / 1e3, edges[i + 1] / 1e3, x.size, np.median(x), np.percentile(x, 90),
                                                         np.median(np.concatenate(sgn[i]))))
