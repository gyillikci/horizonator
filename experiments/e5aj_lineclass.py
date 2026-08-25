#!/usr/bin/env python3
"""Is the levelled line a SEA HORIZON or a terrain-backed WATERLINE?

The shipped discriminator is photometric: a large brightness step
across the line means land behind it (E4q). AK3 (2026-08-25) broke it
— in haze a forested coast and the water below it differ by 0.005 in
brightness, so a waterline 3.9 mrad below the true horizon was
accepted as a horizon and given the tight +-2 mrad band, which then
clamped the elevation offset at exactly its edge.

This measures a GEOMETRIC discriminator instead: a true sea horizon
has SKY immediately above it (that is what makes it a horizon); a
waterline has LAND standing above it. Per frame it reports the
fraction of columns where the extracted terrain boundary stands more
than gap_px above the levelled line.
"""
import sys
import numpy as np

import extract
import skyline as S
from e5l_samsea import sam_sea_line

P = '/home/user/celestial-navigation/peakfinder'
FRAMES = [
    ('AK3   ', f'{P}/PF_new_akyaka3.jpg', 73.74, 16.0),
    ('AK2   ', f'{P}/PF_new_akyaka2.jpg', 73.74, 17.0),
    ('AKclean', f'{P}/PF_20260824_195733_clean.jpg', 73.70, 3.0),
    ('t2    ', 'out/blindfield/t2/redacted.jpg', 73.74, 3.0),
    ('t3    ', 'out/blindfield/t3/redacted.jpg', 73.74, 6.0),
    ('175709', f'{P}/PF_20260823_175709.jpg', 38.07, 4.0),
    ('175647', f'{P}/PF_20260823_175647.jpg', 38.07, 4.0),
    ('175124', f'{P}/PF_20260823_175124.jpg', 38.07, 4.0),
    ('165639', f'{P}/PF_20260823_165639.jpg', 38.07, 4.0),
    ('153457', f'{P}/PF_20260823_153457.jpg', 38.07, 4.0),
]

print(f"{'frame':8s} {'source':10s} {'pitch':>7s} {'contrast':>9s} "
      f"{'land-above':>11s} {'median gap':>11s}")
for tag, path, fov, z in FRAMES:
    img = extract.load_image(path)
    H, W = img.shape[:2]
    f_px = (W / 2) / np.tan(np.radians(fov) / 2)
    dip = S.horizon_dip_rad(max(z, 0.5))
    rows, conf = extract.skyline_seam(img)
    got = sam_sea_line((img * 255).astype(np.uint8), ceiling_rows=rows)
    seed = [(got[0], got[1])] if got else None
    lv = extract.sea_horizon_attitude_radon(img, f_px, dip, max_step=0.20,
                                            extra_candidates=seed)
    if lv is None:
        print(f'{tag:8s} {"DECLINED":10s}')
        continue
    u = np.arange(W) - (W - 1) / 2.0
    a = np.tan(-dip - np.radians(lv['pitch_deg']))
    b = -np.tan(np.radians(lv['roll_deg']))
    line = (H - 1) / 2.0 - (a * np.hypot(u, f_px) + b * u)
    ok = np.asarray(conf) > 0
    gap = (line - np.asarray(rows, float))[ok]     # + = land above line
    gap_px = 0.012 * H                            # ~1.2% of frame height
    print(f'{tag:8s} {lv["source"]:10s} {lv["pitch_deg"]:+7.2f} '
          f'{lv["contrast"]:+9.3f} {float((gap > gap_px).mean()):11.2f} '
          f'{float(np.median(gap)):10.0f}px')
