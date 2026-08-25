#!/usr/bin/env python3
"""What line did the level chain actually lock onto in AK3?

Draws, on the solver's own working frame:
  the SAM water-mask proposal (the 'waterline' seed),
  the native radon candidate (no seed),
  the line the solver ACCEPTED, with its source label.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extract
import skyline as S
from e5l_samsea import sam_sea_line

path = '/home/user/celestial-navigation/peakfinder/PF_new_akyaka3.jpg'
img = extract.load_image(path)
H, W = img.shape[:2]
f_px = (W / 2) / np.tan(np.radians(73.74) / 2)
dip = S.horizon_dip_rad(16.0)
u = np.arange(W) - (W - 1) / 2.0
Hu = np.hypot(u, f_px)

rows, conf = extract.skyline_seam(img)
got = sam_sea_line((img * 255).astype(np.uint8), ceiling_rows=rows)
seed = [(got[0], got[1])] if got else None
print('SAM seed:', None if not got else (round(got[0], 1), round(got[1], 4)))

native = extract.sea_horizon_attitude_radon(img, f_px, dip, max_step=0.20)
seeded = extract.sea_horizon_attitude_radon(img, f_px, dip, max_step=0.20,
                                            extra_candidates=seed)
for nm, lv in (('native (no seed)', native), ('seeded (as solved)', seeded)):
    if lv is None:
        print(f'{nm}: DECLINED')
    else:
        print(f"{nm}: source={lv['source']} pitch={lv['pitch_deg']:+.2f} "
              f"roll={lv['roll_deg']:+.2f} contrast={lv['contrast']:+.3f} "
              f"frac={lv['frac']:.2f} span={lv['span_frac']:.2f} "
              f"rms_px={lv['rms_px']:.2f} score={lv['score']:.1f}")


def line_of(lv):
    a = np.tan(-dip - np.radians(lv['pitch_deg']))
    b = -np.tan(np.radians(lv['roll_deg']))
    return (H - 1) / 2.0 - (a * Hu + b * u)


fig, ax = plt.subplots(figsize=(15, 15 * H / W + 1.2))
ax.imshow(img)
if got:
    ax.plot(np.arange(W), got[0] + got[1] * u, '-', lw=1.6, color='magenta',
            label=f'SAM water-mask proposal (row {got[0]:.0f})')
if native is not None:
    ax.plot(np.arange(W), line_of(native), '--', lw=1.6, color='yellow',
            label=f"native radon [{native['source']}] "
                  f"pitch {native['pitch_deg']:+.2f}")
if seeded is not None:
    ax.plot(np.arange(W), line_of(seeded), '-', lw=2.2, color='red',
            label=f"ACCEPTED [{seeded['source']}] "
                  f"pitch {seeded['pitch_deg']:+.2f}, "
                  f"roll {seeded['roll_deg']:+.2f}")
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis('off')
ax.legend(loc='lower left', fontsize=10)
ax.set_title('AK3 level chain: which line did it lock onto?', fontsize=12)
fig.tight_layout()
fig.savefig('out/ak3/AK3_level.png', dpi=110)
print('-> out/ak3/AK3_level.png')
