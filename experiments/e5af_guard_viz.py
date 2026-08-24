#!/usr/bin/env python3
"""E5af visual: what the two extraction guards did to the t3 frame.

Top: the raw eWaSR boundary as the solver used to see it (every
confident column, including the rectangular haze-notch dropouts).
Bottom: the guarded extraction — the extractor-disagreement pre-check
(seam computed in parallel; fall back when >20% of mutually-confident
columns disagree by >1.5% of frame height) and the terrain-plausibility
filter (no vertical walls: U-notch / n-bump interiors zero-weighted).
"""

import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import skyfix
import extract

photo = sys.argv[1] if len(sys.argv) > 1 else 'out/blindfield/t3/redacted.jpg'
out = sys.argv[2] if len(sys.argv) > 2 else 'out/guard/t3_guard_viz.png'

# the solver's own loader (downscale to 1600 px + float): the guards run
# on THIS image, and the seam behaves differently at full resolution
img = extract.load_image(photo)
H, W = img.shape[:2]

# raw eWaSR boundary (the pre-guard view): reuse skyfix's extractor but
# bypass the guards by calling the pieces directly
skyfix.EXTRACTOR = 'ewasr'
rows_g, conf_g = skyfix.extract_boundary(img)          # guarded path
# raw path: replicate the ewasr block without guards
if skyfix._EWASR is None:
    raise SystemExit('eWaSR did not load')
cls = skyfix._EWASR.predict(img)
sky = cls == 2
rows_r = np.full(W, H - 1, float)
conf_r = np.zeros(W)
has = sky.any(0)
top = np.where(has, sky.argmax(0), 0)
for x in range(W):
    if not has[x]:
        continue
    below = ~sky[top[x]:, x]
    if below.any():
        k = int(np.argmax(below))
        if k >= 3:
            rows_r[x] = top[x] + k
            conf_r[x] = 1.0
rows_s, conf_s = extract.skyline_seam(img)

fell_back = not np.array_equal(np.asarray(rows_g, float), rows_r)

fig, axes = plt.subplots(2, 1, figsize=(14, 2 * 14 * H / W + 1))
for ax, (rows, conf, ttl) in zip(axes, [
        (rows_r, conf_r, 'raw eWaSR boundary (pre-guard)'),
        (rows_g, conf_g,
         'guarded extraction'
         + (' — pre-check fell back to the seam' if fell_back else
            ' — eWaSR kept, implausible columns zero-weighted'))]):
    ax.imshow(img)
    x = np.arange(W)
    ok = conf > 0
    ax.plot(x[ok], np.asarray(rows, float)[ok], '.', ms=1.5,
            color='lime', label='used')
    ax.plot(x[~ok], np.asarray(rows, float)[~ok], '.', ms=1.5,
            color='red', label='zero-weighted')
    ax.set_title(ttl, fontsize=11)
    ax.legend(loc='lower right', fontsize=8, markerscale=6)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
fig.tight_layout()
fig.savefig(out, dpi=110)
kept_r, kept_g = int((conf_r > 0).sum()), int((np.asarray(conf_g) > 0).sum())
print(f'raw kept {kept_r}/{W} cols; guarded kept {kept_g}/{W}; '
      f'fallback={fell_back} -> {out}')
