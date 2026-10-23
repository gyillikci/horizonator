#!/usr/bin/env python3
"""E5bz, Bodrum 9: what the DEM skyline holds that the photo does not.

At the EXIF position, fits the heading and elevation offsets with the
offset bound lifted, splits the residual by DEM crest range and by
azimuth, and draws the photo skyline against the COP30 skyline
coloured by crest range and against the same skyline cut at 15 km.

    python3 e5bz_bodrum9.py      -> out/compass/bodrum9_range.png
"""

import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skyfix as F  # noqa: E402
import skyline as S  # noqa: E402
import extract  # noqa: E402
from assets import celnav  # noqa: E402

LAT, LON, Z, HD, FOV = 37.01992, 27.44426, 9.0, 167.093, 73.7


def skyline(cm):
    el, r = cm.skyline(LAT, LON, Z, F.AZ)
    return el + np.where(np.isfinite(r) & (r > 0),
                         9.0 / np.maximum(r, 1000.0), 0.0), r


def main():
    F.EXTRACTOR = 'ewasr'
    img = extract.load_image(celnav('peakfinder/PF_new_bodrum9.jpg'))
    el_obs, w, d = F.observation(img, FOV, HD, 0.0)
    dem = os.path.expanduser('~/.horizonator/DEMs_COP30')
    box = ((LAT - .6, LAT + .6), (LON - .8, LON + .8))
    el, r = skyline(S.CMarcher(dem, *box, d_min=1000.0))
    el15, _ = skyline(S.CMarcher(dem, *box, d_min=1000.0, d_max=15000.0))
    c, s, b = F.photo_cost(el_obs, w, np.nan_to_num(el),
                           np.arange(-300, 301, 2),
                           np.linspace(-0.04, 0.04, 161))
    print('at the truth, offsets unbounded: psi %+.1f deg, beta %+.1f mrad,'
          ' rms %.2f mrad' % (s * 0.1, b * 1e3, math.sqrt(2 * c) * 1e3))
    eo, m = np.roll(el_obs, s), np.roll(w, s) > 0
    res, rr = (eo[m] + b - el[m]) * 1e3, r[m] / 1e3
    for lo, hi in ((0, 6), (6, 15), (15, 41)):
        k = (rr >= lo) & (rr < hi)
        if k.any():
            print('crest %2d-%2d km: %3.0f%% of columns, median photo - DEM '
                  '%+6.1f mrad' % (lo, hi, 100 * k.mean(),
                                   np.median(res[k])))
    near = m & (r < 15e3)
    rel = (F.AZ[near] - (HD + 0.1 * s) + 180) % 360 - 180
    A = np.c_[np.ones(rel.size), np.radians(rel)]
    coef = np.linalg.lstsq(A, el[near] - eo[near], rcond=None)[0]
    print('tilt of the residual across the frame (near columns): '
          '%.2f deg of roll' % math.degrees(coef[1]))

    H, W, _ = img.shape
    f = (W / 2) / math.tan(math.radians(FOV) / 2)

    def proj(e):
        rl = (F.AZ - (HD + 0.1 * s) + 180) % 360 - 180
        u = f * np.tan(np.radians(rl))
        v = np.tan(e - b) * np.hypot(u, f)
        k = np.abs(rl) < FOV / 2
        x, y = u[k] + (W - 1) / 2, (H - 1) / 2 - v[k]
        o = np.argsort(x)
        return x[o], y[o], k, o

    fig, ax = plt.subplots(2, 1, figsize=(13, 9),
                           gridspec_kw=dict(height_ratios=[1.1, 1]))
    ax[0].imshow(img)
    ax[0].plot(np.arange(W), d['rows'], 'w-', lw=1.2,
               label='photo skyline (eWaSR)')
    x, y, k, o = proj(el)
    sc = ax[0].scatter(x, y, c=(r[k] / 1e3)[o], s=3, cmap='viridis',
                       vmin=0, vmax=25,
                       label='COP30 skyline at truth, colour = crest range')
    x, y, _, _ = proj(el15)
    ax[0].plot(x, y, 'r--', lw=1, label='COP30 skyline cut at 15 km')
    ax[0].set_ylim(H * 0.62, H * 0.30)
    ax[0].set_xlim(0, W)
    ax[0].legend(loc='upper left', fontsize=8)
    plt.colorbar(sc, ax=ax[0], label='crest range km', fraction=.025)
    ax[0].set_title('Bodrum 9 at the EXIF position: heading %+.1f deg from '
                    'the compass, elevation offset %+.1f mrad'
                    % (0.1 * s, b * 1e3))
    az = (F.AZ[m] + 360) % 360
    o = np.argsort(az)
    ax[1].plot(az[o], (eo[m] + b)[o] * 1e3, 'k.', ms=2, label='photo')
    ax[1].scatter(az, el[m] * 1e3, c=r[m] / 1e3, s=3, cmap='viridis',
                  vmin=0, vmax=25, label='DEM, 40 km')
    ax[1].plot(az[o], el15[m][o] * 1e3, 'r--', lw=1, label='DEM, 15 km cut')
    ax[1].axhline(-S.horizon_dip_rad(Z) * 1e3, color='0.5', lw=.8, ls=':',
                  label='sea horizon')
    ax[1].set_xlabel('true azimuth deg')
    ax[1].set_ylabel('elevation mrad')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'out', 'compass', 'bodrum9_range.png'),
                dpi=110)


if __name__ == '__main__':
    main()
