#!/usr/bin/env python3
"""E5z: every boundary front-end in the toolkit, on one frame, no
exceptions.

One extremely hazy PeakFinder frame (Kumlubük looking 254 deg toward
the Datça/Symi far coast, ~20-40 km through haze, the app's overlay
floating well above the real ridge) is pushed through EVERY boundary
determiner this project owns:

  1  seam detector             extract.skyline_seam
  2  learned template + DP     e4m_diverse.seam_extract (Ahmad-style)
  3  radon line candidates     extract.horizon_candidates (top 5)
  4  radon accepted level      extract.sea_horizon_attitude_radon
  5  SAM waterline chain       e5l_samsea.sam_sea_line (+ceiling)
  6  SAM+radon hybrid level    radon refinement of the SAM seed
  7  eWaSR segmentation        ewasr_bridge (water/sky/obstacle)
  8  PeakFinder ink recovery   e5w stroke_rows
  9  DEM as-shot               skyline at EXIF pose, camera level
 10  DEM attitude-fitted       E5s joint heading/pitch/roll fit

Output: a per-method panel grid cropped to the horizon band, plus a
single full-frame composite. Every method reports either its curve
or its explicit refusal — absence is a result too.

Run:  python3 e5z_allfronts.py     (writes out/e5z/*.png)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extract
import skyline as S
import skyfix as SF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'e5z')
DEM = os.path.expanduser('~/.horizonator/DEMs_SRTM1_WM')
# photo id and pose from the command line; defaults = the E5z frame
_id = sys.argv[1] if len(sys.argv) > 1 else 'PF_20260824_115038'
_pose = ([float(x) for x in sys.argv[2].split(',')]
         if len(sys.argv) > 2 else [36.64088, 28.09631, 254.5, 20.0])
PHOTO = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                     'celestial-navigation', 'peakfinder', _id + '.jpg')
LAT, LON, HEADING, Z = _pose
SCRATCH = ('/tmp/claude-0/-home-user/'
           '792503f9-74c5-5111-83ca-eeeda63e838d/scratchpad')


def main():
    img = extract.load_image(PHOTO)
    H, W, _ = img.shape
    f_px = H * 24.0 / 36.0
    fov = np.degrees(2 * np.arctan((W / 2) / f_px))
    dip = S.horizon_dip_rad(Z)
    u = np.arange(W) - (W - 1) / 2
    g = np.asarray(img, float).mean(axis=2)
    results = {}

    # 1 seam
    seam, conf = extract.skyline_seam(img)
    results['1 seam detector'] = ('line', seam)

    # 2 learned template + DP
    try:
        from e4m_diverse import seam_extract
        w_svm = np.load(os.path.join(HERE, 'out', 'e4m_svm.npz'))['w']
        results['2 learned template+DP'] = ('line', seam_extract(img, w_svm))
    except Exception as ex:
        results['2 learned template+DP'] = ('refused', str(ex)[:60])

    # 3 radon candidates
    cands = extract.horizon_candidates(g)
    results['3 radon candidates'] = ('lines', [
        r0 + m * u for r0, m, sc in cands[:5]])

    # 4 radon accepted level
    lev = extract.sea_horizon_attitude_radon(img, f_px, dip)
    if lev:
        a = np.tan(-dip - np.radians(lev['pitch_deg']))
        b = -np.radians(lev['roll_deg'])
        results['4 radon level'] = ('line', (H - 1) / 2
                                    - (a * np.hypot(u, f_px) + b * u))
    else:
        results['4 radon level'] = ('refused', 'no acceptable sea horizon')

    # 5 SAM waterline chain
    from e5l_samsea import sam_sea_line
    got = sam_sea_line((img * 255).astype(np.uint8), ceiling_rows=seam)
    if got:
        r0, sl, frac, mask = got
        results['5 SAM waterline'] = ('line+mask',
                                      r0 + sl * u, mask)
        # 6 hybrid: radon refinement of the SAM seed
        lev2 = extract.sea_horizon_attitude_radon(
            img, f_px, dip, extra_candidates=[(r0, sl)])
        if lev2:
            a = np.tan(-dip - np.radians(lev2['pitch_deg']))
            b = -np.radians(lev2['roll_deg'])
            results['6 SAM+radon level'] = (
                'line', (H - 1) / 2 - (a * np.hypot(u, f_px) + b * u))
        else:
            results['6 SAM+radon level'] = ('refused',
                                            'seed rejected by gates')
    else:
        results['5 SAM waterline'] = ('refused', 'no anchored water mask')
        results['6 SAM+radon level'] = ('refused', 'no seed available')

    # 7 eWaSR
    try:
        from ewasr_bridge import EWasr
        seg = EWasr(os.path.join(SCRATCH, 'eWaSR'),
                    os.path.join(SCRATCH, 'ewasr_resnet18.pth'))
        cls = seg.predict(img)
        sky = cls == 2
        skyline_ew = np.where(sky.any(0),
                              sky.shape[0] - sky[::-1].argmax(0) - 1,
                              np.nan).astype(float)
        skyline_ew[~sky.any(0)] = np.nan
        # last sky row from the top, per column: first non-sky below
        first_non = np.argmax(~sky, axis=0).astype(float)
        first_non[sky.all(0)] = np.nan
        results['7 eWaSR sky boundary'] = ('line+cls', first_non, cls)
    except Exception as ex:
        results['7 eWaSR sky boundary'] = ('refused', str(ex)[:60])

    # 8 PeakFinder ink recovery
    from e5w_pfline import stroke_rows
    results['8 PeakFinder ink'] = ('dots', stroke_rows(img, seam))

    # 9/10 DEM lines
    cm = S.CMarcher(DEM, (LAT - 0.6, LAT + 0.6), (LON - 0.8, LON + 0.8))
    az0 = HEADING + np.degrees(np.arctan2(u, f_px))
    el0, _ = cm.skyline(LAT, LON, Z, az0)
    el0 = S.seahorizon_fill(el0, Z)
    results['9 DEM as-shot'] = ('line', (H - 1) / 2
                                - np.tan(el0) * np.hypot(u, f_px))
    el_full, _ = cm.skyline(LAT, LON, Z, SF.AZ)
    el_full = S.seahorizon_fill(el_full, Z)
    best = (np.inf,)
    for roll in np.arange(-3, 3.01, 0.5):
        eo, w, _ = SF.observation(img, fov, HEADING, roll, 0.0)
        c, s, b = SF.fast_photo_cost(eo, w, el_full, range(-100, 101),
                                     betas=np.arange(-0.10, 0.3001,
                                                     0.002))
        if c < best[0]:
            best = (c, s, b, roll)
    c, sft, pitch, roll = best
    az1 = (HEADING + sft * 0.1) + np.degrees(np.arctan2(u, f_px))
    el1, _ = cm.skyline(LAT, LON, Z, az1)
    el1 = S.seahorizon_fill(el1, Z)
    fit_line = (H - 1) / 2 - (np.tan(el1 - pitch) * np.hypot(u, f_px)
                              - np.radians(roll) * u)
    results['10 DEM fitted'] = ('line', fit_line)
    print(f'attitude fit: dheading {sft*0.1:+.1f} deg, pitch '
          f'{pitch*1e3:+.0f} mrad, roll {roll:+.1f}, rms '
          f'{np.sqrt(2*c)*1e3:.1f} mrad')

    # ---- panel grid, cropped to the horizon band
    r_lo, r_hi = int(0.18 * H), int(0.62 * H)
    names = list(results)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for ax, name in zip(axes.ravel(), names):
        kind = results[name][0]
        ax.imshow(img[r_lo:r_hi])
        if kind == 'refused':
            ax.text(0.5, 0.5, f'REFUSED:\n{results[name][1]}',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=10, color='#ff3b30', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.85))
        elif kind == 'line':
            ax.plot(np.arange(W), results[name][1] - r_lo,
                    color='#ff3b30', lw=1.6)
        elif kind == 'lines':
            for ln in results[name][1]:
                ax.plot(np.arange(W), ln - r_lo, lw=1.2, alpha=0.85)
        elif kind == 'dots':
            ax.plot(np.arange(W), results[name][1] - r_lo, '.',
                    color='#ff9f0a', ms=2)
        elif kind == 'line+mask':
            mask = results[name][2][r_lo:r_hi]
            over = np.zeros((*mask.shape, 4))
            over[mask] = (0.0, 0.5, 1.0, 0.35)
            ax.imshow(over)
            ax.plot(np.arange(W), results[name][1] - r_lo,
                    color='#ff3b30', lw=1.6)
        elif kind == 'line+cls':
            cls = results[name][2][r_lo:r_hi]
            over = np.zeros((*cls.shape, 4))
            over[cls == 1] = (0.0, 0.5, 1.0, 0.35)     # water
            over[cls == 0] = (1.0, 0.2, 0.2, 0.35)     # obstacle/land
            over[cls == 2] = (1.0, 1.0, 0.2, 0.15)     # sky
            ax.imshow(over)
            ax.plot(np.arange(W), results[name][1] - r_lo,
                    color='#1c1c1e', lw=1.4)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylim(r_hi - r_lo, 0)
    fig.suptitle(f'{_id} — every boundary front-end, '
                 'horizon band crop', fontsize=12)
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, _id + '_grid.png'), dpi=100)
    plt.close(fig)
    print(os.path.join(OUT, _id + '_grid.png'))

    # ---- full-frame composite
    fig, ax = plt.subplots(figsize=(8, 17))
    ax.imshow(img)
    styles = {'1 seam detector': ('#ff3b30', '-', 1.4),
              '2 learned template+DP': ('#bf5af2', '-', 1.2),
              '4 radon level': ('#8e8e93', '--', 1.2),
              '5 SAM waterline': ('#0a84ff', '-', 1.4),
              '6 SAM+radon level': ('#5ac8fa', '--', 1.2),
              '7 eWaSR sky boundary': ('#ffd60a', '-', 1.4),
              '9 DEM as-shot': ('#ff9f0a', ':', 1.6),
              '10 DEM fitted': ('#34c759', '-', 2.0)}
    for name, (col, ls, lw) in styles.items():
        kind = results[name][0]
        if kind in ('line', 'line+mask', 'line+cls'):
            ax.plot(np.arange(W), results[name][1], color=col, ls=ls,
                    lw=lw, label=name)
    pf = results['8 PeakFinder ink'][1]
    ax.plot(np.arange(W), pf, '.', color='#ff2d55', ms=1.5,
            label='8 PeakFinder ink')
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.set_title('composite — all boundaries on the full frame',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, _id + '_composite.png'), dpi=100)
    plt.close(fig)
    print(os.path.join(OUT, _id + '_composite.png'))
    for n in names:
        k = results[n][0]
        print(f'  {n}: {"REFUSED - " + results[n][1] if k == "refused" else k}')


if __name__ == '__main__':
    main()
