#!/usr/bin/env python3
"""E4q: the auto-leveller against INDEPENDENTLY MEASURED horizons.

Every previous auto-level result (E4g onward) was checked against the
solver's own geometry — the +-2 mrad beta window after levelling was an
assumption, never a measurement. MaSTr1325 (Bovcon et al. 2019, ViCoS)
closes that: 1325 real maritime images, each with a per-image IMU mask
whose boundary IS the inertially measured horizon line, plus a
sea/sky/obstacle segmentation mask.

What the dataset actually is matters for the experiment design: it was
shot from a USV inside and around a marina, so in the MEDIAN image the
sea horizon is completely occluded by land (measured here: 0% of
columns have sky directly above water; only 111 of 1325 images show an
open horizon over more than half their width). That makes MaSTr1325
two tests, not one:

  A  VETO CORRECTNESS (all 1325) — the auto-leveller is supposed to
     REFUSE a scene whose horizon is hidden behind land, using its
     one-sided below-line veto and photometric water check. The
     segmentation mask says which images truly have an open horizon,
     so this is the first ground-truthed measurement of the false-
     horizon defence: accepts on open scenes = availability, accepts
     on occluded scenes = the dangerous case.

  B  ACCURACY (open-horizon subset) — where the horizon IS visible,
     how far is the estimated attitude from the IMU's?

Comparison is done in PIXEL space first (row offset at image center,
slope across the frame) because that part is independent of the
unknown focal length, then converted to angles with --fov (default
65 deg horizontal) for the mrad numbers the beta window cares about.

Run:   python3 e4q_imu_horizon.py [n_images] [--fov 65] [--open 0.5]
       (writes out/e4q_results.json)
"""

import os
import sys
import glob
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import extract
import skyline as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
DATA = os.environ.get(
    'MASTR1325',
    os.path.join(os.path.dirname(os.path.dirname(HERE)),
                 'celestial-navigation', 'MaSTr1325'))
CAM_Z = 1.0            # USV camera height above waterline, meters
LBL_WATER, LBL_SKY = 1, 2

FOV_DEG, OPEN_T = 65.0, 0.5
for i, a in enumerate(sys.argv):
    if a == '--fov':
        FOV_DEG = float(sys.argv[i + 1])
    elif a == '--open':
        OPEN_T = float(sys.argv[i + 1])


def open_horizon_frac(seg):
    """Fraction of columns where the topmost non-sky pixel is WATER —
    i.e. the sea horizon itself is what bounds the sky. Land in front
    of the horizon (the marina case) drives this to zero."""
    H, W = seg.shape
    nonsky = seg != LBL_SKY
    top = np.where(nonsky.any(0), nonsky.argmax(0), -1)
    cols = np.arange(W)
    v = top >= 0
    return float((seg[top[v], cols[v]] == LBL_WATER).sum() / W)


def fit_line(cols, rows, W):
    """Least-squares row = r0 + s*u about the image center.
    Returns (row at center, slope in px per px, rms residual)."""
    u = cols - (W - 1) / 2.0
    A = np.column_stack([np.ones_like(u, dtype=float), u])
    sol, *_ = np.linalg.lstsq(A, rows.astype(float), rcond=None)
    rms = float(np.sqrt(np.mean((A @ sol - rows) ** 2)))
    return float(sol[0]), float(sol[1]), rms


def imu_line(imu):
    """The IMU horizon as a pixel line. The mask marks everything below
    the inertial horizon, so its upper boundary is the line."""
    H, W = imu.shape
    cols = np.where(imu.any(axis=0))[0]
    if cols.size < W // 3:
        return None
    rows = imu.argmax(axis=0)[cols]
    if np.median(rows) <= 1 or np.median(rows) >= H - 2:
        return None          # horizon outside the frame
    return fit_line(cols, rows, W)


def est_line(est, f_px, H, W):
    """The estimated attitude as the same pixel line:
    v = tan(-dip-pitch)*hypot(u,f) - roll*u, linearised about u=0."""
    dip = S.horizon_dip_rad(CAM_Z)
    p = np.radians(est['pitch_deg'])
    r = np.radians(est['roll_deg'])
    u = np.array([-(W - 1) / 2.0, 0.0, (W - 1) / 2.0])
    v = np.tan(-dip - p) * np.hypot(u, f_px) - r * u
    rows = (H - 1) / 2.0 - v
    return fit_line(u + (W - 1) / 2.0, rows, W)


def main(n=None):
    imgs = sorted(glob.glob(os.path.join(DATA, '*.jpg')))
    if n:
        imgs = imgs[:n]
    dip = S.horizon_dip_rad(CAM_Z)
    rows_out = []
    for k, ip in enumerate(imgs):
        stem = os.path.basename(ip)[:-4]
        mp = os.path.join(DATA, stem + 'm.png')
        up = os.path.join(DATA, 'imus', stem + '.png')
        if not (os.path.exists(mp) and os.path.exists(up)):
            continue
        rgb = np.asarray(Image.open(ip).convert('RGB'), float) / 255.0
        seg = np.asarray(Image.open(mp))
        imu = np.asarray(Image.open(up))
        H, W = seg.shape
        f_px = (W / 2) / np.tan(np.radians(FOV_DEG) / 2)

        openf = open_horizon_frac(seg)
        truth = imu_line(imu)
        srows, conf = extract.skyline_seam(rgb)
        est = extract.sea_horizon_attitude(srows, conf, rgb.shape, f_px,
                                           dip, rgb=rgb)
        r = dict(image=stem, open_frac=openf,
                 water_frac=float((seg == LBL_WATER).mean()),
                 truth_line=truth, accepted=est is not None)
        if est is not None:
            r['est'] = {k2: float(v2) for k2, v2 in est.items()}
            el = est_line(est, f_px, H, W)
            r['est_line'] = el
            if truth is not None:
                d_row = el[0] - truth[0]
                d_slope = el[1] - truth[1]
                r['err'] = dict(
                    d_row_px=float(d_row),
                    d_slope_px_per_px=float(d_slope),
                    pitch_deg=float(np.degrees(np.arctan(d_row / f_px))),
                    roll_deg=float(np.degrees(np.arctan(d_slope))),
                    edge_mrad=float((abs(d_row) + abs(d_slope)
                                     * (W - 1) / 2.0) / f_px * 1e3))
        rows_out.append(r)
        if (k + 1) % 200 == 0:
            print(f'  [{k+1}/{len(imgs)}]', flush=True)

    # ---------- A: veto correctness
    openi = [r for r in rows_out if r['open_frac'] >= OPEN_T]
    occl = [r for r in rows_out if r['open_frac'] < 0.05]
    acc_open = sum(1 for r in openi if r['accepted'])
    acc_occl = sum(1 for r in occl if r['accepted'])
    print(f'\n=== A. veto correctness ({len(rows_out)} images) ===')
    print(f'  open horizon (>={OPEN_T:.0%} of columns): {len(openi)} '
          f'images, accepted {acc_open} '
          f'({100*acc_open/max(len(openi),1):.0f}% availability)')
    print(f'  occluded (<5% open):  {len(occl)} images, accepted '
          f'{acc_occl} ({100*acc_occl/max(len(occl),1):.0f}% '
          f'FALSE ACCEPTS)')

    # ---------- B: accuracy where the horizon is real
    good = [r for r in openi if r.get('err')]
    print(f'\n=== B. accuracy on open-horizon accepts ({len(good)}) ===')
    summary = dict(n=len(rows_out), fov_deg=FOV_DEG, open_thresh=OPEN_T,
                   n_open=len(openi), acc_open=acc_open,
                   n_occluded=len(occl), acc_occluded=acc_occl)
    if good:
        dr = np.array([r['err']['d_row_px'] for r in good])
        ds = np.array([r['err']['d_slope_px_per_px'] for r in good])
        ed = np.array([r['err']['edge_mrad'] for r in good])
        pit = np.array([r['err']['pitch_deg'] for r in good])
        rol = np.array([r['err']['roll_deg'] for r in good])
        print(f'  row offset at center: median {np.median(dr):+.1f} px  '
              f'MAD {np.median(np.abs(dr-np.median(dr))):.1f}  '
              f'p90|.| {np.percentile(np.abs(dr),90):.1f}')
        print(f'  pitch err: median {np.median(pit):+.2f} deg   '
              f'roll err: median {np.median(rol):+.2f} deg')
        print(f'  edge elevation err: median {np.median(ed):.1f} mrad  '
              f'p90 {np.percentile(ed,90):.1f}')
        print(f'  within  2 mrad: {(ed<=2).mean()*100:.0f}%   '
              f'within 10 mrad: {(ed<=10).mean()*100:.0f}%')
        summary.update(
            n_good=len(good),
            d_row_median_px=float(np.median(dr)),
            d_row_p90abs_px=float(np.percentile(np.abs(dr), 90)),
            pitch_median_deg=float(np.median(pit)),
            roll_median_deg=float(np.median(rol)),
            edge_median_mrad=float(np.median(ed)),
            edge_p90_mrad=float(np.percentile(ed, 90)),
            frac_within_2mrad=float((ed <= 2).mean()),
            frac_within_10mrad=float((ed <= 10).mean()))
    with open(os.path.join(OUT, 'e4q_results.json'), 'w') as f:
        json.dump(dict(summary=summary, images=rows_out), f, indent=1)
    print(f"\nwrote {os.path.join(OUT, 'e4q_results.json')}")
    return summary


if __name__ == '__main__':
    nn = None
    for a in sys.argv[1:]:
        if not a.startswith('--') and a.isdigit():
            nn = int(a)
            break
    main(nn)
