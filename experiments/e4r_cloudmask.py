#!/usr/bin/env python3
"""E4r: can the eWaSR segmenter buy back cloud availability? No.

Two arms against the E4n cloud matrix (same images, same centers):
  A  plain extraction (the E4n baseline)
  B  segmentation-derived skyline (topmost obstacle pixel per column,
     cloud transparent) from the pretrained eWaSR bridge

Both formulations of segmenter assistance FAILED, informatively:
masking cloud-base columns is a no-op (the extractor's surviving
columns still carry the wrong flat deck-bottom geometry), and the
seg-derived skyline recovers solves under stratus but wrong ones
(1.9-3.8 km) while COSTING accuracy on clear scenes (127->253 m,
13->179 m): the 512x384 segmentation grid is 5-7 mrad coarse, and our
synthetic water/terrain is far from its training distribution. Tally:
plain 8 good / 2 wrong, seg 5 good / 3 wrong. The E4n conclusion
stands -- overcast costs availability, honestly -- and the segmenter's
real roles stay diagnostic (labeling WHY a scene is inconclusive:
"overcast detected"; selecting sea spans for auto-level), not
replacement extraction. Revisit only with a fine-resolution model
validated on real maritime imagery (MaSTr1325 push).

Setup: needs the eWaSR clone + weights (see ewasr_bridge docstring);
paths via EWASR_DIR / EWASR_WEIGHTS env or the defaults below.
Run:   python3 e4r_cloudmask.py        (writes out/e4r_results.json)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S
import extract
from skyfix import (AZ, BETAS, basin_margin, fast_photo_cost,
                    observation)
from ewasr_bridge import EWasr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
CLD = os.path.join(OUT, 'clouds')
DIR3 = os.path.expanduser('~/.horizonator/DEMs_SRTM3')
SCRATCH = ('/tmp/claude-0/-home-user/792503f9-74c5-5111-83ca-'
           'eeeda63e838d/scratchpad')
EWASR_DIR = os.environ.get('EWASR_DIR', os.path.join(SCRATCH, 'eWaSR'))
EWASR_W = os.environ.get('EWASR_WEIGHTS',
                         os.path.join(SCRATCH, 'ewasr_resnet18.pth'))

F35 = 24
FOV = np.degrees(2 * np.arctan(0.8 * 21.63 / F35))
BOX, STEP0 = 5000.0, 250.0
CASES = [('strait1', 36.9500, 27.2500, 5.0, 180.0, 0.5, 0.0),
         ('strait2', 36.9622, 27.2384, 5.0, 25.0, -0.5, 0.5),
         ('offshore', 36.6050, 26.8590, 5.0, 60.0, 0.0, 0.0)]
VARIANTS = ['clean', 'stratus0.25', 'stratus0.5', 'stratus0.75',
            'cumulus4', 'cumulus10', 'cumulus20', 'haze0.5', 'haze0.75']
rng = np.random.default_rng(20260826)


def seg_skyline(seg):
    """Per column, the topmost obstacle/land pixel of the segmentation
    (cloud = sky class = transparent). Columns without land get zero
    weight; land touching the frame top is treated as clipped."""
    H, W = seg.shape
    obst = seg == 0
    has = obst.any(axis=0)
    rows = np.where(has, obst.argmax(axis=0), H - 1).astype(float)
    keep = has.astype(float)
    keep[rows <= 2] = 0.0
    return rows, keep


def rows_to_grid(rows, keep, shape, fov_deg, heading, roll_deg,
                 pitch_deg):
    """Map per-column boundary rows to (el, wt) on the global az grid,
    the same pinhole/attitude mapping observation() uses."""
    H, W = shape[:2]
    f = (W / 2) / np.tan(np.radians(fov_deg) / 2)
    u = np.arange(W) - (W - 1) / 2
    v = (H - 1) / 2 - rows
    cr, sr = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    ur = u * cr - v * sr
    vr = u * sr + v * cr
    az_rel = np.degrees(np.arctan2(ur, f))
    el_pt = np.arctan2(vr, np.hypot(ur, f)) + np.radians(pitch_deg)
    el = np.zeros(AZ.size)
    wt = np.zeros(AZ.size)
    rel = (AZ - heading + 180.0) % 360.0 - 180.0
    m = (rel >= az_rel.min()) & (rel <= az_rel.max())
    order = np.argsort(az_rel)
    el[m] = np.interp(rel[m], az_rel[order], el_pt[order])
    wt[m] = np.interp(rel[m], az_rel[order], keep[order]) > 0.5
    return el, wt.astype(float)


def run_case(case, variant, seg_model, cm, center_off):
    name, lat, lon, z, heading, pitch, roll = case
    path = os.path.join(CLD, f'{name}_{variant}.jpg')
    img = extract.load_image(path)
    el_obs, w, diag = observation(img, FOV, heading, roll, pitch)
    seg = seg_model.predict(img)
    rows_s, keep_s = seg_skyline(seg)
    el_seg, w_seg = rows_to_grid(rows_s, keep_s, img.shape, FOV,
                                 heading, roll, pitch)
    masked_frac = float(1 - (w_seg > 0).sum() / max((w > 0).sum(), 1))

    mlat, mlon = S.meters_per_degree(lat)
    lat_c = lat + center_off[0] / mlat
    lon_c = lon + center_off[1] / mlon
    shifts = np.arange(-60, 61, 2)
    g = np.arange(-BOX / 2, BOX / 2 + 1, STEP0)
    ccA = np.zeros((g.size, g.size))
    ccB = np.zeros((g.size, g.size))
    for i, dn in enumerate(g):
        for j, de in enumerate(g):
            el, _ = cm.skyline(lat_c + dn / mlat, lon_c + de / mlon,
                               z, AZ)
            ccA[i, j] = fast_photo_cost(el_obs, w, el, shifts, BETAS)[0]
            ccB[i, j] = fast_photo_cost(el_seg, w_seg, el, shifts,
                                        BETAS)[0]
    out = {}
    for tag, cc, eo, ww in (('plain', ccA, el_obs, w),
                            ('seg', ccB, el_seg, w_seg)):
        if (ww > 0).sum() < 30:
            out[tag] = dict(status='inconclusive', err_m=-1,
                            reasons=['observation fully masked'])
            continue
        i, j = np.unravel_index(np.argmin(cc), cc.shape)
        dn0, de0 = g[i], g[j]
        margin = basin_margin(cc, g, min_sep=4 * STEP0)
        boundary = max(abs(dn0), abs(de0)) >= BOX / 2 - STEP0
        best = (np.inf, dn0, de0)
        for step in (50.0, 12.5):
            best = (np.inf, dn0, de0)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    el, _ = cm.skyline(
                        lat_c + (dn0 + di * step) / mlat,
                        lon_c + (de0 + dj * step) / mlon, z, AZ)
                    c = fast_photo_cost(eo, ww, el, shifts, BETAS)[0]
                    if c < best[0]:
                        best = (c, dn0 + di * step, de0 + dj * step)
            _, dn0, de0 = best
        c0 = best[0]
        rms = float(np.sqrt(2 * c0) * 1e3)
        relief = float(np.std(eo[ww > 0]) * 1e3)
        reasons = []
        if margin < 0.15:
            reasons.append(f'margin {margin:.2f}')
        if boundary:
            reasons.append('boundary')
        if rms > 12.0:
            reasons.append(f'rms {rms:.1f}')
        if relief < 1.5:
            reasons.append(f'relief {relief:.1f}')
        err = float(np.hypot(dn0 - (-center_off[0]),
                             de0 - (-center_off[1])))
        out[tag] = dict(status='ok' if not reasons else 'inconclusive',
                        err_m=err, margin=float(margin), rms=rms,
                        reasons=reasons)
    out['masked_frac'] = masked_frac
    return out


if __name__ == '__main__':
    seg_model = EWasr(EWASR_DIR, EWASR_W)
    results = {}
    for case in CASES:
        name, lat, lon = case[0], case[1], case[2]
        cm = S.CMarcher(DIR3, (lat - .6, lat + .6), (lon - .8, lon + .8),
                        d_min=1000.)
        off = rng.uniform(-1200, 1200, 2)
        results[name] = {}
        for variant in VARIANTS:
            r = run_case(case, variant, seg_model, cm, off)
            results[name][variant] = r
            a, b = r['plain'], r['seg']
            print(f"{name:9s} {variant:12s} "
                  f"A {a['status'][:6]:6s} {a['err_m']:6.0f} m | "
                  f"B {b['status'][:6]:6s} {b['err_m']:6.0f} m "
                  f"(masked {100 * r['masked_frac']:3.0f}%)", flush=True)
    with open(os.path.join(OUT, 'e4r_results.json'), 'w') as f:
        json.dump(results, f, indent=1)

    def tally(tag):
        good = wrong = inc = 0
        for c in results.values():
            for v in c.values():
                s = v[tag]
                if s['status'] == 'ok' and 0 <= s['err_m'] < 500:
                    good += 1
                elif s['status'] == 'ok':
                    wrong += 1
                else:
                    inc += 1
        return good, wrong, inc
    ga, wa, ia = tally('plain')
    gb, wb, ib = tally('seg')
    print(f'\nA plain : {ga} good fixes, {ia} inconclusive, '
          f'{wa} WRONG accepted')
    print(f'B seg   : {gb} good fixes, {ib} inconclusive, '
          f'{wb} WRONG accepted')
