#!/usr/bin/env python3
"""E4f: the attitude-prior A/B — re-run the E4e audit over all 203 CH1
photos WITH attitude priors, holding everything else fixed.

Per photo: the reference attitude (heading, pitch offset) is derived by a
one-off orientation solve at the ground-truth position, then corrupted
with realistic instrument noise (sigma 1 deg heading, 0.5 deg pitch) to
form the prior. The position solve then searches only heading +-2 deg and
pitch +-1 deg around that prior — the instrumented regime — instead of
E4e's free rotation. Gates and verdict classes are identical, so the
FALSE-ACCEPT rates are directly comparable:

    E4e (attitude-free):  TRUE 29, FALSE 29, CAUGHT 134, OVER-CAUTIOUS 11
    E4f (with priors):    this run

Run:   python3 e4f_ab_audit.py     (CSV to out/e4f_audit.csv)
"""

import os
import sys
import glob
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import skyline as S
from skyfix import basin_margin
from e4e_gate_audit import ensure_tiles, CH1, DIR3, OUT, AZ, BOX

HEADING_SIGMA_DEG = 1.0
PITCH_SIGMA_DEG = 0.5
rng = np.random.default_rng(20260819)


def pcost(el_obs, el_syn, shifts, betas):
    best = np.inf
    for s in shifts:
        eo = np.roll(el_obs, s)
        r = el_syn - eo
        rb = np.abs(r[None, :] - betas[:, None])
        h = np.where(rb <= 3e-3, .5 * rb * rb, 3e-3 * (rb - 1.5e-3))
        c = h.mean(1).min()
        if c < best:
            best = c
    return best


def audit_photo(meta):
    v = open(meta).read().split('\n')
    f_px, lat_gt, lon_gt = float(v[0]), float(v[1]), float(v[2])
    W, H = int(v[4]), int(v[5])
    mask = np.asarray(Image.open(meta[:-8] + '-mask.png').convert('L')) > 127
    rows = np.where(mask.any(axis=0), mask.argmax(axis=0), H - 1).astype(float)
    u = np.arange(W) - (W - 1) / 2
    az_rel = np.degrees(np.arctan2(u, f_px))
    el_pt = np.arctan2((H - 1) / 2 - rows, np.hypot(u, f_px))
    el_obs = np.zeros(AZ.size)
    wt = np.zeros(AZ.size)
    m = (AZ >= az_rel.min()) & (AZ <= az_rel.max())
    el_obs[m] = np.interp(AZ[m], az_rel, el_pt)
    wt[m] = 1.0
    # restrict the cost to observed bins once (wt fixed for all shifts is
    # wrong under roll; here we shift el_obs and keep a full-width synth,
    # masking by the shifted weights)
    relief = float(np.std(el_obs[wt > 0]) * 1e3)

    ensure_tiles(lat_gt, lon_gt)
    cm = S.CMarcher(DIR3, (lat_gt - .7, lat_gt + .7),
                    (lon_gt - .9, lon_gt + .9), d_min=1000.)
    mlat, mlon = S.meters_per_degree(lat_gt)

    def z_at(la, lo):
        y = int(round((cm.lat_nw - la) / cm.dpp))
        x = int(round((lo - cm.lon_nw) / cm.dpp))
        return float(cm.mosaic[np.clip(y, 0, cm.mosaic.shape[0] - 1),
                               np.clip(x, 0, cm.mosaic.shape[1] - 1)]) + 2.

    def skyl(dn, de):
        la, lo = lat_gt + dn / mlat, lon_gt + de / mlon
        el, _ = cm.skyline(la, lo, z_at(la, lo), AZ)
        return el

    def masked_cost(el_syn, shifts, betas):
        best = np.inf
        for s in shifts:
            eo = np.roll(el_obs, s)
            ww = np.roll(wt, s)
            mm = ww > 0
            r = el_syn[mm] - eo[mm]
            rb = np.abs(r[None, :] - betas[:, None])
            h = np.where(rb <= 3e-3, .5 * rb * rb, 3e-3 * (rb - 1.5e-3))
            c = h.mean(1).min()
            if c < best:
                best = c
        return best

    # ---- reference attitude at the ground truth (full search, once)
    el_gt = skyl(0.0, 0.0)
    betas_full = np.arange(-0.100, 0.1001, 0.010)
    best = (np.inf, 0, 0.0)
    for s in range(-1800, 1800, 4):
        eo = np.roll(el_obs, s)
        ww = np.roll(wt, s)
        mm = ww > 0
        r = el_gt[mm] - eo[mm]
        rb = np.abs(r[None, :] - betas_full[:, None])
        h = np.where(rb <= 3e-3, .5 * rb * rb, 3e-3 * (rb - 1.5e-3))
        cvec = h.mean(1)
        i = int(np.argmin(cvec))
        if cvec[i] < best[0]:
            best = (float(cvec[i]), s, float(betas_full[i]))
    _, s_true, b_true = best

    # ---- the prior: true attitude + instrument noise
    s_c = s_true + int(round(rng.normal(0, HEADING_SIGMA_DEG) / 0.1))
    b_c = b_true + rng.normal(0, np.radians(PITCH_SIGMA_DEG))
    shifts_p = range(s_c - 20, s_c + 21, 2)               # +-2 deg
    betas_p = np.arange(b_c - 0.0175, b_c + 0.0176, 0.0035)  # +-1 deg

    # ---- position solve within the priors
    step0 = 250.0
    g = np.arange(-BOX / 2, BOX / 2 + 1, step0)
    cc = np.array([[masked_cost(skyl(dn, de), shifts_p, betas_p)
                    for de in g] for dn in g])
    i, j = np.unravel_index(np.argmin(cc), cc.shape)
    dn0, de0 = g[i], g[j]
    err = float(np.hypot(dn0, de0))
    margin = basin_margin(cc, g, min_sep=4 * step0)
    boundary = max(abs(dn0), abs(de0)) >= BOX / 2 - step0
    rms = float(np.sqrt(2 * cc[i, j]) * 1e3)
    ok = (margin >= 0.15) and (not boundary) and (rms <= 12.0) \
        and (relief >= 1.5)
    correct = err < 500
    verdict = ('TRUE-ACCEPT' if correct else 'FALSE-ACCEPT') if ok else \
              ('OVER-CAUTIOUS' if correct else 'CAUGHT')
    return err, margin, boundary, rms, relief, verdict


if __name__ == '__main__':
    metas = sorted(glob.glob(os.path.join(CH1, '*', '*.png.txt')))
    metas = [m for m in metas if os.path.exists(m[:-8] + '-mask.png')]
    if len(sys.argv) > 1:
        metas = metas[:int(sys.argv[1])]
    print(f'{len(metas)} photos, priors: heading +-2 deg around truth+'
          f'N(0,{HEADING_SIGMA_DEG}), pitch +-1 deg around truth+'
          f'N(0,{PITCH_SIGMA_DEG})')
    csv = open(os.path.join(OUT, 'e4f_audit.csv'), 'w', buffering=1)
    csv.write('photo,err_m,margin,boundary,rms_mrad,relief_mrad,verdict\n')
    counts = {}
    t0 = time.time()
    for n, meta in enumerate(metas):
        name = os.path.basename(meta)[:-8]
        try:
            err, margin, boundary, rms, relief, verdict = audit_photo(meta)
        except Exception as e:
            print(f'{name} ERROR: {e}', flush=True)
            continue
        counts[verdict] = counts.get(verdict, 0) + 1
        csv.write(f'{name},{err:.0f},{margin:.3f},{int(boundary)},'
                  f'{rms:.1f},{relief:.1f},{verdict}\n')
        print(f'[{n+1}/{len(metas)} {(time.time()-t0)/60:.0f}min] '
              f'{name}: err {err:6.0f} m  {verdict}', flush=True)
    print('\nconfusion:', counts)
    total_ok = counts.get('TRUE-ACCEPT', 0) + counts.get('FALSE-ACCEPT', 0)
    if total_ok:
        print(f'false-accept rate among accepted fixes: '
              f'{counts.get("FALSE-ACCEPT", 0)}/{total_ok}')
