#!/usr/bin/env python3
"""E4l: multi-hypothesis extraction, selected by basin margin.

The parallel study branch measured (CH1, oracle test) that the right
boundary is often IN a set of extraction hypotheses — an oracle picking
among four halves the error — but selection by solver rms captures none
of it, because rms is anti-predictive across extractions (a worse
boundary scores a lower residual). The imported rule: select on a
dimensionless quantity. Ours is the basin margin.

This runs the real extractor (not the curated masks) on all 203 CH1
photos with FOUR parameterizations (default / loose / strict / deep),
solves each hypothesis over the same 5 km attitude-prior landscape
(E4f regime: reference attitude from the mask observation at truth,
corrupted with instrument noise), and compares three selectors:

    default   always hypothesis 0
    margin    argmax basin margin (the dimensionless selector)
    oracle    argmin error (upper bound, truth-fed)

Run:   python3 e4l_multihyp.py [N]     (CSV to out/e4l_multihyp.csv)
"""

import os
import sys
import glob
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import skyline as S
import extract
from skyfix import basin_margin, fast_photo_cost
from e4e_gate_audit import ensure_tiles, CH1, DIR3, OUT, AZ, BOX

HYPS = [
    ('default', dict()),
    ('loose', dict(nsigma=2.5, m_sustain=5)),
    ('strict', dict(nsigma=6.0, m_sustain=12)),
    ('deep', dict(search_frac=0.95)),
]
HEADING_SIGMA_DEG = 1.0
PITCH_SIGMA_DEG = 0.5
rng = np.random.default_rng(20260821)


def to_grid(rows, meta_f, W, H):
    u = np.arange(W) - (W - 1) / 2
    az_rel = np.degrees(np.arctan2(u, meta_f))
    el_pt = np.arctan2((H - 1) / 2 - rows, np.hypot(u, meta_f))
    el = np.zeros(AZ.size)
    wt = np.zeros(AZ.size)
    m = (AZ >= az_rel.min()) & (AZ <= az_rel.max())
    el[m] = np.interp(AZ[m], az_rel, el_pt)
    wt[m] = 1.0
    return el, wt


def audit_photo(meta):
    v = open(meta).read().split('\n')
    f_px, lat_gt, lon_gt = float(v[0]), float(v[1]), float(v[2])
    W, H = int(v[4]), int(v[5])
    img = np.asarray(Image.open(meta[:-4]).convert('RGB'),
                     dtype=np.float32) / 255.0
    mask = np.asarray(Image.open(meta[:-8] + '-mask.png').convert('L')) > 127
    mrows = np.where(mask.any(axis=0), mask.argmax(axis=0),
                     H - 1).astype(float)
    el_ref, wt_ref = to_grid(mrows, f_px, W, H)

    obs = []
    for hname, kw in HYPS:
        rows, _ = extract.skyline_seam(img, **kw)
        obs.append((hname,) + to_grid(rows, f_px, W, H))

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

    # reference attitude at truth from the MASK observation (instrument
    # proxy), then the noisy prior all hypotheses share
    el_gt = skyl(0.0, 0.0)
    betas_full = np.arange(-0.100, 0.1001, 0.010)
    _, s_true, b_true = fast_photo_cost(el_ref, wt_ref, el_gt,
                                        range(-1800, 1800, 4), betas_full)
    s_c = s_true + int(round(rng.normal(0, HEADING_SIGMA_DEG) / 0.1))
    b_c = b_true + rng.normal(0, np.radians(PITCH_SIGMA_DEG))
    shifts = range(s_c - 20, s_c + 21, 2)
    betas = np.arange(b_c - 0.0175, b_c + 0.0176, 0.0035)

    step0 = 250.0
    g = np.arange(-BOX / 2, BOX / 2 + 1, step0)
    ccs = np.zeros((len(obs), g.size, g.size))
    for i, dn in enumerate(g):          # skyline synthesized ONCE per cell
        for j, de in enumerate(g):
            el = skyl(dn, de)
            for hh, (hname, eo, w) in enumerate(obs):
                ccs[hh, i, j] = fast_photo_cost(eo, w, el, shifts, betas)[0]
    out = []
    for hh, (hname, eo, w) in enumerate(obs):
        cc = ccs[hh]
        i, j = np.unravel_index(np.argmin(cc), cc.shape)
        out.append(dict(h=hname, dn=float(g[i]), de=float(g[j]),
                        err=float(np.hypot(g[i], g[j])),
                        margin=float(basin_margin(cc, g,
                                                  min_sep=4 * step0))))
    return out


if __name__ == '__main__':
    metas = sorted(glob.glob(os.path.join(CH1, '*', '*.png.txt')))
    metas = [m for m in metas if os.path.exists(m[:-8] + '-mask.png')]
    if len(sys.argv) > 1:
        metas = metas[:int(sys.argv[1])]
    print(f'{len(metas)} photos x {len(HYPS)} hypotheses')
    csv = open(os.path.join(OUT, 'e4l_multihyp.csv'), 'w', buffering=1)
    csv.write('photo,' + ','.join(f'err_{h},margin_{h},dn_{h},de_{h}'
                                  for h, _ in HYPS) + '\n')
    rows_all = []
    t0 = time.time()
    for n, meta in enumerate(metas):
        name = os.path.basename(meta)[:-8]
        try:
            res = audit_photo(meta)
        except Exception as e:
            print(f'{name} ERROR: {e}', flush=True)
            continue
        rows_all.append(res)
        csv.write(name + ',' + ','.join(
            f"{r['err']:.0f},{r['margin']:.3f},{r['dn']:.0f},{r['de']:.0f}"
            for r in res) + '\n')
        sel = max(res, key=lambda r: r['margin'])
        orc = min(res, key=lambda r: r['err'])
        print(f"[{n+1}/{len(metas)} {(time.time()-t0)/60:.0f}min] {name}: "
              f"default {res[0]['err']:5.0f}  margin-sel {sel['err']:5.0f} "
              f"({sel['h']})  oracle {orc['err']:5.0f} ({orc['h']})",
              flush=True)

    def stats(errs):
        e = np.array(errs)
        return f'median {np.median(e):5.0f} m, <1km {100*(e<1000).mean():.0f}%'
    d = [r[0]['err'] for r in rows_all]
    ms = [max(r, key=lambda x: x['margin'])['err'] for r in rows_all]
    oc = [min(r, key=lambda x: x['err'])['err'] for r in rows_all]
    print(f'\ndefault        : {stats(d)}')
    print(f'margin-selected: {stats(ms)}')
    print(f'oracle         : {stats(oc)}')
