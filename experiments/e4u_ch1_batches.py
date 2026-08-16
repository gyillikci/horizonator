#!/usr/bin/env python3
"""E4u: CH1 full-set test run, reported in batches of 20 photos.

The current-best single-photo regime (E4f instrumented: attitude priors
sigma 1 deg heading / 0.5 deg pitch around a per-photo reference
attitude, restricted solve +-2 deg heading / +-1 deg pitch, 5 km coarse
box at 250 m, four trust gates) run over all 203 CH1 photos with the
FFT-accelerated solver, emitting a summary block after every 20 photos:

    batch #, n, median err, hits <500 m, verdict counts
    (TRUE/FALSE-ACCEPT, CAUGHT, OVER-CAUTIOUS), median accepted margin

plus a final all-set confusion and the standalone/act-on-it tier counts
(E4k margins 0.7 / 1.5). Same rng seed as E4f (20260819) so the per-
photo priors — and therefore the rows — are comparable with
out/e4f_audit.csv.

--best switches to the current-best configuration: full-resolution
SRTM1 tiles (E4p) and the soft near-field (skyfix --dmin-soft 1000
ramp with the C0_NOINFO coverage charge) instead of the hard 1 km
clip; same priors (same seed), outputs to *_best files, for a full-set
A/B against the E4f-comparable baseline.

Run:   python3 e4u_ch1_batches.py [n] [--best]
       (CSV to out/e4u_ch1[_best].csv, batches to
       out/e4u_batches[_best].json)
"""

import os
import sys
import glob
import json
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import skyline as S
from skyfix import basin_margin, fast_photo_cost
from e4e_gate_audit import ensure_tiles, CH1, DIR3, OUT, AZ, BOX

HEADING_SIGMA_DEG = 1.0
PITCH_SIGMA_DEG = 0.5
BATCH = 20
DIR1 = os.path.expanduser('~/.horizonator/DEMs_SRTM1')
DMIN_SOFT = 1000.0
BEST = '--best' in sys.argv          # SRTM1 + soft near-field (sea config)
SRTM1 = '--srtm1' in sys.argv or BEST  # full-res DEM, hard 1 km clip
rng = np.random.default_rng(20260819)


def ensure_tiles1(lat, lon, margin_lat=0.7, margin_lon=0.9):
    import math
    import gzip
    import urllib.request
    os.makedirs(DIR1, exist_ok=True)
    for la in range(math.floor(lat - margin_lat),
                    math.floor(lat + margin_lat) + 1):
        for lo in range(math.floor(lon - margin_lon),
                        math.floor(lon + margin_lon) + 1):
            t = f"N{la:02d}E{lo:03d}"
            p1 = os.path.join(DIR1, t + '.hgt')
            if not os.path.exists(p1):
                url = ('https://s3.amazonaws.com/elevation-tiles-prod/'
                       f'skadi/{t[:3]}/{t}.hgt.gz')
                print('  fetching', t, flush=True)
                with urllib.request.urlopen(url) as r:
                    open(p1, 'wb').write(gzip.decompress(r.read()))


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
    relief = float(np.std(el_obs[wt > 0]) * 1e3)

    ensure_tiles(lat_gt, lon_gt)
    if SRTM1:
        ensure_tiles1(lat_gt, lon_gt)
        cm = S.CMarcher(DIR1, (lat_gt - .7, lat_gt + .7),
                        (lon_gt - .9, lon_gt + .9),
                        d_min=150. if BEST else 1000.)
    else:
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
        el, r = cm.skyline(la, lo, z_at(la, lo), AZ)
        if BEST:
            ws = np.clip((r - 300.0) / (DMIN_SOFT - 300.0), 0.0, 1.0)
            return el, ws
        return el, None

    # reference attitude at the ground truth (FFT full search, once);
    # the reference stays unweighted so the priors match the baseline
    # run draw-for-draw
    el_gt, _ = skyl(0.0, 0.0)
    betas_full = np.arange(-0.100, 0.1001, 0.010)
    _, s_true, b_true = fast_photo_cost(el_obs, wt, el_gt,
                                        range(-1800, 1800, 4), betas_full)

    # the prior: true attitude + instrument noise
    s_c = s_true + int(round(rng.normal(0, HEADING_SIGMA_DEG) / 0.1))
    b_c = b_true + rng.normal(0, np.radians(PITCH_SIGMA_DEG))
    shifts_p = range(s_c - 20, s_c + 21, 2)                  # +-2 deg
    betas_p = np.arange(b_c - 0.0175, b_c + 0.0176, 0.0035)  # +-1 deg

    # position solve within the priors
    step0 = 250.0
    g = np.arange(-BOX / 2, BOX / 2 + 1, step0)
    def cost_at(dn, de):
        el, ws = skyl(dn, de)
        return fast_photo_cost(el_obs, wt, el, shifts_p, betas_p,
                               w_syn=ws)[0]

    cc = np.array([[cost_at(dn, de) for de in g] for dn in g])
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
    return dict(err=err, margin=margin, boundary=bool(boundary), rms=rms,
                relief=relief, verdict=verdict)


def batch_summary(rows, k):
    errs = np.array([r['err'] for r in rows])
    verd = [r['verdict'] for r in rows]
    acc = [r for r in rows if r['verdict'].endswith('ACCEPT')]
    accm = [r['margin'] for r in acc if np.isfinite(r['margin'])]
    s = dict(batch=k, n=len(rows),
             photos=[r['photo'] for r in rows],
             median_err_m=float(np.median(errs)),
             hits_500m=int((errs < 500).sum()),
             true_accept=verd.count('TRUE-ACCEPT'),
             false_accept=verd.count('FALSE-ACCEPT'),
             caught=verd.count('CAUGHT'),
             over_cautious=verd.count('OVER-CAUTIOUS'),
             median_accept_margin=(float(np.median(accm)) if accm
                                   else None),
             standalone=sum(1 for r in acc if r['margin'] >= 0.7),
             act_on_it=sum(1 for r in acc if r['margin'] >= 1.5))
    print(f"\n===== BATCH {k:2d}  (photos {rows[0]['photo']} .. "
          f"{rows[-1]['photo']}, n={len(rows)}) =====")
    print(f"  median err {s['median_err_m']:7.0f} m   "
          f"<500 m: {s['hits_500m']}/{len(rows)}")
    print(f"  verdicts: TA {s['true_accept']}  FA {s['false_accept']}  "
          f"CAUGHT {s['caught']}  OC {s['over_cautious']}")
    print(f"  accepted: {len(acc)}  median margin "
          f"{s['median_accept_margin'] if s['median_accept_margin'] is not None else float('nan'):.2f}  "
          f"standalone(>=0.7): {s['standalone']}  "
          f"act-on-it(>=1.5): {s['act_on_it']}", flush=True)
    return s


if __name__ == '__main__':
    metas = sorted(glob.glob(os.path.join(CH1, '*', '*.png.txt')))
    metas = [m for m in metas if os.path.exists(m[:-8] + '-mask.png')]
    nargs = [a for a in sys.argv[1:] if not a.startswith('--')]
    if nargs:
        metas = metas[:int(nargs[0])]
    tag = '_best' if BEST else ('_srtm1' if SRTM1 else '')
    print(f'{len(metas)} photos, batches of {BATCH}, instrumented regime '
          f'(priors N(0,{HEADING_SIGMA_DEG}) deg heading / '
          f'N(0,{PITCH_SIGMA_DEG}) deg pitch)'
          + (', BEST config: SRTM1 + soft near-field' if BEST else
             (', SRTM1 full resolution' if SRTM1 else '')),
          flush=True)
    csv = open(os.path.join(OUT, f'e4u_ch1{tag}.csv'), 'w', buffering=1)
    csv.write('photo,err_m,margin,boundary,rms_mrad,relief_mrad,verdict\n')
    rows_all, batch_rows, summaries = [], [], []
    t0 = time.time()
    for n, meta in enumerate(metas):
        name = os.path.basename(meta)[:-8]
        try:
            r = audit_photo(meta)
        except Exception as e:
            print(f'{name} ERROR: {e}', flush=True)
            continue
        r['photo'] = name
        rows_all.append(r)
        batch_rows.append(r)
        csv.write(f"{name},{r['err']:.0f},{r['margin']:.3f},"
                  f"{int(r['boundary'])},{r['rms']:.1f},{r['relief']:.1f},"
                  f"{r['verdict']}\n")
        print(f"[{n+1}/{len(metas)} {(time.time()-t0)/60:.0f}min] "
              f"{name}: err {r['err']:6.0f} m  {r['verdict']}", flush=True)
        if len(batch_rows) == BATCH:
            summaries.append(batch_summary(batch_rows, len(summaries) + 1))
            batch_rows = []
    if batch_rows:
        summaries.append(batch_summary(batch_rows, len(summaries) + 1))

    errs = np.array([r['err'] for r in rows_all])
    verd = [r['verdict'] for r in rows_all]
    counts = {v: verd.count(v) for v in set(verd)}
    total_ok = counts.get('TRUE-ACCEPT', 0) + counts.get('FALSE-ACCEPT', 0)
    print('\n===== TOTAL =====')
    print(f'  n {len(rows_all)}   median err {np.median(errs):.0f} m   '
          f'<500 m: {(errs < 500).sum()}/{len(rows_all)}')
    print(f'  confusion: {counts}')
    if total_ok:
        print(f'  false-accept rate among accepted: '
              f'{counts.get("FALSE-ACCEPT", 0)}/{total_ok}')
    with open(os.path.join(OUT, f'e4u_batches{tag}.json'), 'w') as f:
        json.dump(dict(batches=summaries, confusion=counts,
                       n=len(rows_all),
                       median_err_m=float(np.median(errs)),
                       hits_500m=int((errs < 500).sum())), f, indent=1)
