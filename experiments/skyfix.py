#!/usr/bin/env python3
"""skyfix: estimate camera position from photograph skylines.

    python3 skyfix.py IMAGE [IMAGE2 ...] --center LAT,LON [options]

One photo gives one fix; several photos taken from the SAME position
(the recommended field procedure is a telephoto pan — several narrow-FOV
frames swept across the terrain, see study doc E4h) are fused into one
joint fix, each frame weighted by its angular resolution (a tele frame
is worth (fov_wide/fov_tele)^2 wide frames). Per-photo values for
--fov/--heading/--pitch/--roll are comma lists ('x' = unknown heading);
a single value broadcasts to all photos. The heading-shift search is
FFT-accelerated (exact robust cost at the optimum, ~100x faster on
full-circle searches).

The search box is centered on --center (e.g. a dead-reckoning position or,
for validation, the photo's GPS — used only to place the box). Camera FOV,
heading prior and altitude come from EXIF when present, overridable:

  --center LAT,LON   box center (required; or --center-exif)
  --box M            box size in meters (default 5000)
  --fov DEG          horizontal FOV (default: EXIF FocalLengthIn35mmFilm)
  --heading DEG      heading prior of the image center, true (default:
                     EXIF GPSImgDirection; else a full-circle search)
  --z M              camera height above sea level (default: EXIF
                     GPSAltitude, else 10)
  --dem DIR          .hgt directory (default ~/.horizonator/DEMs_SRTM3)
  --roll DEG         camera roll (default 0)
  --pitch DEG        camera pitch prior, positive up (default 0). Getting
                     this right to ~0.5 deg matters: the residual elevation
                     offset co-estimated per candidate is only +-10 mrad,
                     deliberately tight -- a wide offset window discards the
                     absolute-elevation information that pins range
  --auto-level       estimate pitch/roll from the visible sea horizon
                     (exact dip known from --z) instead of --pitch/--roll,
                     and tighten the elevation-offset window to +-2 mrad.
                     Falls back to the priors when no sea is in view
  --dmin M           near-field mask (default 1000 m). Needed for real
                     land-based observers, where the DEM smears the
                     observer's own bluff into blocking terrain; harmless
                     at sea. Reduce toward ~150 only if the camera's
                     near-field is genuinely open water
  --out PREFIX       write PREFIX.json and PREFIX.png diagnostics

Co-estimated per candidate: azimuth offset (+-6 deg around the prior, or
full circle if no prior) and a residual elevation offset (+-10 mrad).
Position: coarse-to-fine over the box. Output includes a
Laplace-approximation covariance.
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S
import extract

AZ = np.arange(-180.0, 180.0, 0.1) + 0.05
BETAS = np.arange(-0.010, 0.0101, 0.002)  # residual after the pitch prior
# residual after sea-horizon auto-levelling: the horizon line pins pitch
# to sub-mrad, so only DEM/refraction residue is left to absorb
BETAS_TIGHT = np.arange(-0.002, 0.00201, 0.0005)


def basin_margin(cc, g, min_sep):
    """Relative cost margin between the best and second-best coarse
    basins (non-max suppression at min_sep meters). Near zero = ambiguous.
    Returns inf when only one basin exists within the box."""
    order = np.argsort(cc, axis=None)
    kept = []
    for o in order:
        i, j = np.unravel_index(o, cc.shape)
        p = (g[i], g[j])
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= min_sep for q, _ in kept):
            kept.append((p, cc[i, j]))
        if len(kept) == 2:
            return float((kept[1][1] - kept[0][1]) / max(kept[0][1], 1e-12))
    return float('inf')


def observation(img, fov_deg, heading, roll_deg, pitch_deg=0.0):
    """Extract the skyline and map it to the global azimuth grid.
    Returns (el_obs, weights, diag) with diag holding extraction results."""
    rows, conf = extract.skyline_seam(img)
    H, W, _ = img.shape
    f = (W / 2) / np.tan(np.radians(fov_deg) / 2)
    u = np.arange(W) - (W - 1) / 2
    v = (H - 1) / 2 - rows
    cr, sr = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    ur = u * cr - v * sr
    vr = u * sr + v * cr
    az_rel = np.degrees(np.arctan2(ur, f))
    el_pt = np.arctan2(vr, np.hypot(ur, f)) + np.radians(pitch_deg)

    el = np.full(AZ.size, np.nan)
    wt = np.zeros(AZ.size)
    rel = (AZ - heading + 180.0) % 360.0 - 180.0
    m = (rel >= az_rel.min()) & (rel <= az_rel.max())
    order = np.argsort(az_rel)
    el[m] = np.interp(rel[m], az_rel[order], el_pt[order])
    wt[m] = np.interp(rel[m], az_rel[order], conf[order])
    wt = wt / (wt[m].max() + 1e-9)
    return (np.where(np.isfinite(el), el, 0.0), wt,
            dict(rows=rows, conf=conf, az_rel=az_rel, el_pt=el_pt))


def photo_cost(el_obs, w, el_syn, shifts, betas=BETAS):
    best = (np.inf, 0, 0.0)
    for s in shifts:
        eo = np.roll(el_obs, s)
        ww = np.roll(w, s)
        m = ww > 0
        r = el_syn[m] - eo[m]
        wm = ww[m]
        for b in betas:
            rb = np.abs(r - b)
            h = np.where(rb <= 3e-3, 0.5 * rb * rb, 3e-3 * (rb - 1.5e-3))
            c = float(np.sum(h * wm) / np.sum(wm))
            if c < best[0]:
                best = (c, s, b)
    return best


def fast_photo_cost(el_obs, w, el_syn, shifts, betas=BETAS, topk=12):
    """photo_cost with the heading-shift search FFT-accelerated.

    The weighted QUADRATIC cost with its offset beta minimized in closed
    form is, for every circular lag s at once, three cross-correlations:

        S1(s) = (w * el_syn)[s] - sum(w el_obs)          (mean residual)
        S2(s) = (w * el_syn^2)[s] - 2 (w el_obs * el_syn)[s]
                + sum(w el_obs^2)
        C2(s) = (S2 - S1^2/W0) / W0                      (variance)

    computed with FFTs in O(n log n) instead of O(n * |shifts|). The
    quadratic landscape preselects the topk candidate lags (plus their
    immediate neighbours), and the exact Huber cost + beta grid runs
    only on those — so the returned optimum is the same robust cost as
    photo_cost, ~10-100x faster on wide shift ranges. Falls back to the
    exhaustive search when the shift set is already tiny."""
    lags = np.asarray(list(shifts), dtype=int)
    if lags.size <= 3 * topk:
        return photo_cost(el_obs, w, el_syn, lags, betas)
    n = el_obs.size
    corr = lambda a, b: np.fft.irfft(np.conj(np.fft.rfft(a))
                                     * np.fft.rfft(b), n)
    A = corr(w, el_syn)
    B = corr(w * el_obs, el_syn)
    C = corr(w, el_syn ** 2)
    W0 = w.sum()
    S1 = A - (w * el_obs).sum()
    S2 = C - 2 * B + (w * el_obs ** 2).sum()
    C2 = (S2 - S1 ** 2 / W0) / W0
    li = lags % n
    top = lags[np.argsort(C2[li])[:topk]]
    cand = np.unique(np.concatenate([top - 1, top, top + 1]))
    cand = cand[np.isin(cand, lags) | np.isin(cand % n, li)]
    return photo_cost(el_obs, w, el_syn, cand, betas)


def per_photo_vals(spec, n, default=None):
    """Parse a per-photo CLI value: a single value broadcasts to all
    photos, a comma list gives one per photo, 'x' means unknown."""
    if spec is None:
        return [default] * n
    parts = [p.strip() for p in str(spec).split(',')]
    if len(parts) == 1:
        parts = parts * n
    if len(parts) != n:
        sys.exit(f'expected 1 or {n} comma-separated values: {spec}')
    return [default if p in ('', 'x', 'X') else float(p) for p in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+',
                    help='one or more photos taken from the SAME position '
                         '(e.g. a telephoto pan); each frame contributes '
                         'to one joint fix, weighted by its angular '
                         'resolution')
    ap.add_argument('--center')
    ap.add_argument('--center-exif', action='store_true')
    ap.add_argument('--box', type=float, default=5000.0)
    ap.add_argument('--fov', help='deg; single value or comma list')
    ap.add_argument('--heading', help='deg true; single value, comma '
                                      "list, or 'x' for unknown")
    ap.add_argument('--z', type=float)
    ap.add_argument('--roll', help='deg; single value or comma list')
    ap.add_argument('--pitch', help='deg, positive up; single value or '
                                    'comma list. Accuracy to ~0.5 deg '
                                    'matters: the residual elevation '
                                    'offset window is only +-10 mrad')
    ap.add_argument('--auto-level', action='store_true',
                    help='estimate pitch and roll from the visible sea '
                         'horizon (per photo), overriding --pitch/--roll '
                         'and tightening the elevation-offset window to '
                         '+-2 mrad; falls back to the priors when no '
                         'adequate sea segment is visible')
    ap.add_argument('--dem', default=os.path.expanduser(
        os.environ.get('HORIZONATOR_DEMS', '~/.horizonator/DEMs_SRTM3')))
    ap.add_argument('--dmin', type=float, default=1000.0)
    ap.add_argument('--min-margin', type=float, default=0.15,
                    help='inconclusive when the second-best coarse basin '
                         'is within this relative cost margin of the best '
                         '(degenerate/ambiguous landscape). E3: genuine '
                         'fixes showed margins >= 0.29')
    ap.add_argument('--max-rms', type=float, default=12.0,
                    help='inconclusive when the best residual exceeds this '
                         '(mrad): the DEM cannot explain the observation '
                         '(clouds, wrong area, extraction failure)')
    ap.add_argument('--min-relief', type=float, default=1.5,
                    help='inconclusive when no photo shows at least this '
                         'much skyline relief (mrad std): too little '
                         'to localize (e.g. open sea horizon only)')
    ap.add_argument('--pitch-sigma', type=float, default=0.29,
                    help='1-sigma pitch-prior uncertainty (deg); the '
                         'elevation-offset window is +-2 sigma (default '
                         'matches the historical +-10 mrad). Braced IMU '
                         '~0.3; an UNCALIBRATED AR/theodolite app needs '
                         '~1.5 -- a field-measured +1.5 deg platform-'
                         'dependent pitch bias in such an app (parallel '
                         'study branch) would silently poison the '
                         'default window. Sea-horizon auto-levelling '
                         'supersedes this when it succeeds')
    ap.add_argument('--px-err', type=float, default=1.5,
                    help='assumed skyline-extraction error in pixels '
                         '(at the 1600 px working width); raise it for '
                         'coarse/degraded input so narrow-FOV frames '
                         'get their full weight advantage')
    ap.add_argument('--sigma-dem', type=float, default=1.5,
                    help='FOV-independent DEM/model residual (mrad) '
                         'flooring the per-photo weights')
    ap.add_argument('--out',
                    help='write PREFIX.json and PREFIX.png diagnostics')
    args = ap.parse_args()
    N = len(args.images)

    exs = [extract.read_exif(p) for p in args.images]
    fovs = per_photo_vals(args.fov, N)
    fovs = [f if f is not None else exs[i].get('fov_deg')
            for i, f in enumerate(fovs)]
    if any(f is None for f in fovs):
        sys.exit('no FOV for some photo: none in EXIF, pass --fov')
    headings = per_photo_vals(args.heading, N)
    headings = [h if h is not None else exs[i].get('heading_deg')
                for i, h in enumerate(headings)]
    pitches = per_photo_vals(args.pitch, N, 0.0)
    rolls = per_photo_vals(args.roll, N, 0.0)
    z = args.z if args.z is not None else \
        next((e['alt_m'] for e in exs if 'alt_m' in e), 10.0)
    if args.center:
        lat_c, lon_c = (float(x) for x in args.center.split(','))
    elif args.center_exif and any('lat' in e for e in exs):
        e = next(e for e in exs if 'lat' in e)
        lat_c, lon_c = e['lat'], e['lon']
    else:
        sys.exit('no box center: pass --center LAT,LON or --center-exif')

    # precision weight ~ 1/sigma^2 per photo, sigma^2 = extraction + DEM:
    # extraction error is ~constant in pixels (--px-err), so its angular
    # part scales with FOV/width; the DEM/model residual (--sigma-dem,
    # mrad) is FOV-independent and floors the weight. A telephoto frame
    # therefore dominates only when extraction noise dominates (coarse
    # input, hand digitization -- study doc E4h); with a clean full-res
    # extractor the DEM floor makes weights near-equal and azimuth
    # coverage decides (E4i)
    sig2 = [(args.px_err * np.radians(f) / 1600.0) ** 2
            + (args.sigma_dem * 1e-3) ** 2 for f in fovs]
    uw = [min(sig2) / s2 for s2 in sig2]

    half = max(0.010, 2.0 * np.radians(args.pitch_sigma))
    betas_prior = np.arange(-half, half * 1.001, half / 5)

    photos = []
    for i, path in enumerate(args.images):
        img = extract.load_image(path)
        betas = betas_prior
        level = None
        pitch, roll = pitches[i], rolls[i]
        if args.auto_level:
            rows, conf = extract.skyline_seam(img)
            f_px = (img.shape[1] / 2) / np.tan(np.radians(fovs[i]) / 2)
            level = extract.sea_horizon_attitude(
                rows, conf, img.shape, f_px,
                S.horizon_dip_rad(max(z, 0.5)), rgb=img)
            if level:
                pitch, roll = level['pitch_deg'], level['roll_deg']
                betas = BETAS_TIGHT
        el_obs, w, diag = observation(
            img, fovs[i], headings[i] if headings[i] is not None else 0.0,
            roll, pitch)
        shifts = np.arange(-60, 61, 2) if headings[i] is not None \
            else np.arange(-1800, 1800, 2)
        relief = float(np.std(el_obs[w > 0]) * 1e3)
        photos.append(dict(path=path, img=img, el_obs=el_obs, w=w,
                           diag=diag, fov=fovs[i], heading=headings[i],
                           pitch=pitch, roll=roll, betas=betas,
                           shifts=shifts, weight=uw[i], relief=relief,
                           level=level))
        print(f'photo {i}: {os.path.basename(path)}  fov {fovs[i]:.1f}, '
              f'heading '
              f'{"none (full search)" if headings[i] is None else headings[i]}'
              f', pitch {pitch:+.2f}, roll {roll:+.2f}, weight {uw[i]:.2f}'
              + (f', auto-level ({level["n_inl"]} sea cols)' if level
                 else ''), flush=True)
    print(f'z {z:.1f} m, box {args.box:.0f} m at ({lat_c:.5f},{lon_c:.5f})')

    mlat, mlon = S.meters_per_degree(lat_c)
    cm = S.CMarcher(args.dem, (lat_c - 0.6, lat_c + 0.6),
                    (lon_c - 0.8, lon_c + 0.8), d_min=args.dmin)
    usum = sum(uw)

    def C(dn, de):
        el, _ = cm.skyline(lat_c + dn / mlat, lon_c + de / mlon, z, AZ)
        return sum(p['weight'] * fast_photo_cost(
            p['el_obs'], p['w'], el, p['shifts'], p['betas'])[0]
            for p in photos) / usum

    step0 = max(args.box / 20, 100.0)
    g = np.arange(-args.box / 2, args.box / 2 + 1, step0)
    cc = np.array([[C(dn, de) for de in g] for dn in g])
    i, j = np.unravel_index(np.argmin(cc), cc.shape)
    dn0, de0 = g[i], g[j]

    # ---- inconclusiveness checks: a solve that converged somewhere is
    # not automatically a fix. Each failed check adds a reason; any reason
    # makes the result INCONCLUSIVE (status + exit code 2)
    reasons = []
    margin = basin_margin(cc, g, min_sep=4 * step0)
    if margin < args.min_margin:
        reasons.append(f'ambiguous landscape: basin margin {margin:.2f} '
                       f'< {args.min_margin:.2f}')
    if max(abs(dn0), abs(de0)) >= args.box / 2 - step0:
        reasons.append('minimum on the search-box boundary: the true '
                       'position may lie outside the box')
    relief = max(p['relief'] for p in photos)
    if relief < args.min_relief:
        reasons.append(f'insufficient skyline relief: {relief:.1f} mrad '
                       f'std < {args.min_relief:.1f}')
    for step in (step0 / 5, step0 / 20):
        best = (np.inf, dn0, de0)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                c = C(dn0 + di * step, de0 + dj * step)
                if c < best[0]:
                    best = (c, dn0 + di * step, de0 + dj * step)
        _, dn0, de0 = best

    # Laplace covariance from a local quadratic fit of the cost
    h = step0 / 10
    c0 = C(dn0, de0)
    cnn = (C(dn0 + h, de0) - 2 * c0 + C(dn0 - h, de0)) / h ** 2
    cee = (C(dn0, de0 + h) - 2 * c0 + C(dn0, de0 - h)) / h ** 2
    cne = (C(dn0 + h, de0 + h) - C(dn0 + h, de0 - h)
           - C(dn0 - h, de0 + h) + C(dn0 - h, de0 - h)) / (4 * h ** 2)
    Hm = np.array([[cnn, cne], [cne, cee]])
    try:
        cov = 2 * c0 * np.linalg.inv(Hm)  # scaled: residual-level heuristic
        sig = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        sig = [np.nan, np.nan]

    lat_e = lat_c + dn0 / mlat
    lon_e = lon_c + de0 / mlon
    el_syn, _ = cm.skyline(lat_e, lon_e, z, AZ)
    pj = []
    for p in photos:
        cb, sb, bb = fast_photo_cost(p['el_obs'], p['w'], el_syn,
                                     p['shifts'], p['betas'])
        pj.append(dict(file=os.path.basename(p['path']), fov_deg=p['fov'],
                       heading_deg=p['heading'], pitch_deg=p['pitch'],
                       roll_deg=p['roll'], weight=p['weight'],
                       rms_mrad=float(np.sqrt(2 * cb) * 1e3),
                       heading_offset_deg=sb * 0.1,
                       el_offset_mrad=bb * 1e3,
                       attitude_source=('sea-horizon' if p['level']
                                        else 'prior'),
                       best_shift=sb, best_beta=bb))
    rms_mrad = float(np.sqrt(2 * c0) * 1e3)
    if rms_mrad > args.max_rms:
        reasons.append(f'residual {rms_mrad:.1f} mrad > {args.max_rms:.1f}: '
                       'the DEM cannot explain this observation')
    status = 'ok' if not reasons else 'inconclusive'
    for r in reasons:
        print('INCONCLUSIVE:', r, file=sys.stderr)
    result = dict(status=status, reasons=reasons,
                  lat=lat_e, lon=lon_e,
                  fix_ok=(status == 'ok'), basin_margin=float(margin),
                  relief_mrad=relief,
                  dn_m=dn0, de_m=de0,
                  sigma_n_m=float(sig[0]), sigma_e_m=float(sig[1]),
                  cost=c0, rms_mrad=rms_mrad,
                  n_photos=N, z_m=z,
                  photos=[{k: v for k, v in d.items()
                           if k not in ('best_shift', 'best_beta')}
                          for d in pj])
    if N == 1:
        result.update(heading_offset_deg=pj[0]['heading_offset_deg'],
                      el_offset_mrad=pj[0]['el_offset_mrad'],
                      fov_deg=pj[0]['fov_deg'])
    print(json.dumps(result, indent=1))

    if args.out:
        with open(args.out + '.json', 'w') as fjs:
            json.dump(result, fjs, indent=1)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows_n = N + (1 if N == 1 else 0)
        fig, axs = plt.subplots(rows_n, 1,
                                figsize=(11, 3.5 * rows_n), squeeze=False)
        axs = axs[:, 0]
        k = 0
        if N == 1:
            ax = axs[0]
            ax.imshow(photos[0]['img'])
            ax.plot(np.arange(photos[0]['img'].shape[1]),
                    photos[0]['diag']['rows'], color='#d55e00', lw=1.2)
            ax.set_title('extracted skyline', fontsize=10)
            ax.axis('off')
            k = 1
        for p, d in zip(photos, pj):
            ax = axs[k]
            k += 1
            eo = np.roll(p['el_obs'], d['best_shift']) + d['best_beta']
            mm = np.roll(p['w'], d['best_shift']) > 0
            ax.plot(AZ[mm], eo[mm] * 1e3, color='#111111', lw=1.8,
                    label='observed')
            ax.plot(AZ[mm], el_syn[mm] * 1e3, color='#0072b2', lw=1.2,
                    label='predicted at fix')
            ax.set_xlabel('azimuth (deg true)')
            ax.set_ylabel('elevation (mrad)')
            ax.set_title(f"{d['file']}  (fov {d['fov_deg']:.1f}, weight "
                         f"{d['weight']:.2f}, rms {d['rms_mrad']:.1f} mrad)",
                         fontsize=9)
            ax.legend(frameon=False, fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
        fig.tight_layout()
        fig.savefig(args.out + '.png', dpi=110)
        print('wrote', args.out + '.json/.png')
    return 0 if status == 'ok' else 2


if __name__ == '__main__':
    sys.exit(main())
