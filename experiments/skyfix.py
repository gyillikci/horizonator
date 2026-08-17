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


EXTRACTOR = 'seam'      # set from --extractor


def extract_boundary(img):
    """The image-side boundary, from whichever front end is selected.

    'seam'    the mountain seam detector (extract.skyline_seam)
    'learned' E4m's patch template + dynamic-programming seam, trained
              on CH1's even half — the approach Ahmad's skyline work
              takes. E4y found the two do not merely differ in
              accuracy: on a layered coastal scene they lock onto
              DIFFERENT LAYERS (86 px apart on OREJ1026, the seam
              detector following the far coast and the template the
              island), while on a clean single-layer scene they agree
              to 2 px."""
    if EXTRACTOR == 'learned':
        from e4m_diverse import seam_extract
        w = np.load(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'out', 'e4m_svm.npz'))['w']
        rows = seam_extract(img, w)
        return rows, np.ones(img.shape[1])
    return extract.skyline_seam(img)


def observation(img, fov_deg, heading, roll_deg, pitch_deg=0.0):
    """Extract the skyline and map it to the global azimuth grid.
    Returns (el_obs, weights, diag) with diag holding extraction results."""
    rows, conf = extract_boundary(img)
    H, W, _ = img.shape
    f = (W / 2) / np.tan(np.radians(fov_deg) / 2)
    u = np.arange(W) - (W - 1) / 2
    v = (H - 1) / 2 - rows
    cr, sr = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    ur = u * cr - v * sr
    vr = u * sr + v * cr
    az_rel = np.degrees(np.arctan2(ur, f))
    el_pt = np.arctan2(vr, np.hypot(ur, f)) + np.radians(pitch_deg)

    # ---- drop masts, poles and rigging before they reach the cost.
    # A ship's mast or a lamp post is a one- or two-column spike tens
    # of mrad tall that no DEM carries, and the robust cost only
    # softens it — it still drags the fit. Terrain silhouettes have
    # bounded slope in angle; anything far outside the local trend
    # over a few columns is man-made and its columns are dropped.
    if el_pt.size > 32:
        k = max(5, el_pt.size // 200) | 1
        pad = np.pad(el_pt, k // 2, mode='edge')
        med = np.median(np.lib.stride_tricks.sliding_window_view(pad, k),
                        axis=-1)
        dev = el_pt - med
        scale = 1.4826 * np.median(np.abs(dev)) + 1e-6
        spike = np.abs(dev) > max(6.0 * scale, 3e-3)
        if spike.any() and spike.mean() < 0.25:
            conf = conf * (~spike)

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


C0_NOINFO = 1.35e-5     # Huber cost of a 6 mrad residual: what a
                        # suppressed bin is charged, so a candidate
                        # cannot LOWER its cost by hiding terrain behind
                        # its own near field (costs stay comparable
                        # across candidates with different suppression)


def photo_cost(el_obs, w, el_syn, shifts, betas=BETAS, w_syn=None):
    """w_syn: optional per-azimuth synthesis-side weight (NOT rolled with
    the observation) — e.g. the soft near-field ramp for land observers.
    Suppressed weight is charged C0_NOINFO per unit instead of vanishing
    from the average."""
    best = (np.inf, 0, 0.0)
    for s in shifts:
        eo = np.roll(el_obs, s)
        w0 = np.roll(w, s)
        W_obs = w0.sum()
        ww = w0 * w_syn if w_syn is not None else w0
        m = ww > 0
        r = el_syn[m] - eo[m]
        wm = ww[m]
        Weff = wm.sum()
        if Weff < 1e-9:
            continue
        for b in betas:
            rb = np.abs(r - b)
            h = np.where(rb <= 3e-3, 0.5 * rb * rb, 3e-3 * (rb - 1.5e-3))
            c = float((np.sum(h * wm) + C0_NOINFO * (W_obs - Weff))
                      / W_obs)
            if c < best[0]:
                best = (c, s, b)
    return best


def photo_cost_curve(el_obs, w, el_syn, shifts, betas=BETAS):
    """Cost per shift (min over betas) — for RIGID-PAN fusion, where all
    frames of a pan share ONE compass offset (their relative headings
    are gyro-accurate), so the joint cost must be minimized over a
    single shared shift instead of per-photo independent ones."""
    lags = np.asarray(list(shifts), dtype=int)
    out = np.full(lags.size, np.inf)
    for k, s in enumerate(lags):
        eo = np.roll(el_obs, s)
        ww = np.roll(w, s)
        m = ww > 0
        if not m.any():
            continue
        r = el_syn[m] - eo[m]
        wm = ww[m]
        rb = np.abs(r[None, :] - np.asarray(betas)[:, None])
        h = np.where(rb <= 3e-3, 0.5 * rb * rb, 3e-3 * (rb - 1.5e-3))
        out[k] = float(((h * wm[None, :]).sum(1) / wm.sum()).min())
    return out


def fast_photo_cost(el_obs, w, el_syn, shifts, betas=BETAS, topk=12,
                    w_syn=None):
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
        return photo_cost(el_obs, w, el_syn, lags, betas, w_syn)
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
    # the quadratic preselect ignores w_syn (its correlations would need
    # six FFTs); the exact pass applies it, so wide-search results with a
    # soft mask are near-optimal rather than exact
    return photo_cost(el_obs, w, el_syn, cand, betas, w_syn)


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
    ap.add_argument('--dt-air-sea', type=float,
                    help='air minus sea temperature (deg C) for the '
                         'sea-horizon dip anomaly correction: refraction '
                         'near the horizon is driven by this gradient '
                         '(van der Werf 2016), and warm air over cold '
                         'water raises the apparent horizon. Applies the '
                         "navigator's rule of thumb (~0.11 arcmin/degC, "
                         'PROVISIONAL until field-calibrated) to the dip '
                         'used by --auto-level, and widens its offset '
                         'window by 0.15 mrad/degC of |dT| since large '
                         'gradients also mean unstable dip')
    ap.add_argument('--level-detector', default='radon',
                    choices=['radon', 'seam'],
                    help='front end for --auto-level: radon (default) '
                         'searches directly for the long straight '
                         'low-contrast horizon line; seam reuses the '
                         'mountain skyline detector. E4q measured radon '
                         'at 2x the availability and 2.4x the accuracy '
                         'on real maritime imagery (7.6 vs 18.6 mrad '
                         'median edge error, 70% vs 22% within 10 mrad)')
    ap.add_argument('--extractor', default='seam',
                    choices=['seam', 'learned'],
                    help='image-side boundary finder: the mountain seam '
                         'detector, or E4m\'s learned patch template '
                         'with a DP seam')
    ap.add_argument('--heading-window', type=float, default=6.0,
                    help='half-width of the co-estimated azimuth search '
                         'around the heading prior, degrees (default 6; '
                         'set it from the compass accuracy the device '
                         'reports)')
    ap.add_argument('--min-range', type=float, default=None,
                    help='reject the sighting when the terrain forming '
                         'the silhouette is nearer than this many '
                         'meters (3000 is the measured knee, E4x): too '
                         'close and the silhouette is canopy and '
                         'rooftops the DEM does not carry')
    ap.add_argument('--conditioning', action='store_true',
                    help='resolve with parts of the evidence removed '
                         '(leave-one-photo-out, or left/right halves of '
                         'a single frame) and report how far the answer '
                         'moves — the stability test E4v showed is '
                         'needed, since neither the basin margin nor the '
                         'covariance separates a good fix from a bad one')
    ap.add_argument('--max-jackknife', type=float, default=None,
                    help='reject a fix whose conditioning spread exceeds '
                         'this many meters')
    ap.add_argument('--max-step', type=float, default=0.20,
                    help='auto-level water check: largest brightness '
                         'step (mean [0,1] intensity) allowed across '
                         'the horizon. A true sea horizon is nearly '
                         'continuous — the water mirrors the sky at '
                         'grazing incidence (E4q measured 0.03 median '
                         'on 1325 real maritime photos, vs 0.375 at '
                         'land boundaries). Raise it only for imagery '
                         'whose sea is rendered darker than physics '
                         'warrants, e.g. the E4c synthetic composites '
                         '(0.17-0.30); default 0.20')
    ap.add_argument('--auto-level', action='store_true',
                    help='estimate pitch and roll from the visible sea '
                         'horizon (per photo), overriding --pitch/--roll '
                         'and tightening the elevation-offset window to '
                         '+-2 mrad; falls back to the priors when no '
                         'adequate sea segment is visible')
    ap.add_argument('--dem', default=os.path.expanduser(
        os.environ.get('HORIZONATOR_DEMS', '~/.horizonator/DEMs_SRTM3')))
    ap.add_argument('--dmin', type=float, default=1000.0)
    ap.add_argument('--dmin-soft', type=float,
                    help='soft near-field mode for LAND observers: '
                         'instead of hard-masking terrain nearer than '
                         '--dmin (which throws away genuine mid-field '
                         'skyline), march from 150 m and DOWN-WEIGHT '
                         'each azimuth by a ramp from 0 at 300 m to 1 '
                         'at this distance (m; try 1500). The observer\'s '
                         'own bluff still contributes ~nothing, but a '
                         'ridge at 800 m keeps most of its vote')
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
    ap.add_argument('--rigid-pan', action='store_true',
                    help='multi-photo mode: the frames are a PAN whose '
                         'relative headings are gyro-accurate, so all '
                         'photos share ONE compass-offset nuisance '
                         'instead of independent per-photo shifts — '
                         'fewer nuisances, stiffer joint fix. Requires '
                         'a heading for every photo')
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
    global EXTRACTOR
    EXTRACTOR = args.extractor
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
            dip = S.horizon_dip_rad(max(z, 0.5))
            if args.dt_air_sea is not None:
                # dip anomaly ~0.11 arcmin/degC of (T_air - T_sea):
                # warm air over cold water raises the apparent horizon
                # (smaller dip). PROVISIONAL coefficient — see --help
                dip -= 0.032e-3 * args.dt_air_sea
            if args.level_detector == 'radon':
                level = extract.sea_horizon_attitude_radon(
                    img, f_px, dip, max_step=args.max_step)
            else:
                level = extract.sea_horizon_attitude(
                    rows, conf, img.shape, f_px, dip, rgb=img,
                    max_step=args.max_step)
            if level:
                pitch, roll = level['pitch_deg'], level['roll_deg']
                half = 0.002
                if args.dt_air_sea is not None:
                    half += 0.00015 * abs(args.dt_air_sea)
                betas = np.arange(-half, half * 1.001, half / 4)
        el_obs, w, diag = observation(
            img, fovs[i], headings[i] if headings[i] is not None else 0.0,
            roll, pitch)
        # the azimuth window must match the compass that produced the
        # prior, not a fixed guess: Theodolite records the accuracy iOS
        # reported (median +-10 deg on this field set, worst +-41), and
        # searching +-6 around a prior that is +-11 good simply cannot
        # reach the truth (E4x: the DEM was being sampled at azimuths
        # the camera never saw).
        hw = int(round(args.heading_window / 0.1))
        shifts = np.arange(-hw, hw + 1, 2) if headings[i] is not None \
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
                    (lon_c - 0.8, lon_c + 0.8),
                    d_min=150.0 if args.dmin_soft else args.dmin)
    usum = sum(uw)

    # ---- is the subject even far enough for this method?
    # E4x measured that the field failures are largely a subject
    # mismatch: binned by the range of the terrain forming the
    # silhouette, sightings of terrain within 3 km miss by a median
    # 2463 m against 778 m for 3-10 km. At a kilometre, trees and
    # buildings subtend tens of mrad and appear in no DEM, and SRTM's
    # posting cannot shape the ridge — so the silhouette being matched
    # is mostly not in the model at all. The check runs at the BOX
    # CENTRE, the position actually known in the field, never at truth.
    subject_km = None
    if args.min_range:
        el_c, rng_c = cm.skyline(lat_c, lon_c, z, AZ)
        med = []
        for p in photos:
            if p['heading'] is None:
                continue
            rel = (AZ - p['heading'] + 180.0) % 360.0 - 180.0
            m = np.abs(rel) <= p['fov'] / 2
            if m.sum() >= 5:
                med.append(float(np.median(rng_c[m])))
        if med:
            subject_km = float(np.median(med)) / 1000.0
            print(f'subject range at the box centre: '
                  f'{subject_km:.1f} km', flush=True)

    rigid = args.rigid_pan and len(photos) > 1 \
        and all(p['heading'] is not None for p in photos)

    def C(dn, de, plist=None, wsum=None):
        pl = photos if plist is None else plist
        us = usum if wsum is None else wsum
        el, r = cm.skyline(lat_c + dn / mlat, lon_c + de / mlon, z, AZ)
        ws = None
        if args.dmin_soft:
            ws = np.clip((r - 300.0) / (args.dmin_soft - 300.0), 0.0, 1.0)
        if rigid:
            shared = np.arange(-60, 61, 2)
            tot = np.zeros(shared.size)
            for p in pl:
                tot += p['weight'] * photo_cost_curve(
                    p['el_obs'], p['w'], el, shared, p['betas'])
            return float(tot.min()) / us
        return sum(p['weight'] * fast_photo_cost(
            p['el_obs'], p['w'], el, p['shifts'], p['betas'],
            w_syn=ws)[0] for p in pl) / us

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

    # ---- conditioning: how much does the fix depend on the data?
    # E4v measured that neither the basin margin nor the covariance
    # ellipse separates a 400 m fix from a 3 km one on real single-frame
    # sightings — the margin scores how DISTINCT the winning basin is,
    # and the Laplace covariance is uniformly overconfident. What does
    # distinguish them is stability: resolve with part of the evidence
    # removed and see whether the answer moves. Multi-photo fixes drop
    # one frame at a time; a single frame is split into left and right
    # halves of its own field of view, which is the same test at the
    # only scale available.
    def coarse_min(plist, wsum):
        cc2 = np.array([[C(dn, de, plist, wsum) for de in g]
                        for dn in g])
        i2, j2 = np.unravel_index(np.argmin(cc2), cc2.shape)
        return g[i2], g[j2]

    jack = None
    if args.conditioning:
        subs = []
        if N >= 3:
            for k in range(N):
                pl = [p for m, p in enumerate(photos) if m != k]
                subs.append((pl, sum(p['weight'] for p in pl)))
        else:
            for p in photos:
                idx = np.where(p['w'] > 0)[0]
                if idx.size < 20:
                    continue
                mid = idx[idx.size // 2]
                for lo, hi in ((idx[0], mid), (mid, idx[-1])):
                    w2 = np.zeros_like(p['w'])
                    w2[lo:hi + 1] = p['w'][lo:hi + 1]
                    q = dict(p)
                    q['w'] = w2
                    subs.append(([q], p['weight']))
        pts = []
        for pl, wsum in subs:
            try:
                pts.append(coarse_min(pl, wsum))
            except Exception:
                pass
        if len(pts) >= 2:
            d = [float(np.hypot(a - dn0, b - de0)) for a, b in pts]
            jack = dict(n_subsets=len(pts),
                        spread_m=float(np.median(d)),
                        max_m=float(np.max(d)))
            print(f'conditioning: {len(pts)} subset solves, median '
                  f'{jack["spread_m"]:.0f} m from the full fix, max '
                  f'{jack["max_m"]:.0f} m', flush=True)

    # Laplace covariance from a local quadratic fit of the cost
    h = step0 / 10
    c0 = C(dn0, de0)
    cnn = (C(dn0 + h, de0) - 2 * c0 + C(dn0 - h, de0)) / h ** 2
    cee = (C(dn0, de0 + h) - 2 * c0 + C(dn0, de0 - h)) / h ** 2
    cne = (C(dn0 + h, de0 + h) - C(dn0 + h, de0 - h)
           - C(dn0 - h, de0 + h) + C(dn0 - h, de0 - h)) / (4 * h ** 2)
    Hm = np.array([[cnn, cne], [cne, cee]])
    sig_maj = sig_min = np.nan
    maj_brg = np.nan
    try:
        cov = 2 * c0 * np.linalg.inv(Hm)  # scaled: residual-level heuristic
        sig = np.sqrt(np.maximum(np.diag(cov), 0))
        # the ELLIPSE, not just its axis-aligned shadow. A fix from one
        # narrow field of view is a bearing: the cost barely changes
        # along the line of sight, so the ellipse is long and thin, and
        # its length is the honest statement of what was measured
        # (E4v). The diagonal sigmas hide this whenever the ellipse is
        # not aligned with north/east.
        evals, evecs = np.linalg.eigh(0.5 * (cov + cov.T))
        evals = np.maximum(evals, 0.0)
        sig_min, sig_maj = np.sqrt(evals[0]), np.sqrt(evals[1])
        vmaj = evecs[:, 1]                    # (north, east) components
        maj_brg = float(np.degrees(np.arctan2(vmaj[1], vmaj[0])) % 180.0)
    except np.linalg.LinAlgError:
        sig = [np.nan, np.nan]
    aniso = float(sig_maj / sig_min) if sig_min > 1e-9 else np.inf

    lat_e = lat_c + dn0 / mlat
    lon_e = lon_c + de0 / mlon
    el_syn, r_syn = cm.skyline(lat_e, lon_e, z, AZ)
    ws_fix = None
    if args.dmin_soft:
        ws_fix = np.clip((r_syn - 300.0) / (args.dmin_soft - 300.0),
                         0.0, 1.0)
    pj = []
    for p in photos:
        cb, sb, bb = fast_photo_cost(p['el_obs'], p['w'], el_syn,
                                     p['shifts'], p['betas'],
                                     w_syn=ws_fix)
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
    if subject_km is not None \
            and subject_km * 1000.0 < args.min_range:
        reasons.append(f'subject too near: the silhouette comes from '
                       f'terrain a median {subject_km:.1f} km away '
                       f'(limit {args.min_range / 1000:.1f} km), where '
                       f'canopy and buildings dominate a DEM skyline')
    if jack and args.max_jackknife \
            and jack['spread_m'] > args.max_jackknife:
        reasons.append(f"conditioning: the fix moves "
                       f"{jack['spread_m']:.0f} m when part of the "
                       f"evidence is removed (limit "
                       f"{args.max_jackknife:.0f} m) — the geometry does "
                       f"not pin a position")
    status = 'ok' if not reasons else 'inconclusive'
    for r in reasons:
        print('INCONCLUSIVE:', r, file=sys.stderr)
    result = dict(status=status, reasons=reasons,
                  lat=lat_e, lon=lon_e,
                  fix_ok=(status == 'ok'), basin_margin=float(margin),
                  relief_mrad=relief,
                  dn_m=dn0, de_m=de0,
                  sigma_n_m=float(sig[0]), sigma_e_m=float(sig[1]),
                  sigma_major_m=float(sig_maj),
                  sigma_minor_m=float(sig_min),
                  major_bearing_deg=maj_brg, anisotropy=aniso,
                  jackknife=jack, subject_km=subject_km,
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
