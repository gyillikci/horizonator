#!/usr/bin/env python3
"""Automatic skyline extraction from a photograph, plus EXIF utilities.

The extractor finds the sky/terrain boundary by local linear continuation:
per column, each row's color is predicted from the rows immediately above,
which tolerates any smooth vertical sky gradient (graded sky, haze) with
zero model drift; the boundary is the first sustained deviation from the
top. This beats both global sky models (extrapolation drift reads as
terrain) and pure edge detectors (a crisp sea horizon outshouts a faint
distant ridge). Returns per-column row + confidence.

No ML dependency: designed for clear-to-hazy daylight maritime scenes,
the study's target regime. Swap in a CNN front-end later if needed.
"""

import numpy as np
from PIL import Image, ExifTags


def load_image(path, max_w=1600):
    im = Image.open(path)
    im = im.convert('RGB')
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)),
                       Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def skyline_seam(rgb, search_frac=0.85, m_sustain=8, nsigma=4.0,
                 band=7, jump_penalty=6.0):
    """rgb (H,W,3) in [0,1] -> (rows, confidence) per column.

    Per column, each row's color is predicted by LINEAR CONTINUATION of the
    rows immediately above (tolerates any smooth vertical sky gradient with
    zero model drift, unlike a global sky fit). The boundary is the first
    row, from the top, whose deviation from that prediction stays above
    nsigma * (per-column noise) for m_sustain consecutive rows. Column
    outliers are repaired with a running median; sub-pixel refinement by
    linear interpolation of the deviation onset.
    """
    H, W, _ = rgb.shape
    Hs = int(H * search_frac)

    # local linear continuation: pred[r] = 2*mean(rgb[r-5:r-2]) -
    # mean(rgb[r-8:r-5]); implemented with a cumulative sum
    c = np.cumsum(np.vstack([np.zeros((1, W, 3)), rgb[:Hs]]), axis=0)
    rr = np.arange(8, Hs)
    near = (c[rr - 2 + 1] - c[rr - 5 + 1]) / 3.0     # mean rows [r-5, r-3]
    far = (c[rr - 5 + 1] - c[rr - 8 + 1]) / 3.0      # mean rows [r-8, r-6]
    pred = np.full((Hs, W, 3), np.nan)
    pred[8:] = 2.0 * near - far
    dev = np.linalg.norm(rgb[:Hs] - pred, axis=2)
    dev[:8] = 0.0

    # per-column noise from the top quarter (assumed sky)
    ntop = max(20, Hs // 4)
    sigma = np.maximum(1.4826 * np.median(
        np.abs(dev[8:ntop] - np.median(dev[8:ntop], axis=0)), axis=0)
        + np.median(dev[8:ntop], axis=0), 0.010)

    # sustained deviation: forward mean over m_sustain rows
    cs = np.cumsum(np.vstack([dev, np.zeros((m_sustain, W))]), axis=0)
    Sm = (cs[m_sustain:] - cs[:-m_sustain]) / m_sustain   # (Hs,W)
    hit = Sm[:Hs] > (nsigma * sigma)[None, :]
    hit[:8] = False
    first = np.where(hit.any(axis=0), hit.argmax(axis=0), Hs - 1).astype(float)
    conf = Sm[np.minimum(first.astype(int), Hs - 1), np.arange(W)] / \
        (nsigma * sigma)

    # repair outlier columns against a running median (window 21)
    k = 10
    med = np.array([np.median(first[max(0, x - k):x + k + 1])
                    for x in range(W)])
    bad = np.abs(first - med) > 12
    first[bad] = med[bad]
    conf[bad] *= 0.3

    # sub-pixel: the sustained-mean trigger fires up to m_sustain-1 rows
    # ABOVE a crisp boundary (its forward window already contains it), a
    # contrast-dependent bias of several pixels. Advance to the first row
    # where dev itself crosses the threshold, then interpolate across it
    out = first.copy()
    for x in range(W):
        r = int(first[x])
        t = nsigma * sigma[x]
        rr = r
        while rr < min(r + m_sustain, Hs - 1) and dev[rr, x] <= t:
            rr += 1
        if 9 <= rr < Hs - 1:
            d0, d1 = dev[rr - 1, x], dev[rr, x]
            if d1 > d0 and d0 < t <= d1:
                out[x] = rr - 1 + (t - d0) / (d1 - d0)
            else:
                out[x] = rr
    return out, np.maximum(conf, 0.0)


def sea_horizon_attitude(rows, conf, shape, f_px, dip_rad, rgb=None,
                         tol_px=1.5, min_frac=0.12, min_span_frac=0.35,
                         max_step=0.20):
    """Estimate camera (pitch, roll) from the visible sea horizon.

    For an observer at known height h the sea horizon sits at the exactly
    known dip sqrt(2h/Reff) below level, so boundary columns lying on it
    form an absolute, drift-free attitude reference (the levelling stage
    of Grelsson et al. 2020, done in closed form instead of with a CNN).
    For small angles a sea-horizon pixel (u, v) satisfies

        v = tan(-dip - pitch) * hypot(u, f) - roll * u

    which is linear in (tan(-dip-pitch), roll). The line is found by
    2-point RANSAC over the boundary columns with a one-sided veto: a
    candidate is discarded when boundary points lie significantly BELOW
    it, which is impossible for a true sea horizon (terrain only ever
    rises above it) but happens whenever a flat elevated ridge is tried
    while real sea is visible lower in the frame.

    A straight distant ridge or shoreline can still masquerade as the
    horizon (the navigator's "false horizon"). When rgb is given, a
    photometric check rejects such fits — but by CONTINUITY, not by
    darkness. At grazing incidence the sea's Fresnel reflectivity
    approaches 1, so the water immediately below a true horizon mirrors
    the sky immediately above it and the brightness step across the
    boundary is small; land, foliage or a hull against the sky makes a
    large step. Measured on MaSTr1325's 1325 real maritime images
    (E4q): true sea horizons have a median sky-minus-water step of
    0.031 (p90 0.185), land boundaries 0.375 (p10 0.244) — so
    |step| <= max_step separates them cleanly, while the earlier
    "water is darker by >= 0.30" rule had the discriminant backwards
    and did the opposite of its job (it rejected 98% of real horizons
    and admitted the land boundaries). Daylight rule; sun glitter or
    night use still wants a learned water/land/sky front-end.

    rows, conf: skyline_seam() output. shape: image shape. f_px: focal
    length in pixels. dip_rad: horizon dip for the camera height
    (skyline.horizon_dip_rad(z)). rgb: the image in [0,1], enables the
    water check.

    Returns dict(pitch_deg, roll_deg, n_inl, frac, span_frac, rms_px,
    contrast), or None when no adequate sea horizon is visible."""
    H, W = shape[:2]
    u = np.arange(W) - (W - 1) / 2.0
    v = (H - 1) / 2.0 - np.asarray(rows, dtype=float)
    Hu = np.hypot(u, f_px)
    ok = np.asarray(conf) > 0
    idx = np.where(ok)[0]
    if idx.size < 16:
        return None

    rng = np.random.default_rng(0)
    best = None
    for _ in range(300):
        i, j = rng.choice(idx, 2, replace=False)
        if abs(u[i] - u[j]) < W / 8:
            continue
        M = np.array([[Hu[i], u[i]], [Hu[j], u[j]]])
        try:
            a, b = np.linalg.solve(M, [v[i], v[j]])
        except np.linalg.LinAlgError:
            continue
        res = v - (a * Hu + b * u)
        if ((res < -3.0) & ok).sum() > 0.02 * W:
            continue                    # boundary below the line: not sea
        n = int(((np.abs(res) <= tol_px) & ok).sum())
        if best is None or n > best[0]:
            best = (n, a, b)
    if best is None:
        return None
    _, a, b = best

    inl = ok
    for _ in range(3):                  # IRLS refinement
        res = v - (a * Hu + b * u)
        inl = (np.abs(res) <= tol_px) & ok
        if inl.sum() < 8:
            return None
        A = np.stack([Hu[inl], u[inl]], axis=1)
        ww = np.clip(conf[inl], 0.0, 3.0)
        sol, *_ = np.linalg.lstsq(A * ww[:, None], v[inl] * ww, rcond=None)
        a, b = float(sol[0]), float(sol[1])

    res = v - (a * Hu + b * u)
    if ((res < -3.0) & ok).sum() > 0.02 * W:
        return None
    rms = float(np.sqrt(np.mean(res[inl] ** 2)))
    frac = float(inl.mean())
    span = float((u[inl].max() - u[inl].min()) / max(u[-1] - u[0], 1.0))
    if frac < min_frac or span < min_span_frac or rms > tol_px:
        return None

    contrast = None
    if rgb is not None:
        bri = np.asarray(rgb).mean(axis=2)
        above, below = [], []
        for x in np.where(inl)[0]:
            r0 = int(round(rows[x]))
            if r0 - 14 >= 0 and r0 + 14 < H:
                above.append(bri[r0 - 12:r0 - 3, x].mean())
                below.append(bri[r0 + 3:r0 + 12, x].mean())
        if len(above) < 8:
            return None
        contrast = float(np.median(above) - np.median(below))
        if abs(contrast) > max_step:
            return None                 # big step across it: not water
    return dict(pitch_deg=float(np.degrees(-dip_rad - np.arctan(a))),
                roll_deg=float(np.degrees(-np.arctan(b))),
                n_inl=int(inl.sum()), frac=frac, span_frac=span,
                rms_px=rms, contrast=contrast)


def _ratio(v):
    try:
        return float(v[0]) / float(v[1])
    except (TypeError, IndexError):
        return float(v)


def read_exif(path):
    """Return dict with any of: fov_deg (from FocalLengthIn35mmFilm and
    aspect), heading_deg (GPSImgDirection), alt_m (GPSAltitude),
    lat, lon (GPS), model."""
    im = Image.open(path)
    ex = im.getexif()
    out = {}
    tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
    sub = ex.get_ifd(0x8769)  # Exif IFD
    tags.update({ExifTags.TAGS.get(k, k): v for k, v in sub.items()})
    f35 = tags.get('FocalLengthIn35mmFilm')
    if f35:
        w, h = im.size
        a = max(w, h) / min(w, h)
        # f35 is defined against the 36x24 frame's 43.27 mm diagonal:
        # tan(HFOV/2) = (a/sqrt(a^2+1)) * (21.63/f35)
        out['fov_deg'] = float(np.degrees(2 * np.arctan(
            (a / np.hypot(a, 1.0)) * 21.63 / float(f35))))
    gps = ex.get_ifd(0x8825)
    if gps:
        g = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}
        if 'GPSImgDirection' in g:
            out['heading_deg'] = _ratio(g['GPSImgDirection'])
        if 'GPSAltitude' in g:
            out['alt_m'] = _ratio(g['GPSAltitude'])
        if 'GPSLatitude' in g:
            d, m, s = (_ratio(v) for v in g['GPSLatitude'])
            out['lat'] = (d + m / 60 + s / 3600) * \
                (-1 if g.get('GPSLatitudeRef') == 'S' else 1)
            d, m, s = (_ratio(v) for v in g['GPSLongitude'])
            out['lon'] = (d + m / 60 + s / 3600) * \
                (-1 if g.get('GPSLongitudeRef') == 'W' else 1)
    if 'Model' in tags:
        out['model'] = str(tags['Model'])
    return out
