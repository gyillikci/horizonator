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

    # sub-pixel: linear interpolation of dev across the onset row
    out = first.copy()
    for x in range(W):
        r = int(first[x])
        if 9 <= r < Hs - 1:
            d0, d1 = dev[r - 1, x], dev[r, x]
            t = nsigma * sigma[x]
            if d1 > d0 and d0 < t <= d1:
                out[x] = r - 1 + (t - d0) / (d1 - d0)
    return out, np.maximum(conf, 0.0)


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
