#!/usr/bin/env python3
"""Night-watch front end: turn a dark video into identified light
bearings.

Three stages, numpy only:

  track_points(frames)   bright-blob detection + nearest-neighbour
                         tracking across frames -> per-track brightness
                         trace and mean pixel position
  classify_trace(t, y)   flash-character classification of one trace:
                         pattern (F/Iso/Oc/Fl/LFl/Q), group, period
  identify(...)          classified tracks x LightDB.match within a
                         range gate from the DR position -> identified
                         lights; a track that does not match EXACTLY
                         ONE charted light is dropped (never guess a
                         landmark)

Classification is deliberately chart-like: threshold the trace into
lit/dark, measure the cycle with autocorrelation of the binary wave,
count flashes per cycle, and read the duty cycle the way a navigator
reads a light list — flashing = short lit fraction, occulting = short
dark fraction, isophase = equal, quick = >= 50 flashes/min. Two or
three charted periods of video (~30-45 s for a 15 s light) suffice.
"""

import numpy as np


def track_points(frames, nsig=10.0, min_sep=6, max_jump=8.0):
    """frames: (T, H, W) grayscale float array of a NIGHT scene.
    Returns list of dict(u, v, trace) — mean pixel position and
    per-frame peak brightness of each persistent point light.
    Detection threshold is median + nsig * MAD-sigma: a point light is
    tens of sigma above a night sky's noise floor, and a percentile
    threshold would sit inside the noise tail and spawn junk tracks."""
    frames = np.asarray(frames, float)
    T = frames.shape[0]
    med = float(np.median(frames))
    sig = 1.4826 * float(np.median(np.abs(frames - med))) + 1e-9
    thr = med + nsig * sig
    tracks = []
    for k in range(T):
        f = frames[k]
        cand = []
        m = f > thr
        ys, xs = np.where(m)
        used = np.zeros(len(ys), bool)
        order = np.argsort(f[ys, xs])[::-1]
        for o in order:
            if used[o]:
                continue
            y0, x0 = ys[o], xs[o]
            close = (np.abs(ys - y0) < min_sep) & (np.abs(xs - x0) < min_sep)
            used |= close
            cand.append((x0, y0, float(f[y0, x0])))
        alive = {id(t): False for t in tracks}
        for x0, y0, b in cand:
            best, bd = None, max_jump
            for t in tracks:
                d = np.hypot(t['u'][-1] - x0, t['v'][-1] - y0)
                if d < bd and not alive[id(t)]:
                    best, bd = t, d
            if best is None:
                t = dict(u=[x0], v=[y0], trace=[0.0] * k + [b])
                tracks.append(t)
                alive[id(t)] = True
            else:
                best['u'].append(x0)
                best['v'].append(y0)
                best['trace'].append(b)
                alive[id(best)] = True
        for t in tracks:
            if len(t['trace']) < k + 1:
                t['trace'].append(0.0)   # dark this frame — keep time base
    out = []
    for t in tracks:
        tr = np.array(t['trace'])
        if (tr > thr).sum() >= max(3, 0.02 * T):
            out.append(dict(u=float(np.mean(t['u'])),
                            v=float(np.mean(t['v'])), trace=tr))
    return out


def classify_trace(t, y, min_cycles=2.0):
    """One brightness trace -> character dict compatible with
    lights.LightDB.match, or None if not classifiable.
    t: seconds (uniform), y: brightness."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t.size < 8:
        return None
    dt = float(np.median(np.diff(t)))
    lo, hi = np.percentile(y, [10, 98])
    if hi - lo < 1e-9:
        return None
    on = y > (lo + 0.4 * (hi - lo))
    # close sub-half-second dark gaps: a wave hiding the light for a
    # frame must not split a flash (or an isophase half) in two
    max_gap = max(2, int(round(0.4 / dt)))
    dark_start = None
    for k in range(1, on.size):
        if not on[k] and on[k - 1]:
            dark_start = k
        elif on[k] and not on[k - 1] and dark_start is not None:
            if k - dark_start <= max_gap:
                on[dark_start:k] = True
            dark_start = None
    frac = on.mean()
    if frac > 0.97:
        return dict(pattern='F', group=1, period_s=None, colour=None)
    if frac < 0.02:
        return None

    # cycle length: unbiased autocorrelation of the binary wave. The
    # full period aligns every cycle (ac ~ 1); the intra-group flash
    # spacing of a group light only partially aligns (ac ~ (g-1)/g),
    # so the fundamental is the SHORTEST lag reaching near-max ac.
    b = on.astype(float) - frac
    n = b.size
    raw = np.correlate(b, b, 'full')[n - 1:]
    ac = raw / (np.arange(n, 0, -1) * max(b.var(), 1e-12))
    min_lag = max(2, int(round(0.5 / dt)))
    seg = ac[min_lag:n // 2]
    if seg.size < 3:
        return None
    peak = float(seg.max())
    if peak < 0.5:
        return None
    per = None
    for lag in range(min_lag, n // 2):
        if ac[lag] >= 0.9 * peak and ac[lag] >= ac[lag - 1] \
                and ac[lag] >= ac[lag + 1]:
            per = lag * dt
            break
    if per is None or t[-1] - t[0] < min_cycles * per:
        return None

    # flashes per cycle: rising edges, then intervals split at the
    # cycle gap (group separation > intra-group spacing)
    edges = np.where(on[1:] & ~on[:-1])[0] + 1
    if edges.size < 2:
        group = 1
    else:
        iv = np.diff(edges) * dt
        intra = iv[iv < 0.5 * per]
        n_cycles = max(1, int(round((t[-1] - t[0]) / per)))
        group = int(round(edges.size / n_cycles))
        group = max(1, group)
        if intra.size == 0:
            group = 1
        # a UNIFORM train that classified as (k flashes, k*period) is
        # really the ungrouped fundamental — e.g. Fl.2.5s at 5 Hz reads
        # Fl(2)5s first; equal edge spacing gives it away. A true group
        # light (Fl(3)15s: gaps 2,2,11) never passes this test.
        if group > 1 and iv.size >= 2 \
                and iv.max() - iv.min() <= 2.5 * dt:
            per = float(iv.mean())
            group = 1

    # flash length statistics for the pattern class
    runs, cur = [], 0
    for v in on:
        if v:
            cur += 1
        elif cur:
            runs.append(cur * dt)
            cur = 0
    if cur:
        runs.append(cur * dt)
    flash = float(np.median(runs)) if runs else 0.0

    rate_per_min = 60.0 * edges.size / (t[-1] - t[0])
    if rate_per_min >= 50 and group == 1:
        return dict(pattern='Q', group=1, period_s=None, colour=None)
    if 0.4 <= frac <= 0.6 and group == 1:
        return dict(pattern='Iso', group=1, period_s=per, colour=None)
    if frac > 0.6:
        return dict(pattern='Oc', group=group, period_s=per, colour=None)
    pattern = 'LFl' if flash >= 2.0 else 'Fl'
    return dict(pattern=pattern, group=group, period_s=per, colour=None)


def bearing_of(u, v, f_px, heading_rad, width, height=None):
    """Compass bearing (rad, CW from north) of pixel column u for a
    camera at `heading_rad` with focal length f_px — the same pinhole
    the skyline channel uses."""
    du = u - (width - 1) / 2
    return heading_rad + np.arctan2(du, f_px)


def identify(tracks, t, db, lat, lon, radius_m, f_px, heading_rad,
             width):
    """Classify every track and keep those matching exactly one
    charted light. Returns list of (light, char, bearing_rad)."""
    out = []
    for tr in tracks:
        char = classify_trace(t, tr['trace'])
        if char is None:
            continue
        cand = db.match(char, lat, lon, radius_m)
        if len(cand) != 1:
            continue
        out.append((cand[0], char,
                    float(bearing_of(tr['u'], tr['v'], f_px,
                                     heading_rad, width))))
    return out
