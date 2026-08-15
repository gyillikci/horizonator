#!/usr/bin/env python3
"""E4j: the coastline as a second matched curve — stadimetric range.

The skyline constrains bearings well but range weakly (cost basins
elongate along the viewing ray). The visible WATERLINE at the foot of
each shore is a second 1D curve with the opposite property: for a
known-height observer the depression of the shore at distance d is

    delta(az) = atan( -(z + d^2/2Reff) / d )  ~  -z/d - d/2Reff

so its shape directly encodes range (the classical "dip short of the
horizon" stadimeter). Range sensitivity sigma_d ~ (d^2/z) sigma_delta:
strong for near shores and high observers, useless far away — this
experiment quantifies whether adding the coastline term rounds out the
basins in practice, BEFORE we invest in a water/land segmenter for the
observation side (study doc section 3.4, Grelsson's three-class CNN).

Synthetic closed loop on the E4c sea sites: observed skyline+coastline
curves generated from the truth (independent noise on each), candidate
curves synthesized per grid cell; compare skyline-only vs joint cost
maps — position error, basin margin, and the Laplace sigma along the
worst axis. Writes out/e4j_results.json.
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S
from skyfix import basin_margin, fast_photo_cost

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
DIR3 = os.path.expanduser('~/.horizonator/DEMs_SRTM3')
AZ = np.arange(-180.0, 180.0, 0.1) + 0.05
BOX = 5000.0
STEP0 = 250.0

# (name, lat, lon, z): E4c sea sites + one elevated observer (a bluff /
# ship's bridge), where the z/d^2 stadimetric sensitivity is 3-4x larger
SITES = [
    ('strait1', 36.9500, 27.2500, 5.0),
    ('strait2', 36.9622, 27.2384, 5.0),
    ('offshore', 36.6050, 26.8590, 5.0),
    ('strait2_z18', 36.9622, 27.2384, 18.0),
]
rng = np.random.default_rng(20260820)


def coast_curve(cm, lat, lon, z, step_m=45.0):
    """Depression angle of the visible waterline per azimuth.

    Marches each azimuth ray over the DEM mosaic until the first land
    sample (> 1 m), giving the distance d(az) to the shore; the curve is
    the depression of the sea-level point there (curvature-corrected).
    Azimuths with no land before the sea horizon get the horizon dip and
    zero weight (no range information)."""
    mlat, mlon = S.meters_per_degree(lat)
    dmax = min(S.horizon_distance_m(max(z, 1.0)) * 1.05, 25000.0)
    n_steps = int(dmax / step_m)
    d = (np.arange(1, n_steps + 1) * step_m)[:, None]     # (steps, az)
    azr = np.radians(AZ)[None, :]
    la = lat + (d * np.cos(azr)) / mlat
    lo = lon + (d * np.sin(azr)) / mlon
    y = np.clip(((cm.lat_nw - la) / cm.dpp).astype(int), 0,
                cm.mosaic.shape[0] - 1)
    x = np.clip(((lo - cm.lon_nw) / cm.dpp).astype(int), 0,
                cm.mosaic.shape[1] - 1)
    land = cm.mosaic[y, x] > 1.0
    first = np.where(land.any(axis=0), land.argmax(axis=0), -1)
    has = first >= 0
    dsh = np.where(has, (first + 1) * step_m, dmax)
    delta = np.arctan(-(z + dsh ** 2 / (2 * S.REFF)) / dsh)
    delta[~has] = -S.horizon_dip_rad(max(z, 1.0))
    return delta, dsh, has.astype(float)


def solve(cost_fn):
    g = np.arange(-BOX / 2, BOX / 2 + 1, STEP0)
    cc = np.array([[cost_fn(dn, de) for de in g] for dn in g])
    i, j = np.unravel_index(np.argmin(cc), cc.shape)
    dn0, de0 = g[i], g[j]
    margin = basin_margin(cc, g, min_sep=4 * STEP0)
    for step in (50.0, 12.5):
        best = (np.inf, dn0, de0)
        for di in range(-2, 3):
            for dj in range(-2, 3):
                c = cost_fn(dn0 + di * step, de0 + dj * step)
                if c < best[0]:
                    best = (c, dn0 + di * step, de0 + dj * step)
        _, dn0, de0 = best
    h = 25.0
    c0 = cost_fn(dn0, de0)
    cnn = (cost_fn(dn0 + h, de0) - 2 * c0 + cost_fn(dn0 - h, de0)) / h ** 2
    cee = (cost_fn(dn0, de0 + h) - 2 * c0 + cost_fn(dn0, de0 - h)) / h ** 2
    cne = (cost_fn(dn0 + h, de0 + h) - cost_fn(dn0 + h, de0 - h)
           - cost_fn(dn0 - h, de0 + h)
           + cost_fn(dn0 - h, de0 - h)) / (4 * h ** 2)
    try:
        cov = 2 * c0 * np.linalg.inv(np.array([[cnn, cne], [cne, cee]]))
        ev = np.sqrt(np.maximum(np.linalg.eigvalsh(cov), 0))
        sig_worst = float(ev.max())
    except np.linalg.LinAlgError:
        sig_worst = float('nan')
    return dict(err_m=float(np.hypot(dn0, de0)), margin=float(margin),
                sigma_worst_m=sig_worst)


if __name__ == '__main__':
    results = {}
    for name, lat, lon, z in SITES:
        cm = S.CMarcher(DIR3, (lat - .6, lat + .6), (lon - .8, lon + .8),
                        d_min=1000.)
        mlat, mlon = S.meters_per_degree(lat)

        el_true, _ = cm.skyline(lat, lon, z, AZ)
        co_true, dsh, w_c = coast_curve(cm, lat, lon, z)
        # observation = truth + independent 1 mrad noise on each curve;
        # coastline weight only where land is within stadimetric reach
        el_obs = el_true + rng.normal(0, 1e-3, AZ.size)
        co_obs = co_true + rng.normal(0, 1e-3, AZ.size)
        w_sky = np.ones(AZ.size)
        w_c = w_c * (dsh < 15000.0)
        shifts = np.arange(-60, 61, 2)

        def sky_cost(dn, de):
            el, _ = cm.skyline(lat + dn / mlat, lon + de / mlon, z, AZ)
            return fast_photo_cost(el_obs, w_sky, el, shifts)[0]

        def joint_cost(dn, de):
            la, lo = lat + dn / mlat, lon + de / mlon
            el, _ = cm.skyline(la, lo, z, AZ)
            cs = fast_photo_cost(el_obs, w_sky, el, shifts)[0]
            co, _, wc2 = coast_curve(cm, la, lo, z)
            wj = w_c * wc2
            if wj.sum() < 10:
                return cs
            r = np.abs(co - co_obs)[wj > 0]
            hub = np.where(r <= 3e-3, .5 * r * r, 3e-3 * (r - 1.5e-3))
            return cs + float(hub.mean())

        A = solve(sky_cost)
        Bv = solve(joint_cost)
        results[name] = dict(z=z, skyline=A, joint=Bv,
                             coast_frac=float((w_c > 0).mean()),
                             d_shore_med=float(np.median(dsh[w_c > 0]))
                             if (w_c > 0).any() else None)
        print(f"{name:12s} z={z:4.1f}  skyline-only: err {A['err_m']:5.1f} m "
              f"margin {A['margin']:5.2f} sig_worst {A['sigma_worst_m']:6.1f} m"
              f"  |  +coastline: err {Bv['err_m']:5.1f} m "
              f"margin {Bv['margin']:5.2f} sig_worst {Bv['sigma_worst_m']:6.1f} m"
              f"  (coast in {100*results[name]['coast_frac']:.0f}% of az)",
              flush=True)
    with open(os.path.join(OUT, 'e4j_results.json'), 'w') as f:
        json.dump(results, f, indent=1)
