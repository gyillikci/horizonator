#!/usr/bin/env python3
"""E5q: a lost boat cruising the Aegean — the instrument end to end.

The scenario the whole study builds toward: a boat with NO position
knowledge sails a nine-leg route through the Izmir gulf, the Mytilene
strait and the Chios strait. At each stop it takes a 360-degree
skyline panorama and asks the instrument where it is.

Honesty rules, all learned the hard way in E4-E5:

  TWO WORLDS   the panorama is rendered from GLO-30 and matched
               against water-masked SRTM1 (E5p) — independent DEM
               families, so the model error between world and map is
               the real, measured kind (E5o), not an inverse crime
  NOISE        each stop draws a compass bias (sigma 2.5 deg), an
               elevation bias (sigma 1.5 mrad, the pitch-prior error
               of E5n) and correlated per-azimuth extraction noise
               (sigma 1 mrad, 3-deg correlation, the E4q floor)
  NO PRIOR     the position search covers the whole 155 x 104 km
               region at 1.5 km resolution, restricted to sea cells
               (a boat knows it floats), then refines to 50 m
  DEAD RECKONING  between stops the boat also keeps a log+compass
               track (5% speed error, 2 deg course error); each fix
               is fused with the DR prediction by inverse covariance,
               and a stop whose basin margin fails the gate falls
               back to DR alone — exactly the E4k contract

Run:  python3 e5q_voyage.py          (writes out/e5q/*.png + json)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import skyline as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'e5q')
WORLD_DEM = os.path.expanduser('~/.horizonator/DEMs_GLO30_1')
MAP_DEM = os.path.expanduser('~/.horizonator/DEMs_SRTM1_WM')

LAT_R = (38.05, 39.55)
LON_R = (25.95, 27.15)
Z = 3.0                                  # eye height on deck, m
AZ = np.arange(0.0, 360.0, 0.5)          # observation grid
AZ_C = np.arange(0.0, 360.0, 2.0)        # coarse-search grid
SEED = 42

# the cruise: Izmir gulf -> Candarli bay -> Mytilene strait north ->
# down the Lesbos east coast -> open water -> Chios strait -> home
STOPS = [
    (38.62, 26.55), (38.85, 26.75), (38.95, 26.65), (39.25, 26.65),
    (39.05, 26.80), (38.45, 26.90), (38.37, 26.22), (38.22, 26.35),
    (38.55, 26.30),
]


def observe(world, lat, lon, rs):
    """One noisy 360-deg panorama from the truth world."""
    el, _ = world.skyline(lat, lon, Z, AZ)
    el = S.seahorizon_fill(el, Z)
    bias_az = rs.normal(0.0, 2.5)
    bias_el = rs.normal(0.0, 1.5e-3)
    n = rs.normal(0.0, 1.0e-3, AZ.size)
    k = np.exp(-0.5 * (np.arange(-6, 7) / 3.0) ** 2)
    n = np.convolve(np.concatenate([n[-6:], n, n[:6]]), k / k.sum(),
                    'same')[6:-6]
    # the boat's compass mislabels the azimuths: what it files under
    # az was really seen at az - bias
    src = (AZ - bias_az) % 360.0
    el_obs = np.interp(src, AZ, el, period=360.0) + bias_el + n
    return el_obs, bias_az, bias_el


def solve_stop(mp, sea, el_obs):
    """Whole-region staged search. Returns fix + quality statistics."""
    obs_c = el_obs[::4]                          # onto the 2-deg grid
    la_g = np.arange(LAT_R[0] + 0.03, LAT_R[1] - 0.03, 1500 / 111132.)
    costs = []
    for la in la_g:
        mlon = 111320.0 * np.cos(np.radians(la))
        for lo in np.arange(LON_R[0] + 0.04, LON_R[1] - 0.04,
                            1500 / mlon):
            if sea.sample(la, lo) > 0.5:
                continue
            el = S.seahorizon_fill(mp.skyline(la, lo, Z, AZ_C)[0], Z)
            costs.append((S.cost_azshift(obs_c, el, max_shift_px=3),
                          la, lo))
    costs.sort()
    c0, la0, lo0 = costs[0]
    # basin margin in the E4k sense: best cost elsewhere, 8 km+ away
    far = [c for c, la, lo in costs
           if np.hypot((la - la0) * 111132.0,
                       (lo - lo0) * 111320.0 * np.cos(np.radians(la0)))
           > 8000.0]
    margin = (min(far) - c0) / c0 if far else 10.0

    # refine: 300 m then 60 m grids on the 0.5-deg observation
    for step, half in ((300.0, 1200.0), (60.0, 240.0)):
        mlat, mlon = 111132.0, 111320.0 * np.cos(np.radians(la0))
        g = np.arange(-half, half + 1, step)
        cc = np.array([[S.cost_azshift(
            el_obs, S.seahorizon_fill(
                mp.skyline(la0 + dn / mlat, lo0 + de / mlon, Z, AZ)[0],
                Z), max_shift_px=12)
            for de in g] for dn in g])
        i, j = np.unravel_index(np.argmin(cc), cc.shape)
        la0 += g[i] / mlat
        lo0 += g[j] / mlon
    # 1-sigma from the fine surface curvature (60 m grid)
    i = np.clip(i, 1, cc.shape[0] - 2)
    j = np.clip(j, 1, cc.shape[1] - 2)
    sig = []
    for cm, c0f, cp in ((cc[i - 1, j], cc[i, j], cc[i + 1, j]),
                        (cc[i, j - 1], cc[i, j], cc[i, j + 1])):
        d2 = max(cm - 2 * c0f + cp, 1e-12)
        sig.append(60.0 * np.sqrt(max(c0f, 1e-12) / d2))
    return dict(lat=la0, lon=lo0, cost=costs[0][0], margin=margin,
                sigma_m=float(np.clip(np.hypot(*sig), 40.0, 5000.0)))


def main():
    rs = np.random.RandomState(SEED)
    print('building the two worlds...', flush=True)
    world = S.CMarcher(WORLD_DEM, LAT_R, LON_R)
    mp = S.CMarcher(MAP_DEM, LAT_R, LON_R)
    sea = S.Dem(MAP_DEM)

    mlat = 111132.0
    rows = []
    dr_lat = dr_lon = None               # dead-reckoning track
    dr_sig = None
    for k, (la, lo) in enumerate(STOPS):
        el_obs, b_az, b_el = observe(world, la, lo, rs)
        fix = solve_stop(mp, sea, el_obs)
        mlon = 111320.0 * np.cos(np.radians(la))
        err = np.hypot((fix['lat'] - la) * mlat, (fix['lon'] - lo) * mlon)
        ok = fix['margin'] > 0.15

        # dead reckoning: previous fused position + the true leg
        # corrupted by log/compass noise
        if k == 0:
            dr = None
        else:
            pla, plo = STOPS[k - 1]
            dn, de = (la - pla) * mlat, (lo - plo) * mlon
            leg = np.hypot(dn, de)
            th = np.arctan2(de, dn) + np.radians(rs.normal(0, 2.0))
            sp = leg * (1 + rs.normal(0, 0.05))
            dr_lat = rows[-1]['fused_lat'] + sp * np.cos(th) / mlat
            dr_lon = rows[-1]['fused_lon'] + sp * np.sin(th) / mlon
            dr_sig = np.hypot(rows[-1]['fused_sig'], 0.06 * leg)
            dr = np.hypot((dr_lat - la) * mlat, (dr_lon - lo) * mlon)

        if ok and dr is None:
            f_lat, f_lon, f_sig = fix['lat'], fix['lon'], fix['sigma_m']
        elif ok:
            wf = 1 / fix['sigma_m'] ** 2
            wd = 1 / dr_sig ** 2
            f_lat = (fix['lat'] * wf + dr_lat * wd) / (wf + wd)
            f_lon = (fix['lon'] * wf + dr_lon * wd) / (wf + wd)
            f_sig = 1 / np.sqrt(wf + wd)
        else:                            # gate refused: coast on DR
            f_lat, f_lon, f_sig = dr_lat, dr_lon, dr_sig
        f_err = np.hypot((f_lat - la) * mlat, (f_lon - lo) * mlon)

        rows.append(dict(
            stop=k, lat=la, lon=lo, fix_lat=fix['lat'],
            fix_lon=fix['lon'], err_m=float(err),
            margin=float(fix['margin']), sigma_m=fix['sigma_m'],
            gate_ok=bool(ok), compass_bias=float(b_az),
            el_bias_mrad=float(b_el * 1e3),
            dr_err_m=None if dr is None else float(dr),
            fused_lat=float(f_lat), fused_lon=float(f_lon),
            fused_sig=float(f_sig), fused_err_m=float(f_err)))
        print(f"stop {k}: fix err {err:6.0f} m  margin "
              f"{fix['margin']:5.2f} {'ok  ' if ok else 'VETO'}"
              f"  DR err {dr if dr else 0:6.0f} m  fused "
              f"{f_err:6.0f} m  (compass bias {b_az:+.1f} deg)",
              flush=True)

    e = np.array([r['err_m'] for r in rows])
    f = np.array([r['fused_err_m'] for r in rows])
    print(f'\nblind fixes: median {np.median(e):.0f} m  worst '
          f'{e.max():.0f} m   fused track: median {np.median(f):.0f} m'
          f'  worst {f.max():.0f} m')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'voyage.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)
    chart(rows, mp, world)


def chart(rows, mp, world):
    # coastline backdrop from the map DEM, coarse land mask
    la = np.linspace(*LAT_R, 320)
    lo = np.linspace(*LON_R, 320)
    dem = S.Dem(MAP_DEM)
    land = np.array([[dem.sample(a, o) > 0.5 for o in lo] for a in la])

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2),
                             gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax = axes[0]
    ax.contourf(lo, la, land, levels=[0.5, 1.5], colors=['#d9cfa8'])
    ax.contour(lo, la, land, levels=[0.5], colors=['#8e8e93'],
               linewidths=0.7)
    t_la = [r['lat'] for r in rows]
    t_lo = [r['lon'] for r in rows]
    ax.plot(t_lo, t_la, '-o', color='#34c759', lw=1.6, ms=5,
            label='true route (GPS, scoring only)')
    ax.plot([r['fix_lon'] for r in rows], [r['fix_lat'] for r in rows],
            'x', color='#ff3b30', ms=8, mew=2,
            label='blind fix (no prior, whole-region search)')
    ax.plot([r['fused_lon'] for r in rows],
            [r['fused_lat'] for r in rows], '-s', color='#0a84ff',
            lw=1.1, ms=4, alpha=0.9, label='fused track (fix + DR)')
    for r in rows:
        ax.annotate(str(r['stop']), (r['lon'], r['lat']), fontsize=8,
                    textcoords='offset points', xytext=(6, 5))
        if not r['gate_ok']:
            ax.annotate('veto', (r['fix_lon'], r['fix_lat']),
                        fontsize=7, color='#ff3b30',
                        textcoords='offset points', xytext=(5, -9))
    ax.set_xlim(*LON_R)
    ax.set_ylim(*LAT_R)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(LAT_R))))
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    e = np.array([r['err_m'] for r in rows])
    f = np.array([r['fused_err_m'] for r in rows])
    ax.set_title(f'a lost boat, nine stops — blind fix median '
                 f'{np.median(e):.0f} m, fused {np.median(f):.0f} m',
                 fontsize=10)
    ax.legend(fontsize=8, loc='lower left')

    ax = axes[1]
    ks = [r['stop'] for r in rows]
    ax.plot(ks, [r['err_m'] for r in rows], '-x', color='#ff3b30',
            label='blind fix error')
    ax.plot(ks[1:], [r['dr_err_m'] for r in rows[1:]], '--',
            color='#8e8e93', label='dead reckoning alone')
    ax.plot(ks, [r['fused_err_m'] for r in rows], '-s',
            color='#0a84ff', label='fused')
    for r in rows:
        if not r['gate_ok']:
            ax.axvline(r['stop'], color='#ff9f0a', alpha=0.4, lw=6)
    ax.set_yscale('log')
    ax.set_xlabel('stop')
    ax.set_ylabel('error (m)')
    ax.grid(alpha=0.25, which='both')
    ax.legend(fontsize=8)
    ax.set_title('per-stop error (orange band = gate veto, DR coast)',
                 fontsize=10)

    fig.tight_layout()
    p = os.path.join(OUT, 'voyage.png')
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(p)

    # one match panel: the worst accepted stop, obs vs predicted
    acc = [r for r in rows if r['gate_ok']]
    r = max(acc, key=lambda r: r['err_m'])
    el_w, _ = world.skyline(r['lat'], r['lon'], Z, AZ)
    el_m, _ = mp.skyline(r['fix_lat'], r['fix_lon'], Z, AZ)
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(AZ, S.seahorizon_fill(el_w, Z) * 1e3, color='#34c759',
            lw=1.2, label='what the boat saw (world DEM, true spot)')
    ax.plot(AZ, S.seahorizon_fill(el_m, Z) * 1e3, color='#0a84ff',
            lw=1.0, ls='--', label='map prediction at the blind fix')
    ax.set_title(f"stop {r['stop']}: worst accepted fix, "
                 f"{r['err_m']:.0f} m off after searching "
                 f"155 x 104 km", fontsize=10)
    ax.set_xlabel('azimuth (deg true)')
    ax.set_ylabel('skyline elevation (mrad)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(OUT, 'match.png')
    fig.savefig(p2, dpi=120)
    print(p2)


if __name__ == '__main__':
    main()
