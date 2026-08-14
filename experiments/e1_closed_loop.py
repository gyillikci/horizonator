#!/usr/bin/env python3
"""E1: synthetic closed-loop localization (render-vs-render) in a 1km x 1km
box, observer at sea.

Per the study doc (doc/skyline-matching-study.md, section 6). Two sites in
the Bodrum/Kos region of the Aegean:

  A "strait":   mid-strait between the Bodrum peninsula and Kos. Terrain in
                nearly every azimuth, 5-15 km away: the strong-geometry case.
  B "offshore": open water SW of Kos. Land (Kefalos hills, Kalymnos, Nisyros)
                only in a ~NNE-ENE sector, 15-28 km away: the weak,
                anisotropic case.

For each site: ground-truth skylines are rendered at random positions in the
box (viewer z = 5 m) and the position is re-estimated with the coarse-to-fine
search of skyline.solve_position(). All candidate skylines are precomputed on
a 25 m lattice over the box, which also yields the full cost surface.

Configs per site:
  clean/360    quantization-noise only, full panorama
  clean/90     same, but only a 90 deg azimuth sector is used (limited FOV)
  noise/360    1 mrad gaussian noise added to the observed skyline
  bias/360     0.2 deg heading bias applied to the observed skyline (the
               dominant real-world systematic; expect ~ range * bias error)

Run headlessly:   xvfb-run -a python3 e1_closed_loop.py
"""

import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import skyline as S

DIR_DEMS = os.environ.get('HORIZONATOR_DEMS',
                          os.path.expanduser('~/.horizonator/DEMs_SRTM3'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)

Z = 5.0            # viewer height above the sea, m
BOX = 1000.0       # search box, m
LATTICE = 25.0     # candidate lattice pitch, m
NGT = 20           # ground-truth positions per site
WIDTH, HEIGHT = 3600, 400   # 0.1 deg/pixel panorama

SITES = {
    'A-strait':   dict(lat=36.95, lon=27.25, sector_center_deg=180.0),
    'B-offshore': dict(lat=36.60, lon=26.85, sector_center_deg=60.0),
}

rng = np.random.default_rng(20260814)
results = {}

for name, site in SITES.items():
    print(f'== site {name} ({site["lat"]}, {site["lon"]}) ==')
    lat_c, lon_c = site['lat'], site['lon']
    mlat, mlon = S.meters_per_degree(lat_c)
    gl = S.GlSkyline(lat_c, lon_c, width=WIDTH, height=HEIGHT,
                     render_radius_m=45000., dir_dems=DIR_DEMS)

    def render_at(dn, de):
        az, el, r = gl.skyline(lat_c + dn / mlat, lon_c + de / mlon, Z)
        return az, S.seahorizon_fill(el, Z).astype(np.float32), r

    # ---- precompute candidate skylines on the lattice over the box
    # (persisted to disk: reruns and replots are cheap)
    n2 = int(BOX / 2 / LATTICE)             # 20 -> 41x41 lattice
    idx = np.arange(-n2, n2 + 1)
    lattice_file = os.path.join(OUT, f'e1_{name}_lattice.npz')
    t0 = time.time()
    if os.path.exists(lattice_file):
        el_lattice = np.load(lattice_file)['el']  # (Nlat, Nlat, WIDTH)
        print(f'  loaded lattice skylines from {lattice_file}')
    else:
        el_lattice = np.empty((idx.size, idx.size, WIDTH), dtype=np.float32)
        for a, i in enumerate(idx):          # dn
            for b, j in enumerate(idx):      # de
                _, el, _ = render_at(i * LATTICE, j * LATTICE)
                el_lattice[a, b] = el
        dt = time.time() - t0
        np.savez_compressed(lattice_file, el=el_lattice)
        print(f'  precomputed {idx.size**2} lattice skylines in {dt:.0f}s '
              f'({dt/idx.size**2*1e3:.0f} ms/render)')
    cache = {(i, j): el_lattice[a, b]
             for a, i in enumerate(idx) for b, j in enumerate(idx)}
    dt = time.time() - t0

    def lattice_skyline(dn, de):
        # clamp to the box: the fine grid may probe past the lattice edge
        # when the coarse minimum lands on the boundary
        i = int(np.clip(round(dn / LATTICE), -n2, n2))
        j = int(np.clip(round(de / LATTICE), -n2, n2))
        return cache[(i, j)]

    # ---- observation info at the box center, for weighting/reporting
    az, el_c, r_c = render_at(0, 0)
    r_land = r_c[np.isfinite(r_c)]
    mean_range = float(np.mean(r_land)) if r_land.size else np.nan
    land_frac = float(np.mean(np.isfinite(r_c)))
    print(f'  land in {land_frac*100:.0f}% of azimuths, '
          f'mean skyline range {mean_range/1e3:.1f} km')

    # ---- FOV masks
    def sector_mask(center_deg, fov_deg):
        d = (az - center_deg + 180.0) % 360.0 - 180.0
        return (np.abs(d) <= fov_deg / 2).astype(float)

    configs = {
        'clean/360': dict(noise=0.0, bias_deg=0.0, weights=None),
        'clean/90':  dict(noise=0.0, bias_deg=0.0,
                          weights=sector_mask(site['sector_center_deg'], 90.0)),
        'noise/360': dict(noise=1e-3, bias_deg=0.0, weights=None),
        'bias/360':  dict(noise=0.0, bias_deg=0.2, weights=None),
    }

    # ---- ground truth + solve
    gt = rng.uniform(-BOX / 2, BOX / 2, size=(NGT, 2))  # (dn, de) meters
    site_res = {k: [] for k in configs}
    for g in gt:
        _, el_obs_clean, _ = render_at(g[0], g[1])
        for cname, cfg in configs.items():
            el_obs = el_obs_clean.copy()
            if cfg['bias_deg']:
                el_obs = np.roll(el_obs, int(round(cfg['bias_deg'] /
                                                   (360.0 / WIDTH))))
            if cfg['noise']:
                el_obs = el_obs + rng.normal(0, cfg['noise'], el_obs.shape)
            dn, de, info = S.solve_position(lattice_skyline, el_obs, Z,
                                            box_m=BOX, coarse_n=9,
                                            fine_step_m=LATTICE,
                                            weights=cfg['weights'])
            site_res[cname].append([g[0], g[1], dn, de])

    # ---- stats
    print(f'  {"config":<10} {"median":>7} {"CEP50":>7} {"CEP95":>7} {"max":>7}  (m)')
    stats = {}
    for cname, rows in site_res.items():
        rows = np.array(rows)
        err = np.hypot(rows[:, 2] - rows[:, 0], rows[:, 3] - rows[:, 1])
        stats[cname] = dict(err=err.tolist(),
                            cep50=float(np.percentile(err, 50)),
                            cep95=float(np.percentile(err, 95)),
                            max=float(err.max()))
        print(f'  {cname:<10} {np.median(err):7.1f} '
              f'{stats[cname]["cep50"]:7.1f} {stats[cname]["cep95"]:7.1f} '
              f'{stats[cname]["max"]:7.1f}')

    # ---- full cost surface for the first GT position (from the cache)
    _, el_obs0, _ = render_at(gt[0, 0], gt[0, 1])
    csurf = np.array([[S.cost(el_obs0, cache[(i, j)]) for j in idx]
                      for i in idx])
    csurf90 = np.array([[S.cost(el_obs0, cache[(i, j)],
                                weights=configs['clean/90']['weights'])
                         for j in idx] for i in idx])

    results[name] = dict(site=site, gt=gt.tolist(), stats=stats,
                         res={k: np.array(v).tolist() for k, v in site_res.items()},
                         mean_range_m=mean_range, land_frac=land_frac,
                         ms_per_render=dt / len(cache) * 1e3)
    np.savez_compressed(os.path.join(OUT, f'e1_{name}.npz'),
                        idx=idx, csurf=csurf, csurf90=csurf90,
                        az=az, el_center=el_c, r_center=r_c,
                        el_obs0=el_obs0, gt=gt,
                        **{f'res_{k.replace("/", "_")}': np.array(v)
                           for k, v in site_res.items()})

with open(os.path.join(OUT, 'e1_results.json'), 'w') as f:
    json.dump(results, f, indent=1)
print('wrote', os.path.join(OUT, 'e1_results.json'))
