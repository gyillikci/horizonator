#!/usr/bin/env python3
"""Plan boat routes that stay on the water and keep land in view, from COP30.

A route is a random walk of 1 km legs from a sea point near a region's centre: each leg
keeps 400 m of water to every side, and looks at land 3-20 km ahead (+-35 deg) standing at
least 50 m, so every frame has a skyline to match. Regions sample the Aegean's coast types.

    python3 plan_routes.py [--legs 8] [--seed 1]  ->  routes/<name>.json
"""
import argparse, json, math, os, sys
import numpy as np
EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP)
import skyline as S

REGIONS = {   # name: (lat, lon, coast type)
    'bodrum': (37.00, 27.38, 'rocky peninsula, Kos across the strait'),
    'gokova': (36.97, 28.05, 'deep gulf between mountain ranges'),
    'datca': (36.70, 27.85, 'narrow peninsula, Symi and Rhodes beyond'),
    'kusadasi': (37.83, 27.15, 'bay facing Samos'),
    'cesme': (38.35, 26.20, 'open strait, Chios across'),
    'ayvalik': (39.28, 26.62, 'low islands archipelago'),
    'edremit': (39.50, 26.75, 'gulf under Kaz Dagi'),
}
dem = S.Dem(os.path.expanduser('~/.horizonator/DEMs_COP30'))


def offset(lat, lon, d, hdg):
    mlat, mlon = S.meters_per_degree(lat)
    h = math.radians(hdg)
    return lat + d * math.cos(h) / mlat, lon + d * math.sin(h) / mlon


def wet_with_clearance(lat, lon, clear=400.0):
    pts = [(lat, lon)] + [offset(lat, lon, clear, a) for a in range(0, 360, 30)]
    h = dem.sample([p[0] for p in pts], [p[1] for p in pts])
    return bool(np.all(h <= 0.5))


def land_ahead(lat, lon, hdg, rmin=3000, rmax=20000, half=35, hmin=50):
    for a in np.linspace(-half, half, 15):
        rs = np.arange(rmin, rmax, 500)
        pts = [offset(lat, lon, r, hdg + a) for r in rs]
        if np.max(dem.sample([p[0] for p in pts], [p[1] for p in pts])) >= hmin:
            return True
    return False


def geoid_n(lat, lon):
    # EGM96 separation (as cesium_render/make_poses.py); the grid comes over PROJ's network
    import pyproj
    pyproj.network.set_network_enabled(True)
    t = pyproj.Transformer.from_crs('EPSG:4326+5773', 'EPSG:4979', always_xy=True)
    n = t.transform(lon, lat, 0.0)[2]
    if not math.isfinite(n) or abs(n) > 150 or n == 0.0:
        raise RuntimeError('no geoid separation at %.5f %.5f' % (lat, lon))
    return float(n)


def plan(name, lat0, lon0, legs, rng):
    # the nearest sea point with clearance, on rings around the centre
    start = None
    for r in range(0, 15000, 250):
        for a in range(0, 360, 15):
            p = offset(lat0, lon0, r, a)
            if wet_with_clearance(*p, 600):
                start = p; break
        if start:
            break
    if not start:
        return None
    best = None
    for attempt in range(60):
        hdg = rng.uniform(0, 360)
        while not land_ahead(*start, hdg):
            hdg = rng.uniform(0, 360)
        wps, cur, h = [start], start, hdg
        for _ in range(legs):
            ok = []
            for dh in rng.permutation(np.arange(-50, 51, 10)):
                nh = (h + dh) % 360
                nxt = offset(*cur, 1000, nh)
                mid = offset(*cur, 500, nh)
                if wet_with_clearance(*nxt) and wet_with_clearance(*mid) and land_ahead(*nxt, nh):
                    ok.append((nh, nxt))
                    break
            if not ok:
                break
            h, cur = ok[0]
            wps.append(cur)
        if best is None or len(wps) > len(best[0]):
            best = (wps, hdg)
        if len(wps) == legs + 1:
            break
    wps, hdg = best
    return dict(name=name, coast=REGIONS[name][2], lat=wps[0][0], lon=wps[0][1], heading=hdg,
                n=geoid_n(*wps[0]), waypoints=[list(map(float, p)) for p in wps[1:]], legs=len(wps) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legs', type=int, default=8); ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs('routes', exist_ok=True)
    for name, (la, lo, _) in REGIONS.items():
        r = plan(name, la, lo, a.legs, rng)
        if not r:
            print('%-9s no sea start found' % name); continue
        json.dump(r, open(os.path.join('routes', name + '.json'), 'w'), indent=1)
        print('%-9s start %.4f,%.4f  N %.1f m  %d legs  (%s)' % (name, r['lat'], r['lon'], r['n'], r['legs'], r['coast']))


if __name__ == '__main__':
    main()
