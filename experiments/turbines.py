#!/usr/bin/env python3
"""Wind turbines as charted daytime landmarks (E5h).

A wind turbine is a surveyed point (OSM: power=generator +
generator:source=wind, usually with hub height), 80-150 m tall, white,
on ridgelines and capes — visible for 10-20 km in daylight, i.e. a
daytime equivalent of the night channel's lighthouses. Three uses:

  presence filter   a scene WITH turbines excludes every candidate
                    position that has none in view (and vice versa:
                    a bare horizon excludes wind-farm coastline)
  constellation fix turbines in a farm are individually anonymous,
                    but the PATTERN of bearings to a farm from a
                    viewpoint is distinctive; a 1-D Hough over the
                    unknown compass offset aligns measured bearings
                    with charted ones (unknown correspondence AND
                    unknown compass bias solved together)
  bearing factors   once matched, each turbine bearing enters the
                    graph through the same shared-compass-bias factor
                    the lights use (skyline_factor.light_bearing_factor)

The Overpass fetcher mirrors lights.fetch_overpass (blocked from this
container's egress — run user-side and commit the JSON). demo_farm()
is a SYNTHETIC stand-in ridge farm for tests.
"""

import json
import math
import urllib.parse
import urllib.request

import numpy as np

OVERPASS = 'https://overpass-api.de/api/interpreter'
QUERY = """
[out:json][timeout:120];
(
  node["power"="generator"]["generator:source"="wind"]({s},{w},{n},{e});
  node["man_made"="windmill"]({s},{w},{n},{e});
);
out body;
"""


def fetch_overpass(south, west, north, east, out_path,
                   endpoint=OVERPASS):
    """All charted wind turbines in the bbox -> TurbineDB JSON."""
    q = QUERY.format(s=south, w=west, n=north, e=east)
    req = urllib.request.Request(
        endpoint, data=urllib.parse.urlencode({'data': q}).encode(),
        headers={'User-Agent': 'horizonator-turbines/1.0'})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    tb, seen = [], set()
    for el in data.get('elements', []):
        if el.get('type') != 'node' or el['id'] in seen:
            continue
        seen.add(el['id'])
        t = el.get('tags', {})
        h = t.get('height:hub') or t.get('generator:height') \
            or t.get('height')
        try:
            h = float(str(h).replace(' m', '')) if h else None
        except ValueError:
            h = None
        tb.append(dict(id=f"osm:{el['id']}", lat=el['lat'],
                       lon=el['lon'], hub_m=h, synthetic=False))
    with open(out_path, 'w') as f:
        json.dump(dict(source=f'overpass {south},{west},{north},{east}',
                       turbines=tb), f, indent=1)
    return len(tb)


def demo_farm(lat0=36.985, lon0=27.395, n=8, spacing_m=420.0,
              bearing_deg=195.0):
    """SYNTHETIC ridge farm: n turbines in a line (typical crest
    layout), for tests only."""
    mlat = 111132.0
    mlon = 111320.0 * math.cos(math.radians(lat0))
    b = math.radians(bearing_deg)
    return TurbineDB(turbines=[
        dict(id=f'demo:t{k}',
             lat=lat0 + k * spacing_m * math.cos(b) / mlat,
             lon=lon0 + k * spacing_m * math.sin(b) / mlon,
             hub_m=95.0, synthetic=True)
        for k in range(n)])


class TurbineDB:
    def __init__(self, path=None, turbines=None):
        if path is not None:
            with open(path) as f:
                turbines = json.load(f)['turbines']
        self.turbines = turbines or []

    def __len__(self):
        return len(self.turbines)

    def enu(self, lat, lon):
        """(east, north) meters of every turbine relative to (lat, lon)."""
        mlat = 111132.0
        mlon = 111320.0 * math.cos(math.radians(lat))
        return np.array([[(t['lon'] - lon) * mlon, (t['lat'] - lat) * mlat]
                         for t in self.turbines]).reshape(-1, 2)

    def bearings_from(self, lat, lon, max_m):
        """Predicted compass bearings (rad) and indices of turbines
        within max_m of the viewpoint."""
        en = self.enu(lat, lon)
        r = np.hypot(en[:, 0], en[:, 1])
        keep = np.where(r < max_m)[0]
        return np.arctan2(en[keep, 0], en[keep, 1]), keep


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def hough_align(meas, pred, tol_rad):
    """Align measured bearings with predicted ones under an UNKNOWN
    common offset (compass bias): every (meas, pred) pair votes an
    offset; the offset explaining the most measurements wins.
    Returns (offset, n_inliers, rms_rad, pairs) with unique greedy
    assignment at the winning offset."""
    meas = np.asarray(meas, float)
    pred = np.asarray(pred, float)
    if meas.size == 0 or pred.size == 0:
        return 0.0, 0, np.inf, []
    votes = wrap(meas[:, None] - pred[None, :]).ravel()
    if votes.size > 24:
        # coarse Hough: keep only offsets in the most-voted 1-deg bins
        h, edges = np.histogram(votes, bins=360, range=(-np.pi, np.pi))
        top = np.argsort(h)[::-1][:4]
        keep = np.zeros(votes.size, bool)
        for b in top:
            if h[b]:
                keep |= (votes >= edges[b]) & (votes < edges[b + 1])
        votes = votes[keep]
    best = (0, np.inf, 0.0, [])          # (inliers, rms, offset, pairs)
    for off in votes:
        d = np.abs(wrap(meas[:, None] - pred[None, :] - off))
        pairs, used_p = [], set()
        for i in np.argsort(d.min(axis=1)):
            for j in np.argsort(d[i]):
                if d[i, j] > tol_rad:
                    break
                if j not in used_p:
                    pairs.append((int(i), int(j)))
                    used_p.add(int(j))
                    break
        if pairs:
            res = np.array([d[i, j] for i, j in pairs])
            rms = float(np.sqrt((res ** 2).mean()))
            if (len(pairs), -rms) > (best[0], -best[1]):
                off_ref = off + np.mean(wrap(
                    meas[[i for i, _ in pairs]]
                    - pred[[j for _, j in pairs]] - off))
                best = (len(pairs), rms, float(off_ref), pairs)
    return best[2], best[0], best[1], best[3]


def constellation_score(db, meas, lat, lon, max_m, tol_rad):
    """Score a candidate viewpoint against a measured bearing set.
    Presence filter included: no turbines in view -> -1 (impossible
    when the scene shows turbines). Higher = better."""
    pred, _ = db.bearings_from(lat, lon, max_m)
    if pred.size == 0:
        return -1.0, 0.0
    off, n_in, rms, _ = hough_align(meas, pred, tol_rad)
    if n_in == 0:
        return 0.0, 0.0
    return n_in - rms / tol_rad, off
