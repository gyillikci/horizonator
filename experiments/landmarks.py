#!/usr/bin/env python3
"""Power-line pylons and communication masts as charted landmarks
(E5i) — the third and fourth point-landmark classes after lighthouses
(E5f/g) and wind turbines (E5h).

Both are surveyed in OSM and both crown exactly the terrain the
camera looks at:

  pylons   power=tower nodes — transmission towers march in charted
           rows over ridges and straits; each row is a picket fence
           of anonymous points (the E5h Hough + class gates handle
           picket fences)
  masts    man_made=mast / tower / communications_tower with a
           communication tower:type — isolated summit points, taller
           than pylons, and usually carrying RED AVIATION OBSTRUCTION
           LIGHTS, which makes a charted mast a NIGHT landmark too:
           `as_light_entries` exports lit masts into the E5g
           lights.LightDB so the flash classifier can identify them
           in the dark alongside the sea lights.

All point sets load into turbines.TurbineDB (it is class-agnostic —
each entry may carry a 'cls' key: 'turbine' | 'pylon' | 'mast') and
run through the same constellation machinery with class-aware
assignment. Fetchers mirror lights.fetch_overpass (blocked from this
container's egress; run user-side, commit the JSON). demo_* builders
are SYNTHETIC stand-ins for tests.
"""

import json
import math
import urllib.parse
import urllib.request

from turbines import TurbineDB
from lights import parse_character_string

OVERPASS = 'https://overpass-api.de/api/interpreter'
QUERY_PYLONS = """
[out:json][timeout:120];
node["power"="tower"]({s},{w},{n},{e});
out body;
"""
QUERY_MASTS = """
[out:json][timeout:120];
(
  node["man_made"="communications_tower"]({s},{w},{n},{e});
  node["man_made"~"^(mast|tower)$"]
      ["tower:type"~"communication"]({s},{w},{n},{e});
);
out body;
"""


def _fetch(query, south, west, north, east, out_path, cls,
           endpoint=OVERPASS):
    q = query.format(s=south, w=west, n=north, e=east)
    req = urllib.request.Request(
        endpoint, data=urllib.parse.urlencode({'data': q}).encode(),
        headers={'User-Agent': 'horizonator-landmarks/1.0'})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    pts, seen = [], set()
    for el in data.get('elements', []):
        if el.get('type') != 'node' or el['id'] in seen:
            continue
        seen.add(el['id'])
        t = el.get('tags', {})
        h = t.get('height')
        try:
            h = float(str(h).replace(' m', '')) if h else None
        except ValueError:
            h = None
        pts.append(dict(id=f"osm:{el['id']}", lat=el['lat'],
                        lon=el['lon'], hub_m=h, cls=cls,
                        synthetic=False))
    with open(out_path, 'w') as f:
        json.dump(dict(source=f'overpass {cls} '
                              f'{south},{west},{north},{east}',
                       turbines=pts), f, indent=1)
    return len(pts)


def fetch_pylons(south, west, north, east, out_path,
                 endpoint=OVERPASS):
    """All transmission towers in the bbox -> TurbineDB JSON."""
    return _fetch(QUERY_PYLONS, south, west, north, east, out_path,
                  'pylon', endpoint)


def fetch_masts(south, west, north, east, out_path,
                endpoint=OVERPASS):
    """All communication masts/towers in the bbox -> TurbineDB JSON."""
    return _fetch(QUERY_MASTS, south, west, north, east, out_path,
                  'mast', endpoint)


def demo_pylon_line(lat0=36.925, lon0=27.205, n=10, spacing_m=350.0,
                    bearing_deg=68.0):
    """SYNTHETIC transmission line marching over the coastal hills —
    for tests only."""
    mlat = 111132.0
    mlon = 111320.0 * math.cos(math.radians(lat0))
    b = math.radians(bearing_deg)
    return [dict(id=f'demo:p{k}',
                 lat=lat0 + k * spacing_m * math.cos(b) / mlat,
                 lon=lon0 + k * spacing_m * math.sin(b) / mlon,
                 hub_m=42.0, cls='pylon', synthetic=True)
            for k in range(n)]


def demo_masts():
    """SYNTHETIC summit communication masts, red obstruction lights
    (Fl.R.1.5s aviation character) — for tests only."""
    return [dict(id='demo:m1', lat=36.940, lon=27.215, hub_m=70.0,
                 cls='mast', light='Fl.R.1.5s', synthetic=True),
            dict(id='demo:m2', lat=37.030, lon=27.330, hub_m=55.0,
                 cls='mast', light='Fl.R.1.5s', synthetic=True)]


def as_light_entries(masts):
    """Lit masts -> lights.LightDB rows: the night crossover. A mast's
    red obstruction light has a charted position and a (generic but
    real) flash character; within a radius gate it identifies exactly
    like a sea light."""
    out = []
    for m in masts:
        if not m.get('light'):
            continue
        out.append(dict(id=m['id'] + ':light', name=f"mast {m['id']}",
                        lat=m['lat'], lon=m['lon'],
                        char=parse_character_string(m['light']),
                        range_nm=8.0, height_m=m.get('hub_m'),
                        synthetic=m.get('synthetic', True)))
    return out


def combined_db(*groups):
    """One TurbineDB over any mix of point-landmark lists."""
    pts = []
    for g in groups:
        pts.extend(g.turbines if isinstance(g, TurbineDB) else g)
    return TurbineDB(turbines=pts)
