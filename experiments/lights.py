#!/usr/bin/env python3
"""Charted-light database for the night channel (E5f/E5g).

Sources, in order of preference:

  1. OSM seamark data via Overpass (`fetch_overpass`) — free, global,
     carries position + flash character for lighthouses, sector lights
     and lit buoys.  Blocked from this dev container's egress; run it
     on a machine with normal network access and commit the JSON.
  2. A JSON file previously fetched (or hand-built from a List of
     Lights) loaded with `LightDB(path)`.
  3. `demo_db()` — the three SYNTHETIC stand-in lights of the E5f
     passage with assigned characters, for end-to-end tests only.
     They are NOT real aids to navigation.

Character model (the navigational "light character", e.g. Fl(3)W.15s):

    dict(pattern, group, period_s, colour)

    pattern: 'F' fixed | 'Fl' flashing | 'LFl' long-flash |
             'Oc' occulting | 'Iso' isophase | 'Q' quick
    group:   flashes per cycle (1 when ungrouped)
    period_s: full cycle length in seconds (None for F/Q continuous)
    colour:  'W' | 'R' | 'G' (default 'W')

`match` compares a classifier output (lightscan.classify_trace) against
the database within a radius gate: pattern class must agree, group must
agree, period within a relative tolerance.  Identification is what
turns an anonymous blinking dot into a surveyed position.
"""

import os
import json
import math
import urllib.parse
import urllib.request

OVERPASS = 'https://overpass-api.de/api/interpreter'
QUERY = """
[out:json][timeout:120];
(
  node["seamark:light:character"]({s},{w},{n},{e});
  node["seamark:light:1:character"]({s},{w},{n},{e});
);
out body;
"""


def parse_character(tags, prefix='seamark:light:'):
    """Character dict from OSM seamark tags (or the sectored variant
    'seamark:light:1:...'). Returns None when no character is tagged."""
    for p in (prefix, 'seamark:light:1:'):
        char = tags.get(p + 'character')
        if char:
            group = tags.get(p + 'group')
            period = tags.get(p + 'period')
            colour = (tags.get(p + 'colour', 'white') or 'white')[0].upper()
            try:
                g = int(str(group).split(';')[0]) if group else 1
            except ValueError:
                g = 1
            try:
                per = float(period) if period else None
            except ValueError:
                per = None
            return dict(pattern=char, group=g, period_s=per, colour=colour)
    return None


def parse_character_string(s):
    """Compact chart notation, e.g. 'Fl(3)W.15s', 'Iso.4s', 'Oc(2)R.10s'."""
    import re
    m = re.match(r'^(LFl|Fl|Oc|Iso|Q|F)(?:\((\d+)\))?\.?([WRG])?\.?'
                 r'(?:(\d+(?:\.\d+)?)s)?$', s.strip())
    if not m:
        return None
    return dict(pattern=m.group(1), group=int(m.group(2) or 1),
                period_s=float(m.group(4)) if m.group(4) else None,
                colour=m.group(3) or 'W')


def fetch_overpass(south, west, north, east, out_path,
                   endpoint=OVERPASS):
    """Fetch all charted lights in the bbox from OSM and write the
    LightDB JSON. Needs open network access (Overpass is blocked from
    the dev container's egress)."""
    q = QUERY.format(s=south, w=west, n=north, e=east)
    req = urllib.request.Request(
        endpoint, data=urllib.parse.urlencode({'data': q}).encode(),
        headers={'User-Agent': 'horizonator-lights/1.0'})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    lights, seen = [], set()
    for el in data.get('elements', []):
        if el.get('type') != 'node' or el['id'] in seen:
            continue
        seen.add(el['id'])
        char = parse_character(el.get('tags', {}))
        if char is None:
            continue
        t = el['tags']
        rng = t.get('seamark:light:range') or t.get('seamark:light:1:range')
        hgt = t.get('seamark:light:height') \
            or t.get('seamark:light:1:height')
        lights.append(dict(
            id=f"osm:{el['id']}",
            name=t.get('seamark:name', t.get('name', '')),
            lat=el['lat'], lon=el['lon'], char=char,
            range_nm=float(rng) if rng else None,
            height_m=float(hgt) if hgt else None,
            synthetic=False))
    with open(out_path, 'w') as f:
        json.dump(dict(source=f'overpass {south},{west},{north},{east}',
                       lights=lights), f, indent=1)
    return len(lights)


def demo_db():
    """The E5f stand-in lights with assigned characters — SYNTHETIC,
    for tests only (positions are the E5f passage geometry)."""
    mk = parse_character_string
    return LightDB(lights=[
        dict(id='demo:1', name='West Point (synthetic)',
             lat=36.980, lon=27.170, char=mk('Fl(3)W.15s'),
             range_nm=10.0, height_m=30.0, synthetic=True),
        dict(id='demo:2', name='Mid Shoal (synthetic)',
             lat=36.890, lon=27.300, char=mk('Fl.W.5s'),
             range_nm=10.0, height_m=12.0, synthetic=True),
        dict(id='demo:3', name='East Head (synthetic)',
             lat=37.020, lon=27.430, char=mk('Iso.W.4s'),
             range_nm=10.0, height_m=25.0, synthetic=True),
    ])


class LightDB:
    def __init__(self, path=None, lights=None):
        if path is not None:
            with open(path) as f:
                lights = json.load(f)['lights']
        self.lights = lights or []

    def __len__(self):
        return len(self.lights)

    def near(self, lat, lon, radius_m):
        mlat = 111132.0
        mlon = 111320.0 * math.cos(math.radians(lat))
        out = []
        for L in self.lights:
            d = math.hypot((L['lat'] - lat) * mlat, (L['lon'] - lon) * mlon)
            if d <= radius_m:
                out.append((d, L))
        return [L for _, L in sorted(out, key=lambda x: x[0])]

    def match(self, char, lat, lon, radius_m, period_tol=0.15):
        """Lights within radius whose charted character matches a
        classified one. Unique match = identification; zero or several
        = leave the detection unused (never guess a landmark)."""
        if char is None:
            return []
        out = []
        for L in self.near(lat, lon, radius_m):
            c = L['char']
            if c['pattern'] != char['pattern']:
                continue
            if char.get('colour') and c.get('colour') \
                    and char['colour'] != c['colour']:
                continue
            if c.get('group', 1) != char.get('group', 1):
                continue
            pa, pb = c.get('period_s'), char.get('period_s')
            if pa and pb and abs(pa - pb) > period_tol * pa:
                continue
            out.append(L)
        return out
