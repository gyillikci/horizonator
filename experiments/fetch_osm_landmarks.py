#!/usr/bin/env python3
"""Fetch the whole charted-landmark web for one region, in one command.

Runs all four Overpass fetchers — navigation lights (E5f/g), wind
turbines (E5h), transmission pylons and communication masts (E5i) —
over a bounding box and writes the JSON databases the night and day
landmark channels consume.

Overpass is blocked from the dev container's egress, so this is meant
to run on a machine with normal network access; commit the resulting
JSONs into experiments/data/ and the channels work offline forever
after (the databases are small and static).

Examples:
    # the Gulf of Gokova / Bodrum peninsula
    python3 fetch_osm_landmarks.py --bbox 36.7 27.0 37.2 27.9

    # a named preset
    python3 fetch_osm_landmarks.py --region bodrum

    # see what would be fetched, no network
    python3 fetch_osm_landmarks.py --region bodrum --dry-run

Overpass etiquette: the public endpoint is a shared free service.
Fetch a region once, keep the JSON, and use --endpoint to point at a
mirror or your own instance for large or repeated pulls.
"""

import os
import sys
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')

# south, west, north, east
REGIONS = {
    'bodrum': (36.70, 27.00, 37.20, 27.90),
    'gokova': (36.80, 27.90, 37.20, 28.60),
    'bafa': (37.40, 27.30, 37.60, 27.70),
    'izmir': (38.20, 26.20, 38.75, 27.30),
    'canakkale': (39.95, 26.00, 40.45, 26.90),
    'istanbul-bosphorus': (40.95, 28.90, 41.30, 29.30),
}

SOURCES = [
    ('lights', 'navigation lights (night channel)'),
    ('turbines', 'wind turbines (day landmark web)'),
    ('pylons', 'transmission pylons'),
    ('masts', 'communication masts (day + night via obstruction lights)'),
]


def run(bbox, out_dir, which, endpoint, dry_run=False):
    s, w, n, e = bbox
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for key, label in SOURCES:
        if which != 'all' and which != key:
            continue
        path = os.path.join(out_dir, f'{key}.json')
        print(f'{key:9s} {label}')
        print(f'          bbox {s},{w},{n},{e} -> {path}')
        if dry_run:
            results[key] = None
            continue
        if key == 'lights':
            from lights import fetch_overpass
            cnt = fetch_overpass(s, w, n, e, path, endpoint=endpoint)
        elif key == 'turbines':
            from turbines import fetch_overpass
            cnt = fetch_overpass(s, w, n, e, path, endpoint=endpoint)
        elif key == 'pylons':
            from landmarks import fetch_pylons
            cnt = fetch_pylons(s, w, n, e, path, endpoint=endpoint)
        else:
            from landmarks import fetch_masts
            cnt = fetch_masts(s, w, n, e, path, endpoint=endpoint)
        print(f'          {cnt} features')
        results[key] = cnt
    return results


def main():
    ap = argparse.ArgumentParser(
        description='Fetch OSM charted landmarks (lights, turbines, '
                    'pylons, masts) for a region.')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--bbox', nargs=4, type=float,
                   metavar=('SOUTH', 'WEST', 'NORTH', 'EAST'))
    g.add_argument('--region', choices=sorted(REGIONS))
    ap.add_argument('--out', default=DATA,
                    help='output directory (default experiments/data)')
    ap.add_argument('--which', default='all',
                    choices=['all'] + [k for k, _ in SOURCES])
    ap.add_argument('--endpoint',
                    default='https://overpass-api.de/api/interpreter',
                    help='Overpass endpoint (use a mirror for big pulls)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the plan without touching the network')
    a = ap.parse_args()
    bbox = tuple(a.bbox) if a.bbox else REGIONS[a.region]
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        ap.error('bbox must be SOUTH WEST NORTH EAST with S<N and W<E')
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area > 4.0 and not a.dry_run:
        print(f'note: {area:.1f} sq-deg is a large Overpass query; '
              f'consider splitting it or using --endpoint mirror',
              file=sys.stderr)
    res = run(bbox, a.out, a.which, a.endpoint, a.dry_run)
    if not a.dry_run:
        idx = os.path.join(a.out, 'index.json')
        with open(idx, 'w') as f:
            json.dump(dict(bbox=list(bbox), counts=res), f, indent=1)
        print(f'\nwrote {idx}')
        print('commit experiments/data/*.json — the channels then run '
              'offline')


if __name__ == '__main__':
    main()
