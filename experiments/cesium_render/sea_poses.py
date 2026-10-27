#!/usr/bin/env python3
"""Put each pose's camera on the water, Z m above mean sea level.

The Google tiles are a surface model: trees and buildings on a shore stand
above a camera that the EXIF puts 2-4 m above the sea, and a render from there
is from inside them. Every capture here is at sea level instead: the camera
moves to the nearest water point found by `driver.mjs --water` (same attitude)
and sits Z m (default 2.5) above it. Poses with no water nearby are dropped.

    python3 sea_poses.py POSES.json WATER_DIR OUT.json [--z 2.5]
"""

import json
import math
import os
import sys


def main():
    src, wdir, out = sys.argv[1:4]
    z = float(sys.argv[sys.argv.index('--z') + 1]) if '--z' in sys.argv else 2.5
    kept, dropped = [], []
    for p in json.load(open(src)):
        w = os.path.join(wdir, p['tag'] + '_water.json')
        water = json.load(open(w))['water'] if os.path.exists(w) else None
        if water is None:
            dropped.append(p['tag'])
            continue
        q = dict(p)
        q.update(lat=water['lat'], lon=water['lon'], z_msl=z, h_ellipsoid=z + p['geoid_n'],
                 truth=dict(lat=p['lat'], lon=p['lon'], z_msl=p['z_msl']),
                 moved_m=float(water['r']), moved_az_deg=float(water['az']))
        kept.append(q)
    json.dump(kept, open(out, 'w'), indent=1)
    print('%d poses on the water at %.1f m MSL; moved median %.0f m, max %.0f m; dropped (no water): %s'
          % (len(kept), z, sorted(k['moved_m'] for k in kept)[len(kept) // 2], max(k['moved_m'] for k in kept),
             ', '.join(dropped) or 'none'))


if __name__ == '__main__':
    main()
