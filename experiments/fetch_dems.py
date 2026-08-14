#!/usr/bin/env python3
"""Fetch SRTM tiles for the E0/E1 test area (Bodrum/Kos, Aegean) from the AWS
elevation-tiles-prod mirror, and decimate the 1" tiles to 3" (the
horizonator's default resolution).

Note: these tiles contain ocean bathymetry (negative elevations). The
horizonator and experiments/skyline.py both clamp elevations < 0 to sea
level, so this is handled.

Usage: python3 fetch_dems.py [outdir]     (default ~/.horizonator)
"""

import os
import sys
import gzip
import urllib.request
import numpy as np

TILES = ['N36E026', 'N36E027', 'N37E026', 'N37E027']
URL = 'https://s3.amazonaws.com/elevation-tiles-prod/skadi/{band}/{tile}.hgt.gz'

base = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/.horizonator')
d1 = os.path.join(base, 'DEMs_SRTM1')
d3 = os.path.join(base, 'DEMs_SRTM3')
os.makedirs(d1, exist_ok=True)
os.makedirs(d3, exist_ok=True)

for tile in TILES:
    p1 = os.path.join(d1, tile + '.hgt')
    if not os.path.exists(p1):
        url = URL.format(band=tile[:3], tile=tile)
        print('fetching', url)
        with urllib.request.urlopen(url) as r:
            with open(p1, 'wb') as f:
                f.write(gzip.decompress(r.read()))
    p3 = os.path.join(d3, tile + '.hgt')
    if not os.path.exists(p3):
        a = np.fromfile(p1, dtype='>i2').reshape(3601, 3601)
        a[::3, ::3].astype('>i2').tofile(p3)   # stride-3 decimation
        print('decimated ->', p3)
print('done')
