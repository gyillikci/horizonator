#!/usr/bin/env python3
"""Build an SRTM1-compatible .hgt store from Copernicus DEM GLO-30.

Copernicus GLO-30 is a 30 m global DSM derived from TanDEM-X. Two
things make it worth testing against the SRTM1 store this campaign has
used throughout: it is a genuine SURFACE model (so it carries the
canopy that E5aw showed explains about half our crest deficit), and it
is far better behaved on coastlines and small islands, where SRTM's
water masking is a known weakness -- which is exactly the geometry of
the Bodrum frames.

The grids align exactly. A GLO-30 tile is 3600x3600 pixel-is-area with
its upper-left corner half a pixel outside the integer degree, so pixel
centre (r, c) sits at lat = lat1 - r/3600, lon = lon0 + c/3600 -- the
same integer-arcsecond lattice an SRTM1 .hgt samples. The .hgt simply
carries one extra row and column, which belong to the neighbouring
tiles; those are taken from the neighbours when present and edge-
replicated when not (one 30 m row at a tile seam).

Heights are referenced to EGM2008 rather than SRTM's EGM96. The
difference over the Aegean is well under a metre and is NOT corrected
here; it would matter for absolute altimetry, not for the silhouette
shape this instrument matches.

    python3 copernicus_to_hgt.py --bbox 36 27 38 29 --out ~/.horizonator/DEMs_COP30
"""

import argparse
import os
import subprocess
import sys

import numpy as np

BUCKET = 'https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com'


def tile_name(lat, lon):
    return ('Copernicus_DSM_COG_10_%s%02d_00_%s%03d_00_DEM'
            % ('N' if lat >= 0 else 'S', abs(lat),
               'E' if lon >= 0 else 'W', abs(lon)))


def hgt_name(lat, lon):
    return ('%s%02d%s%03d.hgt'
            % ('N' if lat >= 0 else 'S', abs(lat),
               'E' if lon >= 0 else 'W', abs(lon)))


def fetch(lat, lon, cache):
    """Download one GLO-30 COG; returns the local path or None if the
    tile does not exist (all-ocean cells are simply absent)."""
    t = tile_name(lat, lon)
    path = os.path.join(cache, t + '.tif')
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    os.makedirs(cache, exist_ok=True)
    url = '%s/%s/%s.tif' % (BUCKET, t, t)
    r = subprocess.run(['curl', '-sS', '-f', '-m', '600', '-o', path, url])
    if r.returncode != 0:
        if os.path.exists(path):
            os.remove(path)
        return None
    return path


def read(path):
    import rasterio
    with rasterio.open(path) as d:
        a = d.read(1).astype(np.float32)
    if a.shape != (3600, 3600):
        raise SystemExit('unexpected GLO-30 shape %s in %s' % (a.shape, path))
    return a


def build(lat, lon, cache):
    """One 3601x3601 SRTM1-shaped array for the degree cell (lat, lon)."""
    p = fetch(lat, lon, cache)
    if p is None:
        return None
    a = read(p)
    out = np.zeros((3601, 3601), np.float32)
    out[:3600, :3600] = a
    # the extra south row (lat = lat) and east column (lon = lon + 1)
    south = fetch(lat - 1, lon, cache)
    out[3600, :3600] = read(south)[0] if south else a[-1]
    east = fetch(lat, lon + 1, cache)
    out[:3600, 3600] = read(east)[:, 0] if east else a[:, -1]
    se = fetch(lat - 1, lon + 1, cache)
    out[3600, 3600] = read(se)[0, 0] if se else out[3599, 3600]
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bbox', nargs=4, type=float, required=True,
                    metavar=('LAT0', 'LON0', 'LAT1', 'LON1'))
    ap.add_argument('--out', default='~/.horizonator/DEMs_COP30')
    ap.add_argument('--cache', default='~/.horizonator/cop30_cog')
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    cache = os.path.expanduser(a.cache)
    os.makedirs(out, exist_ok=True)
    la0, lo0, la1, lo1 = a.bbox
    made = 0
    for lat in range(int(np.floor(la0)), int(np.ceil(la1))):
        for lon in range(int(np.floor(lo0)), int(np.ceil(lo1))):
            dst = os.path.join(out, hgt_name(lat, lon))
            if os.path.exists(dst):
                print('have', os.path.basename(dst))
                made += 1
                continue
            g = build(lat, lon, cache)
            if g is None:
                print('absent (all ocean?)', tile_name(lat, lon))
                continue
            # SRTM1 .hgt: big-endian int16, row 0 = north edge
            np.rint(g).astype('>i2').tofile(dst)
            print('wrote %s  min %.0f max %.0f m'
                  % (os.path.basename(dst), g.min(), g.max()))
            made += 1
    if not made:
        sys.exit('no tiles written')


if __name__ == '__main__':
    main()
