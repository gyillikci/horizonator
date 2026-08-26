#!/usr/bin/env python3
"""E5av: build a canopy-augmented DEM from Meta/WRI 1 m canopy heights.

E5ae..E5ao measured a CREST DEFICIT: the synthesized silhouette sits
9-12 m below the photographed one, and FABDEM (canopy REMOVED) makes
it worse — so the silhouette the camera sees is the canopy top, not
bare earth. E5p rejected a CONSTANT canopy offset because a constant
is noise the elevation-offset beta absorbs. A per-pixel canopy raster
is a different proposition: it changes the silhouette's SHAPE, which
is what the cost actually reads.

Adds the canopy height onto each 1-arcsec DEM post, taking the MAX
over the post's footprint rather than the mean: a ridge silhouette is
set by the tallest trees on it, not the average ones.

  python3 e5av_canopy_dem.py CHM.tif SRC_DEM_DIR DST_DEM_DIR TILE [TILE...]
"""
import os
import sys
import zlib

import numpy as np
import tifffile

R = 6378137.0
CLAMP_M = 60.0          # taller than any Mediterranean canopy: nodata guard


def main():
    chm, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    tiles = sys.argv[4:]
    os.makedirs(dst, exist_ok=True)

    tf = tifffile.TiffFile(chm)
    pg = tf.pages[0]
    sc = pg.tags[33550].value[0]
    tp = pg.tags[33922].value
    X0, Y0 = tp[3], tp[4]
    NR, NC = pg.shape
    offs, cnts = pg.dataoffsets, pg.databytecounts
    pred = pg.tags[317].value if 317 in pg.tags else 1
    fh = open(chm, 'rb')

    def chm_rows(r0, r1):
        """decoded canopy rows [r0, r1) as uint8 (predictor undone)"""
        r0 = max(0, r0); r1 = min(NR, r1)
        if r1 <= r0:
            return None
        out = np.empty((r1 - r0, NC), np.uint8)
        for k in range(r0, r1):
            fh.seek(offs[k])
            out[k - r0] = np.frombuffer(
                zlib.decompress(fh.read(cnts[k])), dtype=np.uint8)
        if pred == 2:
            out = np.cumsum(out, axis=1, dtype=np.uint8)
        return out

    def y_of(lat):
        return (Y0 - R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))) / sc

    def x_of(lon):
        return (np.radians(lon) * R - X0) / sc

    for name in tiles:
        la0 = int(name[1:3]) * (1 if name[0] == 'N' else -1)
        lo0 = int(name[4:7]) * (1 if name[3] == 'E' else -1)
        p_in = os.path.join(src, name + '.hgt')
        if not os.path.exists(p_in):
            print(f'{name}: kaynak yok, atlandı')
            continue
        hgt = np.fromfile(p_in, dtype='>i2').reshape(3601, 3601).astype(np.int32)
        N = 3601
        lats = la0 + 1.0 - np.arange(N) / 3600.0      # row 0 = north edge
        lons = lo0 + np.arange(N) / 3600.0
        xc = x_of(lons)
        # per-post column window: half a post either side
        half_c = 0.5 * (R * np.radians(1.0 / 3600.0)) / sc
        c_lo = np.clip(np.floor(xc - half_c).astype(int), 0, NC - 1)
        c_hi = np.clip(np.ceil(xc + half_c).astype(int) + 1, 1, NC)

        added = np.zeros((N, N), np.float32)
        nrow = 0
        for i in range(N):
            ytop = y_of(lats[i] + 0.5 / 3600.0)
            ybot = y_of(lats[i] - 0.5 / 3600.0)
            r0, r1 = int(np.floor(ytop)), int(np.ceil(ybot)) + 1
            if r1 <= 0 or r0 >= NR:
                continue
            band = chm_rows(r0, r1)
            if band is None or band.size == 0:
                continue
            col_max = band.max(axis=0)
            # max over each post's column window
            for j in range(N):
                a, b = c_lo[j], c_hi[j]
                if b > a:
                    added[i, j] = col_max[a:b].max()
            nrow += 1
            if i % 500 == 0:
                print(f'  {name} satır {i}/{N}', flush=True)

        added = np.where(added > CLAMP_M, 0.0, added)
        # canopy only on land: the store is water-masked, keep it that way
        added = np.where(hgt > 0, added, 0.0)
        out = np.clip(hgt + np.round(added), -32000, 32000).astype('>i2')
        out.tofile(os.path.join(dst, name + '.hgt'))
        cov = float((added > 0).mean())
        print(f'{name}: {nrow} satır kanopi kapsamı, ekleme>0 %{cov*100:.0f}, '
              f'medyan(ekleme>0) {np.median(added[added>0]) if cov else 0:.1f} m, '
              f'p95 {np.percentile(added[added>0], 95) if cov else 0:.1f} m, '
              f'maks {added.max():.0f} m')


if __name__ == '__main__':
    main()
