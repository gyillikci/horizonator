#!/usr/bin/env python3
"""Landcover correction of a DEM store — the poor man's FABDEM.

SRTM and GLO-30 are surface models: over forest they report a height
somewhere between ground and canopy top, over towns the rooftops, and
raw SRTM reports metres of noise over open water (the E5o autopsy
found a phantom ridge of 13-20 m sea noise off Foça that put a
confident fix 3 km away). FABDEM removes forests and buildings
properly but is CC BY-NC-SA, so this is the licence-clean version:
ESA WorldCover 10 m classes (CC BY 4.0, on a public S3 bucket) drive
a per-class height offset.

    water (80)            -> 0          (also erases sea noise)
    tree (10), mangrove (95) -> max(h - tree_off, 0)
    built (50)            -> max(h - built_off, 0)
    everything else       -> unchanged

Offsets are constants, not per-pixel canopy heights — that is the
"poor man's" part — and the field frames answered the empirical
question (E5p): a constant is the WRONG model for the canopy
(C-band penetrates partway into Mediterranean pine and canopy
height varies, so -8/-15 m swing single frames in both directions
and lose at the median), while the water mask helps or ties on
every frame. Defaults are therefore water mask only.

Run:
  python3 canopy.py --src ~/.horizonator/DEMs_SRTM1 \
                    --dst ~/.horizonator/DEMs_SRTM1_WM
Only tiles covered by a downloaded WorldCover tile are written; the
destination store is regional by design.
"""

import os
import re
import sys
import argparse

import numpy as np
import tifffile

WC_DIR = os.path.expanduser('~/.horizonator/WorldCover')
VOID = -32768


def dem_tiles(src):
    for f in sorted(os.listdir(src)):
        m = re.match(r'N(\d+)E(\d+)\.hgt$', f)
        if m:
            yield f, int(m.group(1)), int(m.group(2))


def wc_name(la0, lo0):
    return f'N{la0 // 3 * 3:02d}E{lo0 // 3 * 3:03d}'


def correct_tile(hgt, wc, wc_top, wc_left, la0, lo0, tree, built):
    n = hgt.shape[0]                              # 3601
    lat = la0 + 1 - np.arange(n) / (n - 1)        # row 0 = north edge
    lon = lo0 + np.arange(n) / (n - 1)
    r = np.clip(np.round((wc_top - lat) * 12000).astype(int), 0, 35999)
    c = np.clip(np.round((lon - wc_left) * 12000).astype(int), 0, 35999)
    cls = wc[np.ix_(r, c)]
    out = hgt.astype(np.int32)
    ok = out != VOID
    water = (cls == 80) & ok
    out[water] = 0
    for mask_cls, off in ((np.isin(cls, (10, 95)), tree),
                          ((cls == 50), built)):
        m = mask_cls & ok & (out > 0)
        out[m] = np.maximum(out[m] - int(round(off)), 0)
    stats = dict(water=int(water.sum()),
                 water_nonzero=int((water & (hgt != 0)).sum()),
                 tree=int((np.isin(cls, (10, 95)) & ok).sum()),
                 built=int(((cls == 50) & ok).sum()))
    return out.astype(np.int16), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--tree', type=float, default=0.0,
                    help='height subtracted under tree cover, m. '
                         'E5p measured a constant offset to be NOISE '
                         'on the field frames (medians: water mask '
                         'only 521 m, +tree-8 637, +tree-15 619, '
                         'single frames swinging both ways), so the '
                         'default is 0: the water mask is the part '
                         'of this correction that earns its keep')
    ap.add_argument('--built', type=float, default=0.0,
                    help='height subtracted under built-up, m '
                         '(default 0, same E5p verdict as --tree)')
    ap.add_argument('--only', nargs='*', default=None,
                    help='restrict to these tiles (e.g. N38E026)')
    args = ap.parse_args()
    src = os.path.expanduser(args.src)
    dst = os.path.expanduser(args.dst)
    os.makedirs(dst, exist_ok=True)

    by_wc = {}
    for f, la0, lo0 in dem_tiles(src):
        if args.only and f[:-4] not in args.only:
            continue
        w = wc_name(la0, lo0)
        if os.path.exists(os.path.join(WC_DIR, w + '.tif')):
            by_wc.setdefault(w, []).append((f, la0, lo0))
        else:
            print(f'{f}: no WorldCover tile {w} downloaded — skipped')

    for w, tiles in sorted(by_wc.items()):
        wc = tifffile.imread(os.path.join(WC_DIR, w + '.tif'))
        wc_la = int(w[1:3]); wc_lo = int(w[4:7])
        for f, la0, lo0 in tiles:
            p = os.path.join(src, f)
            n = int(round(np.sqrt(os.path.getsize(p) / 2)))
            hgt = np.fromfile(p, dtype='>i2').reshape(n, n)
            out, st = correct_tile(hgt, wc, wc_la + 3, wc_lo,
                                   la0, lo0, args.tree, args.built)
            out.astype('>i2').tofile(os.path.join(dst, f))
            npix = n * n
            print(f'{f}: tree {st["tree"]/npix*100:4.1f}%  built '
                  f'{st["built"]/npix*100:4.1f}%  water '
                  f'{st["water"]/npix*100:4.1f}% (of which '
                  f'{st["water_nonzero"]/max(st["water"],1)*100:.0f}% '
                  f'had nonzero height — erased)', flush=True)
        del wc


if __name__ == '__main__':
    main()
