#!/usr/bin/env python3
"""Does the rendered world match the real one? Per frame, three skylines in the photo's pixels:

  render  the exact skyline of the Google Photorealistic 3D Tiles render
          (first non-sky row of its segmentation pass)
  photo   the SegFormer skyline of the real photograph (E5ch's seg boundary)
  cop30   the COP30 skyline at the same pose, projected with the same camera,
          straight rays (k = 0) as the renderer draws them

render vs cop30 come from one pose, so their gap is geometry (heights, datum,
camera conventions) plus the terrain difference between the two models.
photo vs render and photo vs cop30 also carry the attitude estimate's error,
so they are scored after a fitted column/row shift (heading, pitch).

    python3 compare.py POSES.json RENDER_DIR [--sheets]
"""

import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
import skyfix as F  # noqa: E402
import skyline as S  # noqa: E402

SEG = os.path.join(EXP, 'out', 'segformer')
DEM = os.path.expanduser('~/.horizonator/DEMs_COP30')
SKY = np.array([255, 0, 255]); SEA = np.array([0, 0, 255])


def seg_lines(path):
    a = np.asarray(Image.open(path).convert('RGB')).astype(int)
    sky = np.abs(a - SKY).sum(2) < 60
    sea = np.abs(a - SEA).sum(2) < 60
    H, W = sky.shape
    top = np.full(W, np.nan); sea_line = np.full(W, np.nan)
    for x in range(W):
        if sky[0, x] and not sky[:, x].all():
            k = int(np.argmax(~sky[:, x]))
            top[x] = k
            s = np.flatnonzero(sea[k:, x])
            if s.size:
                sea_line[x] = k + s[0]
    return top, sea_line, sky.mean()


def project(az_deg, el_rad, p, W, H):
    """Global azimuth/elevation -> rows per column, the inverse of
    skyfix.observation: a pinhole camera rotated by heading, pitch and roll,
    as CesiumJS draws (skyfix --camera-model exact)."""
    f = (W / 2) / math.tan(math.radians(p['hfov_deg']) / 2)
    d = np.radians((az_deg - p['heading_deg'] + 180) % 360 - 180)
    th = math.radians(p['pitch_deg'])
    X, Y, Z = np.sin(d) * np.cos(el_rad), np.sin(el_rad), np.cos(d) * np.cos(el_rad)
    zc = Y * math.sin(th) + Z * math.cos(th)
    ok = (np.abs(d) < math.radians(89)) & (zc > 1e-6)
    zc = np.where(ok, zc, 1.0)
    ur = f * X / zc
    vr = f * (Y * math.cos(th) - Z * math.sin(th)) / zc
    cr, sr = math.cos(math.radians(p['roll_deg'])), math.sin(math.radians(p['roll_deg']))
    u = ur * cr + vr * sr
    v = -ur * sr + vr * cr
    col = u + (W - 1) / 2
    row = (H - 1) / 2 - v
    rows = np.full(W, np.nan)
    m = ok & np.isfinite(row) & (col >= 0) & (col <= W - 1)
    o = np.argsort(col[m])
    rows_m, cols_m = row[m][o], col[m][o]
    if cols_m.size > 1:
        xs = np.arange(W)
        rows = np.interp(xs, cols_m, rows_m, left=np.nan, right=np.nan)
    return rows


def cop30_rows(p, W, H):
    # near terrain and far terrain both in: the segmentation pass draws both
    # (no fog); 60 km is what the +-0.6 deg mosaic holds
    cm = S.CMarcher(DEM, (p['lat'] - .6, p['lat'] + .6), (p['lon'] - .8, p['lon'] + .8), d_min=50.0,
                    d_max=60000.0, refraction_k=0.0)
    el, r = cm.skyline(p['lat'], p['lon'], p['z_msl'], F.AZ)
    return project(F.AZ, el, p, W, H)


def fit_shift(a, b, max_dx=80, max_dy=120):
    """Best integer column shift and median row offset between two curves; robust rms after."""
    best = None
    for dx in range(-max_dx, max_dx + 1, 2):
        bb = np.roll(b, dx)
        if dx > 0: bb[:dx] = np.nan
        elif dx < 0: bb[dx:] = np.nan
        m = np.isfinite(a) & np.isfinite(bb)
        if m.sum() < 100:
            continue
        r = a[m] - bb[m]
        dy = float(np.median(r))
        if abs(dy) > max_dy:
            continue
        e = r - dy
        rms = float(np.sqrt(np.mean(np.clip(e, -30, 30) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, dx, dy, int(m.sum()))
    return best


def stats(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return None
    r = a[m] - b[m]
    return dict(median=float(np.median(r)), mad=float(1.4826 * np.median(np.abs(r - np.median(r)))),
                cols=int(m.sum()))


def main():
    poses = json.load(open(sys.argv[1]))
    rdir = sys.argv[2]
    sheets = '--sheets' in sys.argv
    out = []
    for p in poses:
        meta = os.path.join(rdir, p['tag'] + '.json')
        if not os.path.exists(meta):
            continue
        m = json.load(open(meta))
        W, H = p['width'], p['height']
        # Cesium's camera collision once lifted a sea-level camera 158 m: check what was used
        cu = m.get('camera_used')
        moved = None if cu is None else abs(cu['h_ellipsoid'] - p['h_ellipsoid'])
        if moved is not None and moved > 0.5:
            print('%-28s CAMERA MOVED by Cesium: %.1f m in height, skipped' % (p['tag'], moved))
            continue
        # WebGL short of GPU memory falls back to a smaller drawing buffer
        seg_png = os.path.join(rdir, p['tag'] + '_seg.png')
        if Image.open(seg_png).size != (W, H):
            print('%-28s rendered at %dx%d, not %dx%d: skipped' % ((p['tag'],) + Image.open(seg_png).size + (W, H)))
            continue
        r_sky, r_sea, sky_frac = seg_lines(seg_png)
        ph = np.load(os.path.join(SEG, p['tag'] + '_seg.npz'))
        ph_rows = np.where(ph['conf'] > 0, ph['rows'].astype(float), np.nan)
        cop = cop30_rows(p, W, H)
        f = (W / 2) / math.tan(math.radians(p['hfov_deg']) / 2)
        rc = stats(r_sky, cop)
        pr = fit_shift(ph_rows, r_sky)
        pc = fit_shift(ph_rows, cop)
        # a view that needed more tile memory than the cache allows was drawn coarser
        lod = m.get('lod')
        if lod and lod['sse_used'] > lod['sse']:
            print('%-28s coarsened by the tile cache: sse %.1f, asked %.1f' % (p['tag'], lod['sse_used'], lod['sse']))
        row = dict(tag=p['tag'], loaded=m['loaded'], settle_s=m['settle_s'], sky_frac=sky_frac,
                   camera_checked=moved is not None, lod=lod,
                   render_vs_cop30=rc, photo_vs_render=pr and dict(rms_px=pr[0], dx=pr[1], dy=pr[2], cols=pr[3]),
                   photo_vs_cop30=pc and dict(rms_px=pc[0], dx=pc[1], dy=pc[2], cols=pc[3]),
                   mrad_per_px=1e3 / f)
        out.append(row)
        fm = lambda d, k: ('%6.1f' % d[k]) if d else '     -'
        print('%-28s loaded %-5s | render-cop30 median %s px (mad %s) | photo-render rms %s (dx %s dy %s) | '
              'photo-cop30 rms %s (dx %s dy %s)' % (
                  p['tag'], m['loaded'], fm(rc, 'median'), fm(rc, 'mad'),
                  fm(row['photo_vs_render'], 'rms_px'), fm(row['photo_vs_render'], 'dx'), fm(row['photo_vs_render'], 'dy'),
                  fm(row['photo_vs_cop30'], 'rms_px'), fm(row['photo_vs_cop30'], 'dx'), fm(row['photo_vs_cop30'], 'dy')),
              flush=True)
        if sheets:
            draw_sheet(p, rdir, ph_rows, r_sky, r_sea, cop)
    json.dump(out, open(os.path.join(rdir, 'compare.json'), 'w'), indent=1)


def draw_sheet(p, rdir, ph_rows, r_sky, r_sea, cop):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import extract
    from assets import celnav
    fr = [f for f in json.load(open(os.path.join(EXP, 'data', 'campaign_frames.json')))['frames']
          if f['tag'] == p['tag']][0]
    img = extract.load_image(celnav(fr['path']))
    ren = np.asarray(Image.open(os.path.join(rdir, p['tag'] + '_photo.png')).convert('RGB'))
    W, H = p['width'], p['height']
    x = np.arange(W)
    lo = np.nanmin(np.r_[r_sky, cop, ph_rows]) - 60
    hi = np.nanmax(np.r_[r_sky[np.isfinite(r_sky)], ph_rows[np.isfinite(ph_rows)]]) + 80
    fig, ax = plt.subplots(2, 1, figsize=(15, 9))
    for a, im, title in ((ax[0], img, 'real photo'), (ax[1], ren, 'Google 3D Tiles render, same pose')):
        a.imshow(im)
        a.plot(x, ph_rows, color='#ffd23f', lw=1.6, label='photo skyline (SegFormer)')
        a.plot(x, r_sky, color='#ff2d2d', lw=1.6, label='render skyline (exact)')
        a.plot(x, cop, color='#38c6f4', lw=1.6, ls='--', label='COP30 skyline, k = 0')
        a.plot(x, r_sea, color='#ff4fd8', lw=1.0, label='render sea line')
        a.set_ylim(min(hi, H), max(lo, 0)); a.set_title(title); a.axis('off')
    ax[0].legend(loc='lower left', fontsize=9)
    fig.suptitle(p['tag'])
    fig.tight_layout()
    fig.savefig(os.path.join(rdir, p['tag'] + '_compare.png'), dpi=90)
    plt.close(fig)


if __name__ == '__main__':
    main()
