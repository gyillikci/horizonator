#!/usr/bin/env python3
"""Render a full 2D scene from a DEM: depth per pixel, not one curve.

The skyline matcher keeps a single number per azimuth -- the topmost
boundary -- and throws away everything below it. `visible_layers` in
skyline.py already argues why that is expensive: a coastal scene is
layered, and the near layers are what carry the position, because a
500 m move swings a 41 km coast by 12 mrad and a 6 km island by 81.

This renders the whole frame instead. For each azimuth column the ray
is marched once, the terrain's elevation-angle profile is turned into a
running maximum (which is what occlusion means along a ray), and every
pixel's first visible surface is then one searchsorted into it. The
result is a depth image: each pixel carries the range to the terrain it
sees, or NaN for sky. No GL context, no tiles, pure numpy.

Two uses. Offline it is the geometry half of scene matching -- a 2D
structure to align against the photograph instead of a 1D curve. And
it is the same output contract as `cesium_render.py`'s depth grid, so a
photorealistic render can replace the appearance while this keeps
supplying the 3D.
"""

import numpy as np

import skyline as S

R_EFF = 6371000.0 * 1.13


def depth_image(dem, lat, lon, z, heading, fov, width, height,
                pitch=0.0, roll=0.0, d_min=150.0, d_max=40000.0,
                d_step=None):
    """Range (m) to the first visible terrain at every pixel; NaN = sky.

    Returns (depth, hit_h, el_img, az_col) with depth and hit_h shaped
    (height, width); hit_h is the height of the surface that was hit,
    which separates sea from land without a second pass.
    """
    if d_step is None:
        d_step = 30.0
    f_px = (width / 2.0) / np.tan(np.radians(fov) / 2.0)
    # pixel bearings: azimuth from the column, elevation from the row,
    # with pitch as a rigid rotation of the whole frame. Roll is applied
    # as a rotation in the image plane before the two are separated,
    # which is exact for the small rolls a hand-held frame carries.
    u = np.arange(width) - (width - 1) / 2.0
    v = (height - 1) / 2.0 - np.arange(height)
    U, V = np.meshgrid(u, v)
    if roll:
        cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        U, V = cr * U + sr * V, -sr * U + cr * V
    az_img = heading + np.degrees(np.arctan2(U, f_px))
    el_img = np.radians(pitch) + np.arctan2(V, np.hypot(U, f_px))

    # march once per COLUMN of the azimuth grid, not per pixel
    az_col = az_img[height // 2]              # azimuth varies only with u
    d = np.arange(max(d_min, d_step), d_max, d_step)
    mlat, mlon = S.meters_per_degree(lat)
    la = lat + (d[None, :] * np.cos(np.radians(az_col))[:, None]) / mlat
    lo = lon + (d[None, :] * np.sin(np.radians(az_col))[:, None]) / mlon
    h = dem.sample(la.ravel(), lo.ravel()).reshape(la.shape)
    el = (h - z - d[None, :] ** 2 / (2 * R_EFF)) / d[None, :]
    run = np.maximum.accumulate(el, axis=1)   # occlusion along the ray

    depth = np.full((height, width), np.nan)
    hit_h = np.full((height, width), np.nan)
    for c in range(width):
        # first d whose running-max elevation reaches this pixel's:
        # run is monotone, so one searchsorted per column does the frame
        idx = np.searchsorted(run[c], el_img[:, c])
        ok = idx < d.size
        j = np.where(ok, np.minimum(idx, d.size - 1), 0)
        depth[ok, c] = d[j[ok]]
        hit_h[ok, c] = h[c, j[ok]]
    return depth, hit_h, el_img, az_col


def shade(depth, hit_h, sea_h=5.0):
    """A grey scene from the depth image: haze with range, sea flat.

    Deliberately crude. The point of the render is its STRUCTURE -- the
    silhouette, the layer edges, the shoreline -- not its radiometry,
    because no DEM shading will ever match a photograph's appearance
    (E5be: that is the domain gap the mesh literature reports). Edges
    are what survives; this exists to produce them.
    """
    img = np.zeros(depth.shape, np.float32)
    sky = ~np.isfinite(depth)
    land = np.isfinite(depth) & (hit_h > sea_h)
    sea = np.isfinite(depth) & ~land
    haze = np.clip(np.nan_to_num(depth) / 25000.0, 0, 1)
    img[sky] = 0.95
    img[land] = (0.30 + 0.55 * haze)[land]
    img[sea] = 0.55
    # a touch of relief so ridges inside the silhouette read as edges
    with np.errstate(invalid='ignore'):
        g = np.gradient(np.nan_to_num(hit_h), axis=0)
    img[land] = np.clip(img[land] - 0.12 * np.tanh(g[land] / 12.0), 0, 1)
    return img
