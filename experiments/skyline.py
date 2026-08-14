#!/usr/bin/env python3
"""Shared machinery for the skyline-matching experiments (E0, E1).

Two skyline-synthesis backends over the same SRTM data:

- GlSkyline: the horizonator OpenGL renderer (requires a GL context; run under
  xvfb-run on a headless box). The skyline is extracted from the range image:
  per azimuth column, the topmost pixel with a valid range.

- RayMarcher: an independent NumPy implementation that marches a ray per
  azimuth over the DEM, applying the same curvature-plus-refraction model the
  patched vertex shader uses. Used to cross-validate the renderer in E0, and
  usable headlessly.

Both produce a skyline as a 1D array of elevation angles (radians) over a
uniform azimuth grid, plus the horizontal range to the skyline point.

Conventions: azimuth 0 = North, 90 deg = East (same as the horizonator).
Elevation angle 0 = the viewer's horizontal plane.
"""

import os
import numpy as np

RVENUS = None  # only Earth is supported :-)
REARTH = 6371000.0
K_REFRACTION = 0.13  # must match HORIZONATOR_REFRACTION_K in horizonator.h
REFF = REARTH / (1.0 - K_REFRACTION)


def horizon_dip_rad(h):
    """Apparent dip of the sea horizon below horizontal for viewer height h"""
    return np.sqrt(2.0 * h / REFF)


def horizon_distance_m(h):
    """Horizontal distance to the sea horizon for viewer height h"""
    return np.sqrt(2.0 * REFF * h)


class Dem:
    """Loads a directory of SRTM .hgt tiles; bilinear elevation sampling.

    Tile resolution (1201 = SRTM3 or 3601 = SRTM1) is auto-detected from file
    size. Missing tiles sample as elevation 0 (open ocean), matching the
    horizonator's convention.
    """

    def __init__(self, dirname):
        self.dirname = os.path.expanduser(dirname)
        self.tiles = {}  # (lat0,lon0) -> 2D array, row 0 = north edge

    def _tile(self, lat0, lon0):
        key = (lat0, lon0)
        if key not in self.tiles:
            name = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}" \
                   f"{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}.hgt"
            path = os.path.join(self.dirname, name)
            if not os.path.exists(path):
                self.tiles[key] = None
            else:
                n = os.path.getsize(path)
                w = 1201 if n == 1201 * 1201 * 2 else 3601
                a = np.fromfile(path, dtype='>i2').reshape(w, w).astype(np.float32)
                # clamp voids AND negative elevations (some sources, e.g. the
                # AWS skadi tiles, contain ocean bathymetry) to sea level --
                # same as horizonator_dem_sample() in dem.c
                a[a < 0] = 0.0
                self.tiles[key] = a
        return self.tiles[key]

    def sample(self, lat, lon):
        """Bilinear elevation at arrays of lat,lon (degrees). Vectorized."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        out = np.zeros(lat.shape, dtype=np.float32)

        lat0 = np.floor(lat).astype(int)
        lon0 = np.floor(lon).astype(int)
        for key in set(zip(lat0.ravel(), lon0.ravel())):
            tile = self._tile(*key)
            m = (lat0 == key[0]) & (lon0 == key[1])
            if tile is None:
                continue
            w = tile.shape[0]
            # fractional position inside the tile. Row 0 is the NORTH edge
            x = (lon[m] - key[1]) * (w - 1)          # 0..w-1 west->east
            y = (1.0 - (lat[m] - key[0])) * (w - 1)  # 0..w-1 north->south
            x0 = np.clip(np.floor(x).astype(int), 0, w - 2)
            y0 = np.clip(np.floor(y).astype(int), 0, w - 2)
            fx = x - x0
            fy = y - y0
            out[m] = (tile[y0,     x0    ] * (1 - fx) * (1 - fy) +
                      tile[y0,     x0 + 1] * fx       * (1 - fy) +
                      tile[y0 + 1, x0    ] * (1 - fx) * fy       +
                      tile[y0 + 1, x0 + 1] * fx       * fy)
        return out


class RayMarcher:
    """Skyline synthesis by marching one ray per azimuth over a Dem.

    Applies the curvature-plus-refraction drop d^2/(2 Reff), the same model as
    the patched vertex shader.
    """

    def __init__(self, dem, d_max=40000.0, d_step=90.0, d_min=150.0):
        self.dem = dem
        # march distances. Constant step ~ the DEM posting
        self.d = np.arange(d_min, d_max, d_step)

    def skyline(self, lat, lon, z, az_deg):
        """Skyline elevation angle (rad) and range (m) at each azimuth.

        az_deg: 1D array of azimuths in degrees (0=N, 90=E)
        """
        az = np.radians(np.asarray(az_deg, dtype=np.float64))
        coslat = np.cos(np.radians(lat))

        # (Naz, Nd) grids of sample positions
        d = self.d[np.newaxis, :]
        e = np.sin(az)[:, np.newaxis] * d
        n = np.cos(az)[:, np.newaxis] * d
        lats = lat + np.degrees(n / REARTH)
        lons = lon + np.degrees(e / (REARTH * coslat))

        ele = self.dem.sample(lats, lons)
        # curvature + refraction drop, and the viewer height
        el_angle = np.arctan2(ele - z - d * d / (2.0 * REFF), d)

        i = np.argmax(el_angle, axis=1)
        rows = np.arange(az.size)
        return el_angle[rows, i], self.d[i]


class GlSkyline:
    """Skyline synthesis with the horizonator's OpenGL renderer.

    width,height set the angular resolution: a 360 deg panorama at width=3600
    gives 0.1 deg/pixel in both azimuth and elevation.
    """

    def __init__(self, lat, lon, width=3600, height=400,
                 render_radius_m=45000.0, dir_dems=None, zfar=40000.0):
        import horizonator
        kwargs = dict(render_radius_m=render_radius_m, allow_downloads=False)
        if dir_dems is not None:
            kwargs['dir_dems'] = dir_dems
        self.h = horizonator.horizonator(lat, lon, width, height, **kwargs)
        self.width = width
        self.height = height
        self.zfar = zfar

    def skyline(self, lat, lon, z, az_deg0=-180.0, az_deg1=180.0):
        """Skyline elevation angle (rad) and range (m) per azimuth column.

        Returns (az_deg, el_rad, range_m). Azimuth bins are the pixel-column
        centers. Columns with no terrain (open sea beyond zfar) get el=NaN and
        range=NaN: the caller decides how to treat pure-sea horizon bins.
        """
        rng = self.h.render(az_deg0, az_deg1, lat=lat, lon=lon, z=z,
                            zfar=self.zfar, return_image=False)
        valid = rng > 0
        has = valid.any(axis=0)
        row = valid.argmax(axis=0)  # topmost valid pixel per column

        az_per_px = (az_deg1 - az_deg0) / self.width
        az = az_deg0 + (np.arange(self.width) + 0.5) * az_per_px
        # the equirectangular render has el=0 at pixel row (height/2 - 0.5),
        # with the same deg/pixel as azimuth
        el = np.radians(((self.height / 2.0 - 0.5) - row) * az_per_px)
        el[~has] = np.nan

        r = rng[row, np.arange(self.width)].astype(np.float64)
        r[~has] = np.nan
        return az, el, r


def seahorizon_fill(el, z):
    """Replace NaN (pure sea horizon) bins with the analytic horizon dip"""
    out = el.copy()
    out[np.isnan(out)] = -horizon_dip_rad(z)
    return out


def cost(el_obs, el_syn, huber_delta=3e-3, weights=None):
    """Robust skyline-mismatch cost: mean Huber loss on the per-bin elevation
    angle residual (radians). huber_delta is the outlier knee."""
    r = el_syn - el_obs
    m = np.isfinite(r)
    r = np.abs(r[m])
    l = np.where(r <= huber_delta,
                 0.5 * r * r,
                 huber_delta * (r - 0.5 * huber_delta))
    if weights is not None:
        w = weights[m]
        return float(np.sum(l * w) / np.sum(w))
    return float(np.mean(l))


def meters_per_degree(lat):
    """(m per degree of latitude, m per degree of longitude)"""
    return (REARTH * np.pi / 180.0,
            REARTH * np.pi / 180.0 * np.cos(np.radians(lat)))


def quadratic_refine(xy, c):
    """Sub-grid minimum from a 3x3 grid of costs.

    xy: (3,) offsets of the grid lines (same for both axes, meters)
    c:  (3,3) costs, c[i,j] at (y=xy[i], x=xy[j])
    Returns (dx, dy) of the fitted paraboloid minimum, clamped to the grid.
    Falls back to (0,0) if the paraboloid is not convex.
    """
    # separable 1D parabola fits through the middle row/column
    def vertex(cm, c0, cp, h):
        den = cm - 2 * c0 + cp
        if den <= 0:
            return 0.0
        return float(np.clip(0.5 * h * (cm - cp) / den, -h, h))
    h = xy[2] - xy[1]
    dx = vertex(c[1, 0], c[1, 1], c[1, 2], h)
    dy = vertex(c[0, 1], c[1, 1], c[2, 1], h)
    return dx, dy


def solve_position(skyline_fn, el_obs, lat_c, lon_c, z,
                   box_m=1000.0, coarse_n=9, fine_step_m=25.0,
                   weights=None, verbose=False):
    """Coarse-to-fine position search over a box centered on (lat_c, lon_c).

    skyline_fn(lat, lon) -> el array on the same azimuth grid as el_obs.
    Returns (lat, lon, info) of the estimated position. Search grid spacing:
    coarse box_m/(coarse_n-1), then a 5x5 fine grid at fine_step_m around the
    coarse minimum, then a quadratic sub-grid refinement.
    """
    mlat, mlon = meters_per_degree(lat_c)
    evals = [0]

    def C(dn, de):
        evals[0] += 1
        el = skyline_fn(lat_c + dn / mlat, lon_c + de / mlon)
        return cost(el_obs, el, weights=weights)

    # coarse grid
    g = np.linspace(-box_m / 2, box_m / 2, coarse_n)
    cc = np.array([[C(dn, de) for de in g] for dn in g])
    i, j = np.unravel_index(np.argmin(cc), cc.shape)
    dn0, de0 = g[i], g[j]
    if verbose:
        print(f'  coarse min at dn={dn0:+.0f} de={de0:+.0f} cost={cc[i,j]:.3e}')

    # fine grid around the coarse minimum
    f = np.arange(-2, 3) * fine_step_m
    cf = np.array([[C(dn0 + dn, de0 + de) for de in f] for dn in f])
    i, j = np.unravel_index(np.argmin(cf), cf.shape)
    i = np.clip(i, 1, 3)
    j = np.clip(j, 1, 3)
    dn1, de1 = dn0 + f[i], de0 + f[j]

    # sub-grid quadratic refinement on the surrounding 3x3
    ddx, ddy = quadratic_refine(np.array([-1, 0, 1]) * fine_step_m,
                                cf[i - 1:i + 2, j - 1:j + 2])
    dn2, de2 = dn1 + ddy, de1 + ddx
    if verbose:
        print(f'  fine   min at dn={dn2:+.1f} de={de2:+.1f} '
              f'({evals[0]} evaluations)')

    info = dict(evals=evals[0], coarse_cost=cc, coarse_grid=g,
                dn=dn2, de=de2)
    return lat_c + dn2 / mlat, lon_c + de2 / mlon, info
