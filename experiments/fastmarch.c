/* Fast skyline ray-marcher over a DEM mosaic.

   Native implementation of skyline.RayMarcher, for on-device use (e.g. a
   Raspberry Pi CM5): one ray per azimuth is marched across a bilinearly
   sampled DEM, applying the same curvature-plus-refraction drop
   d^2/(2 Reff) as the patched vertex shader. The inner loop avoids
   divisions and trig: the elevation-angle maximum is tracked via the
   cross-multiplied tangent comparison h/d > hb/db  <=>  h*db > hb*d
   (both distances positive), and atan2 runs once per azimuth at the end.

   Build (any platform with OpenMP; on the CM5 use -mcpu=native):
     cc -O3 -march=native -fopenmp -shared -fPIC fastmarch.c -o fastmarch.so

   The mosaic is a row-major float32 grid, row 0 = the NORTH edge,
   column 0 = the WEST edge, with uniform spacing of dpp degrees per cell
   in both axes. Elevations must already be clamped to >= 0 (sea level);
   out-of-mosaic samples read as 0 (open ocean).
*/

#include <math.h>

static const double REARTH = 6371000.0;

/* Bilinear sample of the mosaic at fractional cell coordinates (x east,
   y south from the NW corner). Out of bounds -> 0 (ocean) */
static inline float sample(const float *dem, int nrows, int ncols,
                           double x, double y)
{
    if (x < 0.0 || y < 0.0 || x > (double)(ncols - 1) || y > (double)(nrows - 1))
        return 0.0f;
    int x0 = (int)x;
    int y0 = (int)y;
    if (x0 > ncols - 2) x0 = ncols - 2;
    if (y0 > nrows - 2) y0 = nrows - 2;
    float fx = (float)(x - x0);
    float fy = (float)(y - y0);
    const float *p = dem + (long)y0 * ncols + x0;
    return p[0]         * (1.f - fx) * (1.f - fy) +
           p[1]         * fx         * (1.f - fy) +
           p[ncols]     * (1.f - fx) * fy         +
           p[ncols + 1] * fx         * fy;
}

/* Skyline (elevation angle el_out, radians; horizontal range r_out, m) at
   each of naz azimuths (radians, 0=N, pi/2=E) for a viewer at lat,lon
   (degrees) and z meters above sea level. refraction_k: the terrestrial
   refraction coefficient (0.13 = the horizonator's default) */
void fastmarch_skyline(const float *dem, int nrows, int ncols,
                       double lat_nw, double lon_nw, double dpp,
                       double lat, double lon, double z,
                       const double *az_rad, int naz,
                       double dmin, double dmax, double dstep,
                       double refraction_k,
                       double *el_out, double *r_out)
{
    const double reff2inv = (1.0 - refraction_k) / (2.0 * REARTH);
    const double coslat   = cos(lat * M_PI / 180.0);
    const double m2deg    = 180.0 / (M_PI * REARTH);   /* meters -> degrees lat */
    const int    nd       = (int)((dmax - dmin) / dstep);

    /* viewer position in fractional mosaic cells */
    const double xv = (lon - lon_nw) / dpp;
    const double yv = (lat_nw - lat) / dpp;

#pragma omp parallel for schedule(static)
    for (int ia = 0; ia < naz; ia++)
    {
        /* per-dstep increments, in fractional cells */
        const double de = sin(az_rad[ia]) * dstep * m2deg / coslat / dpp;
        const double dn = cos(az_rad[ia]) * dstep * m2deg / dpp;

        double x = xv + de * (dmin / dstep);
        double y = yv - dn * (dmin / dstep);

        /* running maximum of h/d, tracked division-free */
        double hbest = -1.0, dbest = 1.0;  /* h/d = -1: far below anything real */
        for (int id = 0; id < nd; id++)
        {
            const double d = dmin + id * dstep;
            const double h = (double)sample(dem, nrows, ncols, x, y)
                             - z - d * d * reff2inv;
            if (h * dbest > hbest * d)
            {
                hbest = h;
                dbest = d;
            }
            x += de;
            y -= dn;
        }
        el_out[ia] = atan2(hbest, dbest);
        r_out[ia]  = dbest;
    }
}
