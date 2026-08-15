#!/usr/bin/env python3
"""skynav: the live skyline-navigation instrument loop.

Incremental version of E5: odometry legs and skyline fixes are fused as
they arrive with GTSAM's iSAM2 (constant-time updates, no batch re-solve),
and the fused position is emitted as standard NMEA 0183 sentences
($GPGGA/$GPRMC), so a chartplotter or OpenCPN consumes the skyline-derived
position like any GPS.

Interface (sensor-agnostic; see e5b_live.py for a simulated source):

    nav = SkyNav(lat0, lon0, z, dem_dir)
    nav.add_odometry(dist_m, heading_rad)        # per log/compass leg
    nav.take_fix(el_obs)                         # per camera frame
    lat, lon, cov = nav.current()
    print(nav.nmea_rmc(utc, speed_kn, course_deg))

On-device sizing (CM5-class): the iSAM2 update is milliseconds; the fix
solve dominates at ~1-2 s (2 km box, native marcher).
"""

import numpy as np
import gtsam
import skyline as S
from skyline_factor import skyline_factor, heading_bias_factor, laplace_cov
from skyfix import basin_margin, fast_photo_cost

AZ = np.arange(-180.0, 180.0, 0.1) + 0.05
X = gtsam.symbol_shorthand.X
B = gtsam.symbol_shorthand.B


class SkyNav:
    def __init__(self, lat0, lon0, z, dem_dir,
                 lat_range=None, lon_range=None,
                 start_pos=(0.0, 0.0), start_heading=0.0,
                 start_sigma=(200.0, 200.0, 0.1),
                 fix_box_m=2000.0, cov_scale=0.02,
                 min_margin=0.15, estimate_compass_bias=True):
        self.lat0, self.lon0, self.z = lat0, lon0, z
        self.mlat, self.mlon = S.meters_per_degree(lat0)
        self.cm = S.CMarcher(dem_dir,
                             lat_range or (lat0 - 0.5, lat0 + 0.5),
                             lon_range or (lon0 - 0.7, lon0 + 0.7))
        self.fix_box_m = fix_box_m
        self.cov_scale = cov_scale
        self.min_margin = min_margin
        self.fix_fresh = False      # True while the latest fix attempt
                                    # was accepted -> NMEA quality 1;
                                    # else quality 6 (estimated/DR)

        params = gtsam.ISAM2Params()
        # aggressive relinearization: the compass-bias variable couples
        # every pose (a whole-chain-rotation mode) and the default lazy
        # settings leave it stuck near its prior; these cost milliseconds
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(params)
        self.k = 0
        self._factors = []          # keep CustomFactor closures alive
        # NB: Pose2.theta is CCW from +x (east); compass headings are CW
        # from north. Convert at the API boundary: theta = pi/2 - heading
        th0 = np.pi / 2 - start_heading
        graph = gtsam.NonlinearFactorGraph()
        graph.add(gtsam.PriorFactorPose2(
            X(0), gtsam.Pose2(start_pos[0], start_pos[1], th0),
            gtsam.noiseModel.Diagonal.Sigmas(list(start_sigma))))
        vals = gtsam.Values()
        vals.insert(X(0), gtsam.Pose2(start_pos[0], start_pos[1], th0))
        # the compass bias as an estimated VARIABLE (imported from the
        # parallel branch: biases belong in the graph, not in the
        # sigmas). With it, the odometry lateral sigma no longer has to
        # price a constant heading bias, so it can be much tighter
        self.est_bias = estimate_compass_bias
        if self.est_bias:
            graph.add(gtsam.PriorFactorVector(
                B(0), np.zeros(1),
                gtsam.noiseModel.Isotropic.Sigma(1, np.radians(5.0))))
            vals.insert(B(0), np.zeros(1))
        self.isam.update(graph, vals)
        self._last_theta = th0

    # ---------------- sensors in
    def add_odometry(self, dist_m, heading_rad):
        """One dead-reckoning leg: distance run along a measured heading.

        With estimate_compass_bias (default), the constant compass bias
        is a graph variable: each leg adds a heading-measurement factor
        tying the pose's theta to the measured heading THROUGH the bias,
        and the between-factor lateral sigma only has to cover random
        compass noise (0.015*d) instead of pricing a worst-case constant
        bias (0.045*d, the legacy E5 setting kept for the fallback)."""
        est = self.isam.calculateEstimate().atPose2(X(self.k))
        theta = np.pi / 2 - heading_rad
        dtheta = theta - self._last_theta
        lat_sig = (0.015 if self.est_bias else 0.045) * dist_m + 5.0
        graph = gtsam.NonlinearFactorGraph()
        graph.add(gtsam.BetweenFactorPose2(
            X(self.k), X(self.k + 1), gtsam.Pose2(dist_m, 0.0, dtheta),
            gtsam.noiseModel.Diagonal.Sigmas(
                [0.04 * dist_m + 5.0, lat_sig, np.radians(1.0)])))
        if self.est_bias:
            f = heading_bias_factor(X(self.k + 1), B(0), heading_rad,
                                    np.radians(0.5))
            self._factors.append(f)
            graph.add(f)
        vals = gtsam.Values()
        vals.insert(X(self.k + 1), gtsam.Pose2(
            est.x() + dist_m * np.sin(heading_rad),
            est.y() + dist_m * np.cos(heading_rad), theta))
        self.isam.update(graph, vals)
        self.k += 1
        self._last_theta = theta

    def compass_bias_deg(self):
        """Current estimate of the compass bias (deg; heading_meas =
        heading_true + bias). None when not estimated."""
        if not self.est_bias:
            return None
        return float(np.degrees(
            self.isam.calculateEstimate().atVector(B(0))[0]))

    def take_fix(self, el_obs):
        """Solve a skyline fix around the current estimate and fuse it.
        el_obs: observed skyline on the global 0.1-deg azimuth grid (from
        skyfix.observation() on a camera frame, or a simulator).
        Returns (fix_enu, cov, margin, accepted, reasons). A solve that
        converged somewhere is not automatically a fix: ambiguous basins,
        a boundary-railed minimum, too little skyline relief, or an
        unexplainable residual all make the attempt INCONCLUSIVE -- no
        factor is added (accepted=False, reasons non-empty) and the graph
        coasts on dead reckoning."""
        est = self.isam.calculateEstimate().atPose2(X(self.k))
        center = np.array([est.x(), est.y()])

        def C(de, dn):
            el, _ = self.cm.skyline(
                self.lat0 + (center[1] + dn) / self.mlat,
                self.lon0 + (center[0] + de) / self.mlon, self.z, AZ)
            return S.cost_azshift(el_obs, el)

        half = self.fix_box_m / 2
        g = np.arange(-half, half + 1, 250.0)
        cc = np.array([[C(de, dn) for de in g] for dn in g])
        i, j = np.unravel_index(np.argmin(cc), cc.shape)
        dn0, de0 = g[i], g[j]
        margin = basin_margin(cc, g, min_sep=750.0)
        for step in (50.0, 12.5):
            best = (np.inf, de0, dn0)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    c = C(de0 + dj * step, dn0 + di * step)
                    if c < best[0]:
                        best = (c, de0 + dj * step, dn0 + di * step)
            _, de0, dn0 = best
        fix = center + np.array([de0, dn0])
        cov = laplace_cov(lambda e, n: C(e - center[0], n - center[1]),
                          fix[0], fix[1], scale=self.cov_scale)
        reasons = []
        if margin < self.min_margin:
            reasons.append(f'ambiguous: margin {margin:.2f}')
        if max(abs(de0), abs(dn0)) >= half - 250.0:
            reasons.append('minimum on search-box boundary')
        rms = np.sqrt(2 * C(de0, dn0)) * 1e3
        if rms > 12.0:
            reasons.append(f'residual {rms:.1f} mrad unexplained')
        if np.std(el_obs) * 1e3 < 1.5:
            reasons.append('insufficient skyline relief')
        if reasons:
            self.fix_fresh = False
            return fix, cov, margin, False, reasons
        f = skyline_factor(X(self.k), fix[0], fix[1], cov)
        self._factors.append(f)
        graph = gtsam.NonlinearFactorGraph()
        graph.add(f)
        if self.est_bias:
            # the fix's co-estimated azimuth shift is a DIRECT compass-
            # bias measurement: the observation is labeled with compass
            # azimuths, the DEM skyline with true ones, so the aligning
            # shift s reads the bias directly (bias = -s; the same
            # mechanism that caught the 7 deg Theodolite offset, E4h).
            # Without this the bias is only observable through a stiff
            # whole-chain-rotation mode that iSAM2 will not excite
            el_fix, _ = self.cm.skyline(
                self.lat0 + fix[1] / self.mlat,
                self.lon0 + fix[0] / self.mlon, self.z, AZ)
            _, s_best, _ = fast_photo_cost(
                el_obs, np.ones(AZ.size), el_fix,
                np.arange(-100, 101, 1), betas=np.array([0.0]))
            b_meas = -np.radians(s_best * 0.1)
            graph.add(gtsam.PriorFactorVector(
                B(0), np.array([b_meas]),
                gtsam.noiseModel.Isotropic.Sigma(1, np.radians(0.3))))
        self.isam.update(graph, gtsam.Values())
        # extra Gauss-Newton sweeps: a fix pulls on the chain-rotation /
        # compass-bias mode, which one incremental step under-corrects
        for _ in range(3):
            self.isam.update()
        self.fix_fresh = True
        return fix, cov, margin, True, []

    # ---------------- state out
    def current(self):
        """(lat, lon, cov_enu 2x2) of the current fused position"""
        est = self.isam.calculateEstimate().atPose2(X(self.k))
        cov3 = self.isam.marginalCovariance(X(self.k))
        th = est.theta()
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th), np.cos(th)]])
        cov = R @ cov3[:2, :2] @ R.T     # tangent -> ENU
        return (self.lat0 + est.y() / self.mlat,
                self.lon0 + est.x() / self.mlon, cov)

    # ---------------- NMEA 0183 out
    @staticmethod
    def _nmea(body):
        cs = 0
        for ch in body:
            cs ^= ord(ch)
        return f'${body}*{cs:02X}'

    @staticmethod
    def _dm(x, degdigits):
        d = int(abs(x))
        m = (abs(x) - d) * 60.0
        return f'{d:0{degdigits}d}{m:07.4f}'

    def nmea_gga(self, utc_hms):
        lat, lon, cov = self.current()
        hdop = float(np.sqrt(np.trace(cov)) / 5.0)  # sigma_m -> HDOP-ish
        # quality 1 = valid fix; 6 = estimated (dead reckoning) while the
        # latest skyline-fix attempt was inconclusive
        q = 1 if self.fix_fresh else 6
        body = (f'GPGGA,{utc_hms},{self._dm(lat, 2)},'
                f'{"N" if lat >= 0 else "S"},{self._dm(lon, 3)},'
                f'{"E" if lon >= 0 else "W"},{q},08,{hdop:.1f},'
                f'{self.z:.1f},M,0.0,M,,')
        return self._nmea(body)

    def nmea_rmc(self, utc_hms, date_ddmmyy, speed_kn, course_deg):
        lat, lon, _ = self.current()
        body = (f'GPRMC,{utc_hms},A,{self._dm(lat, 2)},'
                f'{"N" if lat >= 0 else "S"},{self._dm(lon, 3)},'
                f'{"E" if lon >= 0 else "W"},{speed_kn:.1f},'
                f'{course_deg:.1f},{date_ddmmyy},,,A')
        return self._nmea(body)
