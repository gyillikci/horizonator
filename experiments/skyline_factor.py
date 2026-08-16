#!/usr/bin/env python3
"""SkylineFactor: DEM skyline fixes as GTSAM factors.

A skyline fix (position + covariance from the match-cost curvature, as
produced by skyfix.py / the solvers in this directory) enters a factor
graph as a unary factor on a Pose2 in a local ENU frame. The anisotropic
covariance matters: a fix with land in only one sector has an
uncertainty ellipse elongated along the line of sight (study doc E1
site B), and the graph exploits exactly that structure when fusing with
dead reckoning.

Works with the stock `pip install gtsam` wheel via CustomFactor. A C++
port for the user's gtsam fork is mechanical: error = t(pose) - fix,
Jacobian [R(theta), 0].
"""

import numpy as np
import gtsam


def skyline_factor(key, fix_e, fix_n, cov):
    """Unary Pose2 factor for a skyline fix at ENU (fix_e, fix_n) meters
    with 2x2 covariance cov [[see, sen],[sen, snn]] in (east, north)."""
    noise = gtsam.noiseModel.Gaussian.Covariance(np.asarray(cov, float))

    def error_func(this, values, H):
        pose = values.atPose2(this.keys()[0])
        e = np.array([pose.x() - fix_e, pose.y() - fix_n])
        if H is not None:
            th = pose.theta()
            # retract: t' = t + R(theta) [dx,dy] -> de/d[dx,dy] = R(theta)
            H[0] = np.array([[np.cos(th), -np.sin(th), 0.0],
                             [np.sin(th), np.cos(th), 0.0]])
        return e

    return gtsam.CustomFactor(noise, gtsam.KeyVector([key]), error_func)


def heading_bias_factor(pose_key, bias_key, heading_meas_rad, sigma_rad):
    """Compass-heading measurement with a shared bias VARIABLE.

    Imported lesson from the parallel study branch (celestial-navigation
    claude/iphone-celestial-sighting-imu-ctwbnf): systematic sensor
    biases belong in the graph as estimated variables, not smeared into
    per-measurement sigmas — on that branch's measured device biases the
    difference was a fix 12.4 km wrong vs 2 m wrong.

    Model: heading_meas = heading_true + bias, with Pose2 convention
    theta = pi/2 - heading_true, so the residual is
        err = wrap(theta - (pi/2 - heading_meas) - b)
    on keys (pose, bias); bias is a 1-d variable shared by all heading
    factors, made observable by the skyline fixes anchoring the chain."""
    noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma_rad)
    target = np.pi / 2 - heading_meas_rad

    def error_func(this, values, H):
        theta = values.atPose2(this.keys()[0]).theta()
        b = values.atVector(this.keys()[1])[0]
        e = (theta - target - b + np.pi) % (2 * np.pi) - np.pi
        if H is not None:
            H[0] = np.array([[0.0, 0.0, 1.0]])
            H[1] = np.array([[-1.0]])
        return np.array([e])

    return gtsam.CustomFactor(noise,
                              gtsam.KeyVector([pose_key, bias_key]),
                              error_func)


def depth_factor(key, depth_meas, bathy_fn, sigma, h=30.0):
    """Echo-sounder depth as a unary Pose2 factor: err = predicted
    depth at the pose's position (from a bathymetric grid, positive
    down) minus the measured depth. A classic TRN observable that is
    fully decorrelated from the skyline channel — it works in fog, at
    night, and with zero terrain relief. Jacobian by finite differences
    of the grid (h meters)."""
    noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma)

    def error_func(this, values, H):
        pose = values.atPose2(this.keys()[0])
        e, n = pose.x(), pose.y()
        d0 = bathy_fn(e, n)
        if H is not None:
            de = (bathy_fn(e + h, n) - bathy_fn(e - h, n)) / (2 * h)
            dn = (bathy_fn(e, n + h) - bathy_fn(e, n - h)) / (2 * h)
            th = pose.theta()
            c, s = np.cos(th), np.sin(th)
            # tangent (dx,dy) moves the position by R(theta)[dx,dy]
            H[0] = np.array([[de * c + dn * s, -de * s + dn * c, 0.0]])
        return np.array([d0 - depth_meas])

    return gtsam.CustomFactor(noise, gtsam.KeyVector([key]), error_func)


def light_bearing_factor(pose_key, bias_key, lm_e, lm_n, bearing_meas,
                         sigma):
    """Bearing to an IDENTIFIED charted light (compass convention, CW
    from north; bearing_meas = true bearing + compass bias). The night
    channel: a lighthouse identified by its flash characteristic is a
    surveyed point, and its bearing is a line of position — the skyline
    instrument's after-dark replacement. Shares the compass-bias
    variable with the heading factors."""
    noise = gtsam.noiseModel.Isotropic.Sigma(1, sigma)

    def error_func(this, values, H):
        pose = values.atPose2(this.keys()[0])
        b = values.atVector(this.keys()[1])[0]
        u = lm_e - pose.x()
        v = lm_n - pose.y()
        r2 = u * u + v * v
        pred = np.arctan2(u, v)
        e = (pred - (bearing_meas - b) + np.pi) % (2 * np.pi) - np.pi
        if H is not None:
            th = pose.theta()
            c, s = np.cos(th), np.sin(th)
            dbx, dby = -v / r2, u / r2
            H[0] = np.array([[dbx * c + dby * s,
                              -dbx * s + dby * c, 0.0]])
            H[1] = np.array([[1.0]])
        return np.array([e])

    return gtsam.CustomFactor(noise,
                              gtsam.KeyVector([pose_key, bias_key]),
                              error_func)


def laplace_cov(cost_fn, e0, n0, h=25.0, floor=8.0, scale=1.0):
    """2x2 (east,north) covariance from the local quadratic shape of the
    match cost around its minimum (the same heuristic skyfix.py uses).
    floor: minimum sigma in meters (DEM/extraction error floor).
    scale: empirical calibration. The raw heuristic is ~10x pessimistic
    against measured closed-loop accuracy (study doc E1: measured 7-15 m
    vs ~50-100 m heuristic sigmas); scale=0.02 calibrates sigma by ~1/7
    to match. Recalibrate against E4c-style end-to-end runs when the
    observation source changes."""
    c0 = cost_fn(e0, n0)
    cee = (cost_fn(e0 + h, n0) - 2 * c0 + cost_fn(e0 - h, n0)) / h ** 2
    cnn = (cost_fn(e0, n0 + h) - 2 * c0 + cost_fn(e0, n0 - h)) / h ** 2
    cen = (cost_fn(e0 + h, n0 + h) - cost_fn(e0 + h, n0 - h)
           - cost_fn(e0 - h, n0 + h) + cost_fn(e0 - h, n0 - h)) / (4 * h ** 2)
    Hm = np.array([[cee, cen], [cen, cnn]])
    try:
        cov = 2.0 * c0 * np.linalg.inv(Hm)
    except np.linalg.LinAlgError:
        cov = np.diag([1e4, 1e4])
    # symmetrize, floor the eigenvalues
    cov = 0.5 * (cov + cov.T) * scale
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, floor ** 2)
    return V @ np.diag(w) @ V.T
