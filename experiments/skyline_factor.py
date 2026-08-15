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
