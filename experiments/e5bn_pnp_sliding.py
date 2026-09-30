#!/usr/bin/env python3
"""E5bn: point-to-point versus point-to-line PnP on a silhouette.

The proposed pipeline (the operator's abstract, E5bk/E5bl) renders the
DEM skyline at the INS pose, knows the 3D crest point behind every
predicted pixel, pairs ~100 predicted points with the observed skyline,
and hands the 2D-3D pairs to PnP. E5bl measured that a silhouette point
is localised perpendicular to the curve and NOT along it, by a factor of
8-26. This measures what that does to the pose.

Setup, on the Bosphorus frame (the one with a hand-drawn skyline and a
known-good 54 m solve):
  * truth is the EXIF GPS position; the camera height is fixed at 7 m
    and the solve is 5-DoF: north, east, heading, pitch, roll;
  * the observation is the operator's curve AS DRAWN, roll included, so
    PnP has to recover the +1.75 deg roll E5bi found;
  * each start pose is the truth displaced by 0/250/500 m in eight
    directions with a heading error of -2/0/+2 deg and zero pitch/roll;
  * every iteration re-ray-casts the DEM from the current pose, takes
    the crest 3D points, projects them, samples 100 uniformly across
    the frame, and pairs each with the observed curve IN THE SAME
    COLUMN -- the natural pairing for a skyline, and the one the
    abstract describes. A MAD filter drops outliers;
  * point-to-point minimises the full 2D reprojection error of each
    pair; point-to-line minimises only its component along the observed
    curve's local normal. Same pairs, same robust loss, same iteration
    budget, no prior on either.

The question is how much of the final position is the START position
talking back. Regressing final offset on initial offset gives it as a
slope: 1 means the solve stayed where it was put, 0 means it forgot.
"""

import json
import os

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import skyline as S

LAT, LON, Z = 41.12590, 29.07076, 7.0
FOV = 73.7
W, H = 4032, 3024
F = (W / 2) / np.tan(np.radians(FOV) / 2)
CREST_DH = 9.0
N_PTS = 100
N_ITER = 10
HAND = 'out/bos/user_skyline.npy'
MLAT, MLON = S.meters_per_degree(LAT)
CM = S.CMarcher(os.path.expanduser('~/.horizonator/DEMs_COP30'),
                (LAT - .6, LAT + .6), (LON - .6, LON + .6), d_min=1000.)

# observation: the hand-drawn curve in centred image coords (u right, v up)
_rows = np.load(HAND)
U_OBS = np.arange(W) - (W - 1) / 2
V_OBS = (H - 1) / 2 - _rows
# local slope of the observed curve, for the point-to-line normal
_dv = np.gradient(np.convolve(V_OBS, np.ones(31) / 31, mode='same'))


def axes(psi, theta, phi):
    """Camera fwd / image-right / image-up in ENU, skyfix's roll sign."""
    p, t = np.radians(psi), np.radians(theta)
    fwd = np.array([np.sin(p) * np.cos(t), np.cos(p) * np.cos(t), np.sin(t)])
    right = np.array([np.cos(p), -np.sin(p), 0.0])
    up = np.cross(right, fwd)
    c, s = np.cos(np.radians(phi)), np.sin(np.radians(phi))
    # skyfix: level = (u c - v s, u s + v c); invert for image axes
    r_img = c * right + s * up
    u_img = -s * right + c * up
    return fwd, r_img, u_img


def project(X, pose):
    dn, de, psi, theta, phi = pose
    d = X - np.array([de, dn, Z])
    fwd, r, u = axes(psi, theta, phi)
    xf = d @ fwd
    return F * (d @ r) / xf, F * (d @ u) / xf


def crest_points(pose):
    """Ray-cast the silhouette from the pose; 3D crest point per azimuth."""
    dn, de, psi = pose[0], pose[1], pose[2]
    az = psi + np.arange(-FOV / 2 - 8, FOV / 2 + 8, 0.02)
    el, r = CM.skyline(LAT + dn / MLAT, LON + de / MLON, Z, az)
    ok = np.isfinite(el) & np.isfinite(r) & (r > 0)
    a = np.radians(az[ok])
    rr = r[ok]
    # apparent height: r*tan(el) already carries curvature + refraction
    up = Z + rr * np.tan(el[ok]) + CREST_DH
    return np.stack([de + rr * np.sin(a), dn + rr * np.cos(a), up], axis=1)


def correspondences(pose):
    X = crest_points(pose)
    u, v = project(X, pose)
    inside = (u > U_OBS[0]) & (u < U_OBS[-1])
    X, u, v = X[inside], u[inside], v[inside]
    order = np.argsort(u)
    X, u, v = X[order], u[order], v[order]
    targets = np.linspace(U_OBS[0] + 20, U_OBS[-1] - 20, N_PTS)
    idx = np.unique(np.clip(np.searchsorted(u, targets), 0, u.size - 1))
    X, u, v = X[idx], u[idx], v[idx]
    col = np.clip(np.rint(u + (W - 1) / 2).astype(int), 0, W - 1)
    uo, vo = U_OBS[col], V_OBS[col]            # SAME COLUMN pairing
    res = vo - v
    med = np.median(res)
    mad = 1.4826 * np.median(np.abs(res - med)) + 1e-6
    keep = np.abs(res - med) < 3 * mad
    t = np.stack([np.ones(col.size), _dv[col]], axis=1)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return X[keep], uo[keep], vo[keep], n[keep]


def solve(pose0, mode):
    pose = np.array(pose0, float)
    for it in range(N_ITER):
        X, uo, vo, n = correspondences(pose)

        def resid(p):
            u, v = project(X, p)
            du, dv = uo - u, vo - v
            if mode == 'p2p':
                return np.concatenate([du, dv])
            return n[:, 0] * du + n[:, 1] * dv

        sol = least_squares(resid, pose, loss='soft_l1', f_scale=3.0,
                            x_scale=[50, 50, 0.5, 0.5, 0.5])
        step = np.hypot(*(sol.x[:2] - pose[:2]))
        pose = sol.x
        if step < 1.0:
            break
    return pose, it + 1


def main():
    starts = []
    for rad in (0, 250, 500):
        dirs = [0] if rad == 0 else range(0, 360, 45)
        for b in dirs:
            for dh in (-2.0, 0.0, 2.0):
                starts.append((rad * np.cos(np.radians(b)),
                               rad * np.sin(np.radians(b)), 48.0 + dh))
    rows = []
    for dn0, de0, h0 in starts:
        for mode in ('p2p', 'p2l'):
            pose, iters = solve((dn0, de0, h0, 0.0, 0.0), mode)
            rows.append(dict(mode=mode, dn0=dn0, de0=de0, h0=h0,
                             dn=pose[0], de=pose[1], psi=pose[2],
                             theta=pose[3], phi=pose[4], iters=iters,
                             err=float(np.hypot(pose[0], pose[1]))))
        a, b = rows[-2], rows[-1]
        print('start %+5.0f %+5.0f h%5.1f | p2p -> %5.0f m roll %+.2f | '
              'p2l -> %5.0f m roll %+.2f'
              % (dn0, de0, h0, a['err'], a['phi'], b['err'], b['phi']),
              flush=True)
    json.dump(rows, open('out/bos/pnp_sliding.json', 'w'), indent=1)

    print()
    for mode in ('p2p', 'p2l'):
        R = [r for r in rows if r['mode'] == mode]
        x0 = np.array([[r['dn0'], r['de0']] for r in R])
        x1 = np.array([[r['dn'], r['de']] for r in R])
        # slope of final on initial, pooled over both axes
        A = x0.ravel()
        slope = float(A @ x1.ravel() / (A @ A))
        err = np.array([r['err'] for r in R])
        far = np.array([np.hypot(r['dn0'], r['de0']) >= 499 for r in R])
        print('%s  talk-back slope %.2f | final error median %4.0f m, '
              'p90 %4.0f m | from 500 m starts: median %4.0f m | roll median %+.2f'
              % (mode, slope, np.median(err), np.percentile(err, 90),
                 np.median(err[far]), np.median([r['phi'] for r in R])))
    draw(rows)


def draw(rows):
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), sharex=True, sharey=True)
    for ax, mode, ttl in ((axs[0], 'p2p', 'nokta-noktaya'),
                          (axs[1], 'p2l', 'nokta-doğru')):
        R = [r for r in rows if r['mode'] == mode]
        for r in R:
            ax.annotate('', xy=(r['de'], r['dn']), xytext=(r['de0'], r['dn0']),
                        arrowprops=dict(arrowstyle='->', lw=0.8,
                                        color='#8A8F98', alpha=0.8))
        ax.scatter([r['de0'] for r in R], [r['dn0'] for r in R], s=16,
                   color='#8A8F98', label='başlangıç pozu', zorder=3)
        ax.scatter([r['de'] for r in R], [r['dn'] for r in R], s=22,
                   color='#C0392B' if mode == 'p2p' else '#1E7F5C',
                   label='PnP sonucu', zorder=4)
        ax.plot([0], [0], marker='*', ms=16, color='#1A2732', zorder=5,
                label='GPS gerçeği')
        ax.set_title(ttl, fontsize=13)
        ax.set_xlabel('doğu (m)')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower left', fontsize=9)
    axs[0].set_ylabel('kuzey (m)')
    fig.suptitle('Boğaz karesi — aynı sütun eşleştirmesi, aynı iterasyon bütçesi',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig('out/bos/BOS_pnp_sliding.png', dpi=110)
    print('wrote out/bos/BOS_pnp_sliding.png')


if __name__ == '__main__':
    main()
