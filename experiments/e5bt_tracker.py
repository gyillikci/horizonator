#!/usr/bin/env python3
"""E5bt: INS + skyline + a hill tracker with handover.

E5bs aided a MEMS INS with skyline photographs. This adds what an
operator with a telephoto camera would do: lock a 250 x 250 px tracker
on a known hill, keep it there while passing, and hand over to the next
hill ahead before the first one goes abeam - all along the course.

  * landmarks: prominent summits found automatically in the DEM
    (>= 150 m high, >= 120 m above their 3-km surroundings). The MAP
    position is the GLO-30 summit; the TRUE tracked point is the SRTM1
    summit plus an offset - a tracker locks on a texture near the top,
    not on the surveyed peak (30 m horizontal, 10 m vertical). So each
    hill is a GTSAM variable with the map position as its prior
    (40 m / 15 m) and is refined by the platform's own bearings;
  * the tracker camera: 10 deg telephoto on a gimbal; the measurement is
    the hill's azimuth and elevation in the BODY frame (gimbal encoders
    + the tracker's pixel), 0.4 mrad jitter + 0.5 mrad encoder, every
    keyframe (drone 5 s, vessel 20 s); Earth curvature with refraction
    in both truth and model;
  * handover: the track ends when the hill passes 100 deg off the bow,
    is hidden by nearer terrain (checked on the TRUE DEM), or comes
    within 1.5 km; the next hill is the visible one most nearly ahead
    (within 60 deg, at least 3 km). Acquisition points the camera at
    the hill's ESTIMATED bearing; if that is off by more than half the
    10 deg field of view it fails and is retried at the next keyframe.

Variants, on the identical IMU stream, skyline noise and landmarks:
  A  INS + skyline photos                  (E5bs, finer keyframes)
  B  INS + skyline photos + hill tracker
  C  INS + hill tracker only

    python3 e5bt_tracker.py run [--only drone_mid_nominal_B ...]
    python3 e5bt_tracker.py report
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'out', 'tracker')
# git clone https://github.com/Aceinna/gnss-ins-sim  next to the
# repository checkouts (<parent>/ext/gnss-ins-sim), or set $GNSS_INS_SIM
SIMDIR = os.environ.get('GNSS_INS_SIM', os.path.join(
    os.path.dirname(os.path.dirname(HERE)), 'ext', 'gnss-ins-sim'))
if not os.path.isdir(SIMDIR):
    SIMDIR = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(HERE))), 'ext', 'gnss-ins-sim')

LAT0, LON0 = 36.95, 27.25
FOV = 73.7
GRID_HALF, GRID_STEP = 2000.0, 100.0
HW = 6.0
LAM = 20 / (3e-3) ** 2

FS = 100.0
D2R = math.pi / 180

PLAT = {
    # start (east, north) m from LAT0/LON0; legs: (yaw change deg, seconds)
    'drone': dict(start=(-6000., -3000.), alt=300.0, speed=20.0, yaw0=60.0,
                  dt=5.0, n=180, photo_every=6, log=None,
                  profile=[(0, 330), (40, 15), (0, 285), (40, 15), (0, 265)],
                  baro_sd=3.0, baro_bias=5.0),
    'vessel': dict(start=(-5000., -2200.), alt=5.0, speed=3.0, yaw0=80.0,
                   # keyframes every 20 s carry the speed log and the
                   # no-sideslip constraint; a photograph every 6th
                   dt=20.0, n=180, photo_every=6,
                   log=dict(scale=1.03, sd=0.05, side_sd=0.3, vert_sd=0.1),
                   profile=[(0, 1800), (30, 60), (0, 1760)],
                   baro_sd=None, baro_bias=0.0),
}
NOISE = {
    'nominal': dict(white=1e-3, corr=1e-3, outlier=0.0, mount=0.1),
    'field': dict(white=2e-3, corr=3e-3, outlier=0.10, mount=0.3),
}
GRADE = {   # gnss-ins-sim built-in models; GTSAM gets the same numbers
    'mid': dict(arw=0.25, gbi=3.5, vrw=0.03, abi=5e-5, tau=100.0),
    'low': dict(arw=0.75, gbi=10.0, vrw=0.05, abi=2e-4, tau=100.0),
}


TRK = dict(jitter=0.4e-3, encoder=0.5e-3, max_rel=100.0, ahead=60.0,
           min_range=1500.0, start_range=3000.0, max_range=25000.0,
           acq=5.0, aim_h=30.0, aim_v=10.0, prior_h=40.0, prior_v=15.0)
K_REFR = 0.13
R_EARTH = 6371000.0
VARIANTS = {'A': (True, False), 'B': (True, True), 'C': (False, True)}


def peaks():
    """Prominent summits near the routes: (lat, lon, h_map, h_true,
    lat_true, lon_true), map from GLO-30, truth from SRTM1. Cached."""
    cache = os.path.join(OUT, 'hills.json')
    if os.path.exists(cache):
        return [tuple(r) for r in json.load(open(cache))]
    import skyline as S
    dm = S.Dem(os.path.expanduser('~/.horizonator/DEMs_COP30'))
    dt_ = S.Dem(os.path.expanduser('~/.horizonator/DEMs_SRTM1'))
    m, la_nw, lo_nw, dpp = S.build_mosaic(dm, 36.6, 37.25, 26.9, 27.7)
    r0, r1 = int((la_nw - 37.25) / dpp), int((la_nw - 36.6) / dpp)
    c0, c1 = int((26.9 - lo_nw) / dpp), int((27.7 - lo_nw) / dpp)
    sub = m[r0:r1:3, c0:c1:3]
    mx = ndimage.maximum_filter(sub, size=25)
    mn = ndimage.minimum_filter(sub, size=35)
    ii, jj = np.nonzero((sub == mx) & (sub >= 150) & (sub - mn >= 120))
    out = []
    for i, j in zip(ii, jj):
        r, c = r0 + 3 * i, c0 + 3 * j
        w = m[r - 3:r + 4, c - 3:c + 4]            # refine at full res
        a, b = np.unravel_index(np.argmax(w), w.shape)
        lat = la_nw - (r - 3 + a) * dpp
        lon = lo_nw + (c - 3 + b) * dpp
        if any(math.hypot(lat - q[0], lon - q[1]) < 0.005 for q in out):
            continue
        # truth: the SRTM1 maximum within ~150 m
        g = np.linspace(-0.0015, 0.0015, 13)
        LA, LO = np.meshgrid(lat + g, lon + g, indexing='ij')
        ht = dt_.sample(LA, LO)
        k = np.unravel_index(np.argmax(ht), ht.shape)
        out.append((lat, lon, float(w.max()), float(ht[k]),
                    float(LA[k]), float(LO[k])))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(cache, 'w'))
    return out


def u(deg):
    r = np.radians(deg)
    return np.array([np.sin(r), np.cos(r)])


def gen_imu(plat, grade, seed, workdir):
    """Run gnss-ins-sim; return truth (t, NED pos, NED vel, ypr) and IMU."""
    sys.path.insert(0, SIMDIR)
    from gnss_ins_sim.sim import imu_model, ins_sim
    import skyline as S
    pc = PLAT[plat]
    mlat, mlon = S.meters_per_degree(LAT0)
    lat = LAT0 + pc['start'][1] / mlat
    lon = LON0 + pc['start'][0] / mlon
    path = os.path.join(workdir, 'motion_%s.csv' % plat)
    with open(path, 'w') as f:
        f.write('ini lat (deg),ini lon (deg),ini alt (m),ini vx_body (m/s),'
                'ini vy_body (m/s),ini vz_body (m/s),ini yaw (deg),'
                'ini pitch (deg),ini roll (deg)\n')
        f.write('%.6f,%.6f,%.1f,%.2f,0,0,%.1f,0,0\n'
                % (lat, lon, pc['alt'], pc['speed'], pc['yaw0']))
        f.write('command type,yaw (deg),pitch (deg),roll (deg),vx_body (m/s),'
                'vy_body (m/s),vz_body (m/s),command duration (s),'
                'GPS visibility\n')
        for dyaw, dur in pc['profile']:
            if dyaw:
                f.write('3,%g,0,0,0,0,0,%g,0\n' % (dyaw, dur))
            else:
                f.write('1,0,0,0,0,0,0,%g,0\n' % dur)
    np.random.seed(seed)
    imu = imu_model.IMU(accuracy=grade + '-accuracy', axis=6, gps=False)
    sim = ins_sim.Sim([FS, 0.0, 0.0], path, ref_frame=1, imu=imu,
                      mode=None, env=None, algorithm=None)
    sim.run(1)

    def get(k):
        v = sim.dmgr.get_data_all(k).data
        return np.asarray(v[0] if isinstance(v, dict) else v, float)
    pos = get('ref_pos')
    # gravity of the virtual-inertial frame, read off the level, steady
    # start rather than assumed
    return dict(t=get('time'), ned=pos - pos[0], vel=get('ref_vel'),
                ypr=get('ref_att_euler'), acc=get('accel'), gyr=get('gyro'),
                alt0=pc['alt'], g=float(-get('ref_accel')[0, 2]))


def simulate(name, seed=20261023):
    import gtsam
    import skyline as S
    import skyfix as F
    plat, grade, lvl, variant = name.split('_')
    use_sky, use_trk = VARIANTS[variant]
    scen = '_'.join((plat, grade, lvl))
    pc, nz, gr = PLAT[plat], NOISE[lvl], GRADE[grade]
    # separate streams, seeded by the SCENARIO, so A, B and C see the same
    # IMU, skyline noise, barometer, log and tracker noise
    base = seed + sum(map(ord, scen))
    rng = np.random.default_rng(base + 1)            # skyline photos
    rng_b = np.random.default_rng(base + 2)          # baro / log
    rng_t = np.random.default_rng(base + 3)          # tracker
    work = os.path.join(OUT, 'work_' + name)
    os.makedirs(work, exist_ok=True)
    tr = gen_imu(plat, grade, base, work)
    mlat, mlon = S.meters_per_degree(LAT0)
    dem = lambda d: os.path.expanduser('~/.horizonator/' + d)
    cm_true = S.CMarcher(dem('DEMs_SRTM1'), (36.4, 37.4), (26.6, 27.9),
                         d_min=300.0)
    cm = S.CMarcher(dem('DEMs_COP30'), (36.4, 37.4), (26.6, 27.9),
                    d_min=300.0)
    AZ = F.AZ
    st0 = np.array(pc['start'])
    enu = lambda ned: st0 + np.array([ned[1], ned[0]])   # east, north
    to_ll = lambda p: (LAT0 + p[1] / mlat, LON0 + p[0] / mlon)
    az_rel = np.arange(-FOV / 2, FOV / 2 + 1e-9, 0.1)
    shifts = np.arange(-int(HW / 0.1), int(HW / 0.1) + 1, 2)

    def aim(p, z):
        best, b_az = -1, 0.0
        la, lo = to_ll(p)
        for c in np.arange(0, 360, 15.0):
            el, _ = cm_true.skyline(la, lo, z, (c + az_rel[::10]) % 360)
            m = np.isfinite(el)
            if m.mean() < .8:
                continue
            if np.std(el[m]) > best:
                best, b_az = np.std(el[m]), c
        return b_az

    def observe(p, z, c_true, c_meas, pitch_err, roll_err):
        la, lo = to_ll(p)
        el, _ = cm_true.skyline(la, lo, z, (c_true + az_rel) % 360)
        el = np.where(np.isfinite(el), el, np.nanmin(el))
        el = el + pitch_err - roll_err * np.tan(np.radians(az_rel))
        el = el + rng.normal(0, nz['white'], el.size)
        cor = ndimage.gaussian_filter1d(rng.normal(0, 1, el.size), 10)
        el = el + nz['corr'] * cor / (cor.std() + 1e-12)
        if nz['outlier']:
            for _ in range(rng.poisson(nz['outlier'] * el.size / 40)):
                i0 = rng.integers(0, el.size - 40)
                el[i0:i0 + 40] += rng.uniform(5e-3, 25e-3)
        el_obs, w = np.zeros(AZ.size), np.zeros(AZ.size)
        idx = np.round(((c_meas + az_rel + 180.0) % 360.0 - 0.05) / 0.1
                       ).astype(int) % AZ.size
        el_obs[idx], w[idx] = el, 1.0
        return el_obs, w

    def solve(el_obs, w, c_meas, centre, z_s):
        need = np.abs(((AZ - c_meas + 180) % 360) - 180) <= FOV / 2 + HW + 1
        azn = AZ[need]
        g = np.arange(-GRID_HALF, GRID_HALF + 1, GRID_STEP)
        el_syn = np.zeros(AZ.size)

        def C(pe, pn):
            la, lo = to_ll((pe, pn))
            e, _ = cm.skyline(la, lo, z_s, azn)
            el_syn[:] = 0.0
            el_syn[need] = np.where(np.isfinite(e), e, 0.0)
            return F.fast_photo_cost(el_obs, w, el_syn, shifts, F.BETAS)
        J = np.empty((g.size, g.size))
        for i, dn in enumerate(g):
            for j, de in enumerate(g):
                J[i, j] = C(centre[0] + de, centre[1] + dn)[0]
        i, j = np.unravel_index(np.argmin(J), J.shape)
        c0, s0, _ = C(centre[0] + g[j], centre[1] + g[i])
        best = (c0, centre[0] + g[j], centre[1] + g[i], s0 * 0.1)
        for step in (50.0, 25.0, 12.5):
            e0, n0 = best[1], best[2]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    c, s, _ = C(e0 + dj * step, n0 + di * step)
                    if c < best[0]:
                        best = (c, e0 + dj * step, n0 + di * step, s * 0.1)
        i = min(max(i, 1), g.size - 2)
        j = min(max(j, 1), g.size - 2)
        h = GRID_STEP
        Hee = (J[i, j + 1] - 2 * J[i, j] + J[i, j - 1]) / h ** 2
        Hnn = (J[i + 1, j] - 2 * J[i, j] + J[i - 1, j]) / h ** 2
        Hen = (J[i + 1, j + 1] - J[i + 1, j - 1] - J[i - 1, j + 1]
               + J[i - 1, j - 1]) / (4 * h ** 2)
        try:
            cov = np.linalg.inv(LAM * np.array([[Hee, Hen], [Hen, Hnn]]))
            if np.linalg.eigvalsh(cov).min() <= 0:
                raise np.linalg.LinAlgError
            cov = cov + np.eye(2) * 30.0 ** 2
        except np.linalg.LinAlgError:
            cov = np.eye(2) * 1000.0 ** 2
        return np.array(best[1:3]), best[3], cov, \
            F.basin_margin(J, g, min_sep=4 * GRID_STEP)

    # ---- GTSAM set-up, noise from the same IMU grade
    P = gtsam.PreintegrationCombinedParams.MakeSharedD(tr['g'])
    P.setGyroscopeCovariance(np.eye(3) * (gr['arw'] * D2R / 60) ** 2)
    P.setAccelerometerCovariance(np.eye(3) * (gr['vrw'] / 60) ** 2)
    P.setIntegrationCovariance(np.eye(3) * 1e-8)
    P.setBiasOmegaCovariance(np.eye(3) * (gr['gbi'] * D2R / 3600) ** 2
                             * 2 / gr['tau'])
    P.setBiasAccCovariance(np.eye(3) * gr['abi'] ** 2 * 2 / gr['tau'])
    X, V, B = (lambda k: gtsam.symbol('x', k), lambda k: gtsam.symbol('v', k),
               lambda k: gtsam.symbol('b', k))
    robust = lambda nm: gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345), nm)

    def yaw_of(pose):
        Rm = pose.rotation().matrix()
        return math.atan2(Rm[1, 0], Rm[0, 0])

    def yaw_factor(key, yaw_meas, sigma):
        def err(this, v, H):
            p = v.atPose3(this.keys()[0])
            e = (yaw_of(p) - yaw_meas + math.pi) % (2 * math.pi) - math.pi
            if H is not None:
                J = np.zeros((1, 6))
                for d in range(3):
                    dv = np.zeros(6)
                    dv[d] = 1e-6
                    y2 = yaw_of(p.retract(dv))
                    J[0, d] = ((y2 - yaw_of(p) + math.pi) % (2 * math.pi)
                               - math.pi) / 1e-6
                H[0] = J
            return np.array([e])
        return gtsam.CustomFactor(
            robust(gtsam.noiseModel.Isotropic.Sigma(1, sigma)), [key], err)

    def down_factor(key, d_meas, sigma):
        def err(this, v, H):
            p = v.atPose3(this.keys()[0])
            if H is not None:
                J = np.zeros((1, 6))
                J[0, 3:] = p.rotation().matrix()[2, :]
                H[0] = J
            return np.array([p.translation()[2] - d_meas])
        return gtsam.CustomFactor(gtsam.noiseModel.Isotropic.Sigma(1, sigma),
                                  [key], err)

    t, ned, vel, ypr, acc, gyr = (tr[k] for k in
                                  ('t', 'ned', 'vel', 'ypr', 'acc', 'gyr'))
    stride = int(round(pc['dt'] * FS))
    pose_true = lambda i: gtsam.Pose3(gtsam.Rot3.Ypr(*ypr[i]), ned[i])
    # known start: position 5 m, yaw 1 deg, level 0.2 deg, speed 0.3 m/s
    x0, v0 = pose_true(0), vel[0]
    b0 = gtsam.imuBias.ConstantBias()
    # QR: with 5-s keyframes and a tight MEMS bias random walk the
    # system is poorly conditioned and Cholesky declares it indeterminate
    ip = gtsam.ISAM2Params()
    ip.setFactorization('QR')
    isam = gtsam.ISAM2(ip)
    fg, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
    fg.add(gtsam.PriorFactorPose3(X(0), x0, gtsam.noiseModel.Diagonal.Sigmas(
        [0.2 * D2R, 0.2 * D2R, 1.0 * D2R, 5, 5, 2])))
    fg.add(gtsam.PriorFactorVector(V(0), v0,
                                   gtsam.noiseModel.Isotropic.Sigma(3, 0.3)))
    fg.add(gtsam.PriorFactorConstantBias(B(0), b0, gtsam.noiseModel.Diagonal.Sigmas(
        np.r_[[gr['abi'] * 20] * 3, [gr['gbi'] * 3 * D2R / 3600] * 3])))
    vals.insert(X(0), x0)
    vals.insert(V(0), v0)
    vals.insert(B(0), b0)
    isam.update(fg, vals)
    est = isam.calculateEstimate()
    # free INS: the same IMU, integrated from the same known start, no aiding
    free_state = gtsam.NavState(x0, v0)

    # ---- landmarks in the navigation frame (NED from the start point)
    LMS = []
    lat_s, lon_s = to_ll(st0)
    rng_l = np.random.default_rng(seed)                # same for all runs
    for (la, lo, hm, ht, lat_t, lon_t) in peaks():
        off = rng_l.normal(0, TRK['aim_h'], 2)
        dv = abs(rng_l.normal(0, TRK['aim_v']))
        mp = np.array([(la - lat_s) * mlat, (lo - lon_s) * mlon,
                       tr['alt0'] - hm])
        tp = np.array([(lat_t - lat_s) * mlat + off[0],
                       (lon_t - lon_s) * mlon + off[1],
                       tr['alt0'] - (ht - dv)])
        LMS.append(dict(map=mp, true=tp))
    L = lambda j: gtsam.symbol('l', j)
    sig_b = math.hypot(TRK['jitter'], TRK['encoder'])

    def body_azel(pose_R, pos, lm):
        """azimuth/elevation of lm in the body frame, with the apparent
        drop of a distant point from Earth curvature and refraction"""
        d = lm - pos
        rng_h = math.hypot(d[0], d[1])
        d = d + np.array([0.0, 0.0, rng_h ** 2 * (1 - K_REFR) / (2 * R_EARTH)])
        b = pose_R.T @ d
        return np.array([math.atan2(b[1], b[0]),
                         math.atan2(-b[2], math.hypot(b[0], b[1]))])

    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def bearing_factor(kx, kl, meas):
        def err(this, v, H):
            p = v.atPose3(this.keys()[0])
            lm = v.atPoint3(this.keys()[1])
            f = lambda pp, ll: body_azel(pp.rotation().matrix(),
                                         np.asarray(pp.translation()), ll)
            e0 = f(p, lm)
            e = np.array([wrap(e0[0] - meas[0]), e0[1] - meas[1]])
            if H is not None:
                Jp, Jl = np.zeros((2, 6)), np.zeros((2, 3))
                for d in range(6):
                    dv = np.zeros(6)
                    dv[d] = 1e-6
                    e1 = f(p.retract(dv), lm)
                    Jp[:, d] = [wrap(e1[0] - e0[0]) / 1e-6,
                                (e1[1] - e0[1]) / 1e-6]
                for d in range(3):
                    dl = np.zeros(3)
                    dl[d] = 1e-3
                    e1 = f(p, lm + dl)
                    Jl[:, d] = [wrap(e1[0] - e0[0]) / 1e-3,
                                (e1[1] - e0[1]) / 1e-3]
                H[0], H[1] = Jp, Jl
            return e
        return gtsam.CustomFactor(
            robust(gtsam.noiseModel.Isotropic.Sigma(2, sig_b)), [kx, kl], err)

    def visible(j, i):
        """truth: in range, not hidden by nearer terrain"""
        lm = LMS[j]['true']
        d = lm - ned[i]
        rh = math.hypot(d[0], d[1])
        if not TRK['min_range'] <= rh <= TRK['max_range']:
            return False, rh, 0.0
        az = math.degrees(math.atan2(d[1], d[0])) % 360
        el = math.atan2(-d[2], rh) - rh * (1 - K_REFR) / (2 * R_EARTH)
        la, lo = to_ll(enu(ned[i]))
        sky, _ = cm_true.skyline(la, lo, tr['alt0'] - ned[i][2], np.array([az]))
        ok = (not np.isfinite(sky[0])) or el >= sky[0] - 1.5e-3
        return ok, rh, az

    active, used, handovers, acq_fail, track_kf = None, [], 0, 0, 0
    steps = []
    t_start = time.time()
    KS = gtsam.symbol('s', 0)        # speed-log scale factor, a state

    def log_factor(kx, kv, meas, sig):
        """speed log + no-sideslip: the log reads s * forward body
        velocity; sideways and vertical body velocity are ~0"""
        def err(this, v, H):
            p = v.atPose3(this.keys()[0])
            vn = v.atVector(this.keys()[1])
            sc = v.atVector(this.keys()[2])[0]
            R = p.rotation().matrix()
            vb = R.T @ vn
            D = np.diag([sc, 1.0, 1.0])
            if H is not None:
                sk = np.array([[0, -vb[2], vb[1]], [vb[2], 0, -vb[0]],
                               [-vb[1], vb[0], 0]])
                H[0] = np.hstack([D @ sk, np.zeros((3, 3))])
                H[1] = D @ R.T
                H[2] = np.array([[vb[0]], [0.0], [0.0]])
            return D @ vb - meas
        return gtsam.CustomFactor(gtsam.noiseModel.Diagonal.Sigmas(sig),
                                  [kx, kv, KS], err)

    if pc.get('log'):
        fg, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
        fg.add(gtsam.PriorFactorVector(KS, np.array([1.0]),
                                       gtsam.noiseModel.Isotropic.Sigma(1, 0.05)))
        vals.insert(KS, np.array([1.0]))
        isam.update(fg, vals)
        est = isam.calculateEstimate()

    n_kf = int(os.environ.get('E5BS_N', pc['n'] // pc['photo_every'])
               ) * pc['photo_every']
    for k in range(n_kf):
        i0, i1 = k * stride, (k + 1) * stride
        if i1 >= t.size:
            break
        prev = gtsam.NavState(est.atPose3(X(k)), est.atVector(V(k)))
        pb = est.atConstantBias(B(k))
        pim = gtsam.PreintegratedCombinedMeasurements(P, pb)
        pim_free = gtsam.PreintegratedCombinedMeasurements(P, b0)
        for i in range(i0, i1):
            pim.integrateMeasurement(acc[i], gyr[i], 1 / FS)
            pim_free.integrateMeasurement(acc[i], gyr[i], 1 / FS)
        pred = pim.predict(prev, pb)
        free_state = pim_free.predict(free_state, b0)
        fg, vals = gtsam.NonlinearFactorGraph(), gtsam.Values()
        fg.add(gtsam.CombinedImuFactor(X(k), V(k), X(k + 1), V(k + 1),
                                       B(k), B(k + 1), pim))
        vals.insert(X(k + 1), pred.pose())
        vals.insert(V(k + 1), pred.velocity())
        vals.insert(B(k + 1), pb)
        if pc['baro_sd']:
            d_baro = ned[i1][2] + pc['baro_bias'] + rng_b.normal(0, pc['baro_sd'])
        else:
            d_baro = 0.0
        fg.add(down_factor(X(k + 1), d_baro,
                           (pc['baro_sd'] or 0.5) + abs(pc['baro_bias'])))
        lg = pc.get('log')
        if lg:
            Rt = gtsam.Rot3.Ypr(*ypr[i1]).matrix()
            vb_true = Rt.T @ vel[i1]
            meas = np.array([lg['scale'] * vb_true[0] + rng_b.normal(0, lg['sd']),
                             0.0, 0.0])
            fg.add(log_factor(X(k + 1), V(k + 1), meas,
                              [lg['sd'], lg['side_sd'], lg['vert_sd']]))
        # ---- the hill tracker
        if use_trk:
            yaw_t = ypr[i1][0]
            if active is not None:
                ok, rh, az = visible(active, i1)
                rel = wrap(math.radians(az) - yaw_t)
                if not ok or abs(math.degrees(rel)) > TRK['max_rel']:
                    active = None
                    handovers += 1
            if active is None:
                best = None
                for j in range(len(LMS)):
                    ok, rh, az = visible(j, i1)
                    if not ok or rh < TRK['start_range']:
                        continue
                    rel = abs(math.degrees(wrap(math.radians(az) - yaw_t)))
                    if rel <= TRK['ahead'] and (best is None or rel < best[0]):
                        best = (rel, j)
                if best is not None:
                    j = best[1]
                    # acquisition: point at the hill's ESTIMATED bearing
                    lm_est = (est.atPoint3(L(j)) if est.exists(L(j))
                              else LMS[j]['map'])
                    pp = pred.pose()
                    pa = body_azel(pp.rotation().matrix(),
                                   np.asarray(pp.translation()), lm_est)
                    ta = body_azel(gtsam.Rot3.Ypr(*ypr[i1]).matrix(), ned[i1],
                                   LMS[j]['true'])
                    if abs(math.degrees(wrap(pa[0] - ta[0]))) <= TRK['acq'] \
                            and abs(math.degrees(pa[1] - ta[1])) <= TRK['acq']:
                        active = j
                    else:
                        acq_fail += 1
            if active is not None:
                j = active
                if not est.exists(L(j)) and j not in used:
                    vals.insert(L(j), LMS[j]['map'])
                    fg.add(gtsam.PriorFactorPoint3(
                        L(j), LMS[j]['map'], gtsam.noiseModel.Diagonal.Sigmas(
                            [TRK['prior_h'], TRK['prior_h'], TRK['prior_v']])))
                    used.append(j)
                m = body_azel(gtsam.Rot3.Ypr(*ypr[i1]).matrix(), ned[i1],
                              LMS[j]['true']) + rng_t.normal(0, sig_b, 2)
                fg.add(bearing_factor(X(k + 1), L(j), m))
                track_kf += 1

        photo = (k + 1) % pc['photo_every'] == 0
        if not photo:
            isam.update(fg, vals)
            est = isam.calculateEstimate()
            continue

        # ---- the photograph (variants A and B)
        p_true = enu(ned[i1])
        z_true = tr['alt0'] - ned[i1][2]
        yaw_t, pit_t, rol_t = ypr[i1]
        Rp = pred.pose().rotation().matrix()
        yaw_e = math.atan2(Rp[1, 0], Rp[0, 0])
        pit_e = math.asin(-Rp[2, 0])
        rol_e = math.atan2(Rp[2, 1], Rp[2, 2])
        p_pred = enu(np.asarray(pred.pose().translation()))
        fix, psi, margin = None, float('nan'), float('nan')
        if use_sky:
            c_true = aim(p_true, z_true)
            gimbal = c_true - math.degrees(yaw_t)        # commanded offset
            c_meas = math.degrees(yaw_e) + gimbal + rng.normal(0, nz['mount'])
            # the gimbal levels the camera with the INS estimate, so the
            # camera's residual tilt is the INS attitude error
            el_obs, w = observe(p_true, z_true, c_true, c_meas,
                                pit_t - pit_e, rol_t - rol_e)
            z_s = tr['alt0'] - d_baro if pc['baro_sd'] else pc['alt']
            fix, psi, cov, margin = solve(el_obs, w, c_meas, p_pred, z_s)
            fix_ned = np.array([fix[1] - st0[1], fix[0] - st0[0],
                                pred.pose().translation()[2]])
            cov3 = np.diag([0.0, 0.0, 1e6])
            cov3[:2, :2] = np.asarray(cov)[::-1, ::-1]   # (e,n) -> (n,e)
            fg.add(gtsam.GPSFactor(
                X(k + 1), fix_ned,
                robust(gtsam.noiseModel.Gaussian.Covariance(cov3))))
            yaw_meas = yaw_e + math.radians(psi)
            fg.add(yaw_factor(X(k + 1), yaw_meas,
                              math.radians(1.0 if lvl == 'nominal' else 2.0)))
        isam.update(fg, vals)
        isam.update()
        est = isam.calculateEstimate()
        on = enu(np.asarray(est.atPose3(X(k + 1)).translation()))
        free = enu(np.asarray(free_state.pose().translation()))
        Re = est.atPose3(X(k + 1)).rotation().matrix()
        steps.append(dict(
            k=k + 1, t_min=t[i1] / 60, truth=p_true.tolist(),
            free=free.tolist(), pred=p_pred.tolist(),
            one=fix.tolist() if fix is not None else None,
            online=on.tolist(), margin=float(margin), psi=float(psi),
            tracking=active, yaw_err=math.degrees(wrap(
                math.atan2(Re[1, 0], Re[0, 0]) - yaw_t)),
            yaw_err_pred=math.degrees(wrap(yaw_e - yaw_t)),
            tilt_err=math.degrees(math.hypot(pit_t - pit_e, rol_t - rol_e))))
        print('%s %2d  free %6.0f  pred %5.0f  one %5s  online %5.0f m  '
              'yaw err %+.2f deg  hill %s  [%.0fs]' % (
                  name, len(steps), np.hypot(*(free - p_true)),
                  np.hypot(*(p_pred - p_true)),
                  '%.0f' % np.hypot(*(fix - p_true)) if fix is not None
                  else '-', np.hypot(*(on - p_true)), steps[-1]['yaw_err'],
                  active, time.time() - t_start), flush=True)
    smooth = [enu(np.asarray(est.atPose3(X(s['k'])).translation())).tolist()
              for s in steps]
    bias = est.atConstantBias(B(steps[-1]['k']))
    res = dict(name=name, platform=pc, noise=nz, grade=gr, steps=steps,
               smooth=smooth,
               gyro_bias_dph=(np.asarray(bias.gyroscope()) / D2R * 3600).tolist(),
               acc_bias=np.asarray(bias.accelerometer()).tolist(),
               log_scale=(float(est.atVector(gtsam.symbol('s', 0))[0])
                          if pc.get('log') else None),
               variant=variant, scenario=scen,
               tracker=dict(hills=len(used), handovers=handovers,
                            acq_fail=acq_fail, tracked_keyframes=track_kf,
                            keyframes=steps[-1]['k'],
                            hill_err_map=[float(np.linalg.norm(
                                (LMS[j]['map'] - LMS[j]['true'])[:2]))
                                for j in used],
                            hill_err_est=[float(np.linalg.norm(
                                (np.asarray(est.atPoint3(L(j)))
                                 - LMS[j]['true'])[:2])) for j in used]))
    json.dump(res, open(os.path.join(OUT, name + '.json'), 'w'))
    return name


SCENS = ['drone_mid_nominal', 'drone_low_nominal', 'drone_mid_field',
         'vessel_mid_nominal']


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {'A': '#2E86C1', 'B': '#1E7F5C', 'C': '#C0392B'}
    lab = {'A': 'A: INS + skyline', 'B': 'B: INS + skyline + hill tracker',
           'C': 'C: INS + hill tracker only'}
    scens = [sc for sc in SCENS if any(os.path.exists(os.path.join(
        OUT, '%s_%s.json' % (sc, v))) for v in 'ABC')]
    fig, axs = plt.subplots(2, len(scens), figsize=(5.2 * len(scens), 9.4),
                            squeeze=False)
    hills = peaks()
    summary = {}
    for c, sc in enumerate(scens):
        ax, axm = axs[0, c], axs[1, c]
        summary[sc] = {}
        free_drawn = False
        for v in 'ABC':
            f = os.path.join(OUT, '%s_%s.json' % (sc, v))
            if not os.path.exists(f):
                continue
            r = json.load(open(f))
            st = r['steps']
            T = np.array([s['truth'] for s in st])
            t = np.array([s['t_min'] for s in st])
            on = np.hypot(*(np.array([s['online'] for s in st]) - T).T)
            sm = np.hypot(*(np.array(r['smooth']) - T).T)
            fr = np.hypot(*(np.array([s['free'] for s in st]) - T).T)
            half = slice(len(st) // 2, None)
            tk = r['tracker']
            summary[sc][v] = dict(
                online_median=float(np.median(on)),
                online_2nd_half=float(np.median(on[half])),
                online_final=float(on[-1]), online_max=float(on.max()),
                smooth_median=float(np.median(sm)),
                yaw_err_median=float(np.median(np.abs(
                    [s['yaw_err'] for s in st]))),
                free_final=float(fr[-1]),
                hills=tk['hills'], handovers=tk['handovers'],
                acq_fail=tk['acq_fail'],
                tracked=tk['tracked_keyframes'] / max(tk['keyframes'], 1),
                hill_err_map=float(np.median(tk['hill_err_map']))
                if tk['hill_err_map'] else None,
                hill_err_est=float(np.median(tk['hill_err_est']))
                if tk['hill_err_est'] else None)
            if not free_drawn:
                ax.plot(t, fr, color='#8A8F98', lw=1.4, label='free INS')
                axm.plot(*T.T, color='#1A2732', lw=2.2, label='truth')
                free_drawn = True
            ax.plot(t, on, color=col[v], lw=1.8, label=lab[v])
            axm.plot(*np.array([s['online'] for s in st]).T, color=col[v],
                     lw=1.1)
            if v in 'BC' and r['tracker']['hills']:
                st0 = np.array(r['platform']['start'])
        ax.set_yscale('log')
        ax.set_ylim(3, 3e4)
        ax.set_xlabel('time (min)')
        ax.set_ylabel('position error (m)')
        ax.set_title(sc.replace('_', ', '), fontsize=11)
        ax.grid(alpha=.3, which='both')
        if c == 0:
            ax.legend(fontsize=8, loc='upper left')
        # the hills, in the same east/north frame as the tracks
        import skyline as S
        mlat, mlon = S.meters_per_degree(LAT0)
        H = np.array([((h[1] - LON0) * mlon, (h[0] - LAT0) * mlat)
                      for h in hills])
        lo, hi = T.min(0) - 6000, T.max(0) + 6000
        m = np.all((H > lo) & (H < hi), axis=1)
        axm.scatter(*H[m].T, marker='^', s=26, color='#8A6D3B',
                    label='hills (DEM summits)')
        axm.set_xlim(lo[0], hi[0])
        axm.set_ylim(lo[1], hi[1])
        axm.set_aspect('equal', adjustable='box')
        axm.set_xlabel('east (m)')
        axm.set_ylabel('north (m)')
        axm.grid(alpha=.3)
        if c == 0:
            axm.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'tracker.png'), dpi=100)
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    for sc, d in summary.items():
        print('\n' + sc)
        for v, x in d.items():
            print('  %s online median %5.0f m | 2nd half %5.0f | final %5.0f '
                  '| max %5.0f | smoothed %5.0f | yaw %.2f deg' % (
                      v, x['online_median'], x['online_2nd_half'],
                      x['online_final'], x['online_max'], x['smooth_median'],
                      x['yaw_err_median']))
            if v in 'BC':
                print('     tracker: %d hills, %d handovers, %d failed '
                      'acquisitions, tracking %.0f %% of the time; hill '
                      'position error map %s m -> estimated %s m' % (
                          x['hills'], x['handovers'], x['acq_fail'],
                          100 * x['tracked'],
                          '%.0f' % x['hill_err_map'] if x['hill_err_map']
                          is not None else '-',
                          '%.0f' % x['hill_err_est'] if x['hill_err_est']
                          is not None else '-'))
    print('\nwrote', os.path.join(OUT, 'tracker.png'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['run', 'report'])
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    names = a.only or ['%s_%s' % (sc, v) for sc in SCENS for v in 'ABC']
    if a.cmd == 'run':
        os.makedirs(OUT, exist_ok=True)
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        peaks()                          # build the hill cache once
        # the slow skyline runs first, so the quick C runs fill the gaps
        names.sort(key=lambda n: n.endswith('_C'))
        with ProcessPoolExecutor(a.jobs) as ex:
            for nm in ex.map(simulate, names):
                print('done', nm, flush=True)
        report()
    else:
        report()
