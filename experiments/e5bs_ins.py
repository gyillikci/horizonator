#!/usr/bin/env python3
"""E5bs: skyline observations aiding a real strapdown INS in GTSAM.

E5br fused skyline fixes with dead-reckoning LEGS (distance + heading).
A drone or a modern vessel navigates with an IMU, so this replaces the
legs with a 100 Hz IMU stream and GTSAM's preintegrated, bias-
estimating CombinedImuFactor.

IMU data come from gnss-ins-sim (Aceinna, open source;
github.com/Aceinna/gnss-ins-sim, cloned beside this repository or
pointed to by $GNSS_INS_SIM): a motion profile is turned into the true
trajectory and attitude, and into accelerometer and gyro samples with
the simulator's built-in MEMS error models - 'mid-accuracy' (IMU381
class: gyro bias instability 3.5 deg/h, ARW 0.25 deg/sqrt(h)) and
'low-accuracy' (10 deg/h, 0.75 deg/sqrt(h)). Virtual-inertial frame:
NED navigation, constant gravity, no Earth rate - matched in GTSAM.

Per keyframe (one photograph):
  * IMU preintegration from the last keyframe -> predicted NavState;
  * the camera, on a gimbal levelled with the INS attitude ESTIMATE,
    is aimed at the most relief; its azimuth is known only through the
    INS yaw, and its residual pitch/roll error is the INS attitude
    error - so INS and skyline errors are coupled as they would be;
  * the skyline (truth ray-cast from SRTM1, matched against GLO-30,
    plus extraction noise, outliers) is solved on a 41x41 x 100 m map
    around the INS prediction -> one-shot position fix, and a YAW
    measurement (INS yaw + the solve's heading offset): the skyline
    is the only heading reference in the system;
  * factors: CombinedImuFactor, robust GPS-like position factor from
    the fix, robust yaw factor, barometric altitude (drone) or the sea
    surface (vessel).
Compared: free INS (IMU only, known start), one-shot fixes, the online
iSAM2 estimate and the smoothed track.

    python3 e5bs_ins.py run [--only drone_mid_nominal ...]
    python3 e5bs_ins.py report
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
OUT = os.path.join(HERE, 'out', 'ins')
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
                  dt=30.0, n=30, photo_every=1, log=None,
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
    plat, grade, lvl = name.split('_')
    pc, nz, gr = PLAT[plat], NOISE[lvl], GRADE[grade]
    rng = np.random.default_rng(seed + sum(map(ord, name)))
    work = os.path.join(OUT, 'work_' + name)
    os.makedirs(work, exist_ok=True)
    tr = gen_imu(plat, grade, seed + sum(map(ord, name)), work)
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
    isam = gtsam.ISAM2()
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
            d_baro = ned[i1][2] + pc['baro_bias'] + rng.normal(0, pc['baro_sd'])
        else:
            d_baro = 0.0
        fg.add(down_factor(X(k + 1), d_baro,
                           (pc['baro_sd'] or 0.5) + abs(pc['baro_bias'])))
        lg = pc.get('log')
        if lg:
            Rt = gtsam.Rot3.Ypr(*ypr[i1]).matrix()
            vb_true = Rt.T @ vel[i1]
            meas = np.array([lg['scale'] * vb_true[0] + rng.normal(0, lg['sd']),
                             0.0, 0.0])
            fg.add(log_factor(X(k + 1), V(k + 1), meas,
                              [lg['sd'], lg['side_sd'], lg['vert_sd']]))
        photo = (k + 1) % pc['photo_every'] == 0
        if not photo:
            isam.update(fg, vals)
            est = isam.calculateEstimate()
            continue

        # ---- the photograph
        p_true = enu(ned[i1])
        z_true = tr['alt0'] - ned[i1][2]
        yaw_t, pit_t, rol_t = ypr[i1]
        Rp = pred.pose().rotation().matrix()
        yaw_e = math.atan2(Rp[1, 0], Rp[0, 0])
        pit_e = math.asin(-Rp[2, 0])
        rol_e = math.atan2(Rp[2, 1], Rp[2, 2])
        c_true = aim(p_true, z_true)
        gimbal = c_true - math.degrees(yaw_t)            # commanded offset
        c_meas = math.degrees(yaw_e) + gimbal + rng.normal(0, nz['mount'])
        # the gimbal levels the camera with the INS estimate, so the
        # camera's residual tilt is the INS attitude error
        el_obs, w = observe(p_true, z_true, c_true, c_meas,
                            pit_t - pit_e, rol_t - rol_e)
        p_pred = enu(np.asarray(pred.pose().translation()))
        z_s = tr['alt0'] - d_baro if pc['baro_sd'] else pc['alt']
        fix, psi, cov, margin = solve(el_obs, w, c_meas, p_pred, z_s)
        fix_ned = np.array([fix[1] - st0[1], fix[0] - st0[0],
                            pred.pose().translation()[2]])
        cov3 = np.diag([0.0, 0.0, 1e6])
        cov3[:2, :2] = np.asarray(cov)[::-1, ::-1]       # (e,n) -> (n,e)
        fg.add(gtsam.GPSFactor(X(k + 1), fix_ned,
                               robust(gtsam.noiseModel.Gaussian.Covariance(cov3))))
        yaw_meas = yaw_e + math.radians(psi)
        fg.add(yaw_factor(X(k + 1), yaw_meas,
                          math.radians(1.0 if lvl == 'nominal' else 2.0)))
        isam.update(fg, vals)
        isam.update()
        est = isam.calculateEstimate()
        on = enu(np.asarray(est.atPose3(X(k + 1)).translation()))
        free = enu(np.asarray(free_state.pose().translation()))
        steps.append(dict(
            k=k + 1, t_min=t[i1] / 60, truth=p_true.tolist(),
            free=free.tolist(), pred=p_pred.tolist(), one=fix.tolist(),
            online=on.tolist(), margin=float(margin), psi=float(psi),
            yaw_err_pred=math.degrees(((yaw_e - yaw_t + math.pi)
                                       % (2 * math.pi)) - math.pi),
            yaw_err_meas=math.degrees(((yaw_meas - yaw_t + math.pi)
                                       % (2 * math.pi)) - math.pi),
            tilt_err=math.degrees(math.hypot(pit_t - pit_e, rol_t - rol_e))))
        print('%s %2d  free %6.0f  pred %5.0f  one %5.0f  online %5.0f m  '
              'yaw err pred %+.2f -> skyline %+.2f deg  [%.0fs]' % (
                  name, len(steps), np.hypot(*(free - p_true)),
                  np.hypot(*(p_pred - p_true)), np.hypot(*(fix - p_true)),
                  np.hypot(*(on - p_true)), steps[-1]['yaw_err_pred'],
                  steps[-1]['yaw_err_meas'], time.time() - t_start), flush=True)
    smooth = [enu(np.asarray(est.atPose3(X(s['k'])).translation())).tolist()
              for s in steps]
    bias = est.atConstantBias(B(steps[-1]['k']))
    res = dict(name=name, platform=pc, noise=nz, grade=gr, steps=steps,
               smooth=smooth,
               gyro_bias_dph=(np.asarray(bias.gyroscope()) / D2R * 3600).tolist(),
               acc_bias=np.asarray(bias.accelerometer()).tolist(),
               log_scale=(float(est.atVector(gtsam.symbol('s', 0))[0])
                          if pc.get('log') else None))
    json.dump(res, open(os.path.join(OUT, name + '.json'), 'w'))
    return name


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [n for n in ('drone_mid_nominal', 'drone_low_nominal',
                         'drone_mid_field', 'vessel_mid_nominal')
             if os.path.exists(os.path.join(OUT, n + '.json'))]
    fig, axs = plt.subplots(2, len(names), figsize=(5.2 * len(names), 9.2),
                            squeeze=False)
    summary = {}
    for col, name in enumerate(names):
        r = json.load(open(os.path.join(OUT, name + '.json')))
        st = r['steps']
        T = np.array([s['truth'] for s in st])
        t = np.array([s['t_min'] for s in st])
        err = lambda key: np.hypot(*(np.array([s[key] for s in st]) - T).T)
        e = dict(free=err('free'), one=err('one'), online=err('online'),
                 smooth=np.hypot(*(np.array(r['smooth']) - T).T))
        ye_p = np.abs([s['yaw_err_pred'] for s in st])
        half = slice(len(st) // 2, None)
        summary[name] = {k: dict(median=float(np.median(v)),
                                 median_2nd_half=float(np.median(v[half])),
                                 final=float(v[-1]), max=float(v.max()))
                         for k, v in e.items()}
        summary[name]['yaw_err_pred_median_deg'] = float(np.median(ye_p))
        summary[name]['tilt_err_median_deg'] = float(
            np.median([s['tilt_err'] for s in st]))
        summary[name]['gyro_bias_dph'] = r['gyro_bias_dph']
        ax = axs[0, col]
        ax.plot(t, e['free'], color='#8A8F98', lw=1.6, label='free INS (IMU only)')
        ax.scatter(t, e['one'], s=16, color='#D68910', zorder=3,
                   label='one-shot skyline fix')
        ax.plot(t, e['online'], color='#2E86C1', lw=2.0,
                label='INS + skyline, online')
        ax.plot(t, e['smooth'], color='#2E86C1', lw=1.4, ls='--',
                label='INS + skyline, smoothed')
        ax.set_yscale('log')
        ax.set_ylim(3, max(5000, e['free'].max() * 1.3))
        ax.set_xlabel('time (min)')
        ax.set_ylabel('position error (m)')
        ax.set_title(name.replace('_', ', '), fontsize=11)
        ax.grid(alpha=.3, which='both')
        if col == 0:
            ax.legend(fontsize=8, loc='upper left')
        ax = axs[1, col]
        ax.plot(*T.T, color='#1A2732', lw=2, label='truth')
        ax.plot(*np.array([s['free'] for s in st]).T, color='#8A8F98',
                lw=1.2, label='free INS')
        ax.plot(*np.array([s['online'] for s in st]).T, color='#2E86C1',
                lw=1.2, label='INS + skyline')
        ax.scatter(*np.array([s['one'] for s in st]).T, s=10,
                   color='#D68910', label='one-shot')
        # free INS leaves for tens of km; frame the true track instead
        lo, hi = T.min(0) - 1500, T.max(0) + 1500
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('east (m)')
        ax.set_ylabel('north (m)')
        ax.grid(alpha=.3)
        if col == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'ins.png'), dpi=100)
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    for name, sm in summary.items():
        print('\n' + name)
        for k in ('free', 'one', 'online', 'smooth'):
            v = sm[k]
            print('  %-7s median %7.0f m | 2nd half %7.0f m | final %7.0f m '
                  '| max %7.0f m' % (k, v['median'], v['median_2nd_half'],
                                     v['final'], v['max']))
        print('  INS yaw error before each photo: median %.2f deg; camera '
              'tilt error median %.2f deg; gyro bias est %s deg/h' % (
                  sm['yaw_err_pred_median_deg'], sm['tilt_err_median_deg'],
                  np.round(sm['gyro_bias_dph'], 1).tolist()))
    print('\nwrote', os.path.join(OUT, 'ins.png'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['run', 'report'])
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    names = a.only or ['drone_mid_nominal', 'drone_low_nominal',
                       'drone_mid_field', 'vessel_mid_nominal']
    if a.cmd == 'run':
        os.makedirs(OUT, exist_ok=True)
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        with ProcessPoolExecutor(len(names)) as ex:
            for nm in ex.map(simulate, names):
                print('done', nm, flush=True)
        report()
    else:
        report()
