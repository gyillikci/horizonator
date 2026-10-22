#!/usr/bin/env python3
"""E5br: a sequence of skyline observations against one shot.

Every field result so far is a single photograph. A platform that moves
takes many, and dead reckoning links them. This asks how much a series
buys, on a vessel and on a drone, from a known start:

  DR only         log/airspeed + compass, integrated;
  one-shot        each observation solved alone (argmin of its cost map,
                  refined to 12.5 m) - what the field campaign did;
  GTSAM           loosely coupled: one-shot fixes as robust position
                  factors, DR legs as between factors whose prediction
                  carries the platform's systematic errors as STATES -
                  a heading correction c (compass bias), a speed scale k
                  (log bias) and a drift velocity w (current or wind):
                      x[k+1] - x[k] = k * R(c) * d_meas + w * dt
                  and every skyline solve also measures c, because the
                  heading offset it co-estimates IS the compass bias.
                  iSAM2, estimate reported online (what the navigator
                  sees at that moment), plus the final smoothed track;
  point-mass      tightly coupled: the cost map of each observation
                  becomes a likelihood exp(-lam * (J - Jmin)) on the
                  position grid; the posterior is shifted by the DR leg,
                  diffused, multiplied by the next likelihood. Ambiguous
                  single maps (two basins) are resolved by the sequence.

Observations are simulated directly as skylines, no segmenter: the
TRUE skyline is ray-cast from SRTM1, the solver matches against
Copernicus GLO-30 - two independent DEMs, so model error is real, not
zero. On top: white and correlated extraction noise, pitch and roll
error, compass bias + noise, and (field-like level) outlier blocks such
as masts and cloud edges. The camera has a 73.7 deg field of view and
is aimed at the bearing with the most terrain relief.

Each observation's cost map is computed once, on a 41 x 41 grid at
100 m centred on the point-mass prediction, and shared by all methods.
That centring is generous to one-shot (its box always contains the
truth while the filter tracks), so the gains reported for fusion are
if anything understated.

    python3 e5br_sequence.py run [--only vessel_nominal ...]
    python3 e5br_sequence.py report
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'out', 'sequence')

LAT0, LON0 = 36.95, 27.25              # ENU origin, Bodrum-Kos strait
FOV = 73.7
GRID_HALF, GRID_STEP = 2000.0, 100.0
HW = 6.0                               # heading window, deg
LAM = 20 / (3e-3) ** 2                 # likelihood temper: ~20 independent
                                       # columns at a 3 mrad model error

SCEN = {
    'vessel': dict(z=5.0, speed=3.0, dt=120.0, n=30, start=(-5000., -2200.),
                   legs=[(0, 80.0), (15, 110.0)],
                   log_scale=1.03, comp_bias=2.0, drift=(0.25, -0.15),
                   drift_prior=0.5, baro_sd=0.0),
    'drone': dict(z=300.0, speed=20.0, dt=30.0, n=30, start=(-6000., -3000.),
                  legs=[(0, 60.0), (12, 100.0), (22, 140.0)],
                  log_scale=1.0, comp_bias=2.5, drift=(3.0, -2.5),
                  drift_prior=6.0, baro_sd=8.0),
}
NOISE = {
    'nominal': dict(white=1e-3, corr=1e-3, pitch=0.2, roll=0.2,
                    comp_noise=0.5, outlier=0.0),
    'field': dict(white=2e-3, corr=3e-3, pitch=0.5, roll=0.5,
                  comp_noise=1.0, outlier=0.10),
}


def u(deg):
    r = np.radians(deg)
    return np.array([np.sin(r), np.cos(r)])


def simulate(name, seed=20261022):
    import skyline as S
    import skyfix as F
    plat, lvl = name.split('_')
    sc, nz = SCEN[plat], NOISE[lvl]
    rng = np.random.default_rng(seed + sum(map(ord, name)))
    sc = dict(sc, n=int(os.environ.get('E5BR_N', sc['n'])))
    mlat, mlon = S.meters_per_degree(LAT0)
    dem = lambda d: os.path.expanduser('~/.horizonator/' + d)
    cm_true = S.CMarcher(dem('DEMs_SRTM1'), (36.4, 37.4), (26.6, 27.9),
                         d_min=300.0)
    cm = S.CMarcher(dem('DEMs_COP30'), (36.4, 37.4), (26.6, 27.9),
                    d_min=300.0)
    AZ = F.AZ
    to_ll = lambda p: (LAT0 + p[1] / mlat, LON0 + p[0] / mlon)

    # ---- truth track and dead reckoning
    def course(k):
        return [h for k0, h in sc['legs'] if k >= k0][-1]
    n, dt, z = sc['n'], sc['dt'], sc['z']
    drift = np.array(sc['drift'])
    truth = [np.array(sc['start'])]
    legs = []
    for k in range(n):
        h = course(k)
        truth.append(truth[-1] + sc['speed'] * dt * u(h) + drift * dt)
        # measured: speed scaled by the log, heading through the biased
        # compass, plus small per-leg noise
        dist = sc['speed'] * dt * sc['log_scale'] * (1 + rng.normal(0, .01))
        hm = h + sc['comp_bias'] + rng.normal(0, 0.3)
        legs.append((dist, hm))
    truth = np.array(truth)
    dr = [truth[0].copy()]
    for dist, hm in legs:
        dr.append(dr[-1] + dist * u(hm))
    dr = np.array(dr)

    # ---- observation model
    az_rel = np.arange(-FOV / 2, FOV / 2 + 1e-9, 0.1)

    def aim(p):
        """bearing with the most relief, as an operator would point"""
        best, b_az = -1, 0.0
        la, lo = to_ll(p)
        for c in np.arange(0, 360, 15.0):
            el, _ = cm_true.skyline(la, lo, z, (c + az_rel[::10]) % 360)
            m = np.isfinite(el)
            if m.mean() < .8:
                continue
            rel = np.std(el[m])
            if rel > best:
                best, b_az = rel, c
        return b_az

    def observe(p):
        c_true = aim(p)
        la, lo = to_ll(p)
        el, _ = cm_true.skyline(la, lo, z, (c_true + az_rel) % 360)
        el = np.where(np.isfinite(el), el, np.nanmin(el))
        el = el + np.radians(rng.normal(0, nz['pitch']))
        el = el - np.radians(rng.normal(0, nz['roll'])) * np.tan(np.radians(az_rel))
        el = el + rng.normal(0, nz['white'], el.size)
        cor = ndimage.gaussian_filter1d(rng.normal(0, 1, el.size), 10)
        el = el + nz['corr'] * cor / (cor.std() + 1e-12)
        if nz['outlier']:
            nblk = rng.poisson(nz['outlier'] * el.size / 40)
            for _ in range(nblk):
                i0 = rng.integers(0, el.size - 40)
                el[i0:i0 + 40] += rng.uniform(5e-3, 25e-3)
        c_meas = c_true + sc['comp_bias'] + rng.normal(0, nz['comp_noise'])
        # place on the global grid where the solver BELIEVES it looks
        el_obs = np.zeros(AZ.size)
        w = np.zeros(AZ.size)
        idx = np.round(((c_meas + az_rel + 180.0) % 360.0 - 0.05) / 0.1
                       ).astype(int) % AZ.size
        el_obs[idx] = el
        w[idx] = 1.0
        return el_obs, w, c_meas, c_true

    shifts = np.arange(-int(HW / 0.1), int(HW / 0.1) + 1, 2)
    betas = F.BETAS

    def cost_map(el_obs, w, c_meas, centre, z_solver):
        need = (np.abs(((AZ - c_meas + 180) % 360) - 180)
                <= FOV / 2 + HW + 1)
        azn = AZ[need]
        g = np.arange(-GRID_HALF, GRID_HALF + 1, GRID_STEP)
        J = np.empty((g.size, g.size))
        PSI = np.empty_like(J)
        el_syn = np.zeros(AZ.size)

        def C(pe, pn):
            la, lo = to_ll((pe, pn))
            e, _ = cm.skyline(la, lo, z_solver, azn)
            el_syn[:] = 0.0
            el_syn[need] = np.where(np.isfinite(e), e, 0.0)
            return F.fast_photo_cost(el_obs, w, el_syn, shifts, betas)
        for i, dn in enumerate(g):
            for j, de in enumerate(g):
                c, s, _ = C(centre[0] + de, centre[1] + dn)
                J[i, j], PSI[i, j] = c, s * 0.1
        # one-shot: argmin, then refine 25 -> 12.5 m
        i, j = np.unravel_index(np.argmin(J), J.shape)
        best = (J[i, j], centre[0] + g[j], centre[1] + g[i], PSI[i, j])
        for step in (50.0, 25.0, 12.5):
            e0, n0 = best[1], best[2]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    c, s, _ = C(e0 + dj * step, n0 + di * step)
                    if c < best[0]:
                        best = (c, e0 + dj * step, n0 + di * step, s * 0.1)
        margin = F.basin_margin(J, g, min_sep=4 * GRID_STEP)
        return g, J, PSI, best, margin

    # ---- run: the point-mass filter drives where each map is centred
    g = np.arange(-GRID_HALF, GRID_HALF + 1, GRID_STEP)
    post = np.zeros((g.size, g.size))
    post[g.size // 2, g.size // 2] = 1.0           # known start
    post = ndimage.gaussian_filter(post, 0.3)
    centre = truth[0].copy()
    steps = []
    t0 = time.time()
    for k in range(n):
        dist, hm = legs[k]
        d = dist * u(hm)
        # predict: shift by the DR leg, diffuse by its uncertainty
        mean_e = centre[0] + (post.sum(0) * g).sum() / post.sum()
        mean_n = centre[1] + (post.sum(1) * g).sum() / post.sum()
        new_c = np.array([mean_e, mean_n]) + d
        q = np.hypot(0.06 * dist + 5.0, sc['drift_prior'] * dt * 0.5)
        E, N = np.meshgrid(g, g)
        src_e = (new_c[0] + E - d[0] - centre[0] - g[0]) / GRID_STEP
        src_n = (new_c[1] + N - d[1] - centre[1] - g[0]) / GRID_STEP
        prior = ndimage.map_coordinates(post, [src_n, src_e], order=1,
                                        cval=0.0)
        prior = ndimage.gaussian_filter(prior, q / GRID_STEP)
        if prior.sum() <= 0:
            prior = np.ones_like(prior)
        prior /= prior.sum()
        centre = new_c
        p_true = truth[k + 1]
        el_obs, w, c_meas, c_true = observe(p_true)
        z_s = z + rng.normal(0, sc['baro_sd']) if sc['baro_sd'] else z
        g, J, PSI, best, margin = cost_map(el_obs, w, c_meas, centre, z_s)
        L = np.exp(-LAM * (J - J.min()))
        post = prior * L
        post /= post.sum()
        pm_e = centre[0] + (post.sum(0) * g).sum()
        pm_n = centre[1] + (post.sum(1) * g).sum()
        var = ((post.sum(0) * (centre[0] + g - pm_e) ** 2).sum()
               + (post.sum(1) * (centre[1] + g - pm_n) ** 2).sum())
        # Laplace covariance of the single map, consistent with LAM
        i, j = np.unravel_index(np.argmin(J), J.shape)
        i = min(max(i, 1), g.size - 2)
        j = min(max(j, 1), g.size - 2)
        h = GRID_STEP
        Hee = (J[i, j + 1] - 2 * J[i, j] + J[i, j - 1]) / h ** 2
        Hnn = (J[i + 1, j] - 2 * J[i, j] + J[i - 1, j]) / h ** 2
        Hen = (J[i + 1, j + 1] - J[i + 1, j - 1] - J[i - 1, j + 1]
               + J[i - 1, j - 1]) / (4 * h ** 2)
        Hm = LAM * np.array([[Hee, Hen], [Hen, Hnn]])
        try:
            cov = np.linalg.inv(Hm)
            ev = np.linalg.eigvalsh(cov)
            if ev.min() <= 0:
                raise np.linalg.LinAlgError
            cov = cov + np.eye(2) * 30.0 ** 2
        except np.linalg.LinAlgError:
            cov = np.eye(2) * 1000.0 ** 2
        one = np.array([best[1], best[2]])
        steps.append(dict(
            k=k + 1, t_min=(k + 1) * dt / 60, truth=p_true.tolist(),
            dr=dr[k + 1].tolist(), one=one.tolist(), one_cost=float(best[0]),
            one_psi=float(best[3]), margin=float(margin),
            cov=cov.tolist(), pmf=[pm_e, pm_n], pmf_sd=float(np.sqrt(var)),
            c_meas=c_meas, c_true=c_true,
            truth_in_grid=bool(np.all(np.abs(p_true - centre) < GRID_HALF))))
        print('%s %2d/%d  DR %5.0f  one %5.0f  pmf %5.0f m (sd %4.0f) '
              'margin %.2f psi %+.1f  [%.0fs]' % (
                  name, k + 1, n, np.hypot(*(dr[k + 1] - p_true)),
                  np.hypot(*(one - p_true)), np.hypot(pm_e - p_true[0],
                                                      pm_n - p_true[1]),
                  np.sqrt(var), margin, best[3], time.time() - t0),
              flush=True)
    res = dict(name=name, scenario=sc, noise=nz, legs=legs,
               truth0=truth[0].tolist(), steps=steps)
    res['gtsam'] = fuse(res)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, name + '.json'), 'w'))
    return name


def fuse(res):
    """Loosely coupled iSAM2 with the platform's biases as states."""
    import gtsam
    sc, legs, steps = res['scenario'], res['legs'], res['steps']
    X = lambda k: gtsam.symbol('x', k)
    Kc, Kk, Kw = gtsam.symbol('c', 0), gtsam.symbol('k', 0), gtsam.symbol('w', 0)
    dt = sc['dt']

    def between(dist, hm):
        def err(this, v, H):
            x0 = v.atPoint2(this.keys()[0])
            x1 = v.atPoint2(this.keys()[1])
            c = v.atVector(this.keys()[2])[0]
            kk = v.atVector(this.keys()[3])[0]
            ww = v.atVector(this.keys()[4])
            a = np.radians(hm) + c
            uu = np.array([np.sin(a), np.cos(a)])
            e = x1 - x0 - (kk * dist * uu + ww * dt)
            if H is not None:
                H[0] = -np.eye(2)
                H[1] = np.eye(2)
                H[2] = (-kk * dist * np.array([np.cos(a), -np.sin(a)])
                        ).reshape(2, 1)
                H[3] = (-dist * uu).reshape(2, 1)
                H[4] = -dt * np.eye(2)
            return e
        return err

    isam = gtsam.ISAM2()
    online, fix_used = [], []
    gr, init = gtsam.NonlinearFactorGraph(), gtsam.Values()
    x0 = np.array(res['truth0'])
    gr.add(gtsam.PriorFactorPoint2(X(0), x0,
                                   gtsam.noiseModel.Isotropic.Sigma(2, 10.0)))
    gr.add(gtsam.PriorFactorVector(Kc, np.array([0.0]),
                                   gtsam.noiseModel.Isotropic.Sigma(1, np.radians(6))))
    gr.add(gtsam.PriorFactorVector(Kk, np.array([1.0]),
                                   gtsam.noiseModel.Isotropic.Sigma(1, 0.05)))
    gr.add(gtsam.PriorFactorVector(Kw, np.zeros(2),
                                   gtsam.noiseModel.Isotropic.Sigma(2, sc['drift_prior'])))
    init.insert(X(0), x0)
    init.insert(Kc, np.array([0.0]))
    init.insert(Kk, np.array([1.0]))
    init.insert(Kw, np.zeros(2))
    isam.update(gr, init)
    est = isam.calculateEstimate()
    robust = lambda cov: gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Gaussian.Covariance(np.asarray(cov)))
    for k, (dist, hm) in enumerate(legs):
        gr, init = gtsam.NonlinearFactorGraph(), gtsam.Values()
        nm = gtsam.noiseModel.Isotropic.Sigma(2, 0.01 * dist + 3.0)
        gr.add(gtsam.CustomFactor(nm, [X(k), X(k + 1), Kc, Kk, Kw],
                                  between(dist, hm)))
        prev = est.atPoint2(X(k))
        c = est.atVector(Kc)[0]
        kk = est.atVector(Kk)[0]
        ww = est.atVector(Kw)
        a = np.radians(hm) + c
        init.insert(X(k + 1), prev + kk * dist * np.array([np.sin(a), np.cos(a)])
                    + ww * dt)
        s = steps[k]
        gr.add(gtsam.PriorFactorPoint2(X(k + 1), np.array(s['one']),
                                       robust(s['cov'])))
        # the solve's heading offset measures the compass correction
        gr.add(gtsam.PriorFactorVector(
            Kc, np.array([np.radians(s['one_psi'])]),
            gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(1.345),
                gtsam.noiseModel.Isotropic.Sigma(1, np.radians(1.0)))))
        isam.update(gr, init)
        isam.update()
        est = isam.calculateEstimate()
        online.append(est.atPoint2(X(k + 1)).tolist())
    smooth = [est.atPoint2(X(k + 1)).tolist() for k in range(len(legs))]
    return dict(online=online, smooth=smooth,
                c_deg=float(np.degrees(est.atVector(Kc)[0])),
                k=float(est.atVector(Kk)[0]),
                w=est.atVector(Kw).tolist())


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [n for n in ('vessel_nominal', 'vessel_field', 'drone_nominal',
                         'drone_field')
             if os.path.exists(os.path.join(OUT, n + '.json'))]
    fig, axs = plt.subplots(2, len(names), figsize=(5.2 * len(names), 9.4),
                            squeeze=False)
    summary = {}
    for col, name in enumerate(names):
        r = json.load(open(os.path.join(OUT, name + '.json')))
        st = r['steps']
        T = np.array([s['truth'] for s in st])
        t = np.array([s['t_min'] for s in st])
        err = lambda P: np.hypot(*(np.asarray(P) - T).T)
        e = dict(dr=err([s['dr'] for s in st]),
                 one=err([s['one'] for s in st]),
                 gtsam=err(r['gtsam']['online']),
                 smooth=err(r['gtsam']['smooth']),
                 pmf=err([s['pmf'] for s in st]))
        acc = np.array([s['margin'] >= 0.15 for s in st])
        half = slice(len(st) // 2, None)
        summary[name] = {k: dict(median=float(np.median(v)),
                                 median_2nd_half=float(np.median(v[half])),
                                 final=float(v[-1]), max=float(v.max()))
                         for k, v in e.items()}
        summary[name]['one_gated'] = dict(
            accepted=int(acc.sum()), n=len(st),
            median=float(np.median(e['one'][acc])) if acc.any() else None,
            wrong_500=int((e['one'][acc] >= 500).sum()))
        summary[name]['states'] = dict(
            c_deg=r['gtsam']['c_deg'], true_c_deg=-r['scenario']['comp_bias'],
            k=r['gtsam']['k'], true_k=1 / r['scenario']['log_scale'],
            w=r['gtsam']['w'], true_w=r['scenario']['drift'])
        ax = axs[0, col]
        ax.plot(t, e['dr'], color='#8A8F98', lw=1.6, label='dead reckoning')
        ax.scatter(t[acc], e['one'][acc], s=18, color='#D68910', zorder=3,
                   label='one-shot (margin ≥ 0.15)')
        ax.scatter(t[~acc], e['one'][~acc], s=18, facecolors='none',
                   edgecolors='#D68910', zorder=3, label='one-shot (refused)')
        ax.plot(t, e['gtsam'], color='#2E86C1', lw=1.8, label='GTSAM online')
        ax.plot(t, e['pmf'], color='#1E7F5C', lw=1.8, label='point-mass filter')
        ax.set_yscale('log')
        ax.set_ylim(5, max(5000, e['dr'].max() * 1.3))
        ax.set_xlabel('time (min)')
        ax.set_ylabel('position error (m)')
        ax.set_title(name.replace('_', ', '), fontsize=11)
        ax.grid(alpha=.3, which='both')
        if col == 0:
            ax.legend(fontsize=8, loc='upper left')
        ax = axs[1, col]
        ax.plot(*T.T, color='#1A2732', lw=2, label='truth')
        ax.plot(*np.array([s['dr'] for s in st]).T, color='#8A8F98', lw=1.2,
                label='DR')
        ax.plot(*np.array(r['gtsam']['online']).T, color='#2E86C1', lw=1.2,
                label='GTSAM')
        ax.plot(*np.array([s['pmf'] for s in st]).T, color='#1E7F5C', lw=1.2,
                label='PMF')
        O = np.array([s['one'] for s in st])
        ax.scatter(*O.T, s=10, color='#D68910', label='one-shot')
        ax.set_aspect('equal')
        ax.set_xlabel('east (m)')
        ax.set_ylabel('north (m)')
        ax.grid(alpha=.3)
        if col == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'sequence.png'), dpi=100)
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    for name, sm in summary.items():
        print('\n' + name)
        for k in ('dr', 'one', 'gtsam', 'smooth', 'pmf'):
            v = sm[k]
            print('  %-7s median %6.0f m | 2nd half %6.0f m | final %6.0f m '
                  '| max %6.0f m' % (k, v['median'], v['median_2nd_half'],
                                     v['final'], v['max']))
        g = sm['one_gated']
        print('  one-shot gated: %d/%d accepted, median %s m, wrong >= 500 m: %d'
              % (g['accepted'], g['n'], '%.0f' % g['median'] if g['median']
                 is not None else '-', g['wrong_500']))
        s = sm['states']
        print('  states: compass corr %+.2f deg (true %+.2f), log scale %.3f '
              '(true %.3f), drift (%+.2f, %+.2f) m/s (true %+.2f, %+.2f)' % (
                  s['c_deg'], s['true_c_deg'], s['k'], s['true_k'],
                  s['w'][0], s['w'][1], s['true_w'][0], s['true_w'][1]))
    print('\nwrote', os.path.join(OUT, 'sequence.png'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['run', 'report', 'fuse'])
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    names = a.only or ['vessel_nominal', 'vessel_field', 'drone_nominal',
                       'drone_field']
    if a.cmd == 'run':
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        with ProcessPoolExecutor(len(names)) as ex:
            for nm in ex.map(simulate, names):
                print('done', nm, flush=True)
        report()
    elif a.cmd == 'fuse':
        # re-run only the factor graph on stored observations
        for nm in names:
            f = os.path.join(OUT, nm + '.json')
            r = json.load(open(f))
            r['gtsam'] = fuse(r)
            json.dump(r, open(f, 'w'))
        report()
    else:
        report()
