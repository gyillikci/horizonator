#!/usr/bin/env python3
"""The particle filter hands over to a factor graph (GTSAM iSAM2) once it has found the boat.

The particle filter (pf_track.py) is the tool while the position is still many-valued (the
6 km box of an uncertain start); once its cloud is one tight blob the posterior is near
Gaussian, and a smoother does the same job with a fraction of the cost evaluations, re-weighs
the past with every new frame, and takes further sensors as factors. Here:

  variables   x_k: position (north, east m), b_k: the compass bias (deg)
  factors     the dead-reckoned leg between x_k-1 and x_k, turned back by b_k (as
              pf_track.PF.predict: 2 % speed error, the compass's white noise across the leg),
              the bias's random walk (0.1 deg a frame), and per frame the skyline: skyfix's
              cost on a 13 x 13 grid about the predicted position (and a finer 9 x 9 where
              that is too coarse for the likelihood), times the prediction, moment-matched,
              the prediction divided out again -- the skyline's own likelihood
              as a Gaussian factor on x_k (Huber-robust) -- and the heading shift that fits
              there as a measurement of b_k (sigma: the compass's white noise; the heading
              searched as wide as the white noise and the bias's own uncertainty)

The hand-over: the cloud's major sigma under 120 m on three updates running, its covariance
doubled in sigma (from a known departure point the graph starts at once). The way back: the skyline fitting badly (rms over
7 mrad) or landing far outside the prediction (Mahalanobis^2 over 25) on three frames
running re-seeds the particle filter about the graph's estimate (a Gaussian three times the
graph's sigma wide, at least 500 m, keeping the bias).

    python3 hybrid.py DIR ROUTE [ROUTE ...] [--known-start M] [--tag NAME]
"""
import argparse, glob, json, math, os, sys, time, zlib
import numpy as np
import gtsam

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_track as P  # noqa: E402

D2R = math.pi / 180.0
HANDOVER_M, BIAS_WALK = 120.0, 0.1


def X(k):
    return gtsam.symbol('x', k)


def B(k):
    return gtsam.symbol('b', k)


def dr_factor(k, d_meas, sigma):
    """x_k - x_k-1 = the logged leg turned back by the bias b_k (deg)."""
    dn, de = float(d_meas[0]), float(d_meas[1])

    def err(this, v, H):
        x0, x1 = v.atPoint2(this.keys()[0]), v.atPoint2(this.keys()[1])
        b = v.atVector(this.keys()[2])[0] * D2R
        c, s = math.cos(b), math.sin(b)
        if H is not None:
            H[0] = -np.eye(2)
            H[1] = np.eye(2)
            H[2] = -D2R * np.array([[de * c - dn * s], [-dn * c - de * s]])
        return (x1 - x0) - np.array([dn * c + de * s, de * c - dn * s])

    return gtsam.CustomFactor(gtsam.noiseModel.Isotropic.Sigma(2, sigma), gtsam.KeyVector([X(k - 1), X(k), B(k)]), err)


KERNEL, INFLATE = 'huber', 4.0     # the skyline factors' kernel; the hand-over's covariance factor
LEG_JITTER = 0.0                   # m added to each leg's sigma (the particle filter roughens 15 m a resample)


def robust(base, k=2.5):
    # the Mahalanobis gate already turns gross outliers away; 'gauss' lets an accepted factor
    # pull with its full weight however far the graph has wandered
    if KERNEL == 'gauss':
        return base
    return gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(k), base)


def skyline_likelihood(pf, ob, mean, cov, b_lin, white, n=13):
    """The skyline about the prediction (mean, cov: north/east m), as a Gaussian likelihood on
    the position and a measurement of the bias. Cost on an n x n grid spanning 3 sigma of the
    prediction (150 m to 1.5 km), posterior = prediction x likelihood moment-matched, the
    prediction's information taken out again."""
    sig = math.sqrt(max(np.linalg.eigvalsh(cov)[-1], 1.0))
    h = min(max(3 * sig, 150.0), 1500.0)
    Ip = np.linalg.inv(cov)
    pre0 = int(round(-b_lin / 0.1))

    def posterior(centre, half, m):
        g = np.linspace(-half, half, m)
        nn, ee = np.meshgrid(g, g, indexing='ij')
        x = centre + np.stack([nn.ravel(), ee.ravel()], 1)
        cost, sh = pf.f(*pf.ll(x), ob, np.full(len(x), pre0))
        d = x - mean
        lw = -P.LAM * cost - 0.5 * np.einsum('ni,ij,nj->n', d, Ip, d)
        w = np.exp(lw - lw.max()); w /= w.sum()
        mu = w @ x
        dd = x - mu
        step = 2 * half / (m - 1)
        cpost = (w[:, None, None] * np.einsum('ni,nj->nij', dd, dd)).sum(0) + np.eye(2) * step ** 2 / 12
        return x, cost, sh, w, mu, cpost, step

    x, cost, sh, w, mu, cpost, step = posterior(mean, h, n)
    cells = len(x)
    if math.sqrt(np.linalg.eigvalsh(cpost)[-1]) < step:
        # the likelihood is sharper than the grid: its moments are a cell's, not its own --
        # again, 9 x 9 at 3/8 of the step about the posterior
        x2, cost2, sh2, w, mu, cpost, _ = posterior(mu, 1.5 * step, 9)
        cost, sh = np.concatenate([cost, cost2]), sh2
        cells += len(x2)
    Iq = np.linalg.inv(cpost)
    # Iq = Ip + Il and Iq mu = Ip mean + Il mu_l, so Il (mu_l - mean) = Iq (mu - mean): along a
    # direction the skyline says nothing about (bearing-only, a straight shore) Il is noise
    # about zero and the division would throw mu_l kilometres off; there the factor keeps the
    # prediction's mean and next to no information
    lam, V = np.linalg.eigh(Iq - Ip)
    keep = lam > 0.05 * np.linalg.eigvalsh(Ip)[0]
    lam = np.where(keep, lam, 1.0 / 5000.0 ** 2)
    Il = V @ np.diag(lam) @ V.T
    r = V @ (np.where(keep, (V.T @ (Iq @ (mu - mean))) / lam, 0.0))
    mu_l = mean + r
    s = float(w @ sh)
    s_sd = math.sqrt(float(w @ (sh - s) ** 2))
    m2 = float(r @ np.linalg.solve(cov + np.linalg.inv(Il), r))
    return dict(mu=mu_l, info=Il, post=mu, post_cov=cpost, m2=m2, best_rms_mrad=math.sqrt(2 * float(cost.min())) * 1e3,
                kept=int(keep.sum()), lk_sigma=[float(1 / math.sqrt(x)) for x in lam], pred_sigma=sig, post_sigma=float(math.sqrt(np.linalg.eigvalsh(cpost)[-1])),
                bias=b_lin - 0.1 * s, bias_sigma=math.sqrt(white ** 2 + (0.1 * s_sd) ** 2), cells=cells, half_m=h)


class Graph:
    """iSAM2 over positions and the compass bias, from the frame k0 on."""

    def __init__(self, k0, mean, cov, bias, bias_sigma):
        p = gtsam.ISAM2Params()
        p.setRelinearizeThreshold(0.01); p.relinearizeSkip = 1
        self.isam, self.k0, self.k = gtsam.ISAM2(p), k0, k0
        g, v = gtsam.NonlinearFactorGraph(), gtsam.Values()
        g.add(gtsam.PriorFactorPoint2(X(k0), np.asarray(mean, float), gtsam.noiseModel.Gaussian.Covariance(np.asarray(cov, float))))
        g.add(gtsam.PriorFactorVector(B(k0), np.array([bias]), gtsam.noiseModel.Isotropic.Sigma(1, bias_sigma)))
        v.insert(X(k0), np.asarray(mean, float)); v.insert(B(k0), np.array([bias]))
        self.isam.update(g, v)

    def predict(self, k, d_meas, sigma):
        est = self.isam.calculateEstimate()
        x0, b = est.atPoint2(X(self.k)), est.atVector(B(self.k))[0]
        c, s = math.cos(b * D2R), math.sin(b * D2R)
        g, v = gtsam.NonlinearFactorGraph(), gtsam.Values()
        g.add(dr_factor(k, d_meas, math.hypot(sigma, LEG_JITTER)))
        g.add(gtsam.BetweenFactorVector(B(k - 1), B(k), np.zeros(1), gtsam.noiseModel.Isotropic.Sigma(1, BIAS_WALK)))
        v.insert(X(k), x0 + np.array([d_meas[0] * c + d_meas[1] * s, d_meas[1] * c - d_meas[0] * s]))
        v.insert(B(k), np.array([b]))
        self.isam.update(g, v)
        self.k = k

    def observe(self, lk):
        g = gtsam.NonlinearFactorGraph()
        g.add(gtsam.PriorFactorPoint2(X(self.k), lk['mu'], robust(gtsam.noiseModel.Gaussian.Information(lk['info']))))
        g.add(gtsam.PriorFactorVector(B(self.k), np.array([lk['bias']]), robust(gtsam.noiseModel.Isotropic.Sigma(1, lk['bias_sigma']))))
        self.isam.update(g, gtsam.Values())
        self.isam.update()

    def now(self):
        est = self.isam.calculateEstimate()
        return est.atPoint2(X(self.k)), self.isam.marginalCovariance(X(self.k)), est.atVector(B(self.k))[0]

    def bias_sigma(self):
        return math.sqrt(float(self.isam.marginalCovariance(B(self.k))[0, 0]))

    def smoothed(self):
        est = self.isam.calculateEstimate()
        return {k: est.atPoint2(X(k)) for k in range(self.k0, self.k + 1)}


def pf_bias_sigma(pf):
    m = float(np.sum(pf.w * pf.b))
    return math.sqrt(float(np.sum(pf.w * (pf.b - m) ** 2)))


def run_route(d, route, n, workers, opts, seed='a'):
    """pf_track.run_route's frames, compass and legs (the same draws), filtered by the hybrid."""
    import solve_cruise as SC
    SC.IMU_SIG = 0.3
    tags = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(d, 'cruise_%s_[0-9]*.json' % route)))
    frames = [SC.frame(d, t) for t in tags]
    cerr, white = P.compass_model('bias', route, len(frames), seed)
    rng = np.random.default_rng(zlib.crc32((route + seed).encode()))
    f0 = frames[0]
    field = P.Field(f0['lat'], f0['lon'], workers)
    pf = P.PF(field, f0['lat'], f0['lon'], n=n, seed=zlib.crc32(route.encode()), bias_state=True)
    pf_opts = dict(dense_init=True)
    graph, tight, bad, segments = None, 0, 0, []
    rows, prev = [], None
    for k, (t, p) in enumerate(zip(tags, frames)):
        prior, _, imu = SC.knows(p, t, 1000.0, 5.0, seed)
        compass = (p['heading_deg'] + cerr[k]) % 360
        t0 = time.perf_counter()
        info = dict(update=False)
        if prev is not None:
            dn = (p['lat'] - prev['lat']) * pf.mlat; de = (p['lon'] - prev['lon']) * pf.mlon
            a = math.radians(cerr[k]); sc = 1 + rng.normal(0, 0.02)
            d_ne = sc * np.array([dn * math.cos(a) - de * math.sin(a), dn * math.sin(a) + de * math.cos(a)])
            leg = math.hypot(dn, de)
            sigma = math.sqrt((0.02 * leg) ** 2 + (math.radians(white) * leg) ** 2 + 20.0 ** 2)
        ob, why = P.observation(os.path.join(d, t + '_segpost.npz'), os.path.join(d, t + '_sealine.npz'), p['width'],
                                p['height'], p['hfov_deg'], p['z_msl'], compass, imu, csig=white, reach=35.0)
        if ob is not None and ob['relief'] < P.FULL_RELIEF:
            ob, why = None, 'relief %.1f mrad < %.1f' % (ob['relief'], P.FULL_RELIEF)
        if prev is None and opts.get('known_start'):
            graph = Graph(k, pf.ne(p['lat'], p['lon']), np.eye(2) * opts['known_start'] ** 2, 0.0, 5.0)   # as pf_track's seed_gauss
        elif prev is None:
            pf.seed_box(*prior)
        if graph is not None:                      # ---- the factor graph
            if k > graph.k:
                graph.predict(k, d_ne, sigma)
            mean, cov, b = graph.now()
            if ob is not None:
                # the heading is searched as wide as the compass is unknown: its white noise and
                # what the graph does not know yet of its bias (a particle filter has one per particle)
                cs = math.hypot(white, graph.bias_sigma())
                hw = int(round(max(6.0, 4 * cs) / 0.1))
                lk = skyline_likelihood(pf, dict(ob, shifts=np.arange(-hw, hw + 1, 2), compass_k=P.compass_k(cs)), mean, cov, b, cs)
                info = dict(update=True, cells=lk['cells'], best_rms_mrad=lk['best_rms_mrad'], m2=lk['m2'], kept=lk['kept'],
                            lk_sigma=lk['lk_sigma'], pred_sigma=lk['pred_sigma'], post_sigma=lk['post_sigma'])
                if lk['best_rms_mrad'] > P.BAD_RMS * 1e3 or lk['m2'] > 25.0:
                    bad += 1; info.update(update=False, why='rejected: rms %.1f mrad, m2 %.0f' % (lk['best_rms_mrad'], lk['m2']))
                else:
                    bad = 0; graph.observe(lk)
            else:
                info['why'] = why
            mean, cov, b = graph.now()
            mode = 'graph'
            if bad >= 3:                           # lost: back to the particle filter about the estimate
                segments.append(graph); graph, bad, tight = None, 0, 0
                sig = math.sqrt(max(np.linalg.eigvalsh(cov)[-1], 0))
                pf.seed_gauss(pf.lat0 + mean[0] / pf.mlat, pf.lon0 + mean[1] / pf.mlon, max(3 * sig, 500.0))
                pf.b = b + pf.rng.normal(0, 1.0, pf.n)
                info['reseeded'] = True
        else:                                      # ---- the particle filter
            if prev is not None:
                pf.predict(d_ne, sigma)
            info = pf.step(ob, why, pf_opts)
            pf.resample()
            lat, lon, cov, b = pf.estimate()
            mean = pf.ne(lat, lon)
            mode = 'pf'
            sig = math.sqrt(max(np.linalg.eigvalsh(cov)[-1], 0))
            tight = tight + 1 if info.get('update') and sig < HANDOVER_M else (tight if not info.get('update') else 0)
            if tight >= 3:                         # one tight blob: hand over
                # a converging cloud is tighter than its error (Kusadasi: 186 m off at sigma
                # under 120 m), and the robust skyline factors would then pull the graph back
                # only slowly: the prior gets twice the cloud's sigma
                graph = Graph(k, mean, cov * INFLATE, b, max(pf_bias_sigma(pf), 0.5))
                info['handover'] = True; tight = 0
        truth = pf.ne(p['lat'], p['lon'])
        e = mean - truth
        m2 = float(e @ np.linalg.solve(cov + np.eye(2), e))
        sig = float(math.sqrt(max(np.linalg.eigvalsh(cov)[-1], 0)))
        rows.append(dict(tag=t, k=k, mode=mode, err=float(np.hypot(*e)), sigma_major=sig, inside95=bool(m2 <= 5.99),
                         bias=float(b), compass_err=float(cerr[k]), dt=time.perf_counter() - t0,
                         cells=info.get('cells') or info.get('dense_cells') or (n if info.get('update') else 0), mean=[float(mean[0]), float(mean[1])],
                         truth=[float(truth[0]), float(truth[1])],
                         **{x: info[x] for x in ('update', 'why', 'handover', 'reseeded', 'm2', 'kept', 'lk_sigma', 'pred_sigma', 'post_sigma') if x in info}))
        prev = p
        print('%-22s %-5s %-3s err %6.0f m  sigma %6.0f m %s  bias %+5.1f (compass off %+5.1f)  %s%s%s' % (
            t, mode, 'upd' if info.get('update') else '-', rows[-1]['err'], sig, 'in95' if rows[-1]['inside95'] else 'OUT ',
            b, cerr[k], info.get('why') or 'rms %.1f' % info.get('best_rms_mrad', float('nan')),
            ' HANDOVER' if info.get('handover') else '', ' RESEED' if info.get('reseeded') else ''), flush=True)
    if graph is not None:
        segments.append(graph)
    for g in segments:                             # the smoothed track: every frame re-weighed by the later ones
        for k, m in g.smoothed().items():
            rows[k]['err_smooth'] = float(np.hypot(*(m - np.array(rows[k]['truth']))))
    for r in rows:
        r.setdefault('err_smooth', r['err'])
    field.ex.shutdown()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('routes', nargs='+')
    ap.add_argument('--n', type=int, default=1000); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--known-start', type=float, default=None, metavar='M',
                    help='the departure point is known to M metres (1 sigma): the graph starts there')
    ap.add_argument('--tag', default='hybrid')
    ap.add_argument('--kernel', default=KERNEL, choices=['huber', 'gauss'], help='the skyline factors\' kernel')
    ap.add_argument('--inflate', type=float, default=INFLATE, help='hand-over covariance factor')
    ap.add_argument('--leg-jitter', type=float, default=LEG_JITTER, help='m added (in quadrature) to each leg\'s sigma')
    ap.add_argument('--rolls', default=','.join('%g' % r for r in P.ROLLS), help='roll offsets (deg) searched about the levelled roll')
    a = ap.parse_args()
    globals().update(KERNEL=a.kernel, INFLATE=a.inflate, LEG_JITTER=a.leg_jitter)
    P.ROLLS = tuple(float(x) for x in a.rolls.split(','))
    d = os.path.abspath(a.dir)
    opts = dict(known_start=a.known_start)
    allr = {}
    for route in a.routes:
        rows = run_route(d, route, a.n, a.workers, opts)
        allr[route] = rows
        e, es = np.array([r['err'] for r in rows]), np.array([r['err_smooth'] for r in rows])
        print('%s: hybrid median %.0f m (smoothed %.0f m), p90 %.0f m, %d/%d under 300 m, 95%% ellipse holds the truth on %d/%d; '
              '%d frames on the graph; %.0f cost evaluations and %.1f s a frame' % (
                  route, np.median(e), np.median(es), np.percentile(e, 90), (e < 300).sum(), e.size,
                  sum(r['inside95'] for r in rows), e.size, sum(r['mode'] == 'graph' for r in rows),
                  np.mean([r['cells'] for r in rows]), np.mean([r['dt'] for r in rows])), flush=True)
    json.dump(dict(opts=opts, routes=allr), open(os.path.join(d, a.tag + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
