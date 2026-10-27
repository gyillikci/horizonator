#!/usr/bin/env python3
"""A particle filter (sequential Monte Carlo) on the boat's track, with skyfix's skyline cost
as the measurement: the terrain-aided navigation particle filter (Gustafsson et al. 2002,
Nordlund's marginalized form) fed by the camera's horizon instead of a radar altimeter.

Each particle is a position and, with --bias-state, a compass bias. Between frames every
particle moves by the dead-reckoned leg (log and compass: 2 % speed error; its own compass
bias taken out of the leg's heading) plus process noise; at a frame the weights take the
skyline's likelihood at every particle,

    log L = -lam * cost,   lam = n_eff / sigma_m^2 = 20 / (3 mrad)^2

the statistics skyfix's own soft compass prior is written in (cost is its mean Huber residual,
minimised over the heading shift, the elevation offset band and the roll). The observation is
built exactly as the boat's skyfix builds it (solve_cruise.boat_cmd: exact camera, straight
rays, no crest, sea line levelled with the IMU as referee, the waterline band, soft compass).

  --dense-init   the first frame that can update seeds the cloud from the likelihood on a
                 75 m grid over the 6 km box about the dead-reckoned prior (a full search,
                 four times finer than skyfix's coarse grid) instead of a uniform box
  --bias-state   particles carry the compass bias (random walk 0.1 deg a frame): the skyline
                 is aligned at compass - bias with the white noise's window, and the leg is
                 turned back by it
  --soft-relief  a skyline under 5 mrad of relief still updates, weighted (relief / 5)^2,
                 down to skyfix's 1.5 mrad floor; the single-frame gate exists for integrity,
                 a filter carries the weak evidence against dead reckoning
A frame whose sea line the levelling refuses gives no update. When even the best particle
fits badly for three updates running the filter re-seeds about its own estimate.

    python3 pf_track.py DIR ROUTE [ROUTE ...] [--compass iid|bias] [--dense-init] [--bias-state] [--soft-relief]
                        [--views EXPLORE_DIR,EXPLOIT_DIR [--switch-sigma 120]]
"""
import argparse, glob, json, math, os, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
sys.path.insert(0, HERE)
import skyfix as F  # noqa: E402
import skyline as S  # noqa: E402
import extract  # noqa: E402

DEM = os.path.expanduser('~/.horizonator/DEMs_COP30')
SIG_M, N_EFF = 3e-3, 20.0          # skyfix's measurement model (soft compass prior)
LAM = N_EFF / SIG_M ** 2
BAD_RMS = 7e-3                     # skyfix's residual gate in the boat's flags
ROLLS = (-0.5, 0.0, 0.5)           # roll offsets searched about the levelled roll
FULL_RELIEF, MIN_RELIEF = 5.0, 1.5


def compass_k(sigma):
    return 0.5 * (0.1 / sigma) ** 2 * SIG_M ** 2 / N_EFF


# ------------------------------------------------------------------ the observation

def resample_rows(src, W):
    src = np.asarray(src, float)
    return np.interp(np.linspace(0, src.size - 1, W), np.arange(src.size), src) * (W / src.size)


def observation(segpost, sealine, W, H, fov, z, compass, imu, csig=5.0, level_tol=2.5, imu_gate=1.5,
                band=(-12.5e-3, 5e-3), reach=20.0):
    """The frame as skyfix would see it, or (None, why) when it gives no update. csig: the
    compass error the heading search is weighed against; reach: the half-width (deg) the
    observation must be placeable over (the window plus any bias pre-rotation)."""
    F.CAMERA, F.EXTRACTOR = 'exact', 'ewasr'
    if not os.path.exists(sealine):
        return None, 'no sea line'
    b = np.load(segpost)
    F.BOUNDARY_ROWS, F.BOUNDARY_CONF = b['rows'], b['conf']
    f_px = (W / 2) / math.tan(math.radians(fov) / 2)
    dip = S.horizon_dip_rad(max(z, 0.5), k=0.0)
    lr = resample_rows(np.load(sealine)['rows'], W)
    lv = extract.sea_horizon_attitude(np.nan_to_num(lr), np.isfinite(lr) * 1.0, (H, W), f_px, dip,
                                      tol_px=level_tol, below_px=2 * level_tol)
    if not lv:
        return None, 'levelling refused the sea line'
    if max(abs(lv['pitch_deg'] - imu[0]), abs(lv['roll_deg'] - imu[1])) > imu_gate:
        return None, 'sea line disagrees with the IMU'
    img = np.zeros((H, W, 3))
    variants = []
    for dr in ROLLS:
        eo, w, _ = F.observation(img, fov, compass, lv['roll_deg'] + dr, lv['pitch_deg'])
        variants.append((eo, w))
    eo0, w0 = variants[1]
    if w0.sum() < 1e-6 or (w0 > 0).sum() < 20:
        return None, 'no terrain silhouette'
    relief = float(np.std(eo0[w0 > 0]) * 1e3)
    hw = int(round(max(6.0, 4 * csig) / 0.1))
    read = np.zeros(F.AZ.size, bool)
    on = np.any([w > 0 for _, w in variants], axis=0)
    for s in range(-int(reach * 10), int(reach * 10) + 1, 2):
        read |= np.roll(on, s)
    return dict(variants=variants, shifts=np.arange(-hw, hw + 1, 2),
                betas=np.linspace(band[0], band[1], int(round((band[1] - band[0]) / 0.00125)) + 1),
                az_idx=np.flatnonzero(read), z=z, compass_k=compass_k(csig), relief=relief,
                pitch=lv['pitch_deg'], roll=lv['roll_deg']), None


# ------------------------------------------------------- the cost at many places

_CM = None


def _init(lat, lon):
    global _CM
    _CM = S.CMarcher(DEM, (lat - .6, lat + .6), (lon - .8, lon + .8), d_min=1000.0, d_max=40000.0, refraction_k=0.0)


def _costs(job):
    """Cost and best heading shift (0.1 deg lags, relative to the pre-rotation) per position;
    pre[i] rotates the observation first: the particle's compass bias."""
    pts, pre, ob = job
    F.COMPASS_K = ob['compass_k']
    idx = ob['az_idx']
    out, sh = np.empty(len(pts)), np.zeros(len(pts), int)
    for i, (la, lo) in enumerate(pts):
        el = np.zeros(F.AZ.size)
        el[idx], _ = _CM.skyline(la, lo, ob['z'], F.AZ[idx])
        best = (np.inf, 0)
        for eo, w in ob['variants']:
            if pre[i]:
                eo, w = np.roll(eo, pre[i]), np.roll(w, pre[i])
            c, s, _ = F.fast_photo_cost(eo, w, el, ob['shifts'], ob['betas'])
            if c < best[0]:
                best = (c, s)
        out[i], sh[i] = best
    return out, sh


class Field:
    """skyfix's cost at arrays of positions, over a pool of processes."""

    def __init__(self, lat, lon, workers):
        self.ex = ProcessPoolExecutor(workers, initializer=_init, initargs=(lat, lon))
        self.workers = workers

    def __call__(self, lat, lon, ob, pre=None):
        pts = list(zip(lat, lon))
        pre = np.zeros(len(pts), int) if pre is None else np.asarray(pre, int)
        k = max(1, len(pts) // (self.workers * 4))
        res = list(self.ex.map(_costs, [(pts[i:i + k], pre[i:i + k], ob) for i in range(0, len(pts), k)]))
        return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


# --------------------------------------------------------------------- the filter

class PF:
    def __init__(self, field, lat0, lon0, n=1000, seed=0, bias_state=False, bias0=5.0, bias_walk=0.1):
        self.f, self.n = field, n
        self.lat0, self.lon0 = lat0, lon0
        self.mlat, self.mlon = S.meters_per_degree(lat0)
        self.rng = np.random.default_rng(seed)
        self.bias_state, self.bias0, self.bias_walk = bias_state, bias0, bias_walk
        self.x = self.w = self.b = None      # positions (k, 2) north/east m; weights; biases (deg)
        self.bad = 0
        self.seeded = False                  # False until the first update: the cloud is the prior box

    def ll(self, x):
        return self.lat0 + x[:, 0] / self.mlat, self.lon0 + x[:, 1] / self.mlon

    def ne(self, lat, lon):
        return np.array([(lat - self.lat0) * self.mlat, (lon - self.lon0) * self.mlon])

    def seed_box(self, lat, lon, box=6000.0, n0=4000):
        self.x = self.ne(lat, lon) + self.rng.uniform(-box / 2, box / 2, (n0, 2))
        self.w = np.full(n0, 1.0 / n0)
        self.b = self.rng.normal(0, self.bias0, n0) if self.bias_state else np.zeros(n0)
        self.seeded = False

    def seed_gauss(self, lat, lon, sigma):
        """A known departure point: the cloud is a Gaussian about it (the harbour, the last
        GNSS fix), so every hill is where the chart says from the first frame on."""
        self.x = self.ne(lat, lon) + self.rng.normal(0, sigma, (self.n, 2))
        self.w = np.full(self.n, 1.0 / self.n)
        self.b = self.rng.normal(0, self.bias0, self.n) if self.bias_state else np.zeros(self.n)
        self.seeded = True

    def seed_dense(self, lat, lon, ob, weight=1.0, box=6000.0, step=75.0):
        """A full search: the likelihood on a grid over the box, the cloud drawn from it (that is
        this frame's posterior already: no second update with it). The bias is not known yet, so
        the heading is searched as wide as the whole compass error (+-20 deg, sigma 5), and the
        shift each cell fits best gives its particles' bias."""
        g = np.arange(-box / 2, box / 2 + 1, step)
        nn, ee = np.meshgrid(g, g, indexing='ij')
        x = self.ne(lat, lon) + np.stack([nn.ravel(), ee.ravel()], 1)
        wide = dict(ob, shifts=np.arange(-200, 201, 2), compass_k=compass_k(5.0))
        cost, sh = self.f(*self.ll(x), wide)
        ll = -LAM * weight * cost
        for a in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01):   # tempered until 1 % of the cells carry weight
            w = np.exp(a * (ll - ll.max())); w /= w.sum()
            if 1.0 / np.sum(w ** 2) >= 0.01 * len(w):
                break
        idx = self.rng.choice(len(w), size=self.n, p=w)
        self.x = x[idx] + self.rng.normal(0, step / 2, (self.n, 2))
        # compass = truth + bias, so the observation lines up shifted by -bias: bias = -0.1 * shift
        self.b = (-0.1 * sh[idx] + self.rng.normal(0, 1.0, self.n)) if self.bias_state else np.zeros(self.n)
        self.w = np.full(self.n, 1.0 / self.n)
        self.seeded = True
        return dict(dense_cells=len(w), temper=a, best_rms_mrad=math.sqrt(2 * float(cost.min())) * 1e3)

    def predict(self, d_meas, sigma):
        """d_meas: the leg as the log and compass gave it; each particle turns it back by its
        own compass bias (a compass reading heading h + b swings the leg by b)."""
        if self.bias_state:
            c, s = np.cos(np.radians(self.b)), np.sin(np.radians(self.b))
            d = np.stack([d_meas[0] * c + d_meas[1] * s, d_meas[1] * c - d_meas[0] * s], 1)
            self.b = self.b + self.rng.normal(0, self.bias_walk, len(self.b))
        else:
            d = d_meas
        self.x = self.x + d + self.rng.normal(0, sigma, self.x.shape)

    def update(self, ob, weight=1.0):
        pre = np.rint(-self.b / 0.1).astype(int) if self.bias_state else None
        cost, _ = self.f(*self.ll(self.x), ob, pre)
        ll = -LAM * weight * cost
        best = float(cost.min())
        # temper until at least 5 % of the particles carry weight: a likelihood this sharp, from
        # a DEM that is not the world, would otherwise collapse the cloud onto one cell
        a = 1.0
        for a in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
            lw = np.log(self.w + 1e-300) + a * (ll - ll.max())
            wn = np.exp(lw - lw.max()); wn /= wn.sum()
            if 1.0 / np.sum(wn ** 2) >= 0.05 * len(wn):
                break
        self.w = wn
        self.bad = self.bad + 1 if math.sqrt(2 * best) > BAD_RMS else 0
        return dict(best_rms_mrad=math.sqrt(2 * best) * 1e3, temper=a, ess=float(1.0 / np.sum(self.w ** 2)))

    def resample(self, jitter=15.0):
        """Systematic resampling to n particles (and a little roughening), when the effective
        sample size has fallen under half, or the initial box gives way to the first update."""
        ess = 1.0 / np.sum(self.w ** 2)
        if ess > 0.5 * len(self.w) and len(self.w) == self.n:
            return False
        u = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), u), len(self.w) - 1)
        self.x = self.x[idx] + self.rng.normal(0, jitter, (self.n, 2))
        self.b = self.b[idx] + (self.rng.normal(0, 0.2, self.n) if self.bias_state else 0.0)
        self.w = np.full(self.n, 1.0 / self.n)
        return True

    def estimate(self):
        m = np.sum(self.w[:, None] * self.x, axis=0)
        d = self.x - m
        cov = (self.w[:, None, None] * np.einsum('ni,nj->nij', d, d)).sum(0)
        return self.lat0 + m[0] / self.mlat, self.lon0 + m[1] / self.mlon, cov, float(np.sum(self.w * self.b))

    def step(self, ob, why, opts):
        """Update with one frame (or not), handle seeding and divergence. Returns info."""
        if ob is None:
            return dict(update=False, why=why)
        weight = 1.0
        if ob['relief'] < FULL_RELIEF:
            if not opts.get('soft_relief') or ob['relief'] < MIN_RELIEF:
                return dict(update=False, why='relief %.1f mrad < %.1f' % (ob['relief'], FULL_RELIEF))
            weight = (ob['relief'] / FULL_RELIEF) ** 2
        if opts.get('dense_init') and not self.seeded:
            lat, lon, _, _ = self.estimate()
            return dict(update=True, seeded=True, weight=weight, **self.seed_dense(lat, lon, ob, weight))
        info = dict(update=True, weight=weight, **self.update(ob, weight))
        self.seeded = True
        if self.bad >= 3:                    # diverged: start again about the estimate
            lat, lon, _, _ = self.estimate()
            if opts.get('dense_init'):
                self.seed_dense(lat, lon, ob, weight)
            else:
                self.seed_box(lat, lon); self.update(ob, weight); self.seeded = True
            self.bad = 0
            info['reseeded'] = True
        return info


# ------------------------------------------------------------- offline, on a route

def compass_model(kind, route, n, seed):
    """Per-frame compass errors (deg): 'iid' N(0, 5) as the single-frame study draws them; 'bias'
    a per-route bias N(0, 5) drifting 0.1 deg a frame plus N(0, 2) white noise, as a real
    compass's deviation behaves."""
    rng = np.random.default_rng(zlib.crc32((route + seed + kind).encode()))
    if kind == 'iid':
        return rng.normal(0, 5.0, n), 5.0
    b = rng.normal(0, 5.0) + np.cumsum(rng.normal(0, 0.1, n))
    return b + rng.normal(0, 2.0, n), 2.0


def run_route(d, route, n, workers, opts, seed='a'):
    import solve_cruise as SC
    SC.IMU_SIG = 0.3
    tags = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(d, 'cruise_%s_[0-9]*.json' % route)))
    frames = [SC.frame(d, t) for t in tags]
    # opts['views'] = (dir, dir): the same instants seen two ways (look_planner.py --views); the
    # filter looks with the first while its cloud is wider than opts['switch_sigma'], else the second
    views = [(v, [SC.frame(v, t) for t in tags]) for v in opts.get('views') or ()]
    cerr, white = compass_model(opts['compass'], route, len(frames), seed)
    rng = np.random.default_rng(zlib.crc32((route + seed).encode()))
    f0 = frames[0]
    field = Field(f0['lat'], f0['lon'], workers)
    pf = PF(field, f0['lat'], f0['lon'], n=n, seed=zlib.crc32(route.encode()), bias_state=opts.get('bias_state', False))
    # the filter's compass model: with a bias state only the white noise is left to search;
    # without one the whole error is (wrongly) taken as white, as before
    csig = white if opts.get('bias_state') else 5.0
    reach = 20.0 + (15.0 if opts.get('bias_state') else 0.0)
    rows, prev = [], None
    for k, (t, p) in enumerate(zip(tags, frames)):
        prior, _, imu = SC.knows(p, t, 1000.0, 5.0, seed)
        compass = (p['heading_deg'] + cerr[k]) % 360
        t0 = time.perf_counter()
        if prev is None and opts.get('known_start'):
            pf.seed_gauss(p['lat'], p['lon'], opts['known_start'])   # the departure point is known
        elif prev is None:
            pf.seed_box(*prior)                         # 1 km off, as the single-frame solves start
        else:                                           # the log and the (same) compass since the last frame
            dn = (p['lat'] - prev['lat']) * pf.mlat; de = (p['lon'] - prev['lon']) * pf.mlon
            a = math.radians(cerr[k] if opts['compass'] == 'bias' else rng.normal(0, 5.0)); sc = 1 + rng.normal(0, 0.02)
            d_ne = sc * np.array([dn * math.cos(a) - de * math.sin(a), dn * math.sin(a) + de * math.cos(a)])
            leg = math.hypot(dn, de)
            pf.predict(d_ne, math.sqrt((0.02 * leg) ** 2 + (math.radians(csig) * leg) ** 2 + 20.0 ** 2))
        dv, pv, view = d, p, None
        if views:
            sig = math.sqrt(max(np.linalg.eigvalsh(pf.estimate()[2])[-1], 0))
            view = 0 if (not pf.seeded or sig > opts['switch_sigma']) else 1
            dv, pv = views[view][0], views[view][1][k]
            _, _, imu = SC.knows(pv, t, 1000.0, 5.0, seed)
            compass = (pv['heading_deg'] + cerr[k]) % 360
        ob, why = observation(os.path.join(dv, t + '_segpost.npz'), os.path.join(dv, t + '_sealine.npz'), pv['width'],
                              pv['height'], pv['hfov_deg'], pv['z_msl'], compass, imu, csig=csig, reach=reach)
        info = pf.step(ob, why, opts)
        pf.resample()
        lat, lon, cov, bias = pf.estimate()
        e = pf.ne(lat, lon) - pf.ne(p['lat'], p['lon'])
        try:
            m2 = float(e @ np.linalg.solve(cov + np.eye(2), e))
        except np.linalg.LinAlgError:
            m2 = float('nan')
        sig = float(math.sqrt(max(np.linalg.eigvalsh(cov)[-1], 0)))
        true_bias = float(cerr[k]) if opts['compass'] == 'bias' else None
        rows.append(dict(tag=t, err=float(np.hypot(*e)), sigma_major=sig, inside95=bool(m2 <= 5.99), bias=bias,
                         compass_err=float(cerr[k]), dt=time.perf_counter() - t0, view=view, **info))
        prev = p
        print('%-22s %-5s err %6.0f m  sigma %6.0f m %s  bias %+5.1f (compass off %+5.1f)  %s' % (
            t, 'upd' if info['update'] else '-', rows[-1]['err'], sig, 'in95' if rows[-1]['inside95'] else 'OUT ',
            bias, cerr[k], info.get('why') or 'rms %.1f w %.2f temper %.2f%s%s' % (
                info['best_rms_mrad'], info.get('weight', 1.0), info['temper'],
                ' SEEDED' if info.get('seeded') else '', ' RESEED' if info.get('reseeded') else '')), flush=True)
    field.ex.shutdown()
    return rows


def main():
    global ROLLS
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('routes', nargs='+')
    ap.add_argument('--n', type=int, default=1000); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--compass', default='iid', choices=['iid', 'bias'])
    ap.add_argument('--dense-init', action='store_true'); ap.add_argument('--bias-state', action='store_true')
    ap.add_argument('--soft-relief', action='store_true')
    ap.add_argument('--known-start', type=float, default=None, metavar='M',
                    help='the departure point is known to M metres (1 sigma): seed there, not a 6 km box')
    ap.add_argument('--tag', default='', help='name for the results file')
    ap.add_argument('--single', default='speed_B', help='single-frame solves to compare with')
    ap.add_argument('--rolls', default=','.join('%g' % r for r in ROLLS), help='roll offsets (deg) searched about the levelled roll')
    ap.add_argument('--views', default=None, metavar='DIR,DIR', help='look with the first view while the cloud is wider '
                    'than --switch-sigma, else with the second (the same instants: look_planner.py --views)')
    ap.add_argument('--switch-sigma', type=float, default=120.0)
    a = ap.parse_args()
    ROLLS = tuple(float(x) for x in a.rolls.split(','))
    opts = dict(compass=a.compass, dense_init=a.dense_init, bias_state=a.bias_state, soft_relief=a.soft_relief,
                known_start=a.known_start, switch_sigma=a.switch_sigma,
                views=[os.path.abspath(v) for v in a.views.split(',')] if a.views else None)
    d = os.path.abspath(a.dir)
    single = {r['tag']: r for r in json.load(open(os.path.join(d, a.single, 'summary.json')))}
    allr = {}
    for route in a.routes:
        rows = run_route(d, route, a.n, a.workers, opts)
        allr[route] = rows
        e = np.array([r['err'] for r in rows])
        s = [single[r['tag']] for r in rows if r['tag'] in single]
        se = np.array([x['err'] if x['status'] == 'ok' else np.nan for x in s])
        print('%s: particle filter median %.0f m, p90 %.0f m, %d/%d frames under 300 m, 95%% ellipse holds the truth on %d/%d; '
              'single-frame accepted %d/%d, median %.0f m, %d under 300 m; %.1f s a frame' % (
                  route, np.median(e), np.percentile(e, 90), (e < 300).sum(), e.size, sum(r['inside95'] for r in rows), e.size,
                  np.isfinite(se).sum(), len(s), np.nanmedian(se), (se < 300).sum(), np.mean([r['dt'] for r in rows])), flush=True)
    name = a.tag or ('pf_' + a.compass + ''.join(x for x, on in (('_dense', a.dense_init), ('_bias', a.bias_state), ('_soft', a.soft_relief)) if on))
    json.dump(dict(opts=opts, routes=allr), open(os.path.join(d, name + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
