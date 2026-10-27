#!/usr/bin/env python3
"""Live position fixes for cruise.mjs --live: a local HTTP server that keeps SegFormer loaded
and answers each frame with a skyfix solve (solve_cruise.boat_cmd: the campaign pipeline for a
boat, sea line vetted by the IMU).

POST /solve  {tag, png (data URL or base64), prior: [lat, lon], compass_deg, imu: [pitch, roll],
              z_msl, hfov_deg, truth: {lat, lon} (optional, for scoring only), yaw_deg (the pan head's)}
  -> {tag, status, lat, lon, err_m, rms_mrad, reasons, seg_s, solve_s, look_yaw (--pf --look)}

    python3 live_solver.py [--port 8765] [--workers 2] [--out out/live]
"""
import argparse, base64, json, math, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solve_cruise as SC  # noqa: E402
import skyline as S  # noqa: E402

PF_STATE = {}                                     # session -> particle filter
LOOK_CM = {}                                      # session -> look_planner's marcher


def track(q, base, has_sea):
    """One particle-filter step: predict by the leg the boat dead-reckoned since its last frame,
    update with this frame's skyline (when the frame gives one), resample, estimate."""
    import numpy as np
    import pf_track as P
    sess = q.get('session', 'default')
    pf = PF_STATE.get(sess)
    # the configuration the Aegean sweep chose (pf_track: --dense-init --bias-state; soft relief
    # updates misled it on flat skylines): the compass bias is a particle state, so only the
    # compass's white noise (2 deg) is left to the heading search
    opts = dict(dense_init=True, bias_state=True)
    if pf is None:
        pf = P.PF(P.Field(q['prior'][0], q['prior'][1], A.pf_workers), q['prior'][0], q['prior'][1], n=A.particles,
                  bias_state=True)
        pf.seed_box(*q['prior'])
        PF_STATE[sess] = pf
    elif q.get('dr'):
        pf.predict(np.array(q['dr'][:2], float), float(q['dr'][2]))
    ob, why = (P.observation(base + '_segpost.npz', base + '_sealine.npz', q['W'], q['H'], q['hfov_deg'], q['z_msl'],
                             q['compass_deg'], q['imu'], csig=2.0, reach=35.0) if has_sea else (None, 'no sea line'))
    info = pf.step(ob, why, opts)
    pf.resample()
    lat, lon, cov, bias = pf.estimate()
    ev = np.linalg.eigvalsh(cov)
    pick = pf.rng.choice(len(pf.w), size=min(300, len(pf.w)), replace=False, p=pf.w)
    plat, plon = pf.ll(pf.x[pick])
    r = dict(pf=dict(lat=float(lat), lon=float(lon), sigma_major=float(math.sqrt(max(ev[-1], 0))),
                     sigma_minor=float(math.sqrt(max(ev[0], 0))), compass_bias_deg=bias),
             particles=[[round(float(a), 6), round(float(b), 6)] for a, b in zip(plat, plon)], **info)
    if q.get('truth'):
        mlat, mlon = S.meters_per_degree(q['truth']['lat'])
        r['pf']['err_m'] = math.hypot((lat - q['truth']['lat']) * mlat, (lon - q['truth']['lon']) * mlon)
    if A.look and q.get('yaw_deg') is not None:       # where the camera should turn for the next frame
        import look_planner as L
        cm = LOOK_CM.get(sess) or LOOK_CM.setdefault(sess, L.marcher(lat, lon))
        hull = q['compass_deg'] - bias - q['yaw_deg']  # the pan head's yaw off the bow is known
        # unconverged, the most informative land; converged, far ridges the DEM gets right
        r['look_mode'] = 'explore' if r['pf']['sigma_major'] > 120.0 else 'exploit'
        r['look_yaw'] = L.best_yaw(cm, lat, lon, q['z_msl'], hull, q['hfov_deg'], r['look_mode'])[0]
    return r


def solve(q):
    tag = q['tag']
    base = os.path.join(OUT, tag)
    png = q['png'].split(',', 1)[1] if q['png'].startswith('data:') else q['png']
    open(base + '_photo.png', 'wb').write(base64.b64decode(png))
    t0 = time.time()
    with SEG_LOCK:
        if A.band and q.get('imu'):
            has_sea, band = SC.segment(SF, base + '_photo.png', base, band=(q['imu'][0], q['imu'][1], q['hfov_deg'], q['z_msl']))
        else:
            has_sea, band = SC.segment(SF, base + '_photo.png', base), None
    seg_s = time.time() - t0
    r = dict(tag=tag, seg_s=seg_s, truth=q.get('truth'), prior=q['prior'], band=band)
    if A.pf:
        t1 = time.time()
        r.update(track(q, base, has_sea), status='pf', solve_s=time.time() - t1)
        if r['pf'].get('err_m') is not None:
            r['err_m'] = r['pf']['err_m']
        r['lat'], r['lon'] = r['pf']['lat'], r['pf']['lon']
        return r
    if not has_sea:
        r.update(status='no sea line', reasons=['no straight sea line from the segmenter'], solve_s=0.0)
        return r
    cmd = SC.boat_cmd(base + '_photo.png', base + '_segpost.npz', base + '_sealine.npz', tuple(q['prior']),
                      q['compass_deg'], q['z_msl'], q['hfov_deg'], base, imu=tuple(q['imu']) if q.get('imu') else None)
    t1 = time.time()
    with SOLVES:
        with open(base + '.log', 'w') as lg:
            subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=SC.EXP, env=dict(os.environ, OMP_NUM_THREADS='1'))
    r['solve_s'] = time.time() - t1
    if not os.path.exists(base + '.json'):
        r.update(status='failed', reasons=['skyfix wrote no result'])
        return r
    j = json.load(open(base + '.json'))
    r.update(status=j['status'], reasons=[x[:80] for x in j.get('reasons', [])], rms_mrad=j.get('rms_mrad'))
    if j.get('dn_m') is not None:
        r.update(lat=j['lat'], lon=j['lon'])
        if q.get('truth'):
            mlat, mlon = S.meters_per_degree(q['truth']['lat'])
            r['err_m'] = math.hypot((j['lat'] - q['truth']['lat']) * mlat, (j['lon'] - q['truth']['lon']) * mlon)
    return r


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/solve':
            self.send_error(404); return
        q = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        try:
            r = solve(q)
        except Exception as e:                      # report, keep serving
            r = dict(tag=q.get('tag'), status='failed', reasons=[repr(e)[:200]])
        LOG.write(json.dumps(r) + '\n'); LOG.flush()
        print('%-24s %-12s %s  seg %.1f s  solve %.1f s' % (r['tag'], r['status'],
              ('%6.0f m' % r['err_m']) if r.get('err_m') is not None else '     - ', r.get('seg_s', 0), r.get('solve_s', 0)), flush=True)
        b = json.dumps(r).encode()
        self.send_response(200)
        self.send_header('content-type', 'application/json'); self.send_header('content-length', str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def log_message(self, *a):
        pass


def main():
    # run in the server process only: Python 3.14's forkserver re-imports this module in every
    # particle-filter worker, which must neither load SegFormer nor bind the port again
    global A, OUT, SF, SEG_LOCK, SOLVES, LOG
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--workers', type=int, default=2, help='skyfix solves at once')
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'live'))
    ap.add_argument('--band', action='store_true', help='SegFormer on the band around the IMU horizon only')
    ap.add_argument('--pf', action='store_true', help='track with the particle filter (pf_track.py) instead of '
                                                      'single-frame skyfix; the driver sends its dead-reckoned legs')
    ap.add_argument('--particles', type=int, default=1000)
    ap.add_argument('--pf-workers', type=int, default=8)
    ap.add_argument('--look', action='store_true', help='with --pf: answer with the camera yaw (look_planner.py) that '
                                                        'sees the most usable land from the estimate; cruise.mjs --look-live turns to it')
    A = ap.parse_args()
    OUT = os.path.join(os.path.abspath(A.out), time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(OUT, exist_ok=True)

    from segformer_bridge import SegFormer
    SF = SegFormer()
    SEG_LOCK = threading.Lock()                       # one GPU segmentation at a time
    SOLVES = threading.BoundedSemaphore(A.workers)
    LOG = open(os.path.join(OUT, 'fixes.jsonl'), 'a')
    print('live solver on http://127.0.0.1:%d, frames and fixes in %s' % (A.port, OUT), flush=True)
    ThreadingHTTPServer(('127.0.0.1', A.port), H).serve_forever()


if __name__ == '__main__':
    main()
