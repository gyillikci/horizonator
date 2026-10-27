#!/usr/bin/env python3
"""Solve the frames a cruise recorded (cruise.mjs --record DIR): SegFormer on each photo,
E5cj's skyline and sea-line post-processing, then skyfix with the campaign pipeline's flags
(campaign_frames.json pipeline_flags: sea line required, soft compass, roll search, gates).

The solver is handed what a boat would know: a position prior OFF metres from the truth
(random direction) and a compass heading with an error drawn from N(0, COMPASS sigma); the
truth pose is only used to score. Straight rays (k = 0): CesiumJS draws no refraction; no
crest offset: the render has none of the real photos' canopy/segmenter bias (E5ck).

    python3 solve_cruise.py DIR [--off 1000] [--compass 5] [--jobs 14]
"""
import argparse, glob, json, math, os, subprocess, sys, zlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
import e5cj_seg_level as J  # noqa: E402
import extract  # noqa: E402
import skyline as S  # noqa: E402

PY = sys.executable
BAND_FRAC = 1 / 3       # segment_band's height, as a fraction of the frame
EXTRA = []              # more skyfix arguments (--extra)
LEVEL_TOL, IMU_SIG, SOLVE_DIR, SEG_SUFFIX = None, 0.0, 'solve', ''     # set from --level-tol / --imu / --out / --band
MAX_W = 1600            # frames wider are downsized on load, by skyfix too (--max-width)
DEM = os.path.expanduser('~/.horizonator/DEMs_COP30')


def prep(d, tags, off, csig, seed):
    from segformer_bridge import SegFormer
    import time
    sf = SegFormer()
    times, grown = [], 0
    for t in tags:
        out = os.path.join(d, t)
        if os.path.exists(out + SEG_SUFFIX + '_segpost.npz'):
            continue
        if SEG_SUFFIX not in ('', '_band'):     # seg_eval.py --save writes these; nothing here may stand in
            print('%s: no %s_segpost.npz, not solved' % (t, SEG_SUFFIX), flush=True)
            continue
        t0 = time.perf_counter()
        if SEG_SUFFIX == '_band':          # the band around the horizon the IMU predicts
            p = frame(d, t)
            _, _, imu = knows(p, t, off, csig, seed)
            _, info = segment(sf, out + '_photo.png', out + SEG_SUFFIX, band=(imu[0], imu[1], p['hfov_deg'], p['z_msl']))
            grown += info[2] > 0
        else:
            segment(sf, out + '_photo.png', out)
        times.append(time.perf_counter() - t0)
    if times:
        print('segmented %d frames: %.3f s median each%s' % (len(times), float(np.median(times)),
              ', band grown on %d' % grown if SEG_SUFFIX == '_band' else ''), flush=True)


def boat_cmd(photo, segpost, sealine, prior, compass, z, fov, out, imu=None, level_tol=2.5):
    """skyfix for a boat: the campaign pipeline's flags without its land observer (a boat is
    on the water), no crest offset and straight rays (CesiumJS renders), the sea line vetted
    by the IMU's pitch and roll when there is one."""
    flags = json.load(open(J.C.FRAMES))['pipeline_flags'].replace('~', os.path.expanduser('~'))
    flags = flags.replace('--crest-dh 9', '--crest-dh 0').replace('--land-observer 2.0', '').replace('--land-only 60.0', '')
    # a flat skyline far off does not pin a boat: every wrong accept of the Aegean sweep was a
    # 15-30 km view under 5 mrad of relief (5 mrad: 11 -> 2 wrong for 10 good fixes)
    flags += ' --min-relief 5'
    # the roll searched in the refinement only: 1.7x faster for one more wrong accept in 305
    flags += ' --roll-stages refine'
    cmd = [PY, os.path.join(EXP, 'skyfix.py'), photo, '--center', '%.7f,%.7f' % prior, '--box', '6000',
           '--z', '%.2f' % z, '--fov', '%.3f' % fov, '--heading', '%.3f' % compass, '--boundary-npy', segpost,
           '--refraction-k', '0', '--out', out] + flags.split()
    cmd += ['--level-npy', sealine, '--require-level']
    if level_tol:
        cmd += ['--level-tol', str(level_tol)]
    if imu is not None:
        cmd += ['--imu-attitude', '%.3f,%.3f' % imu, '--imu-gate', '1.5']
    if MAX_W != 1600:
        cmd += ['--max-width', str(MAX_W)]
    return cmd + EXTRA


def horizon_rows(attitude, fov, W, H, z):
    """The sea horizon's highest and lowest row in a W x H frame for a camera at pitch, roll
    (deg) and height z, exact camera, straight rays."""
    pitch, roll = attitude
    f = (W / 2) / math.tan(math.radians(fov) / 2)
    v0 = f * math.tan(-S.horizon_dip_rad(max(z, 0.5), k=0.0) - math.radians(pitch)) / math.cos(math.radians(roll))
    tilt = math.tan(math.radians(roll)) * (W - 1) / 2
    rows = [(H - 1) / 2 - (v0 - tilt), (H - 1) / 2 - (v0 + tilt)]
    return min(rows), max(rows)


def segment_band(sf, img, attitude, fov, z):
    """SegFormer on a third of the frame around the horizon the IMU predicts (70 % of it above
    the horizon: the skyline is up there), grown when it is too small: upwards while land
    touches its top edge (a mountain taller than the band), downwards while land fills its
    bottom with no water above it (a near shore). Returns the class map and the band."""
    H, W = img.shape[:2]
    bh = int(H * BAND_FRAC)
    # the band starts and ends on SegFormer's own grid (its logits are 4 px of the resized
    # frame, 5 px of a 640 px one): off-grid, the band's lines moved 1-3 px from the frame's
    unit = max(1, int(round(4 * min(H, W) / sf.short_side)))
    lo, hi = horizon_rows(attitude, fov, W, H, z)
    r0 = int(max(0, math.floor((lo - 0.7 * bh) / unit) * unit))
    r1 = int(min(H, math.ceil((hi + 0.3 * bh) / unit) * unit))
    grown = 0
    for _ in range(5):
        cls = sf.predict_band(img, r0, r1)
        # land at the top edge, or a skyline within 16 px of it: a mountain taller than the band
        # (and SegFormer's edge padding bends a boundary that near)
        first_ground = np.argmax(cls[r0:r1] != 2, axis=0)
        top_land = ((cls[r0] != 2) | ((first_ground < 16) & (cls[r0:r1] != 2).any(0))).mean() if r0 > 0 else 0.0
        dry = (cls[r0:r1] == 1).sum(0) == 0
        bottom_land = ((cls[r1 - 1] == 0) & dry).mean() if r1 < H else 0.0
        if top_land > 0.01:
            r0 = max(0, (r0 - bh) // unit * unit); grown += 1
        elif bottom_land > 0.2:
            r1 = min(H, -(-(r1 + bh // 2) // unit) * unit); grown += 1
        else:
            break
    return cls, (r0, r1, grown)


def segment(sf, photo, base, band=None):
    """SegFormer's skyline and sea line for one photo, post-processed as E5cj does; writes
    base_segpost.npz and (if a straight sea line is found) base_sealine.npz. band = (pitch,
    roll, fov, z): segment only the band around the predicted horizon (segment_band)."""
    img = extract.load_image(photo, max_w=MAX_W)
    info = None
    if band:
        cls, info = segment_band(sf, img, band[:2], band[2], band[3])
    else:
        cls = sf.predict(img)
    sky, sea_raw = J.raw_lines(cls)
    conf = np.isfinite(sky) * 1.0
    e = max(1, round(J.EDGE_FRAC * cls.shape[1]))
    conf[:e] = conf[-e:] = 0.0
    np.savez(base + '_segpost.npz', rows=np.nan_to_num(sky), conf=conf)
    sea, fit = J.clean_sea_line(sea_raw)
    if fit:
        np.savez(base + '_sealine.npz', rows=sea)
    return (bool(fit), info) if band else bool(fit)


def frame(d, tag):
    p = json.load(open(os.path.join(d, tag + '.json')))
    p = p.get('pose', p)                                     # cesium_render's renders nest it
    p.setdefault('t_s', 0.0)
    p['roll_deg'] = (p['roll_deg'] + 180) % 360 - 180      # Cesium reports 359.7 for -0.3
    return p


def knows(p, tag, off, csig, seed):
    """What the boat knows for a frame: a position prior off m from the truth, a compass with
    N(0, csig) error and the IMU's pitch and roll (IMU_SIG), seeded by the frame."""
    rng = np.random.default_rng(zlib.crc32((tag + seed).encode()))
    th = rng.uniform(0, 2 * math.pi)
    mlat, mlon = S.meters_per_degree(p['lat'])
    prior = (p['lat'] + off * math.cos(th) / mlat, p['lon'] + off * math.sin(th) / mlon)
    comp = (p['heading_deg'] + rng.normal(0, csig)) % 360
    imu = (p['pitch_deg'] + rng.normal(0, IMU_SIG), p['roll_deg'] + rng.normal(0, IMU_SIG))
    return prior, comp, imu


def solve_one(d, tag, off, csig, seed):
    p = frame(d, tag)
    mlat, mlon = S.meters_per_degree(p['lat'])
    (lat_c, lon_c), comp, imu = knows(p, tag, off, csig, seed)
    out = os.path.join(d, SOLVE_DIR, tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sl = os.path.join(d, tag + SEG_SUFFIX + '_sealine.npz')
    if not os.path.exists(sl):
        return dict(tag=tag, status='no sea line', pose=p, compass_err=comp - p['heading_deg'])
    cmd = boat_cmd(os.path.join(d, tag + '_photo.png'), os.path.join(d, tag + SEG_SUFFIX + '_segpost.npz'), sl, (lat_c, lon_c), comp,
                   p['z_msl'], p['hfov_deg'], out, imu=imu if IMU_SIG else None, level_tol=LEVEL_TOL)
    if not os.path.exists(out + '.json'):        # solved before: keep it
        with open(out + '.log', 'w') as lg:
            subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=EXP, env=dict(os.environ, OMP_NUM_THREADS='1'))
    if not os.path.exists(out + '.json'):
        return dict(tag=tag, status='failed', pose=p, compass_err=comp - p['heading_deg'])
    j = json.load(open(out + '.json'))
    r = dict(tag=tag, status=j['status'], pose=p, compass_err=(comp - p['heading_deg'] + 180) % 360 - 180,
             reasons=[x[:60] for x in j.get('reasons', [])])
    if j.get('lat') is not None and j.get('dn_m') is not None:
        dn, de = (j['lat'] - p['lat']) * mlat, (j['lon'] - p['lon']) * mlon
        h = math.radians(p['heading_deg'])
        r.update(err=math.hypot(dn, de), along=dn * math.cos(h) + de * math.sin(h), cross=-dn * math.sin(h) + de * math.cos(h),
                 rms=j['rms_mrad'], prior_err=off)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir'); ap.add_argument('--off', type=float, default=1000.0)
    ap.add_argument('--compass', type=float, default=5.0); ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--seed', default='a')
    ap.add_argument('--level-tol', type=float, default=None, help='levelling tolerance px (renders: 2.5)')
    ap.add_argument('--imu', type=float, default=0.0, help='simulated IMU sigma deg; vets the sea line (0 = off)')
    ap.add_argument('--out', default='solve', help='subfolder for the solves')
    ap.add_argument('--band', action='store_true', help='segment only the band around the IMU horizon (needs --imu)')
    ap.add_argument('--extra', default='', help='more skyfix arguments, e.g. "--roll-stages refine"')
    ap.add_argument('--seg-suffix', default='', help='solve with <tag><suffix>_segpost/_sealine.npz (seg_eval.py --save)')
    ap.add_argument('--max-width', type=int, default=1600, help='keep frames up to this wide at full resolution')
    a = ap.parse_args()
    global LEVEL_TOL, IMU_SIG, SOLVE_DIR, SEG_SUFFIX, MAX_W
    MAX_W = a.max_width
    LEVEL_TOL, IMU_SIG, SOLVE_DIR = a.level_tol, a.imu, a.out
    SEG_SUFFIX = '_band' if a.band else a.seg_suffix
    EXTRA[:] = a.extra.split()
    a.dir = os.path.abspath(a.dir)
    tags = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(a.dir, 'cruise_*.json')))
    prep(a.dir, tags, a.off, a.compass, a.seed)
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda t: solve_one(a.dir, t, a.off, a.compass, a.seed), tags))
    print('%-12s %6s %6s %7s %7s %-13s %s' % ('frame', 't_s', 'compass', 'err', 'along', 'status', 'reasons'))
    for r in res:
        print('%-12s %6.0f %+6.1f %7s %7s %-13s %s' % (r['tag'], r['pose']['t_s'], r['compass_err'],
              '%.0f' % r['err'] if 'err' in r else '-', '%+.0f' % r['along'] if 'along' in r else '-', r['status'], '; '.join(r.get('reasons', []))))
    ok = [r for r in res if r['status'] == 'ok']
    e = np.array([r['err'] for r in res if 'err' in r])
    print('\n%d frames: %d solved, %d accepted (%d under 300 m, %d over 1 km); median error of all fixes %.0f m' % (
        len(res), len(e), len(ok), sum(r['err'] < 300 for r in ok), sum(r['err'] > 1000 for r in ok), float(np.median(e)) if e.size else float('nan')))
    json.dump(res, open(os.path.join(a.dir, SOLVE_DIR, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
