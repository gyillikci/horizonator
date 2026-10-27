#!/usr/bin/env python3
"""E5ck: the error budget, one input at a time.

Every input the solver takes first gets its true value and no search
around it (heading window 0, elevation offset fixed at 0), and the fix is
made. Then one input at a time gets an error back, or its search window,
to see which error moves the fix, how far, and which way: along the line
of sight (range) or across it (bearing).

Sets
  real     the 44 campaign photos with SegFormer's post-processed lines
           (E5cj). Their attitude truth is the best fit at the GPS position
           with the exact camera (`truth`), so 'oracle' here is optimistic:
           the fitted attitude absorbs part of the DEM's error.
  perfect  the observation IS the solver's model: COP30 at the true pose,
           through a pinhole camera rotated by the true attitude, with the
           solver's d_min, d_max and refraction and no crest offset. What
           is left is the solver's own (grid, search, gates) and whatever
           is injected. At the 44 campaign truths (camp) and the 29 sea
           poses (sea).
  google   CesiumJS renders of Google Photorealistic 3D Tiles at the sea
           poses (cesium_render/out/sealevel16): the render's exact
           skyline or SegFormer's on the rendered photo. Exact truth, a
           different terrain model from COP30 (trees, buildings, k = 0).

Conditions are '/'-joined tokens on top of the oracle, e.g. 'compass:2',
'compass:5/hw:6', 'pitch:0.5', 'pitch:0.5/beta:10', 'fov:2' (percent),
'z:5', 'cam:legacy', 'level', 'obs:noise2', 'off:2000', 'pipeline'.
See COND_HELP.

    python3 e5ck_oracle.py truth                    # real: attitude at truth
    python3 e5ck_oracle.py world                    # perfect-world lines
    python3 e5ck_oracle.py synth                    # google-world lines
    python3 e5ck_oracle.py solve SET COND [COND ...] [--jobs 14]
    python3 e5ck_oracle.py report SET COND [COND ...]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from assets import celnav  # noqa: E402

OUT = os.path.join(HERE, 'out', 'oracle')
FRAMES = os.path.join(HERE, 'data', 'campaign_frames.json')
SEGL = os.path.join(HERE, 'out', 'seg_level')
CES = os.path.join(HERE, 'cesium_render', 'out')
RENDERS = os.path.join(CES, 'sealevel16')
DEM = os.path.expanduser('~/.horizonator/DEMs_COP30')
PY = sys.executable
BOX = 6000.0
OFF = 1000.0        # default prior-position error: the truth is not a grid node
LAND_H, LAND_R = 2.0, 60.0

COND_HELP = """tokens, on top of the oracle (true value, no search):
  compass:D    heading prior = truth + D deg ('compass:real': the photo's own compass)
  hw:W         heading search window +-W deg (oracle 0)
  csig:S       soft compass prior sigma S deg (widens the window to 4 S);
               csig:acc = Theodolite's recorded compass accuracy (else 10)
  rw:W rs:S    search the roll +-W deg in S deg steps (skyfix --roll-window)
  pitch:D      pitch = truth + D deg      roll:D   roll = truth + D deg
  beta:B       elevation-offset band +-B mrad ('beta:free' = the prior's +-10)
  level        pitch and roll from the sea line (the levelling), its own band
  lpitch/lroll that one angle from the levelling, the other true
  band:LO,HI   elevation-offset band LO..HI mrad
  fov:P        field of view x (1 + P/100)
  z:D          camera height + D m        land / landonly   E5cj's land arms
  wb           waterline band -12.5,5 mrad (E5cj)
  cam:legacy   the historical camera model (oracle and skyfix default: exact)
  reqlevel     with level: refuse a photo the levelling cannot level
  crest:C      crest offset C m (oracle: 0 perfect, 9 real/google)
  k:K          the solver's refraction (oracle: the world's)
  off:R        prior position R m from the truth (oracle 1000)
  ri:N         refinement passes per level (skyfix --refine-iter; study 10,
               pipeline 1)
  obs:V        observation variant (perfect: k0.30, near, noise2;
               google: exact, segf)
  peak:T       peak-height gate T m (skyfix --max-peak-dh; study 30)
  jack         skyfix --conditioning: re-solve each half of the frame
  par:X rms:X  gates: skyfix --min-parallax X (per km), --max-rms X (mrad)
  pipeline     real: the campaign pipeline, campaign_frames.json's
               pipeline_flags (compass:real/csig:5/rw:1/peak:100/rms:7/
               par:0.03/level/wb/land/landonly/reqlevel/crest:9, pitch and
               band free)
  pipeline_v3  the same without rms:7/par:0.03
  pipeline_v2  pipeline_v3 with hw:6, no roll search, peak:30
  pipeline_v1  E5cj's level_landonly_wb12_hard6 as it ran: cam:legacy/ri:1,
               no reqlevel"""


# ---------------------------------------------------------------- frames

def campaign():
    return json.load(open(FRAMES))['frames']


def sea_poses():
    return json.load(open(os.path.join(CES, 'poses_sealevel.json')))


def records(set_):
    """Per frame: the truth, the image, the observation files."""
    recs = []
    if set_ in ('real', 'perfect_camp'):
        tr = json.load(open(os.path.join(OUT, 'truth_real.json')))
        tf = os.path.join(OUT, 'theodolite_meta.json')
        theo = json.load(open(tf)) if os.path.exists(tf) else {}
        for fr in campaign():
            t = tr.get(fr['tag'])
            if t is None:
                continue
            r = dict(tag=fr['tag'], image=celnav(fr['path']), lat=fr['lat'],
                     lon=fr['lon'], z=fr['z'], fov=fr['fov'],
                     heading=t['heading'], pitch=t['pitch'], roll=t['roll'],
                     compass=fr['heading'], W=t['W'], H=t['H'])
            r['lev'] = t.get('level')       # the exact levelling's pitch and roll
            # iOS's compass accuracy, as Theodolite recorded it (PF frames: none)
            r['acc'] = theo.get(fr['tag'], {}).get('azimuth_acc_deg')
            if set_ == 'real':
                r['obs'] = {'': os.path.join(SEGL, fr['tag'] + '_segpost.npz')}
                sl = os.path.join(SEGL, fr['tag'] + '_sealine.npz')
                r['sea'] = {'': sl} if os.path.exists(sl) else {}
                r['crest'], r['k'] = 9.0, None
            else:
                r.update(perfect_files('camp', fr['tag']))
                r['crest'], r['k'] = 0.0, None
            recs.append(r)
    elif set_ in ('perfect_sea', 'google'):
        for p in sea_poses():
            r = dict(tag=p['tag'], image=os.path.join(RENDERS, p['tag'] + '_photo.png'),
                     lat=p['lat'], lon=p['lon'], z=p['z_msl'], fov=p['hfov_deg'],
                     heading=p['heading_deg'], pitch=p['pitch_deg'], roll=p['roll_deg'],
                     compass=None, W=p['width'], H=p['height'])
            if set_ == 'google':
                d = os.path.join(OUT, 'google')
                r['obs'] = {v: os.path.join(d, '%s_%s.npz' % (p['tag'], v))
                            for v in ('exact', 'segf')}
                r['sea'] = {v: os.path.join(d, '%s_%s_sea.npz' % (p['tag'], v))
                            for v in ('exact', 'segf')}
                # CesiumJS draws straight rays; and no crest offset: the 9 m
                # that real photos need (SegFormer's boundary sits above the
                # ridge, and canopy) moved render fixes 86 -> 200 m (E5ck)
                r['crest'], r['k'] = 0.0, 0.0
            else:
                r.update(perfect_files('sea', p['tag']))
                r['crest'], r['k'] = 0.0, None
            recs.append(r)
    else:
        sys.exit('unknown set ' + set_)
    return recs


def perfect_files(geom, tag):
    d = os.path.join(OUT, 'perfect', geom)
    return dict(obs={v: os.path.join(d, '%s_%s.npz' % (tag, v)) for v in WORLDS},
                sea={v: os.path.join(d, '%s_%s_sea.npz' % (tag, v)) for v in WORLDS})


# --------------------------------------------------------------- cameras

def project_exact(az_deg, el, heading, pitch, roll, fov, W, H):
    """Global azimuth/elevation -> image row per column for a pinhole camera
    rotated by heading, pitch and roll: the inverse of skyfix.pixel_angles
    with --camera-model exact (and how CesiumJS draws)."""
    f = (W / 2) / math.tan(math.radians(fov) / 2)
    a = np.radians((np.asarray(az_deg) - heading + 180) % 360 - 180)
    th = math.radians(pitch)
    X, Y, Z = np.sin(a) * np.cos(el), np.sin(el), np.cos(a) * np.cos(el)
    yc, zc = Y * math.cos(th) - Z * math.sin(th), Y * math.sin(th) + Z * math.cos(th)
    ok = zc > 1e-6
    zs = np.where(ok, zc, 1.0)
    ur, vr = f * X / zs, f * yc / zs
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    u, v = ur * cr + vr * sr, -ur * sr + vr * cr
    col, row = u + (W - 1) / 2, (H - 1) / 2 - v
    m = ok & np.isfinite(row) & (np.abs(a) < math.radians(89))
    if m.sum() < 2:
        return np.full(W, np.nan)
    o = np.argsort(col[m])
    return np.interp(np.arange(W), col[m][o], row[m][o], left=np.nan, right=np.nan)


def resample(src, W):
    src = np.asarray(src, float)
    return np.interp(np.linspace(0, src.size - 1, W), np.arange(src.size), src) * (W / src.size)


# ----------------------------------------------------------------- truth

def fit_truth(fr, span=30.0):
    """Heading, pitch and roll that best explain the photo at its GPS
    position, exact camera, the solver's model (crest 9, k 0.13, d_min
    1 km): roll from the sea line when there is one, then a pitch x roll
    grid with the heading searched +-30 deg at each, then sub-lag heading."""
    import extract
    import skyfix as F
    import skyline as S
    F.CAMERA, F.EXTRACTOR, F.COMPASS_K = 'exact', 'ewasr', 0.0
    b = np.load(os.path.join(SEGL, fr['tag'] + '_segpost.npz'))
    F.BOUNDARY_ROWS, F.BOUNDARY_CONF = b['rows'], b['conf']
    img = extract.load_image(celnav(fr['path']))
    H, W = img.shape[:2]
    f_px = (W / 2) / math.tan(math.radians(fr['fov']) / 2)
    dip = S.horizon_dip_rad(max(fr['z'], 0.5))
    lev = None
    sl = os.path.join(SEGL, fr['tag'] + '_sealine.npz')
    if os.path.exists(sl):
        lr = resample(np.load(sl)['rows'], W)
        lev = extract.sea_horizon_attitude(np.nan_to_num(lr), np.isfinite(lr) * 1.0,
                                           img.shape, f_px, dip, camera='exact')
    cm = S.CMarcher(DEM, (fr['lat'] - .6, fr['lat'] + .6), (fr['lon'] - .8, fr['lon'] + .8),
                    d_min=1000.0, d_max=40000.0)
    el, r = cm.skyline(fr['lat'], fr['lon'], fr['z'], F.AZ)
    el = el + np.where(np.isfinite(r) & (r > 0), 9.0 / np.maximum(r, 1000.0), 0.0)
    shifts = np.arange(-int(span * 10), int(span * 10) + 1)

    def cost(h, p, ro, sh=shifts):
        eo, w, _ = F.observation(img, fr['fov'], h, ro, p)
        return F.photo_cost(eo, w, el, sh, betas=[0.0])     # (cost, shift, 0)

    roll0 = lev['roll_deg'] if lev else 0.0
    rolls = roll0 + (np.arange(-1.0, 1.01, 0.5) if lev else np.arange(-4.0, 4.01, 0.5))
    best = None
    for p in np.arange(-9.0, 9.01, 0.25):
        for ro in rolls:
            c = cost(fr['heading'], p, ro)
            if best is None or c[0] < best[0]:
                best = (c[0], fr['heading'] + 0.1 * c[1], p, ro)
    _, h0, p0, r0 = best
    for p in p0 + np.arange(-0.3, 0.301, 0.02):
        for ro in r0 + np.arange(-0.5, 0.501, 0.1):
            c = cost(h0, p, ro, np.arange(-10, 11))
            if c[0] < best[0]:
                best = (c[0], h0 + 0.1 * c[1], p, ro)
    _, h0, p0, r0 = best
    for dh in np.arange(-0.1, 0.101, 0.02):
        c = cost(h0 + dh, p0, r0, [0])
        if c[0] < best[0]:
            best = (c[0], h0 + dh, p0, r0)
    c, h, p, ro = best
    return dict(heading=float(h % 360.0), pitch=float(p), roll=float(ro),
                rms_mrad=float(math.sqrt(2 * c) * 1e3), W=int(W), H=int(H),
                compass_error_deg=float((fr['heading'] - h + 180) % 360 - 180),
                level=lev and dict(pitch=lev['pitch_deg'], roll=lev['roll_deg']))


def cmd_truth(a):
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, 'truth_real.json')
    res = json.load(open(f)) if os.path.exists(f) else {}
    todo = [fr for fr in campaign() if fr['tag'] not in res
            and os.path.exists(os.path.join(SEGL, fr['tag'] + '_segpost.npz'))]
    with ThreadPoolExecutor(a.jobs) as ex:
        # one process per frame: the fit sets skyfix's module state
        outs = ex.map(lambda fr: subprocess.run(
            [PY, __file__, '_truth1', fr['tag']], capture_output=True, text=True,
            env=dict(os.environ, OMP_NUM_THREADS='1')), todo)
        for fr, o in zip(todo, outs):
            try:
                res[fr['tag']] = json.loads(o.stdout.strip().splitlines()[-1])
            except Exception:
                print(fr['tag'], 'FAILED', o.stderr[-300:]); continue
            t = res[fr['tag']]
            print('%-28s heading %6.1f (compass off %+5.1f)  pitch %+5.2f  roll %+5.2f  rms %4.1f mrad%s' % (
                fr['tag'], t['heading'], t['compass_error_deg'], t['pitch'], t['roll'], t['rms_mrad'],
                '  level pitch %+5.2f roll %+5.2f' % (t['level']['pitch'], t['level']['roll']) if t['level'] else ''),
                flush=True)
    json.dump(res, open(f, 'w'), indent=1)


# ----------------------------------------------------------------- worlds

# perfect-world variants: refraction of the world, near terrain, row noise
WORLDS = {'w': dict(k=None, dmin=1000.0, noise=0.0),
          'k0.00': dict(k=0.0, dmin=1000.0, noise=0.0),
          'k0.07': dict(k=0.07, dmin=1000.0, noise=0.0),
          'k0.20': dict(k=0.20, dmin=1000.0, noise=0.0),
          'k0.30': dict(k=0.30, dmin=1000.0, noise=0.0),
          'near': dict(k=None, dmin=50.0, noise=0.0),
          'noise0.5': dict(k=None, dmin=1000.0, noise=0.5),
          'noise1': dict(k=None, dmin=1000.0, noise=1.0),
          'noise2': dict(k=None, dmin=1000.0, noise=2.0),
          'noise4': dict(k=None, dmin=1000.0, noise=4.0)}


def make_world(r, geom):
    import skyfix as F
    import skyline as S
    d = os.path.join(OUT, 'perfect', geom)
    os.makedirs(d, exist_ok=True)
    cms = {}
    for v, wv in WORLDS.items():
        key = (wv['k'], wv['dmin'])
        if key not in cms:
            k = S.K_REFRACTION if wv['k'] is None else wv['k']
            cm = S.CMarcher(DEM, (r['lat'] - .6, r['lat'] + .6), (r['lon'] - .8, r['lon'] + .8),
                            d_min=wv['dmin'], d_max=40000.0, refraction_k=k)
            el, _ = cm.skyline(r['lat'], r['lon'], r['z'], F.AZ)
            rows = project_exact(F.AZ, el, r['heading'], r['pitch'], r['roll'], r['fov'], r['W'], r['H'])
            # the sea horizon, everywhere: an ideal level reference
            dip = S.horizon_dip_rad(max(r['z'], 0.5), k=k)
            sea = project_exact(F.AZ, np.full(F.AZ.size, -dip), r['heading'], r['pitch'], r['roll'],
                                r['fov'], r['W'], r['H'])
            cms[key] = rows, sea
        rows, sea = cms[key]
        rows = rows.copy()
        if wv['noise']:
            rng = np.random.default_rng(zlib.crc32(r['tag'].encode()))
            rows += rng.normal(0.0, wv['noise'], rows.size)
        ok = np.isfinite(rows) & (rows >= 0) & (rows <= r['H'] - 1)
        np.savez(os.path.join(d, '%s_%s.npz' % (r['tag'], v)), rows=np.where(ok, rows, 0.0), conf=ok * 1.0)
        sk = np.isfinite(sea) & (sea >= 0) & (sea <= r['H'] - 1)
        np.savez(os.path.join(d, '%s_%s_sea.npz' % (r['tag'], v)), rows=np.where(sk, sea, np.nan))
    return r['tag']


def cmd_world(a):
    jobs = [(r, 'camp') for r in records_raw('camp')] + [(r, 'sea') for r in records_raw('sea')]
    with ThreadPoolExecutor(a.jobs) as ex:
        for t in ex.map(lambda j: make_world(*j), jobs):
            pass
    print('perfect-world lines for %d poses x %d variants' % (len(jobs), len(WORLDS)))


def records_raw(geom):
    """Truth poses without observation files (for building them)."""
    if geom == 'camp':
        tr = json.load(open(os.path.join(OUT, 'truth_real.json')))
        return [dict(tag=fr['tag'], lat=fr['lat'], lon=fr['lon'], z=fr['z'], fov=fr['fov'],
                     heading=tr[fr['tag']]['heading'], pitch=tr[fr['tag']]['pitch'],
                     roll=tr[fr['tag']]['roll'], W=tr[fr['tag']]['W'], H=tr[fr['tag']]['H'])
                for fr in campaign() if fr['tag'] in tr]
    return [dict(tag=p['tag'], lat=p['lat'], lon=p['lon'], z=p['z_msl'], fov=p['hfov_deg'],
                 heading=p['heading_deg'], pitch=p['pitch_deg'], roll=p['roll_deg'],
                 W=p['width'], H=p['height']) for p in sea_poses()]


def cmd_synth(a):
    """Google world: the render's exact lines (its segmentation pass) and
    SegFormer's on the rendered photo, post-processed as E5cj does."""
    import extract
    import e5cj_seg_level as J
    from segformer_bridge import SegFormer
    sys.path.insert(0, os.path.join(HERE, 'cesium_render'))
    import compare as K
    d = os.path.join(OUT, 'google')
    os.makedirs(d, exist_ok=True)
    sf = SegFormer()
    for p in sea_poses():
        t = p['tag']
        sky, sea, _ = K.seg_lines(os.path.join(RENDERS, t + '_seg.png'))
        conf = np.isfinite(sky) * 1.0
        np.savez(os.path.join(d, t + '_exact.npz'), rows=np.nan_to_num(sky), conf=conf)
        # the render's sea line, kept where it is straight (as E5cj keeps SegFormer's)
        s_clean, _ = J.clean_sea_line(np.where(np.isfinite(sky), sea, np.nan))
        np.savez(os.path.join(d, t + '_exact_sea.npz'), rows=s_clean)
        img = extract.load_image(os.path.join(RENDERS, t + '_photo.png'))
        cls = sf.predict(img)
        sky2, sea2 = J.raw_lines(cls)
        c2 = np.isfinite(sky2) * 1.0
        e = max(1, round(J.EDGE_FRAC * cls.shape[1]))
        c2[:e] = c2[-e:] = 0.0
        np.savez(os.path.join(d, t + '_segf.npz'), rows=np.nan_to_num(sky2), conf=c2)
        s2, _ = J.clean_sea_line(sea2)
        np.savez(os.path.join(d, t + '_segf_sea.npz'), rows=s2)
        m = np.isfinite(sky) & np.isfinite(sky2)
        print('%-28s exact %4.0f%% cols, segformer %4.0f%%, |segformer - exact| median %.1f px' % (
            t, 100 * conf.mean(), 100 * c2.mean(), float(np.median(np.abs(sky2[m] - sky[m]))) if m.any() else float('nan')),
            flush=True)


# ------------------------------------------------------------- conditions

def parse(cond):
    c = {}
    for tok in cond.split('/'):
        if not tok or tok == 'oracle':
            continue
        k, _, v = tok.partition(':')
        c[k] = v if v else True
    names = ('pipeline', 'pipeline_v1', 'pipeline_v2', 'pipeline_v3')
    if any(n in c for n in names):
        # the campaign pipeline (campaign_frames.json pipeline_flags, decided
        # 2026-10-25): land observer, land only, waterline band, sea line
        # required, soft compass 5 deg, roll +-1 deg, peak gate 100 m,
        # residual gate 7 mrad, parallax gate 0.03/km, exact camera,
        # iterated refinement. _v3: the same without the residual and
        # parallax gates. _v2: before the compass, roll and peak-gate
        # decisions (hard +-6, no roll search, peak gate 30 m).
        # _v1: E5cj's level_landonly_wb12_hard6 as it ran (legacy camera,
        # one refinement pass, the pitch prior where the levelling failed)
        base = dict(compass='real', hw='6', level=True, land=True, landonly=True, wb=True,
                    crest='9', beta='free', reqlevel=True)
        if 'pipeline_v1' in c:
            base.update(cam='legacy', ri='1')
            del base['reqlevel']
        elif 'pipeline' in c or 'pipeline_v3' in c:
            base.update(csig='5', rw='1', peak='100')
            if 'pipeline' in c:
                base.update(rms='7', par='0.03')
        c = dict(base, **{k: v for k, v in c.items() if k not in names})
    return c


def slug(cond):
    return cond.replace('/', '__').replace(':', '=') or 'oracle'


def mpd(lat):
    import skyline as S
    return S.meters_per_degree(lat)


def offset_of(r, radius):
    th = math.radians((zlib.crc32(r['tag'].encode()) % 3600) / 10.0)
    return radius * math.cos(th), radius * math.sin(th)


def build(r, cond, set_):
    c = parse(cond)
    obs_v = c.get('obs', 'w' if set_.startswith('perfect') else ('exact' if set_ == 'google' else ''))
    mlat, mlon = mpd(r['lat'])
    on, oe = offset_of(r, float(c.get('off', OFF)))
    heading = r['heading'] + float(c.get('compass', 0) if c.get('compass') != 'real' else 0)
    if c.get('compass') == 'real':
        heading = r['compass']
    fov = r['fov'] * (1 + float(c.get('fov', 0)) / 100)
    z = r['z'] + float(c.get('z', 0))
    flags = ['--extractor', 'ewasr', '--dem', DEM, '--max-peak-dh', c.get('peak', '30'), '--box', str(BOX),
             '--center', '%.7f,%.7f' % (r['lat'] + on / mlat, r['lon'] + oe / mlon),
             '--z', '%.3f' % z, '--fov', '%.4f' % fov, '--heading', '%.4f' % heading,
             '--heading-window', c.get('hw', '0'),
             '--boundary-npy', r['obs'][obs_v],
             '--camera-model', 'legacy' if c.get('cam') == 'legacy' else 'exact',
             '--crest-dh', c.get('crest', str(r['crest']))]
    k = c.get('k', r['k'])
    if k is not None:
        flags += ['--refraction-k', str(k)]
    if 'csig' in c:
        # csig:acc = the photo's own reported compass accuracy (10 deg without one)
        sig = (r.get('acc') or 10.0) if c['csig'] == 'acc' else float(c['csig'])
        flags += ['--compass-sigma', str(sig)]
    if 'rw' in c:
        flags += ['--roll-window', c['rw'], '--roll-step', c.get('rs', '0.25')]
    if 'jack' in c:
        flags += ['--conditioning']
    if 'par' in c:
        flags += ['--min-parallax', c['par']]
    if 'rms' in c:
        flags += ['--max-rms', c['rms']]
    # the study searches with iterated refinement (one pass strands a fix
    # down the range valley even in the perfect world); 'pipeline' keeps E5cj's
    flags += ['--refine-iter', c.get('ri', '10')]
    if 'level' in c:
        sea = r['sea'].get(obs_v)
        if sea and os.path.exists(sea):
            flags += ['--level-npy', sea]
        if 'reqlevel' in c:
            flags += ['--require-level']
    else:
        # lpitch / lroll: that angle from the levelling, the other true
        lev = r.get('lev') or {}
        pitch = lev['pitch'] if 'lpitch' in c and lev else r['pitch']
        roll = lev['roll'] if 'lroll' in c and lev else r['roll']
        flags += ['--pitch', '%.4f' % (pitch + float(c.get('pitch', 0))),
                  '--roll', '%.4f' % (roll + float(c.get('roll', 0)))]
    beta = c.get('beta', 'free' if 'level' in c else '0')
    if 'band' in c:
        flags += ['--beta-band', c['band']]
    elif beta != 'free':
        flags += ['--beta-band', '%s,%s' % (-float(beta), float(beta))]
    if 'land' in c:
        flags += ['--land-observer', str(LAND_H)]
    if 'landonly' in c:
        flags += ['--land-only', str(LAND_R)]
    if 'wb' in c:
        flags += ['--waterline-band', '-12.5,5']
    return [PY, os.path.join(HERE, 'skyfix.py'), r['image']] + flags


def run_one(set_, cond, r):
    d = os.path.join(OUT, 'runs', set_, slug(cond))
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, r['tag'])
    if os.path.exists(out + '.json'):
        return out
    c = parse(cond)
    sea = r['sea'].get(c.get('obs', 'w' if set_.startswith('perfect') else ('exact' if set_ == 'google' else '')))
    if 'reqlevel' in c and 'level' in c and not (sea and os.path.exists(sea)):
        # no sea line at all: the photo is not solved
        json.dump(dict(status='inconclusive', fix_ok=False,
                       reasons=['no usable sea line: none from the segmenter']), open(out + '.json', 'w'))
        return out
    cmd = build(r, cond, set_) + ['--out', out]
    with open(out + '.log', 'w') as lg:
        lg.write(' '.join(cmd) + '\n'); lg.flush()
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE,
                           env=dict(os.environ, OMP_NUM_THREADS='1'))
        lg.write(p.stdout)
    if not os.path.exists(out + '.json'):
        # an early refusal prints its verdict instead of writing it
        try:
            txt = p.stdout[p.stdout.index('{'):]
            json.dump(json.loads(txt[:txt.rindex('}') + 1]), open(out + '.json', 'w'))
        except ValueError:
            pass
    return out


def cmd_solve(a):
    recs = records(a.set)
    jobs = [(c, r) for c in a.conds for r in recs]
    with ThreadPoolExecutor(a.jobs) as ex:
        for n, _ in enumerate(ex.map(lambda j: run_one(a.set, *j), jobs), 1):
            if n % 50 == 0 or n == len(jobs):
                print('%d/%d' % (n, len(jobs)), flush=True)


# ---------------------------------------------------------------- report

def outcome(r, j, cond):
    """Error of the fix from the truth (m), split along and across the true
    line of sight."""
    if not j or j.get('lat') is None or j.get('dn_m') is None:
        return None
    mlat, mlon = mpd(r['lat'])
    dn, de = (j['lat'] - r['lat']) * mlat, (j['lon'] - r['lon']) * mlon
    h = math.radians(r['heading'])
    return dict(err=math.hypot(dn, de), along=dn * math.cos(h) + de * math.sin(h),
                cross=-dn * math.sin(h) + de * math.cos(h), ok=j['status'] == 'ok',
                margin=j.get('basin_margin'), rms=j.get('rms_mrad'),
                hoff=(j.get('photos') or [{}])[0].get('heading_offset_deg'),
                reasons=j.get('reasons') or [])


def load(set_, cond):
    out = {}
    for r in records(set_):
        f = os.path.join(OUT, 'runs', set_, slug(cond), r['tag'] + '.json')
        out[r['tag']] = outcome(r, json.load(open(f)) if os.path.exists(f) else None, cond)
    return out


def summary(res):
    v = [x for x in res.values() if x]
    if not v:
        return None
    e = np.array([x['err'] for x in v])
    acc = [x for x in v if x['ok']]
    return dict(n=len(v), med=float(np.median(e)), p75=float(np.percentile(e, 75)),
                p90=float(np.percentile(e, 90)), lt100=int((e < 100).sum()), lt300=int((e < 300).sum()),
                gross=int((e > 1000).sum()), acc=len(acc), acc_good=sum(x['err'] < 300 for x in acc),
                acc_wrong=sum(x['err'] > 1000 for x in acc),
                along=float(np.median([abs(x['along']) for x in v])),
                cross=float(np.median([abs(x['cross']) for x in v])))


def cmd_report(a):
    rows = []
    print('%-34s %3s %7s %7s %7s %5s %5s %5s | %4s %4s %5s | %7s %7s' % (
        'condition', 'n', 'median', 'p75', 'p90', '<100', '<300', '>1km', 'acc', 'good', 'wrong',
        '|along|', '|cross|'))
    for c in a.conds:
        s = summary(load(a.set, c))
        if not s:
            print('%-34s  (no runs)' % c); continue
        rows.append(dict(cond=c, **s))
        print('%-34s %3d %7.0f %7.0f %7.0f %5d %5d %5d | %4d %4d %5d | %7.0f %7.0f' % (
            c, s['n'], s['med'], s['p75'], s['p90'], s['lt100'], s['lt300'], s['gross'],
            s['acc'], s['acc_good'], s['acc_wrong'], s['along'], s['cross']))
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, 'report_%s.json' % a.set)
    allr = json.load(open(f)) if os.path.exists(f) else {}
    allr.update({r['cond']: r for r in rows})
    json.dump(allr, open(f, 'w'), indent=1)


GATES = (('ambiguous landscape', 'margin'), ('minimum on the search-box', 'edge'),
         ('insufficient skyline relief', 'relief'), ('residual', 'rms'),
         ('subject too near', 'range'), ('DEM families', 'dem2'),
         ('covariance degenerate', 'cov'), ('peak heights', 'peaks'),
         ('conditioning', 'jack'), ('no terrain silhouette', 'nosil'),
         ('no usable sea line', 'nosea'), ('bearing only', 'parallax'))


def gate_of(reason):
    return next((g for pre, g in GATES if reason.startswith(pre)), 'other')


def cmd_gates(a):
    """The constraints against the truth: per gate, how many fixes it
    refused that were good (< 300 m), middling, or wrong (> 1 km); and
    how many wrong fixes no gate stopped."""
    cls = lambda e: 'good' if e < 300 else ('wrong' if e > 1000 else 'mid')
    tab, acc = {}, {'good': 0, 'mid': 0, 'wrong': 0}
    n = 0
    for c in a.conds:
        for x in load(a.set, c).values():
            if not x:
                continue
            n += 1
            k = cls(x['err'])
            if x['ok']:
                acc[k] += 1
            for g in sorted({gate_of(r) for r in x['reasons']}):
                tab.setdefault(g, {'good': 0, 'mid': 0, 'wrong': 0})[k] += 1
    print('%d fixes over %d conditions' % (n, len(a.conds)))
    print('%-8s %6s %6s %6s   (fixes each gate refused, by true error)' % ('gate', 'good', 'mid', 'wrong'))
    for g, d in sorted(tab.items(), key=lambda kv: -sum(kv[1].values())):
        print('%-8s %6d %6d %6d' % (g, d['good'], d['mid'], d['wrong']))
    print('%-8s %6d %6d %6d   (accepted)' % ('ACCEPT', acc['good'], acc['mid'], acc['wrong']))


FAMILIES = (('compass', 'Pusula hatası, arama penceresi 0', '°', (0.2, 0.5, 1, 2, 5)),
            ('pitch', 'Eğim (pitch) hatası, β sabit', '°', (0.05, 0.1, 0.2, 0.5, 1)),
            ('roll', 'Yatış (roll) hatası', '°', (0.1, 0.25, 0.5, 1, 2)),
            ('fov', 'Görüş açısı (FOV) hatası', '%', (0.5, 1, 2, 5)),
            ('z', 'Kamera yüksekliği hatası', 'm', (1, 2, 5, 10, 20)),
            ('obs:noise', 'Skyline gürültüsü (σ)', 'px', (0.5, 1, 2, 4)))


def cmd_plot(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, ink2, grid = '#0b0b0b', '#52514e', '#e4e3df'
    c_med, c_p90 = '#2a78d6', '#eb6834'
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.2), facecolor='#fcfcfb')
    base = summary(load(a.set, 'oracle'))
    for ax, (key, title, unit, mags) in zip(axs.ravel(), FAMILIES):
        xs, med, p90 = [0.0], [base['med']], [base['p90']]
        for m in mags:
            c = ('%s%g' % (key, m)) if key.startswith('obs:') else ('%s:%g' % (key, m))
            s = summary(load(a.set, c))
            if s:
                xs.append(m); med.append(s['med']); p90.append(s['p90'])
        ax.set_facecolor('#fcfcfb')
        for y, col, lab in ((med, c_med, 'medyan'), (p90, c_p90, '%90')):
            ax.plot(xs, y, color=col, lw=2, solid_capstyle='round', marker='o', ms=8,
                    mec='#fcfcfb', mew=2, label=lab, zorder=3)
            ax.annotate('%.0f m' % y[-1], (xs[-1], y[-1]), xytext=(6, 0), textcoords='offset points',
                        va='center', fontsize=9, color=ink2)
        ax.set_yscale('log')
        ax.set_ylim(3, 20000)
        ax.set_title(title, fontsize=11, color=ink, loc='left')
        ax.set_xlabel('eklenen hata (%s)' % unit, fontsize=9, color=ink2)
        ax.grid(True, color=grid, lw=1)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=ink2, labelsize=9, length=0)
        ax.set_xlim(-0.05 * xs[-1], xs[-1] * 1.25)
    axs[0, 0].set_ylabel('konum hatası (m)', fontsize=9, color=ink2)
    axs[1, 0].set_ylabel('konum hatası (m)', fontsize=9, color=ink2)
    axs[0, 0].legend(frameon=False, fontsize=9, loc='upper left')
    fig.suptitle('Tek değişkenli hata büyümesi, mükemmel dünya (%d kare): diğer her şey gerçek değerinde'
                 % base['n'], fontsize=12, color=ink, x=0.01, ha='left')
    fig.tight_layout()
    f = os.path.join(OUT, 'sensitivity_%s.png' % a.set)
    fig.savefig(f, dpi=110, facecolor='#fcfcfb')
    print('wrote', f)


def _arg(cmd, flag, value):
    cmd = list(cmd)
    if flag in cmd:
        cmd[cmd.index(flag) + 1] = value
    else:
        cmd += [flag, value]
    return cmd


def _photo_line(log):
    import re
    m = re.search(r'pitch ([-+\d.]+), roll ([-+\d.]+)', open(log).read()) if os.path.exists(log) else None
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def cmd_diag(a):
    """Why a fix went wrong: the same solve's cost at the truth (box 0, one
    refinement pass, so it stays there) against the cost at the fix. Truth
    cheaper: the search missed it. Truth dearer: the model or the
    observation prefers the wrong place."""
    recs = {r['tag']: r for r in records(a.set)}
    tf = os.path.join(OUT, 'theodolite_meta.json')
    theo = json.load(open(tf)) if os.path.exists(tf) else {}
    tr = json.load(open(os.path.join(OUT, 'truth_real.json'))) if a.set == 'real' else {}
    d = os.path.join(OUT, 'diag', a.set, slug(a.cond))
    os.makedirs(d, exist_ok=True)
    for tag in a.tags:
        r = recs[tag]
        base = os.path.join(OUT, 'runs', a.set, slug(a.cond), tag)
        j = json.load(open(base + '.json'))
        o = outcome(r, j, a.cond)
        cmd = build(r, a.cond, a.set)
        cmd = _arg(_arg(_arg(cmd, '--center', '%.7f,%.7f' % (r['lat'], r['lon'])), '--box', '0'),
                   '--refine-iter', '1')
        out = os.path.join(d, tag + '_attruth')
        if not os.path.exists(out + '.json'):
            with open(out + '.log', 'w') as lg:
                subprocess.run(cmd + ['--out', out], stdout=lg, stderr=subprocess.STDOUT, cwd=HERE,
                               env=dict(os.environ, OMP_NUM_THREADS='1'))
        t = json.load(open(out + '.json'))
        pf, rf = _photo_line(base + '.log')
        ph = (j.get('photos') or [{}])[0]
        pt = (t.get('photos') or [{}])[0]
        print('== %s  (%s)' % (tag, a.cond))
        if o:
            print('fix   %5.0f m from truth (along %+5.0f, across %+5.0f)  %s  margin %.2f  rms %.2f mrad  cost %.3g  '
                  'heading %+5.1f  beta %+5.1f mrad  roll %+4.2f  peaks %s' % (
                      o['err'], o['along'], o['cross'], j['status'], j['basin_margin'], j['rms_mrad'], j['cost'],
                      ph.get('heading_offset_deg', 0), ph.get('el_offset_mrad', 0), ph.get('roll_offset_deg', 0),
                      '%+.0f/%.0f m' % (j['peak_dh_m'], j['peak_dh_mad_m']) if j.get('peak_dh_m') is not None else '-'))
        print('truth            rms %.2f mrad  cost %.3g  heading %+5.1f  beta %+5.1f mrad  roll %+4.2f  -> %s' % (
            t['rms_mrad'], t['cost'], pt.get('heading_offset_deg', 0), pt.get('el_offset_mrad', 0),
            pt.get('roll_offset_deg', 0),
            'SEARCH missed a cheaper truth' if t['cost'] < j.get('cost', np.inf) else 'MODEL prefers the fix'))
        tt = tr.get(tag, {})
        th = theo.get(tag, {})
        print('attitude used: pitch %s roll %s | truth fit: pitch %+.2f roll %+.2f, compass off %+.1f | IMU: %s' % (
            '%+.2f' % pf if pf is not None else '-', '%+.2f' % rf if rf is not None else '-',
            tt.get('pitch', np.nan), tt.get('roll', np.nan), tt.get('compass_error_deg', np.nan),
            'pitch %+.2f roll %+.2f' % (th['imu_pitch_deg'], th['imu_roll_deg']) if th else '-'))
        if j.get('reasons'):
            print('reasons: ' + '; '.join(x[:70] for x in j['reasons']))
        print('profiles: %s.png  %s.png' % (base, out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=COND_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    for name in ('truth', 'world', 'synth'):
        s = sp.add_parser(name)
        s.add_argument('--jobs', type=int, default=14)
    s = sp.add_parser('_truth1')
    s.add_argument('tag')
    s.add_argument('--span', type=float, default=30.0)
    s = sp.add_parser('diag')
    s.add_argument('set', choices=['real', 'perfect_camp', 'perfect_sea', 'google'])
    s.add_argument('cond')
    s.add_argument('tags', nargs='+')
    s = sp.add_parser('plot')
    s.add_argument('set', choices=['real', 'perfect_camp', 'perfect_sea', 'google'])
    for name in ('solve', 'report', 'gates'):
        s = sp.add_parser(name)
        s.add_argument('set', choices=['real', 'perfect_camp', 'perfect_sea', 'google'])
        s.add_argument('conds', nargs='+')
        s.add_argument('--jobs', type=int, default=14)
    a = ap.parse_args()
    if a.cmd == '_truth1':
        fr = [f for f in campaign() if f['tag'] == a.tag][0]
        print(json.dumps(fit_truth(fr, a.span)))
        return
    {'truth': cmd_truth, 'world': cmd_world, 'synth': cmd_synth, 'solve': cmd_solve,
     'report': cmd_report, 'gates': cmd_gates, 'plot': cmd_plot, 'diag': cmd_diag}[a.cmd](a)


if __name__ == '__main__':
    main()
