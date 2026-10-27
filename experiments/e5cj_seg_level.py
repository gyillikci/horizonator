#!/usr/bin/env python3
"""E5cj: SegFormer lines after post-processing, and the sea line as the level.

Per frame, from the SegFormer (ADE20K) sky/water/land map:
  skyline   e5cf_faint_ridge.ewasr_boundary (the E5ch 'seg' curve), with
            the outer 1% of columns zero-weighted on each side: the
            segmenter bends at the frame edge. The solver's own screens
            (terrain_plausible, the spike filter) still run on top.
  sea line  first water row at or below the skyline, per column; kept
            only where it lies within TOL_PX of a robust straight-line
            fit. A far shoreline or the sea horizon projects to a
            straight line in a pinhole frame (E5au: eWaSR's top of water
            fits one to 1.2 px rms), so masts, hulls and the near
            harbour fall off it.

Arms, each with the campaign flags (hard +-6) and with --compass-sigma 5:
  post      the post-processed skyline
  level     the same, plus --level-npy: the sea line pins pitch and roll
            (classed waterline, +-5 mrad elevation band)
  pipeline  the campaign pipeline decided after E5ck (2026-10-25): all of
            campaign_frames.json's pipeline_flags (land observer, land
            only, waterline band, soft compass 5 deg, roll searched +-1 deg,
            peak gate 100 m, residual gate 7 mrad, parallax gate 0.03 per
            km) with the sea line; a photo whose sea line is
            missing or refused by the levelling is not solved. The level
            arms refuse those photos too (--require-level)

    python3 e5cj_seg_level.py prep [TAG ... | all]
    python3 e5cj_seg_level.py solve [TAG ... | all] [--jobs 8]
    python3 e5cj_seg_level.py report [TAG ... | all]
    python3 e5cj_seg_level.py replay [TAG ... | all | pf]   (level_soft5, for pf_debug)
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from assets import celnav  # noqa: E402
import e5ch_segformer as C  # noqa: E402
import e5cf_faint_ridge as M  # noqa: E402

OUT = os.path.join(HERE, 'out', 'seg_level')
TOL_PX = 3.0
EDGE_FRAC = 0.01
MIN_SEA_COLS = 50
LAND_H = 2.0      # EXIF altitude - COP30 ground at the 44 truths: median +2.0 m
LAND_R = 60.0     # every truth has land within 60 m (5 sample as sea)
WB = {'_wb12': '-12.5,5', '_wb20': '-20,5'}   # --waterline-band, mrad
ARMS = ('post_hard6', 'post_soft5', 'level_hard6', 'level_soft5',
        'level_land_hard6', 'level_land_soft5',
        'level_landonly_hard6', 'level_landonly_soft5',
        'level_landonly_wb12_hard6', 'level_landonly_wb12_soft5',
        'level_landonly_wb20_hard6', 'level_landonly_wb20_soft5',
        'pipeline')


def raw_lines(cls):
    """Skyline (E5ch rule) and the first water row at or below it."""
    sky = M.ewasr_boundary(cls)
    W = cls.shape[1]
    sea = np.full(W, np.nan)
    for x in range(W):
        top = int(sky[x]) if np.isfinite(sky[x]) else 0
        w = np.flatnonzero(cls[top:, x] == 1)
        if w.size:
            sea[x] = top + w[0]
    return sky, sea


def clean_sea_line(sea, tol=TOL_PX, iters=20):
    """Keep the columns within tol px of a robust straight-line fit."""
    x = np.arange(sea.size)
    ok = np.isfinite(sea)
    if ok.sum() < MIN_SEA_COLS:
        return np.full(sea.size, np.nan), None
    for _ in range(iters):
        a, b = np.polyfit(x[ok], sea[ok], 1)
        res = sea - (a * x + b)
        s = 1.4826 * np.nanmedian(np.abs(res[ok]))
        new = np.isfinite(sea) & (np.abs(res) <= max(2.5 * s, tol))
        if new.sum() < MIN_SEA_COLS or (new == ok).all():
            break
        ok = new
    keep = np.isfinite(sea) & (np.abs(sea - (a * x + b)) <= tol)
    if keep.sum() < MIN_SEA_COLS:
        return np.full(sea.size, np.nan), None
    r = (sea - (a * x + b))[keep]
    return np.where(keep, sea, np.nan), dict(
        slope=float(a), icpt=float(b), kept=float(keep.mean()),
        rms_px=float(np.sqrt(np.mean(r ** 2))))


def prep(a):
    import extract
    from segformer_bridge import SegFormer
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    sf = SegFormer()
    info = {}
    for fr in C.frames(a.tags):
        img = extract.load_image(celnav(fr['path']))
        H, W = img.shape[:2]
        cls = sf.predict(img)
        sky, sea_raw = raw_lines(cls)
        conf = np.isfinite(sky) * 1.0
        e = max(1, round(EDGE_FRAC * W))
        conf[:e] = conf[-e:] = 0.0
        np.savez(os.path.join(OUT, fr['tag'] + '_segpost.npz'),
                 rows=np.nan_to_num(sky), conf=conf)
        sea, fit = clean_sea_line(sea_raw)
        if fit:
            np.savez(os.path.join(OUT, fr['tag'] + '_sealine.npz'), rows=sea)
        info[fr['tag']] = dict(sky_cols=float(conf.mean()), sea=fit)

        fig, ax = plt.subplots(figsize=(12, 12 * H / W))
        ax.imshow(img)
        x = np.arange(W)
        ax.plot(x, sea_raw, color='#00e5ff', lw=0.8, alpha=0.35)
        ax.plot(x, np.where(conf > 0, sky, np.nan), color='#ff2d2d', lw=1.3)
        ax.plot(x, sea, color='#00e5ff', lw=1.3)
        ax.set_title('%s  skyline %.0f%% cols, sea line %s' % (
            fr['tag'], 100 * conf.mean(),
            '%.0f%% cols, rms %.1f px' % (100 * fit['kept'], fit['rms_px'])
            if fit else 'none'))
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fr['tag'] + '_lines.png'), dpi=80)
        plt.close(fig)
        print('%-26s skyline %3.0f%%  sea line %s' % (
            fr['tag'], 100 * conf.mean(),
            '%3.0f%% rms %.1f px tilt %+.2f deg' % (
                100 * fit['kept'], fit['rms_px'],
                math.degrees(math.atan(fit['slope']))) if fit else 'none'),
            flush=True)
    f = os.path.join(OUT, 'prep.json')
    allr = json.load(open(f)) if os.path.exists(f) else {}
    allr.update(info)
    json.dump(allr, open(f, 'w'), indent=1)


def replay_dir(arm):
    return 'replay' if arm == 'level_soft5' else 'replay_' + arm


def solve_one(fr, arm, flags, view=False):
    # 'pipeline': the campaign pipeline decided after E5ck, all of its
    # options in campaign_frames.json's pipeline_flags
    if arm == 'pipeline':
        flags = json.load(open(C.FRAMES))['pipeline_flags'].replace(
            '~', os.path.expanduser('~'))
    hard = arm.endswith('hard6')
    out = os.path.join(OUT, '%s_%s' % (fr['tag'], arm))
    if view:
        # the same solve again, writing the pf_debug replay file
        out = os.path.join(OUT, replay_dir(arm), fr['tag'])
        flags = flags + ' --view-json ' + out + '.view.json'
    if os.path.exists(out + '.json'):
        return out
    cmd = [sys.executable, os.path.join(HERE, 'skyfix.py'),
           celnav(fr['path']),
           '--center', '%.6f,%.6f' % (fr['lat'], fr['lon']),
           '--box', str(fr['box']), '--z', str(fr['z']),
           '--fov', str(fr['fov']), '--heading', str(fr['heading']),
           '--boundary-npy', os.path.join(OUT, fr['tag'] + '_segpost.npz'),
           '--out', out] + flags.split()
    sl = os.path.join(OUT, fr['tag'] + '_sealine.npz')
    if arm.startswith('level') or arm == 'pipeline':
        # a photo whose sea line is missing, or not a straight horizon the
        # levelling accepts, is not solved: its pitch is unknown, and the
        # campaign's frames are pitched up to 9 deg (E5ck, 2026-10-25)
        if not os.path.exists(sl):
            json.dump(dict(status='inconclusive', fix_ok=False,
                           reasons=['no usable sea line: none from the segmenter']),
                      open(out + '.json', 'w'))
            return out
        cmd += ['--level-npy', sl, '--require-level']
    if arm == 'pipeline':
        return run_solve(cmd, out)
    # every shot taken from land: the camera stands LAND_H above the ground
    # at each candidate; _landonly also keeps the search off open water
    if '_land' in arm:
        cmd += ['--land-observer', str(LAND_H)]
    if '_landonly' in arm:
        cmd += ['--land-only', str(LAND_R)]
    # one-sided elevation-offset band for the waterline-levelled frames
    for tag_, band in WB.items():
        if tag_ in arm:
            cmd += ['--waterline-band', band]
    if not hard:
        cmd += ['--compass-sigma', '5']
    return run_solve(cmd, out)


def run_solve(cmd, out):
    with open(out + '.log', 'w') as lg:
        subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=HERE,
                       env=dict(os.environ, OMP_NUM_THREADS='1'))
    return out


def solve(a):
    flags = json.load(open(C.FRAMES))['flags'].replace(
        '~', os.path.expanduser('~'))
    arms = a.arms.split(',') if a.arms else ARMS
    jobs = [(fr, arm) for arm in arms
            for fr in C.frames(a.tags)
            if os.path.exists(os.path.join(OUT, fr['tag'] + '_segpost.npz'))]
    with ThreadPoolExecutor(a.jobs) as ex:
        for n, out in enumerate(ex.map(lambda j: solve_one(*j, flags), jobs),
                                1):
            print('%3d/%d %s %s' % (n, len(jobs), os.path.basename(out),
                                    '' if os.path.exists(out + '.json')
                                    else 'NO RESULT'), flush=True)


def replay(a):
    """pf_debug replay files for the level_soft5 arm; 'pf' = every PF_ frame.

    python3 pf_debug/serve.py out/seg_level/replay/*.view.json"""
    flags = json.load(open(C.FRAMES))['flags'].replace(
        '~', os.path.expanduser('~'))
    arm = a.arms or 'level_soft5'
    os.makedirs(os.path.join(OUT, replay_dir(arm)), exist_ok=True)
    tags = a.tags
    if tags == ['pf']:
        tags = [f['tag'] for f in C.frames(['all']) if '_PF_' in f['tag']]
    frs = [fr for fr in C.frames(tags)
           if os.path.exists(os.path.join(OUT, fr['tag'] + '_segpost.npz'))]
    with ThreadPoolExecutor(a.jobs) as ex:
        for n, out in enumerate(ex.map(
                lambda fr: solve_one(fr, arm, flags, view=True),
                frs), 1):
            print('%3d/%d %s %s' % (n, len(frs), os.path.basename(out),
                                    '' if os.path.exists(out + '.view.json')
                                    else 'NO VIEW'), flush=True)


def levelled(tag):
    """pitch, roll and inlier count the solver took from the sea line."""
    lg = os.path.join(OUT, tag + '_level_soft5.log')
    if not os.path.exists(lg):
        return None
    m = re.search(r'pitch ([-+\d.]+), roll ([-+\d.]+), .*auto-level \((\d+) '
                  r'sea cols\)', open(lg).read())
    return (float(m.group(1)), float(m.group(2)), int(m.group(3))) if m \
        else None


def report(a):
    arms = {'ewasr_hard6': ('baseline', '%s_huber.json'),
            'ewasr_soft5': ('compass', '%s_soft5.json'),
            'seg_hard6': ('segformer', '%s_seg_hard6.json'),
            'seg_soft5': ('segformer', '%s_seg_soft5.json'),
            'post_hard6': ('seg_level', '%s_post_hard6.json'),
            'post_soft5': ('seg_level', '%s_post_soft5.json'),
            'level_hard6': ('seg_level', '%s_level_hard6.json'),
            'level_soft5': ('seg_level', '%s_level_soft5.json'),
            'land_hard6': ('seg_level', '%s_level_land_hard6.json'),
            'land_soft5': ('seg_level', '%s_level_land_soft5.json'),
            'lonly_hard6': ('seg_level', '%s_level_landonly_hard6.json'),
            'lonly_soft5': ('seg_level', '%s_level_landonly_soft5.json'),
            'wb12_hard6': ('seg_level', '%s_level_landonly_wb12_hard6.json'),
            'wb12_soft5': ('seg_level', '%s_level_landonly_wb12_soft5.json'),
            'wb20_hard6': ('seg_level', '%s_level_landonly_wb20_hard6.json'),
            'wb20_soft5': ('seg_level', '%s_level_landonly_wb20_soft5.json'),
            'pipeline': ('seg_level', '%s_pipeline.json')}
    rows, missing, kept_out = [], [], []
    for fr in C.frames(a.tags):
        r = dict(tag=fr['tag'])
        for k, (d, p) in arms.items():
            r[k] = C.load_one(os.path.join(HERE, 'out', d, p % fr['tag']))
        pj = os.path.join(OUT, fr['tag'] + '_pipeline.json')
        if os.path.exists(pj) and any('no usable sea line' in x for x in
                                      json.load(open(pj)).get('reasons', [])):
            kept_out.append(fr['tag'])       # the pipeline does not solve it
        elif all(r[k] for k in arms):
            rows.append(r)
        else:
            missing.append(fr['tag'])
    print('frames with every arm: %d%s' % (
        len(rows), ' (missing: %s)' % ', '.join(missing) if missing else ''))
    if kept_out:
        print('not solved, no usable sea line: %d (%s)' % (
            len(kept_out), ', '.join(t[3:] for t in kept_out)))
    print('levelled from the sea line: %d of %d' % (
        sum(levelled(r['tag']) is not None for r in rows), len(rows)))
    print('%-12s %7s %5s %5s | %4s %4s %5s' % (
        'arm', 'median', '<300', '<500', 'acc', 'good', 'wrong'))
    summ = {}
    for k in arms:
        e = np.array([r[k]['err'] for r in rows])
        # the soft-prior arms' heading-offset gate; the pipeline's own
        # acceptance is skyfix's verdict alone (E5ck measured it so)
        gate = 10.0 if 'soft5' in k else 1e9
        acc = [r for r in rows if r[k]['ok'] and abs(r[k]['hoff']) <= gate]
        summ[k] = dict(median=float(np.median(e)),
                       within300=int((e < C.GOOD).sum()),
                       within500=int((e < C.BAD).sum()), accepted=len(acc),
                       good=sum(r[k]['err'] < C.GOOD for r in acc),
                       wrong=sum(r[k]['err'] >= C.BAD for r in acc),
                       wrong_tags=[r['tag'] for r in acc
                                   if r[k]['err'] >= C.BAD])
        print('%-12s %6.0fm %5d %5d | %4d %4d %5d  %s' % (
            k, summ[k]['median'], summ[k]['within300'],
            summ[k]['within500'], summ[k]['accepted'], summ[k]['good'],
            summ[k]['wrong'],
            ', '.join(t[3:] for t in summ[k]['wrong_tags'])))
    print('\nper frame (err m, A = accepted; level = pitch, roll, sea cols):')
    print('  %-24s ' % 'tag' + ' | '.join('%6s' % k[:6] for k in arms))
    for r in rows:
        lv = levelled(r['tag'])
        print('  %-24s ' % r['tag'][:24] + ' | '.join(
            '%5.0f%s' % (r[k]['err'], 'A' if r[k]['ok'] else '-')
            for k in arms) + ('   level %+.2f %+.2f %d' % lv if lv else ''))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(
        OUT, 'report_%d.json' % len(rows)), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    for n in ('prep', 'solve', 'report', 'replay'):
        p = sp.add_parser(n)
        p.add_argument('tags', nargs='*')
        p.add_argument('--jobs', type=int, default=8)
        p.add_argument('--arms', help='comma list (solve: which arms; '
                                      'replay: one arm, default level_soft5)')
    a = ap.parse_args()
    dict(prep=prep, solve=solve, report=report, replay=replay)[a.cmd](a)
