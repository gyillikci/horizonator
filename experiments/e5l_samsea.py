#!/usr/bin/env python3
"""E5l: MobileSAM as a sea-horizon finder, scored against annotation.

Prompt SAM with a handful of points low in the frame (open water from
a boat), take the water mask's upper boundary as the sea line, fit
the same line model E4q uses, and score it on the SAME ground truth
as the other detectors: MaSTr1325's hand-annotated water/sky boundary
over the open-horizon subset. Direct comparison:

    seam   2.9 px median row offset, 17% availability   (E4q)
    radon  2.1 px median row offset, 32% availability   (E4q-2)
    SAM    measured here

Availability here means: SAM returned a water mask whose upper
boundary spans most of the frame and passes the same acceptance the
others face (a line, not a blob edge).

Run:  python3 e5l_samsea.py [n]      (writes out/e5l_results.json)
      python3 e5l_samsea.py --photo ID   (visual on a field frame)
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = ('/tmp/claude-0/-home-user/'
           '792503f9-74c5-5111-83ca-eeeda63e838d/scratchpad')
sys.path.insert(0, os.path.join(SCRATCH, 'MobileSAM'))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                    'celestial-navigation', 'MaSTr1325')
sys.path.insert(0, HERE)
from e4q_imu_horizon import seg_line, open_horizon_frac, fit_line

_PRED = None


def predictor():
    global _PRED
    if _PRED is None:
        from mobile_sam import sam_model_registry, SamPredictor
        sam = sam_model_registry['vit_t'](checkpoint=os.path.join(
            SCRATCH, 'MobileSAM', 'weights', 'mobile_sam.pt'))
        sam.to('cpu').eval()
        _PRED = SamPredictor(sam)
    return _PRED


def sam_sea_line(rgb_u8, n_pts=5):
    """Water mask from low-frame prompts; upper boundary; line fit.
    Returns (r0, slope, frac_cols, mask) or None."""
    H, W, _ = rgb_u8.shape
    pred = predictor()
    pred.set_image(rgb_u8)
    xs = np.linspace(0.15 * W, 0.85 * W, n_pts)
    pts = np.stack([xs, np.full(n_pts, 0.90 * H)], axis=1)
    masks, scores, _ = pred.predict(point_coords=pts,
                                    point_labels=np.ones(n_pts),
                                    multimask_output=True)
    best = None
    for m in masks[np.argsort(scores)[::-1]]:
        cover = m.mean()
        if 0.05 < cover < 0.85:          # not a sliver, not the frame
            best = m
            break
    if best is None:
        return None
    top = np.where(best.any(0), best.argmax(0), -1)
    cols = np.where(top > 0)[0]
    if cols.size < 0.6 * W:
        return None
    rows = top[cols].astype(float)
    # the same robust straight-line acceptance the others face
    r0, sl, rms = fit_line(cols, rows, W)
    inl = np.abs(rows - (r0 + sl * (cols - (W - 1) / 2))) <= 4.0
    if inl.mean() < 0.7:
        return None
    r0, sl, rms = fit_line(cols[inl], rows[inl], W)
    return r0, sl, float(cols[inl].size / W), best


def sam_attitude(img01, f_px, dip_rad):
    """skyfix-compatible attitude from the SAM sea line: same output
    dict as extract.sea_horizon_attitude_radon. img01: float RGB 0-1."""
    H, W, _ = img01.shape
    got = sam_sea_line((img01 * 255).astype(np.uint8))
    if got is None:
        return None
    r0, sl, frac, _ = got
    v_c = (H - 1) / 2.0 - r0
    pitch = float(np.degrees(-dip_rad - np.arctan2(v_c, f_px)))
    roll = float(np.degrees(-np.arctan(sl)))
    return dict(pitch_deg=pitch, roll_deg=roll, n_inl=int(frac * W),
                frac=frac, span_frac=frac, rms_px=0.0, contrast=0.0,
                score=1.0, source='mobilesam')


def benchmark(n=None):
    imgs = sorted(glob.glob(os.path.join(DATA, '*.jpg')))
    rows_out, n_open = [], 0
    for ip in imgs:
        stem = os.path.basename(ip)[:-4]
        segp = os.path.join(DATA, stem + 'm.png')
        if not os.path.exists(segp):
            continue
        seg = np.asarray(Image.open(segp))
        if open_horizon_frac(seg) < 0.5:
            continue
        n_open += 1
        if n and n_open > n:
            break
        truth = seg_line(seg)
        if truth is None:
            continue
        rgb = np.asarray(Image.open(ip).convert('RGB'))
        got = sam_sea_line(rgb)
        if got is None:
            rows_out.append(dict(image=stem, ok=False))
            continue
        r0, sl, frac, _ = got
        rows_out.append(dict(image=stem, ok=True,
                             d_row=float(r0 - truth[0]),
                             d_slope=float(sl - truth[1]), frac=frac))
        if n_open % 20 == 0:
            print(f'  [{n_open}]', flush=True)
    ok = [r for r in rows_out if r['ok']]
    dr = np.array([abs(r['d_row']) for r in ok])
    ds = np.array([abs(r['d_slope']) for r in ok])
    print(f'\nopen-horizon images tried: {len(rows_out)}, SAM line '
          f'accepted on {len(ok)} ({100*len(ok)/max(len(rows_out),1):.0f}%'
          f' availability)')
    if ok:
        print(f'  |row offset|  median {np.median(dr):.1f} px  '
              f'p90 {np.percentile(dr, 90):.1f}')
        print(f'  |slope|error  median {np.degrees(np.arctan(np.median(ds))):.2f} deg')
        print(f'  reference: seam 2.9 px @ 17%,  radon 2.1 px @ 32%')
    with open(os.path.join(HERE, 'out', 'e5l_results.json'), 'w') as f:
        json.dump(rows_out, f, indent=1)


if __name__ == '__main__':
    if '--photo' in sys.argv:
        import extract
        import skyline as S
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        sid = sys.argv[sys.argv.index('--photo') + 1]
        idx = {x['id']: x for x in json.load(open(os.path.join(
            HERE, 'out', 'theodolite', 'index.json')))['sightings']}
        s = idx[sid]; e, a = s['exif'], s['attitude']
        img = extract.load_image(os.path.join(
            os.path.dirname(os.path.dirname(HERE)),
            'celestial-navigation', 'theodolite', s['raw']))
        H, W, _ = img.shape
        got = sam_sea_line((img * 255).astype(np.uint8))
        f_px = (W / 2) / np.tan(np.radians(a['fov_deg']) / 2)
        z = max(e.get('alt_m') or 5.0, 2.0)
        lvl = extract.sea_horizon_attitude_radon(img, f_px,
                                                 S.horizon_dip_rad(z))
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.imshow(img)
        if got:
            r0, sl, frac, mask = got
            u = np.arange(W) - (W - 1) / 2
            ax.plot(np.arange(W), r0 + sl * u, color='#bf5af2', lw=1.6,
                    label=f'MobileSAM sea line ({100*frac:.0f}% cols)')
        if lvl:
            dip = S.horizon_dip_rad(z)
            u = np.arange(W) - (W - 1) / 2
            v = (np.tan(-dip - np.radians(lvl['pitch_deg']))
                 * np.hypot(u, f_px)
                 - np.radians(lvl['roll_deg']) * u)
            ax.plot(np.arange(W), (H - 1) / 2 - v, color='#34c759',
                    lw=1.4, ls='--', label='radon sea horizon')
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='lower left', fontsize=9)
        os.makedirs(os.path.join(HERE, 'out', 'e5l'), exist_ok=True)
        p = os.path.join(HERE, 'out', 'e5l', f'{sid}_sea.png')
        fig.tight_layout(); fig.savefig(p, dpi=110)
        print(p)
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else None
        benchmark(n)
