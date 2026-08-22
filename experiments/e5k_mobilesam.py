#!/usr/bin/env python3
"""E5k: MobileSAM as the foreground extractor.

E5b's weakness was measured, not guessed: its atmospheric-perspective
threshold finds the island in only 12-15% of columns and a 0.1 nudge
of the threshold swings the fix from 559 m to 3250 m. MobileSAM
(Apache-2.0, weights in-repo, CPU-fast) segments by appearance
coherence instead of a hand threshold: prompt it with a few points
inside the near layer and take the mask's upper contour as the
foreground silhouette.

Prompts come from the same physics as before — the darkest, most
saturated pixels above the sea horizon — but now they only need to be
ROUGHLY right: SAM turns a handful of seeds into a coherent region
with a sharp boundary. Stability is tested by re-running with
different prompt seeds, which is the knob that used to break E5b.

Run:  python3 e5k_mobilesam.py ID [ID ...]
      (panels to out/e5k/, contour JSON to out/e5k/<id>.json)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import extract, skyline as S
from e5b_foreground import nearness, foreground_rows

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'e5k')
PHOTOS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      'celestial-navigation', 'theodolite')
SCRATCH = ('/tmp/claude-0/-home-user/'
           '792503f9-74c5-5111-83ca-eeeda63e838d/scratchpad')
sys.path.insert(0, os.path.join(SCRATCH, 'MobileSAM'))


def load_sam():
    import torch
    from mobile_sam import sam_model_registry, SamPredictor
    sam = sam_model_registry['vit_t'](
        checkpoint=os.path.join(SCRATCH, 'MobileSAM', 'weights',
                                'mobile_sam.pt'))
    sam.to('cpu').eval()
    return SamPredictor(sam)


def horizon_rows_of(img, f_px, z, pitch, roll):
    H, W, _ = img.shape
    dip = S.horizon_dip_rad(z)
    u = np.arange(W) - (W - 1) / 2
    v = (np.tan(-dip - np.radians(pitch)) * np.hypot(u, f_px)
         - np.radians(roll) * u)
    return (H - 1) / 2 - v


def sam_foreground(pred, img, hz, rng, thr_rows, n_pts=6):
    """Seed MobileSAM FROM the sparse E5b detections. The hand
    threshold is fragile as a contour (12% of columns) but its hits
    are correct — and a first attempt that prompted from 'dark pixels
    above the horizon' segmented the hazy FAR COAST instead, because
    the island body sits below the drawn horizon line (its waterline
    hides behind the sea horizon, so it rises from that line, and
    attitude error puts much of it below). Sparse-but-right seeds,
    pushed a few pixels into the land, are exactly what SAM needs."""
    H, W, _ = img.shape
    cols = np.where(np.isfinite(thr_rows))[0]
    if cols.size < 4:
        return None, None
    pick = cols[np.linspace(0, cols.size - 1,
                            min(n_pts, cols.size)).astype(int)]
    jitter = rng.integers(6, 18, pick.size)
    pts = np.stack([np.clip(thr_rows[pick] + jitter, 1, H - 2),
                    pick.astype(float)], axis=1)
    pred.set_image((img * 255).astype(np.uint8))
    masks, scores, _ = pred.predict(
        point_coords=pts[:, ::-1].astype(float),
        point_labels=np.ones(pick.size), multimask_output=True)
    m = masks[int(np.argmax(scores))]
    # generous clip: keep the body below the drawn horizon (attitude
    # error and the hidden waterline put it there), drop only deep sea
    for x in range(W):
        m[int(np.clip(hz[x] + 0.18 * H, 1, H - 1)):, x] = False
    rows = np.where(m.any(0), m.argmax(0), np.nan)
    return m, rows


def run(sid, idx, pred):
    s = idx[sid]; e, a = s['exif'], s['attitude']
    img = extract.load_image(os.path.join(PHOTOS, s['raw']))
    H, W, _ = img.shape
    f_px = (W / 2) / np.tan(np.radians(a['fov_deg']) / 2)
    z = max(e.get('alt_m') or 5.0, 2.0)
    lvl = extract.sea_horizon_attitude_radon(img, f_px,
                                             S.horizon_dip_rad(z))
    pitch = lvl['pitch_deg'] if lvl else (a.get('pitch_deg') or 0.0)
    roll = lvl['roll_deg'] if lvl else (a.get('roll_deg') or 0.0)
    hz = horizon_rows_of(img, f_px, z, pitch, roll)

    # stability across prompt seeds — the knob that used to break E5b
    thr_rows = foreground_rows(img, hz, frac=0.55)
    contours = []
    mask0 = None
    for seed in (1, 2, 3):
        m, rows = sam_foreground(pred, img, hz,
                                 np.random.default_rng(seed), thr_rows)
        if rows is not None:
            contours.append(rows)
            if mask0 is None:
                mask0 = m
    if not contours:
        print(f'{sid}: no foreground prompts found')
        return
    Cs = np.array(contours)
    ok = np.isfinite(Cs).all(0)
    spread = float(np.nanmedian(np.abs(Cs[1:] - Cs[0])[:, ok])) \
        if ok.any() else float('nan')
    fig, ax = plt.subplots(2, 1, figsize=(12, 9))
    over = img.copy()
    if mask0 is not None:
        for c, w_ in ((0, 1.0), (1, 0.45), (2, 0.25)):
            over[..., c] = np.where(mask0, 0.5 * over[..., c] + 0.5 * w_,
                                    over[..., c])
    ax[0].imshow(over)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    cov = [100 * np.isfinite(c).mean() for c in contours]
    ax[0].set_title(f'{sid} — MobileSAM foreground mask '
                    f'(coverage {cov[0]:.0f}% of columns; seed spread '
                    f'{spread:.1f} px)', fontsize=10)
    ax[1].imshow(img)
    for k, c in enumerate(contours):
        ax[1].plot(np.arange(W), c, lw=1.2,
                   color=['#34c759', '#30b0c7', '#32d74b'][k],
                   alpha=0.9,
                   label=f'SAM contour, prompt seed {k+1}' if k < 1
                   else None)
    ax[1].plot(np.arange(W), thr_rows, color='#ff9f0a', lw=1.2, ls=':',
               label='E5b hand threshold (the fragile one)')
    ax[1].plot(np.arange(W), hz, color='#8e8e93', lw=1.0, ls='--',
               label='sea horizon')
    ax[1].set_xlim(0, W); ax[1].set_ylim(H, 0)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].legend(loc='lower left', fontsize=8, framealpha=0.9)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'{sid}_sam.png')
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)

    med = np.nanmedian(Cs, axis=0)
    with open(os.path.join(OUT, f'{sid}.json'), 'w') as f:
        json.dump(dict(id=sid, rows=[None if not np.isfinite(v)
                                     else float(v) for v in med],
                       coverage=cov, seed_spread_px=spread), f)
    thr_cov = 100 * np.isfinite(thr_rows).mean()
    print(f'{sid}: SAM coverage {cov[0]:.0f}% vs threshold {thr_cov:.0f}%'
          f' of columns; contour spread across prompt seeds '
          f'{spread:.1f} px -> {p}', flush=True)


if __name__ == '__main__':
    idx = {x['id']: x for x in json.load(open(os.path.join(
        HERE, 'out', 'theodolite', 'index.json')))['sightings']}
    pred = load_sam()
    for sid in (sys.argv[1:] or ['OREJ1026']):
        if sid in idx:
            run(sid, idx, pred)
