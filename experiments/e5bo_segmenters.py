#!/usr/bin/env python3
"""E5bo: foundation-model segmenters against the extraction failures.

The extraction side is where this campaign keeps breaking: both front
ends rode cumulus edges on the Bosphorus frame (E5bh), an aircraft wing
took the Milas correlation from +0.89 to -0.10 (E5bd), and eWaSR walked
onto a crane line at Bodrum (E5bb). This asks whether a modern
foundation model does better.

Access shaped the choice. Hugging Face and Meta's weight host are both
blocked from this container, so DINOv3, DINOv2 and the official SAM 2
checkpoints are unreachable, and MaSTr1325's hosts are blocked too, so
no head can be trained. What is reachable is SAM 2.1, redistributed by
Ultralytics on GitHub releases. So:

  sam2.1-t / sam2.1-b, prompted
      eight positive points across the top band (sky) and eight
      negative points across the bottom band (water and foreground),
      one object, one mask;

  sam2.1-b features, zero-shot
      the DINO recipe with a different backbone: the frozen encoder's
      64x64x256 image embedding, k-means prototypes from the top band
      (sky, including clouds) and the bottom band (water, land), each
      cell labelled by its nearest prototype, the probability map
      upsampled. No training, no labels. Coarse by construction --
      ~25 px per cell at working resolution, the same limit DINO's
      16 px patches impose;

against the two shipped front ends, seam and eWaSR.

Every mask goes through the same post-processing: fill holes (a cloud
inside the sky is still sky), keep the connected sky component that
touches the top edge (a sky reflection in the water is not), and take
each column's boundary as the row below that component's lowest pixel.

Ground truth is the operator's hand-drawn Bosphorus skyline (E5bh).
"""

import json
import os
import sys
import time

import numpy as np
import cv2
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extract

SAMDIR = os.path.expanduser('~/.horizonator/assets/sam2')
P = '/home/user/celestial-navigation/peakfinder/'
_MODELS = {}


def sam(name):
    if name not in _MODELS:
        from ultralytics import SAM
        _MODELS[name] = SAM(os.path.join(SAMDIR, name))
    return _MODELS[name]


def mask_to_boundary(sky):
    """Fill holes, keep the top-touching component, lowest row + 1."""
    sky = ndimage.binary_fill_holes(sky)
    lab, n = ndimage.label(sky)
    if n == 0:
        return None, None
    # "touches the top" means the top band, not row 0: SAM upsamples a
    # 256x256 mask and routinely leaves the first rows outside it, which
    # made an otherwise correct sky mask yield no boundary at all
    band = lab[:max(3, sky.shape[0] // 50)]
    top = np.unique(band[band > 0])
    keep = np.isin(lab, top)
    H, W = keep.shape
    rows = np.full(W, np.nan)
    has = keep.any(0)
    last = H - 1 - np.argmax(keep[::-1], axis=0)
    rows[has] = last[has] + 1
    return rows, has.astype(float)


def prompts(H, W):
    xs = np.linspace(0.06 * W, 0.94 * W, 8)
    pos = [[float(x), 0.06 * H] for x in xs]
    neg = [[float(x), 0.92 * H] for x in xs]
    return pos + neg, [1] * 8 + [0] * 8


LAST_SCORE = {}


def _fresh(m):
    """Drop cached image features. Ultralytics' SAM predictor reuses
    self.features whenever it is set, so after any set_image() every
    later prompted call silently segments THE PREVIOUS IMAGE. The first
    version of this script hit exactly that on its second and third
    frames."""
    if m.predictor is not None:
        m.predictor.reset_image()


def sam_prompted(img8, name):
    H, W = img8.shape[:2]
    pts, lab = prompts(H, W)
    m = sam(name)
    _fresh(m)
    # conf=0: keep the mask even when SAM is unsure; the score is
    # recorded, because an unsure "sky" is itself a finding
    r = m(img8[..., ::-1].copy(), points=[pts], labels=[lab],
          conf=0.0, verbose=False)
    b = r[0].boxes
    LAST_SCORE[name] = float(b.conf[0]) if b is not None and len(b) else float('nan')
    _fresh(m)
    return r[0].masks.data.cpu().numpy()[0] > 0.5


def sam_features_zeroshot(img8, name='sam2.1_b.pt', k=4):
    """DINO-style zero-shot: nearest-prototype on frozen dense features."""
    m = sam(name)
    if m.predictor is None:          # the predictor is built on first call
        m(img8[..., ::-1].copy(), points=[[10, 10]], labels=[1], verbose=False)
    p = m.predictor
    p.set_image(img8[..., ::-1].copy())
    emb = p.features['image_embed'][0].float().cpu().numpy()     # C,64,64
    _fresh(m)
    C, G, _ = emb.shape
    H, W = img8.shape[:2]
    # Ultralytics letterboxes to 1024 square, image top-left, padded
    # bottom/right: the image occupies the first gh x gw cells
    s = 1024.0 / max(H, W)
    gh, gw = int(round(H * s / 16)), int(round(W * s / 16))
    f = emb[:, :gh, :gw].reshape(C, -1).T
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    f = f.reshape(gh, gw, C)
    top = f[:max(2, gh // 8)].reshape(-1, C)
    bot = f[-max(2, gh // 5):].reshape(-1, C)

    def kmeans(x, k):
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        _, _, c = cv2.kmeans(x.astype(np.float32), k, None, crit, 4,
                             cv2.KMEANS_PP_CENTERS)
        return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
    ps, pg = kmeans(top, k), kmeans(bot, k)
    sim_s = (f @ ps.T).max(-1)
    sim_g = (f @ pg.T).max(-1)
    prob = 1.0 / (1.0 + np.exp(-(sim_s - sim_g) * 20.0))
    prob = cv2.resize(prob.astype(np.float32), (W, H),
                      interpolation=cv2.INTER_LINEAR)
    return prob > 0.5


def run_all(path):
    img = extract.load_image(path)
    img8 = (img * 255).astype(np.uint8)
    out = {}
    t = time.time()
    r, c = extract.skyline_seam(img)
    out['seam'] = (np.asarray(r, float), np.asarray(c, float), time.time() - t)
    try:
        import skyfix
        skyfix.EXTRACTOR = 'ewasr'
        t = time.time()
        # the raw eWaSR boundary, not the guarded path: the pre-check
        # would silently hand back the seam when they disagree
        cls = skyfix._EWASR.predict(img) if skyfix._EWASR else None
        if cls is None:
            skyfix.extract_boundary(img)
            cls = skyfix._EWASR.predict(img)
        dt = time.time() - t
        # shipped logic (skyfix.extract_boundary's eWaSR branch, before
        # the pre-check): first non-sky row below the first sky run
        sky = cls == 2
        H, W = sky.shape
        rs = np.full(W, np.nan)
        has = sky.any(0)
        top = np.where(has, sky.argmax(0), 0)
        for x in np.where(has)[0]:
            below = ~sky[top[x]:, x]
            if below.any() and int(np.argmax(below)) >= 3:
                rs[x] = top[x] + int(np.argmax(below))
        out['ewasr (shipped)'] = (rs, np.isfinite(rs).astype(float), dt)
        rr, cc = mask_to_boundary(sky)
        out['ewasr + same post-proc'] = (rr, cc, dt)
    except Exception as e:
        print('ewasr unavailable:', e, file=sys.stderr)
    for name, tag in (('sam2.1_t.pt', 'sam2.1-t'), ('sam2.1_b.pt', 'sam2.1-b')):
        t = time.time()
        rr, cc = mask_to_boundary(sam_prompted(img8, name))
        out[tag] = (rr, cc, time.time() - t)
        print('   %s SAM confidence %.3f' % (tag, LAST_SCORE[name]))
    t = time.time()
    rr, cc = mask_to_boundary(sam_features_zeroshot(img8))
    out['sam2.1-b features (zero-shot)'] = (rr, cc, time.time() - t)
    return img, out


def slug(k):
    return (k.replace('.', '').replace(' + ', '_').replace(' ', '_')
             .replace('(', '').replace(')', ''))


def score(rows, gt, tol=(3, 10)):
    ok = np.isfinite(rows) & np.isfinite(gt)
    e = np.abs(rows - gt)[ok]
    if e.size == 0:
        return dict(cov=0.0)
    d = rows[ok] - gt[ok]
    off = float(np.median(d))
    sc = np.abs(d - off)
    return dict(cov=float(ok.mean()), off=off, sc_med=float(np.median(sc)),
                sc_p90=float(np.percentile(sc, 90)), sc_max=float(sc.max()),
                sc_w3=float((sc <= 3).mean()), med=float(np.median(e)),
                p90=float(np.percentile(e, 90)), max=float(e.max()),
                w3=float((e <= tol[0]).mean()), w10=float((e <= tol[1]).mean()))


COLORS = {'seam': '#E67E22', 'ewasr (shipped)': '#8E44AD',
          'ewasr + same post-proc': '#D98CE8', 'sam2.1-t': '#2E86C1',
          'sam2.1-b': '#17A589', 'sam2.1-b features (zero-shot)': '#C0392B'}


def figure(img, out, gt, title, fn):
    H, W = img.shape[:2]
    n = len(out)
    fig, axs = plt.subplots(2, 1, figsize=(15, 11),
                            gridspec_kw=dict(height_ratios=[1.35, 1]))
    axs[0].imshow(img)
    x = np.arange(W)
    if gt is not None:
        axs[0].plot(x, gt, '-', lw=3.2, color='white', alpha=.9, label='elle çizilmiş (gerçek)')
    for k, (r, c, _) in out.items():
        axs[0].plot(x, r, '-', lw=1.3, color=COLORS[k], label=k)
    axs[0].set_xlim(0, W); axs[0].set_ylim(H * 0.78, H * 0.02); axs[0].axis('off')
    axs[0].legend(loc='upper right', fontsize=9, framealpha=.85)
    axs[0].set_title(title, fontsize=12)
    if gt is not None:
        for k, (r, c, _) in out.items():
            axs[1].plot(x, np.clip(r - gt, -300, 300), lw=1.1, color=COLORS[k], label=k)
        axs[1].axhline(0, color='k', lw=.8)
        axs[1].set_ylabel('satır hatası (px), + = aşağıda')
        axs[1].set_xlabel('sütun')
        axs[1].set_ylim(-320, 320)
        axs[1].grid(alpha=.3)
        axs[1].legend(fontsize=8, ncol=3)
    else:
        axs[1].axis('off')
    fig.tight_layout()
    fig.savefig(fn, dpi=100)


def main():
    os.makedirs('out/seg', exist_ok=True)
    # --- quantitative: the Bosphorus frame against the hand-drawn curve
    img, out = run_all(P + 'PF_new_bosphorus1.jpg')
    H, W = img.shape[:2]
    hand = np.load('out/bos/user_skyline.npy')
    gt = np.interp(np.linspace(0, hand.size - 1, W), np.arange(hand.size),
                   hand) * (W / hand.size)
    rows = []
    print('%-30s %8s %9s %9s %9s %7s %6s'
          % ('method', 'offset', 'scat med', 'scat p90', 'scat max', '<=3px', 'sec'))
    for k, (r, c, dt) in out.items():
        s = score(r, gt)
        rows.append(dict(method=k, sec=dt, **s))
        print('%-30s %+7.1fpx %8.1fpx %8.1fpx %8.0fpx %6.0f%% %6.1f'
              % (k, s['off'], s['sc_med'], s['sc_p90'], s['sc_max'],
                 100 * s['sc_w3'], dt))
        np.save('out/seg/bosphorus_%s.npy' % slug(k),
                r)   # working-resolution rows: skyfix infers the row scale
                     # from the column count, so rows and columns must agree
    json.dump(rows, open('out/seg/bosphorus_scores.json', 'w'), indent=1)
    figure(img, out, gt, 'Boğaz — bulutlu gökyüzü, elle çizilmiş gerçeğe karşı',
           'out/seg/seg_bosphorus.png')

    # --- qualitative: the other two known failures
    for fn, ttl, tag in (('PF_new_milas1.jpg', 'Milas — sol üstte uçak kanadı', 'milas'),
                         ('PF_new_bodrum9.jpg', 'Bodrum 9 — solda vinç hattı', 'bodrum9')):
        img, out = run_all(P + fn)
        figure(img, out, None, ttl, 'out/seg/seg_%s.png' % tag)
        W = img.shape[1]
        for k, (r, c, _) in out.items():
            np.save('out/seg/%s_%s.npy' % (tag, slug(k)),
                    r)
        print('wrote out/seg/seg_%s.png' % tag)


if __name__ == '__main__':
    main()
