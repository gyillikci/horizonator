#!/usr/bin/env python3
"""E5bo visuals: what SAM 2.1 actually segments, frame by frame.

Three panels per frame:
  1. prompted SAM 2.1-b - the sky mask tinted, its boundary, and the
     prompts that produced it (+ sky, x not-sky);
  2. SAM 2.1-b "segment everything" - every mask the automatic grid
     finds, each in its own colour. This is the panel that shows WHY
     the prompted mask stops halfway down a gradient sky: SAM itself
     carves the sky into separate objects;
  3. zero-shot sky probability from SAM 2.1-b's frozen features
     (the DINO-style probe of e5bo_segmenters.py), with its 0.5 contour.

    python3 e5bo_sam_visuals.py compute bosphorus|milas|bodrum9
    python3 e5bo_sam_visuals.py draw
"""

import os
import sys
import time

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extract
import e5bo_segmenters as E

P = '/home/user/celestial-navigation/peakfinder/'
FRAMES = {
    'bosphorus': (P + 'PF_new_bosphorus1.jpg', 'Boğaz — bulutlu gökyüzü'),
    'milas': (P + 'PF_new_milas1.jpg', 'Milas — pus, sol üstte uçak kanadı'),
    'bodrum9': (P + 'PF_new_bodrum9.jpg', 'Bodrum 9 — gün batımı geçişi'),
}
CACHE = 'out/seg/sam_%s.npz'


def compute(key):
    path, _ = FRAMES[key]
    img = extract.load_image(path)
    img8 = (img * 255).astype(np.uint8)
    t = time.time()
    prompted_t = E.sam_prompted(img8, 'sam2.1_t.pt')
    score_t = E.LAST_SCORE['sam2.1_t.pt']
    prompted = E.sam_prompted(img8, 'sam2.1_b.pt')
    score_b = E.LAST_SCORE['sam2.1_b.pt']
    t1 = time.time() - t
    # zero-shot probability, kept continuous for the heatmap
    m = E.sam('sam2.1_b.pt')
    p = m.predictor
    p.set_image(img8[..., ::-1].copy())
    emb = p.features['image_embed'][0].float().cpu().numpy()
    C = emb.shape[0]
    H, W = img8.shape[:2]
    s = 1024.0 / max(H, W)
    gh, gw = int(round(H * s / 16)), int(round(W * s / 16))
    f = emb[:, :gh, :gw].reshape(C, -1).T
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    f = f.reshape(gh, gw, C)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    def km(x):
        _, _, c = cv2.kmeans(x.astype(np.float32), 4, None, crit, 4,
                             cv2.KMEANS_PP_CENTERS)
        return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
    ps = km(f[:max(2, gh // 8)].reshape(-1, C))
    pg = km(f[-max(2, gh // 5):].reshape(-1, C))
    prob = 1 / (1 + np.exp(-((f @ ps.T).max(-1) - (f @ pg.T).max(-1)) * 20))
    prob = cv2.resize(prob.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    E._fresh(m)        # never let set_image's cached features leak onward
    t = time.time()
    r = m(img8[..., ::-1].copy(), verbose=False)      # segment everything
    t2 = time.time() - t
    allm = (r[0].masks.data.cpu().numpy() > 0.5) if r[0].masks is not None \
        else np.zeros((0, H, W), bool)
    np.savez_compressed(CACHE % key, prompted=prompted, prob=prob,
                        prompted_t=prompted_t, score_t=score_t, score_b=score_b,
                        everything=np.packbits(allm, axis=-1),
                        shape=np.array(allm.shape))
    print('%s: prompted %.1fs, everything %.1fs, %d masks'
          % (key, t1, t2, allm.shape[0]))


PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
    [210, 245, 60], [250, 190, 212], [0, 128, 128], [170, 110, 40],
], float) / 255


def panels(ax, key):
    ax_all = list(ax)
    path, title = FRAMES[key]
    img = extract.load_image(path)
    H, W = img.shape[:2]
    z = np.load(CACHE % key)
    n, h, w = z['shape']
    allm = np.unpackbits(z['everything'], axis=-1)[..., :w].astype(bool)
    prompted = z['prompted']
    prob = z['prob']

    # 1-2. prompted, both model sizes
    pts, lab = E.prompts(H, W)
    for a, mk, sc, nm in ((ax[0], z['prompted_t'], float(z['score_t']), '2.1-t'),
                          (ax[1], prompted, float(z['score_b']), '2.1-b')):
        over = img.copy()
        tint = np.array([0.15, 0.85, 1.0])
        over[mk] = 0.45 * over[mk] + 0.55 * tint
        a.imshow(over)
        rows, _ = E.mask_to_boundary(mk)
        if rows is not None:
            a.plot(np.arange(W), rows, '-', color='#FFD400', lw=1.6)
        for (x, y), l in zip(pts, lab):
            a.plot(x, y, '+' if l else 'x', ms=11, mew=2.6,
                   color='#1ED760' if l else '#FF3B30')
        a.set_title('SAM %s, istemli — maske %%%.0f, güven %.2f'
                    % (nm, 100 * mk.mean(), sc), fontsize=10.5)
    ax[0].set_title(title + '\n' + ax[0].get_title(), fontsize=10.5)
    ax = [None, None, ax[2], ax[3]]

    # 2. segment everything, largest drawn first so small ones sit on top
    over = img.copy() * 0.55
    order = np.argsort(allm.reshape(n, -1).sum(1))[::-1]
    for i, k in enumerate(order):
        c = PALETTE[i % len(PALETTE)]
        mk = allm[k]
        over[mk] = 0.35 * img[mk] + 0.65 * c
        cnt, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
        for cc in cnt:
            ax[2].plot(cc[:, 0, 0], cc[:, 0, 1], '-', color='white', lw=0.7)
    ax[2].imshow(np.clip(over, 0, 1))
    covered = allm.any(0).mean()
    ax[2].set_title('SAM 2.1-b, her şeyi segmentle: %d maske, %%%.0f kaplama'
                    % (n, 100 * covered), fontsize=10.5)

    # 3. zero-shot probability
    ax[3].imshow(img)
    ax[3].imshow(prob, cmap='coolwarm_r', alpha=0.55, vmin=0, vmax=1)
    ax[3].contour(prob, levels=[0.5], colors='#FFD400', linewidths=1.6)
    ax[3].set_title('öznitelik prototipi, sıfır-atış: gökyüzü olasılığı',
                    fontsize=10.5)
    for a in ax_all:
        a.set_xlim(0, W); a.set_ylim(H, 0); a.axis('off')


def draw():
    keys = [k for k in FRAMES if os.path.exists(CACHE % k)]
    fig, axs = plt.subplots(len(keys), 4, figsize=(24, 4.6 * len(keys)))
    axs = np.atleast_2d(axs)
    for row, k in zip(axs, keys):
        panels(row, k)
    fig.tight_layout()
    fig.savefig('out/seg/sam_segmentation.png', dpi=95)
    for k in keys:
        f, a = plt.subplots(2, 2, figsize=(17, 11.5))
        a = a.ravel()
        panels(a, k)
        f.tight_layout()
        f.savefig('out/seg/sam_segmentation_%s.png' % k, dpi=110)
        plt.close(f)
    print('wrote out/seg/sam_segmentation.png and one per frame')


if __name__ == '__main__':
    if sys.argv[1] == 'compute':
        compute(sys.argv[2])
    else:
        draw()
