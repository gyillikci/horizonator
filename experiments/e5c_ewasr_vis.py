#!/usr/bin/env python3
"""Run the pretrained eWaSR segmenter on field photographs and look.

E4r rejected eWaSR as an extraction front end (a stratus deck reads as
sky; its boundaries are segmentation-grade, not matching-grade). This
renders what it actually produces on the Theodolite frames, alongside
the two things we ended up using instead: the radon sea-horizon line
and the silhouette the matcher is fed.

Per photo: the frame with the three classes tinted, and the class
boundaries drawn against our own lines.

Run:  python3 e5c_ewasr_vis.py ID [ID ...]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import extract, skyline as S
from ewasr_bridge import EWasr

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'e5c')
PHOTOS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      'celestial-navigation', 'theodolite')
SCRATCH = ('/tmp/claude-0/-home-user/'
           '792503f9-74c5-5111-83ca-eeeda63e838d/scratchpad')
TINT = {0: (1.0, 0.25, 0.2), 1: (0.05, 0.5, 1.0), 2: (1.0, 0.85, 0.1)}
NAME = {0: 'obstacle', 1: 'water', 2: 'sky'}


def panel(sid, s, seg):
    e, a = s['exif'], s['attitude']
    img = extract.load_image(os.path.join(PHOTOS, s['raw']))
    H, W, _ = img.shape
    cls = seg.predict(img)
    z = max(e.get('alt_m') or 5.0, 2.0)
    f_px = (W / 2) / np.tan(np.radians(a['fov_deg']) / 2)
    dip = S.horizon_dip_rad(z)
    lvl = extract.sea_horizon_attitude_radon(img, f_px, dip)
    rows, _ = extract.skyline_seam(img)

    over = img.copy()
    for k, col in TINT.items():
        m = cls == k
        for c in range(3):
            over[..., c][m] = 0.55 * over[..., c][m] + 0.45 * col[c]

    # eWaSR's own water/sky boundary: topmost water row per column
    wb = np.where((cls == 1).any(0), (cls == 1).argmax(0), np.nan)
    sb = np.where((cls != 2).any(0), (cls != 2).argmax(0), np.nan)

    fig, ax = plt.subplots(2, 1, figsize=(12, 9))
    ax[0].imshow(over)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    frac = {k: 100 * (cls == k).mean() for k in TINT}
    ax[0].set_title(f'{sid} — eWaSR: water {frac[1]:.0f}%, sky '
                    f'{frac[2]:.0f}%, obstacle {frac[0]:.0f}%',
                    fontsize=10)
    ax[1].imshow(img)
    ax[1].plot(np.arange(W), rows, color='#ff3b30', lw=1.3,
               label='silhouette fed to the matcher (seam)')
    ax[1].plot(np.arange(W), sb, color='#ffd60a', lw=1.3, ls='-',
               label='eWaSR sky boundary')
    ax[1].plot(np.arange(W), wb, color='#0a84ff', lw=1.3, ls=':',
               label='eWaSR topmost water')
    if lvl:
        u = np.arange(W) - (W - 1) / 2
        v = (np.tan(-dip - np.radians(lvl['pitch_deg']))
             * np.hypot(u, f_px) - np.radians(lvl['roll_deg']) * u)
        ax[1].plot(np.arange(W), (H - 1) / 2 - v, color='#34c759',
                   lw=1.3, ls='--', label='our sea horizon (radon)')
    ax[1].set_xlim(0, W); ax[1].set_ylim(H, 0)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].legend(loc='lower left', fontsize=8, framealpha=0.9)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'{sid}_ewasr.png')
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)

    d = np.abs(sb - rows)
    ok = np.isfinite(d)
    msg = (f'{sid}: eWaSR sky boundary vs our silhouette median '
           f'{np.nanmedian(d[ok]):.0f} px')
    if lvl:
        hz = (H - 1) / 2 - v
        dh = np.abs(wb - hz)
        msg += (f'; eWaSR water top vs our horizon median '
                f'{np.nanmedian(dh[np.isfinite(dh)]):.0f} px')
    print(msg + f' -> {p}', flush=True)
    return p


if __name__ == '__main__':
    idx = {x['id']: x for x in json.load(open(os.path.join(
        HERE, 'out', 'theodolite', 'index.json')))['sightings']}
    seg = EWasr(os.path.join(SCRATCH, 'eWaSR'),
                os.path.join(SCRATCH, 'ewasr_resnet18.pth'))
    for sid in (sys.argv[1:] or ['OREJ1026', 'SRYK4301', 'APST5638']):
        if sid in idx:
            panel(sid, idx[sid], seg)
        else:
            print(f'{sid}: not in the curated index')
