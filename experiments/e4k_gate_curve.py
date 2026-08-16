#!/usr/bin/env python3
"""E4k: the trust-gate operating curve, tuned on the full CH1 audits.

Sweeps the basin-margin acceptance threshold over the recorded E4e
(attitude-free) and E4f (attitude-prior) audits and reports, per
threshold: availability (accepted/203), false-accept share among
accepted (err >= 500 m), and far impostors (err > 1.5 km) — the curve
that turns --min-margin from a first guess into a chosen operating
point. Writes out/e4k_curve.png and out/e4k_curve.json.
"""

import os
import sys
import csv
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
BLUE, ORANGE = '#2a78d6', '#eb6834'
THRESH = [0.15, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


def load(path):
    rows = list(csv.DictReader(open(path)))
    err = np.array([float(r['err_m']) for r in rows])
    marg = np.array([np.inf if r['margin'] == 'inf' else float(r['margin'])
                     for r in rows])
    bnd = np.array([r['boundary'] == '1' for r in rows])
    rms = np.array([float(r['rms_mrad']) for r in rows])
    rel = np.array([float(r['relief_mrad']) for r in rows])
    return err, marg, bnd, rms, rel


def curve(err, marg, bnd, rms, rel):
    pts = []
    base_ok = (~bnd) & (rms <= 12.0) & (rel >= 1.5)   # the other 3 gates
    for t in THRESH:
        acc = base_ok & (marg >= t)
        n = int(acc.sum())
        if n == 0:
            pts.append(dict(thresh=t, n=0))
            continue
        wrong = int((err[acc] >= 500).sum())
        far = int((err[acc] > 1500).sum())
        pts.append(dict(thresh=t, n=n, avail=n / err.size,
                        false_rate=wrong / n, far=far,
                        median_acc_err=float(np.median(err[acc]))))
    return pts


if __name__ == '__main__':
    out = {}
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    for sp in ax.spines.values():
        sp.set_color(INK2), sp.set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    for path, name, color in (
            ('e4e_audit.csv', 'attitude-free (E4e)', BLUE),
            ('e4f_audit.csv', 'attitude priors (E4f)', ORANGE)):
        pts = curve(*load(os.path.join(OUT, path)))
        out[name] = pts
        p = [q for q in pts if q['n'] > 0]
        ax.plot([100 * q['avail'] for q in p],
                [100 * q['false_rate'] for q in p],
                '-o', color=color, lw=1.8, ms=7, mec='white', mew=1.2,
                label=name)
        for q in p:
            if q['thresh'] in (0.15, 0.5, 1.0, 2.0):
                ax.annotate(f"{q['thresh']:g}",
                            (100 * q['avail'], 100 * q['false_rate']),
                            textcoords='offset points', xytext=(7, 5),
                            fontsize=8, color=INK)
        print(f'\n{name}:')
        for q in p:
            print(f"  margin>={q['thresh']:4g}: accepted {q['n']:3d} "
                  f"({100*q['avail']:2.0f}%)  wrong {100*q['false_rate']:3.0f}%"
                  f"  >1.5km {q['far']:2d}  median(acc) "
                  f"{q['median_acc_err']:4.0f} m")
    ax.set_xlabel('availability: photos accepted (%)', fontsize=9, color=INK)
    ax.set_ylabel('false accepts among accepted (%)', fontsize=9, color=INK)
    ax.set_title('trust-gate operating curve, 203 CH1 photos '
                 '(labels = margin threshold)', fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'e4k_curve.png'), dpi=120,
                facecolor=SURF, bbox_inches='tight')
    with open(os.path.join(OUT, 'e4k_curve.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote out/e4k_curve.png/.json')
