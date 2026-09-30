#!/usr/bin/env python3
"""E5bm report: what the +-6 deg heading window costs, frame by frame.

Reads out/resweep/<tag>_w6.json and <tag>_w30.json -- the same frame
solved twice with identical flags except --heading-window -- and scores
both against the EXIF GPS truth. Classifies each frame by what widening
the window did to it:

  false->refused  w6 ACCEPTED a wrong fix (>= GOOD m) and w30 refused it
                  -- the Bodrum 9 pattern (E5bf)
  rescued         w30 more than halves the error, with a heading offset
                  outside +-6 (so the narrow window could not reach it)
                  and NOT pinned at the +-30 edge
  pinned          w30's heading sits on its own +-30 edge: the wide
                  window is clamped too, and its answer means nothing
  broken          w30 more than doubles the error, or accepts a wrong
                  fix that w6 did not
  unchanged       everything else

Frames without EXIF GPS are excluded: their "truth" is whatever centre
the solve was given, which for the blind tests was deliberately wrong.

and, separately, tallies the verdicts that changed: accepts that became
refusals and the reverse, and whether each accept was right.
"""

import glob
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'out/resweep'
GOOD = 300.0          # an "accepted and right" fix, metres


def load():
    pairs = {}
    for f in glob.glob(os.path.join(OUT, '*_w*.json')):
        tag, w = os.path.basename(f)[:-5].rsplit('_w', 1)
        try:
            j = json.load(open(f))
        except Exception:
            continue
        p = j['photos'][0]
        pairs.setdefault(tag, {})[int(w)] = dict(
            err=math.hypot(j['dn_m'], j['de_m']), dn=j['dn_m'], de=j['de_m'],
            ok=j['status'] == 'ok', margin=j['basin_margin'],
            hoff=p['heading_offset_deg'])
    return {t: d for t, d in pairs.items() if 6 in d and 30 in d}


def classify(a, b):
    if abs(b['hoff']) >= 29.5:
        return 'pinned'
    if a['ok'] and a['err'] >= GOOD and not b['ok']:
        return 'false->refused'
    if b['ok'] and b['err'] >= GOOD and not (a['ok'] and a['err'] >= GOOD):
        return 'broken'
    if a['err'] > 2 * b['err'] and abs(b['hoff']) > 6.0:
        return 'rescued'
    if b['err'] > 2 * a['err']:
        return 'broken'
    return 'unchanged'


def main():
    P = load()
    S = '/tmp/claude-0/-home-user/792503f9-74c5-5111-83ca-eeeda63e838d/scratchpad/'
    fr = json.load(open(S + 'frames_u.json'))
    no_truth = {'%02d_%s' % (i, f['file'].rsplit('.', 1)[0][:28])
                for i, f in enumerate(fr) if f.get('truth_src') != 'exif'}
    for t in no_truth:
        P.pop(t, None)
    if no_truth:
        print('excluded, no EXIF truth:', ', '.join(sorted(no_truth)))
    rows = []
    for t in sorted(P):
        a, b = P[t][6], P[t][30]
        rows.append(dict(tag=t, a=a, b=b, cls=classify(a, b)))

    print('%-34s %8s %-5s %6s | %8s %-5s %6s | %s'
          % ('frame', 'w6 err', 'acc', 'hoff', 'w30 err', 'acc', 'hoff', 'effect'))
    for r in rows:
        a, b = r['a'], r['b']
        print('%-34s %7.0fm %-5s %+6.1f | %7.0fm %-5s %+6.1f | %s'
              % (r['tag'][:34], a['err'], 'yes' if a['ok'] else 'no', a['hoff'],
                 b['err'], 'yes' if b['ok'] else 'no', b['hoff'], r['cls']))

    n = len(rows)
    ea = np.array([r['a']['err'] for r in rows])
    eb = np.array([r['b']['err'] for r in rows])
    print('\nframes compared: %d' % n)
    for k in ('false->refused', 'rescued', 'pinned', 'broken', 'unchanged'):
        print('  %-9s %d' % (k, sum(r['cls'] == k for r in rows)))
    print('median error: w6 %.0f m, w30 %.0f m' % (np.median(ea), np.median(eb)))
    print('w30 heading offset outside +-6 deg on %d frames'
          % sum(abs(r['b']['hoff']) > 6 for r in rows))

    def verdicts(key):
        acc = [r for r in rows if r[key]['ok']]
        right = [r for r in acc if r[key]['err'] < GOOD]
        return len(acc), len(right)
    for key, name in (('a', 'w6 '), ('b', 'w30')):
        na, nr = verdicts(key)
        print('%s accepted %2d, of which within %d m: %2d, false accepts: %d'
              % (name, na, GOOD, nr, na - nr))

    json.dump(rows, open(os.path.join(OUT, 'ab_report.json'), 'w'), indent=1)

    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    col = {'false->refused': '#2E86C1', 'rescued': '#1E7F5C',
           'pinned': '#D68910', 'broken': '#C0392B', 'unchanged': '#8A8F98'}
    for r in rows:
        m = 'o' if r['b']['ok'] else 'x'
        ax.scatter(r['a']['err'], r['b']['err'], s=46, marker=m,
                   color=col[r['cls']], zorder=3)
    lim = [30, max(ea.max(), eb.max()) * 1.4]
    ax.plot(lim, lim, color='#1A2732', lw=1, alpha=.6)
    ax.plot(lim, [l / 2 for l in lim], color='#1A2732', lw=.7, ls='--', alpha=.4)
    ax.plot(lim, [l * 2 for l in lim], color='#1A2732', lw=.7, ls='--', alpha=.4)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('konum hatası, ±6° pencere (m)')
    ax.set_ylabel('konum hatası, ±30° pencere (m)')
    for k, c in col.items():
        ax.scatter([], [], color=c, s=46, label='%s (%d)' % (
            {'false->refused': 'sahte kabul → ret', 'rescued': 'kurtarıldı',
             'pinned': '±30 kenarında', 'broken': 'bozuldu',
             'unchanged': 'değişmedi'}[k], sum(r['cls'] == k for r in rows)))
    ax.scatter([], [], color='#8A8F98', marker='x', s=46, label='±30° reddetti')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=.3, which='both')
    ax.set_title('Heading penceresi: ±6° ile ±30°, %d kare' % n, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'ab_window.png'), dpi=110)
    print('wrote', os.path.join(OUT, 'ab_window.png'))


if __name__ == '__main__':
    main()
