#!/usr/bin/env python3
"""E4l part 2: hypothesis CONSENSUS as a selector and as a trust gate.

Reads out/e4l_multihyp.csv (with per-hypothesis positions). Two
questions: (a) does picking the medoid hypothesis (closest to the
others) beat picking by margin? (b) does hypothesis AGREEMENT — how many
of the four extractions land their fix within r of the medoid — work as
an integrity gate, alone and combined with the margin?
"""

import os
import sys
import csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
H = ['default', 'loose', 'strict', 'deep']
R_AGREE = 500.0


def main():
    rows = list(csv.DictReader(open(os.path.join(OUT,
                                                 'e4l_multihyp.csv'))))
    per = []
    for r in rows:
        pos = {h: np.array([float(r[f'dn_{h}']), float(r[f'de_{h}'])])
               for h in H}
        errs = {h: float(r[f'err_{h}']) for h in H}
        margs = {h: (np.inf if r[f'margin_{h}'] == 'inf'
                     else float(r[f'margin_{h}'])) for h in H}
        # medoid: hypothesis minimizing summed distance to the others
        dsum = {h: sum(np.linalg.norm(pos[h] - pos[k]) for k in H)
                for h in H}
        med = min(H, key=lambda h: dsum[h])
        agree = sum(1 for h in H
                    if np.linalg.norm(pos[h] - pos[med]) <= R_AGREE)
        per.append(dict(errs=errs, margs=margs, med=med, agree=agree,
                        marg_sel=max(H, key=lambda h: margs[h])))

    def stats(errs):
        e = np.array(errs)
        return (f'median {np.median(e):5.0f} m, <1km '
                f'{100 * (e < 1000).mean():2.0f}%')

    print(f'n = {len(per)}')
    print('selector comparison:')
    print(f"  default   : {stats([p['errs']['default'] for p in per])}")
    print(f"  margin    : {stats([p['errs'][p['marg_sel']] for p in per])}")
    print(f"  consensus : {stats([p['errs'][p['med']] for p in per])}")
    print(f"  oracle    : {stats([min(p['errs'].values()) for p in per])}")

    print(f'\nagreement gate (k of 4 within {R_AGREE:.0f} m of medoid), '
          'fix = medoid hypothesis:')
    for k in (2, 3, 4):
        sel = [p for p in per if p['agree'] >= k]
        if not sel:
            continue
        e = np.array([p['errs'][p['med']] for p in sel])
        print(f'  k>={k}: accepted {len(sel):3d} '
              f'({100 * len(sel) / len(per):2.0f}%)  '
              f'wrong(>=500 m) {100 * (e >= 500).mean():3.0f}%  '
              f'>1.5km {int((e > 1500).sum()):2d}  median {np.median(e):4.0f} m')

    print('\ncombined gates (medoid fix): agreement k AND margin t:')
    for k in (3, 4):
        for t in (0.3, 0.5, 1.0):
            sel = [p for p in per
                   if p['agree'] >= k and p['margs'][p['med']] >= t]
            if not sel:
                continue
            e = np.array([p['errs'][p['med']] for p in sel])
            print(f'  k>={k} & margin>={t:3g}: accepted {len(sel):3d} '
                  f'({100 * len(sel) / len(per):2.0f}%)  '
                  f'wrong {100 * (e >= 500).mean():3.0f}%  '
                  f'>1.5km {int((e > 1500).sum()):2d}  '
                  f'median {np.median(e):4.0f} m')


if __name__ == '__main__':
    main()
