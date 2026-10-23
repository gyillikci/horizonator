#!/usr/bin/env python3
"""E5ci: accept a fix only when two front ends agree.

E5ch: SegFormer finds more good fixes than eWaSR but also accepts more
wrong ones, and the two segmenters fail on different frames. E5bq did
the same with two MATCHERS (huber vs ncc): accept only when both pass
the gates and land within T m of each other. Here the two are two
FRONT ENDS, same matcher and same DEM.

    accept = gates pass on both solves AND |fix_a - fix_b| <= T
    reported fix = the mean of the two

Pairs (all from stored solves, nothing is re-run):
  hard   eWaSR hard +-6      with  SegFormer hard +-6
  soft   eWaSR soft 5 + gate with  SegFormer soft 5 + gate
  f_hard eWaSR hard +-6      with  SegFormer+detector hard +-6
  f_soft eWaSR soft 5 + gate with  SegFormer+detector soft 5 + gate

where "gate" is |psi| <= 2 sigma = 10 deg. Truth = the EXIF fix, so
the error of a fix is hypot(dn, de).

    python3 e5ci_agreement.py
"""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = os.path.join(HERE, 'data', 'campaign_frames.json')
BAD, GOOD = 500.0, 300.0
SRC = dict(
    ewasr_hard=('baseline', '%s_huber.json'),
    ewasr_soft=('compass', '%s_soft5.json'),
    seg_hard=('segformer', '%s_seg_hard6.json'),
    seg_soft=('segformer', '%s_seg_soft5.json'),
    segf_hard=('segformer', '%s_seg_f_hard6.json'),
    segf_soft=('segformer', '%s_seg_f_soft5.json'))
PAIRS = dict(hard=('ewasr_hard', 'seg_hard'), soft=('ewasr_soft', 'seg_soft'),
             f_hard=('ewasr_hard', 'segf_hard'),
             f_soft=('ewasr_soft', 'segf_soft'))


def load(tag, key):
    d, pat = SRC[key]
    f = os.path.join(HERE, 'out', d, pat % tag)
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    if j.get('dn_m') is None:
        return None
    gate = 10.0 if key.endswith('soft') else 1e9
    return dict(dn=j['dn_m'], de=j['de_m'],
                ok=j['status'] == 'ok'
                and abs(j['photos'][0]['heading_offset_deg']) <= gate)


def main():
    tags = [f['tag'] for f in json.load(open(FRAMES))['frames']]
    R = {t: {k: load(t, k) for k in SRC} for t in tags}
    out = {}
    print('%-8s %5s | %4s %4s %5s | %s' % ('pair', 'T', 'acc', 'good',
                                           'wrong', 'accepted fixes (err m)'))
    for name, (a, b) in PAIRS.items():
        for T in (300, 500, 1000, 1e9):
            acc = []
            for t in tags:
                x, y = R[t][a], R[t][b]
                if not (x and y and x['ok'] and y['ok']):
                    continue
                if math.hypot(x['dn'] - y['dn'], x['de'] - y['de']) > T:
                    continue
                dn, de = (x['dn'] + y['dn']) / 2, (x['de'] + y['de']) / 2
                acc.append((t, math.hypot(dn, de)))
            good = sum(e < GOOD for _, e in acc)
            wrong = [(t, e) for t, e in acc if e >= BAD]
            out['%s_T%g' % (name, T)] = dict(
                accepted=len(acc), good=good, wrong=len(wrong),
                wrong_tags=[t for t, _ in wrong])
            print('%-8s %5s | %4d %4d %5d | %s' % (
                name, 'any' if T > 1e8 else '%d' % T, len(acc), good,
                len(wrong), ', '.join('%s %.0f' % (t[3:], e)
                                      for t, e in acc)))
        print()
    # for reference: each solve alone, same gates
    print('single solves, same gates:')
    for key in SRC:
        acc = [(t, math.hypot(R[t][key]['dn'], R[t][key]['de']))
               for t in tags if R[t][key] and R[t][key]['ok']]
        print('  %-10s %2d accepted, %d good, %d wrong' % (
            key, len(acc), sum(e < GOOD for _, e in acc),
            sum(e >= BAD for _, e in acc)))
    json.dump(out, open(os.path.join(HERE, 'out', 'segformer',
                                     'agreement.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
