#!/usr/bin/env python3
"""Single-frame solves of the same poses side by side (solve_cruise.py summary.json files):
accepted, within 300 m, over 1 km off, the median error of every fix (accepted or not), and,
against the first run, frame by frame on the frames both accepted (the medians of two
different accepted sets mislead: 38 common frames read 97 against 139 m where the paired
difference was 0 m).

    python3 res_compare.py out/res1280/solve out/res2560/solve [...]
"""
import json, os, sys
import numpy as np

runs = {p: {r['tag']: r for r in json.load(open(os.path.join(p, 'summary.json')))} for p in sys.argv[1:]}
first = list(runs.values())[0]
print('%-30s %6s %9s %7s %6s %10s %10s %26s' % ('run', 'frames', 'accepted', '<300 m', '>1 km', 'accepted', 'all fixes',
                                                 'paired vs first: median d'))
for p, rr in runs.items():
    ok = [r for r in rr.values() if r['status'] == 'ok']
    e = np.array([r['err'] for r in ok])
    alle = np.array([r['err'] for r in rr.values() if 'err' in r])
    pair = [t for t in rr if rr[t]['status'] == 'ok' and first.get(t, {}).get('status') == 'ok']
    dd = np.array([rr[t]['err'] - first[t]['err'] for t in pair])
    print('%-30s %6d %9d %7d %6d %8.0f m %8.0f m %9s (%d frames, %d better, %d worse)' % (
        p, len(rr), len(ok), (e < 300).sum(), (e > 1000).sum(), np.median(e) if e.size else np.nan, np.median(alle),
        '%+.0f m' % np.median(dd) if dd.size else '-', dd.size, (dd < 0).sum(), (dd > 0).sum()))
