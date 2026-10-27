#!/usr/bin/env python3
"""Trackers run on the same instants seen different ways (look_planner.py --views), side by side:
median, p90, frames within 300 m, the 95 % ellipse's hold, per-route medians, and for a
switching run how many frames it looked with each view.

    python3 views_summary.py out/views NAME:VIEW_DIR/RESULT [...]
      e.g. "known, ahead:ahead/pfk"  reads out/views/ahead/pfk.json
"""
import json, os, sys
import numpy as np

root = sys.argv[1]
print('%-30s %7s %7s %9s %9s  %s' % ('run', 'median', 'p90', '<300 m', 'in 95 %', 'per route (median m); views used'))
for arg in sys.argv[2:]:
    name, path = arg.rsplit(':', 1)
    f = os.path.join(root, path + '.json')
    if not os.path.exists(f):
        print('%-30s (not yet)' % name); continue
    R = json.load(open(f))['routes']
    rows = [r for v in R.values() for r in v]
    e = np.array([r['err'] for r in rows])
    used = [r.get('view') for r in rows if r.get('view') is not None]
    tail = ' '.join('%s %.0f' % (k[:4], np.median([r['err'] for r in v])) for k, v in R.items())
    if used:
        tail += ';  first view %d, second %d' % (used.count(0), used.count(1))
    print('%-30s %7.0f %7.0f %4d/%-4d %4d/%-4d  %s' % (name, np.median(e), np.percentile(e, 90), (e < 300).sum(), e.size,
                                                    sum(r['inside95'] for r in rows), e.size, tail))
