"""Summarise particle-filter result files (pf_track.py) side by side."""
import json, os, sys
import numpy as np
d = sys.argv[1]
print('%-12s %7s %7s %10s %10s %8s %s' % ('run', 'median', 'p90', '<300 m', 'in 95 %', 'bias err', 'per route (median m)'))
for name in sys.argv[2:]:
    f = os.path.join(d, name + '.json')
    if not os.path.exists(f):
        print('%-12s (not yet)' % name); continue
    R = json.load(open(f))['routes']
    rows = [r for v in R.values() for r in v]
    e = np.array([r['err'] for r in rows])
    be = [abs(r['bias'] - r['compass_err']) for r in rows if r.get('bias') is not None]
    print('%-12s %7.0f %7.0f %4d/%-5d %4d/%-5d %8.1f %s' % (name, np.median(e), np.percentile(e, 90), (e < 300).sum(), e.size,
          sum(r['inside95'] for r in rows), e.size, np.median(be) if any(r['bias'] for r in rows) else float('nan'),
          ' '.join('%s %.0f' % (k[:4], np.median([r['err'] for r in v])) for k, v in R.items())))
