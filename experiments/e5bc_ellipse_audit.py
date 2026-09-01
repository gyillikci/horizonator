#!/usr/bin/env python3
"""E5bc: audit the reported covariance against the actual errors, and
group position error by the attitude source that sets the beta band.

Backs the two measured claims in doc/angle-only-navigation-review.md:
  - sigma_major is optimistic (median err/sigma ~1.5-1.7) and its
    orientation carries no information about the true error direction;
  - median position error falls with the beta band / with how often
    beta is pinned at the band edge.

Reads every solve JSON under out/, de-duplicating repeated solves by
(dn, de, sigma, heading).
"""

import glob
import json
import math
import statistics as st
from collections import defaultdict


def _sep(a, b):
    """Smallest angle between two undirected directions, degrees."""
    d = abs(a % 180.0 - b % 180.0) % 180.0
    return min(d, 180.0 - d)


def load(pattern='out/*/*.json'):
    ell, bysrc = {}, defaultdict(dict)
    for f in sorted(glob.glob(pattern)):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if not isinstance(j, dict) or 'dn_m' not in j:
            continue
        ph = j.get('photos') or []
        if not ph:
            continue
        p = ph[0]
        dn, de = j.get('dn_m'), j.get('de_m')
        if dn is None or de is None:
            continue
        tot = math.hypot(dn, de)
        if tot < 1:
            continue
        src, beta = p.get('attitude_source'), p.get('el_offset_mrad')
        if src is not None:
            bysrc[src][(round(dn), round(de), src)] = (
                tot, abs(beta) if beta is not None else None)
        sm, an, mb, hd = (j.get('sigma_major_m'), j.get('anisotropy'),
                          j.get('major_bearing_deg'), p.get('heading_deg'))
        if any(v is None or not math.isfinite(v) for v in (sm, an, mb, hd)):
            continue
        if an <= 1.15:                     # a round ellipse has no direction
            continue
        ell[(round(dn), round(de), round(sm), round(hd, 1))] = (
            tot, sm, _sep(mb, math.degrees(math.atan2(de, dn))),
            j.get('status'))
    return ell, bysrc


def main():
    ell, bysrc = load()
    v = list(ell.values())
    acc = [x for x in v if x[3] == 'ok']
    for lbl, s in (('all', v), ('accepted', acc)):
        print('%-9s n=%3d  err/sigma median %.2f  '
              '|major axis - true error| median %2.0f deg '
              '(random 45), within 30 deg %d/%d'
              % (lbl, len(s), st.median(x[0] / x[1] for x in s),
                 st.median(x[2] for x in s),
                 sum(x[2] < 30 for x in s), len(s)))
    print()
    half = {'horizon': 2, 'radon': 2, 'waterline': 5, 'prior': 10}
    for src in sorted(bysrc):
        vals = list(bysrc[src].values())
        h = half.get(src)
        pinned = sum(1 for _, b in vals
                     if b is not None and h and b >= 0.95 * h)
        print('%-11s n=%3d  median err %5.0f m  band +-%-4s  '
              'beta at band edge %d/%d'
              % (src, len(vals), st.median(x[0] for x in vals),
                 h if h else '?', pinned, len(vals)))


if __name__ == '__main__':
    main()
