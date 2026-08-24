#!/usr/bin/env python3
"""Blind-protocol harness: is the solver really centre-independent?

The field validation so far centred the search box on the GPS truth,
which leaves one attack open: a solver secretly biased toward the
box centre would score well without measuring anything. This harness
closes it with a SEPARATE PROCESS holding the secret:

  make    draws N random box centres offset 500-2000 m from the
          truth (OS-entropy seed, recorded), writes centers.txt for
          the solver — truth does NOT appear in that file — and
          seals truth+offsets+seed in sealed.json, which the solve
          step never reads
  score   after the solves, opens the seal and reports each fix
          against the truth (dlat/dlon), plus the spread of the
          ABSOLUTE fixed positions across trials — the number that
          proves centre-independence: if the fix follows the box
          centre, the spread matches the 1.5 km centre scatter; if
          the solver measures terrain, the fixes cluster regardless
          of where the box was placed

Run:
  python3 blindbox.py make  --truth LAT,LON --n 8 --dir out/blind
  ... solve each line of centers.txt with skyfix --center <line> ...
  python3 blindbox.py score --dir out/blind --fixes fixes.json
"""

import os
import sys
import json
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['make', 'score'])
    ap.add_argument('--truth', default=None)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--min-off', type=float, default=500.0)
    ap.add_argument('--max-off', type=float, default=2000.0)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--fixes', default=None)
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    if args.cmd == 'make':
        la, lo = [float(x) for x in args.truth.split(',')]
        mlat = 111132.0
        mlon = 111320.0 * np.cos(np.radians(la))
        seed = int.from_bytes(os.urandom(8), 'big')
        rs = np.random.default_rng(seed)
        r = rs.uniform(args.min_off, args.max_off, args.n)
        th = rs.uniform(0, 2 * np.pi, args.n)
        centers = [(la + r[k] * np.cos(th[k]) / mlat,
                    lo + r[k] * np.sin(th[k]) / mlon)
                   for k in range(args.n)]
        with open(os.path.join(args.dir, 'centers.txt'), 'w') as f:
            for c in centers:
                f.write(f'{c[0]:.6f},{c[1]:.6f}\n')
        with open(os.path.join(args.dir, 'sealed.json'), 'w') as f:
            json.dump(dict(truth=[la, lo], seed=seed,
                           offsets_m=[[float(r[k] * np.cos(th[k])),
                                       float(r[k] * np.sin(th[k]))]
                                      for k in range(args.n)]), f)
        print(f'{args.n} blind centers written to '
              f'{args.dir}/centers.txt (offsets {args.min_off:.0f}-'
              f'{args.max_off:.0f} m, seed sealed)')
        return

    with open(os.path.join(args.dir, 'sealed.json')) as f:
        seal = json.load(f)
    la, lo = seal['truth']
    mlat = 111132.0
    mlon = 111320.0 * np.cos(np.radians(la))
    with open(args.fixes) as f:
        fixes = json.load(f)
    print(f"seed was {seal['seed']}")
    good = []
    for k, fx in enumerate(fixes):
        off = seal['offsets_m'][k]
        if fx is None:
            print(f'trial {k}: solver refused/failed '
                  f'(centre offset dlat {off[0]:+.0f} dlon {off[1]:+.0f})')
            continue
        dn = (fx['lat'] - la) * mlat
        de = (fx['lon'] - lo) * mlon
        cn = (fx['lat'] - float(fx['center_lat'])) * mlat
        ce = (fx['lon'] - float(fx['center_lon'])) * mlon
        good.append((dn, de))
        print(f'trial {k}: centre offset dlat {off[0]:+5.0f} dlon '
              f'{off[1]:+5.0f} m | fix error dlat {dn:+5.0f} dlon '
              f'{de:+5.0f} m | fix-from-centre {np.hypot(cn, ce):4.0f} m'
              f' | ok={fx.get("fix_ok")}')
    if len(good) >= 2:
        g = np.array(good)
        sp = g - g.mean(0)
        print(f'\nabsolute-fix scatter across trials: sigma_lat '
              f'{sp[:,0].std():.0f} m, sigma_lon {sp[:,1].std():.0f} m '
              f'(centre scatter was ~{np.std([o[0] for o in seal["offsets_m"]]):.0f}/'
              f'{np.std([o[1] for o in seal["offsets_m"]]):.0f} m)')
        print(f'mean fix error: dlat {g[:,0].mean():+.0f} m, '
              f'dlon {g[:,1].mean():+.0f} m')


if __name__ == '__main__':
    main()
