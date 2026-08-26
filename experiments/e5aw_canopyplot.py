#!/usr/bin/env python3
"""E5aw: does a per-pixel canopy raster explain the crest deficit?"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

la, lo, hd = 37.05190, 28.32292, 196.5
DHS = [0, 3, 6, 9, 12]


def err(f, la, lo, hd):
    j = json.load(open(f))
    dn = (j['lat'] - la) * 111132.
    de = (j['lon'] - lo) * 111320. * np.cos(np.radians(la))
    h = np.radians(hd)
    return np.hypot(dn, de), dn * np.cos(h) + de * np.sin(h)


fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for tag, col, nm in (('base', '#c0563a', 'bare DEM (SRTM1, water-masked)'),
                     ('can', '#2f7d4f', '+ Meta 1 m canopy, per-post max')):
    tot = [err(f'out/canopy/{tag}_dh{d}.json', la, lo, hd)[0] for d in DHS]
    alo = [err(f'out/canopy/{tag}_dh{d}.json', la, lo, hd)[1] for d in DHS]
    ax[0].plot(DHS, tot, 'o-', color=col, lw=2, ms=6, label=nm)
    ax[1].plot(DHS, alo, 'o-', color=col, lw=2, ms=6, label=nm)
ax[0].set_ylabel('total position error (m)')
ax[1].set_ylabel('along-sight error (m)')
ax[1].axhline(0, color='#bbb', lw=0.8, zorder=0)
for a in ax:
    a.set_xlabel('--crest-dh (m)')
    a.grid(alpha=0.25)
    a.legend(fontsize=9)
ax[0].set_title('Akyaka clean: canopy alone removes 105 m at dh = 0')
ax[1].set_title('and halves the along-sight bias the deficit produces')
fig.suptitle('E5aw — the crest deficit is about half canopy, half 30 m cell clipping',
             fontsize=11)
fig.tight_layout()
fig.savefig('out/canopy/canopy_test.png', dpi=115)

print('confirmation across three frames, along-sight bias at dh=0:')
for tag, (a, b, c) in (('AKc', (la, lo, hd)),
                       ('AK3', (37.04859, 28.29302, 172.6)),
                       ('AK2', (37.04860, 28.29299, 177.0))):
    p = 'out/canopy/' + ('base_dh0.json' if tag == 'AKc' else f'{tag}_base_dh0.json')
    q = 'out/canopy/' + ('can_dh0.json' if tag == 'AKc' else f'{tag}_can_dh0.json')
    t0, a0 = err(p, a, b, c)
    t1, a1 = err(q, a, b, c)
    print(f'  {tag}: {t0:.0f} -> {t1:.0f} m   along {a0:+.0f} -> {a1:+.0f} '
          f'({100*(a1-a0)/a0:+.0f}%)')
print('-> out/canopy/canopy_test.png')
