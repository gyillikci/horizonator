#!/usr/bin/env python3
"""E5ao: the crest-deficit sweep on the Theodolite material.

E5al reproduced dh = 9 on the Marmaris/Akyaka coast. This asks the
same question of a different, independently attitude-referenced set,
to test the standing claim that the crest deficit is REGIONAL.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S = {s['id']: s for s in json.load(open('out/theodolite/index.json'))['sightings']}
IDS = [('HATY3309', '#2a6fb0'), ('XKYU3498', '#c0563a'),
       ('LHCG7767', '#4a8c5c'), ('PQBC6867', '#8a6bbf'),
       ('YBPL9738', '#c8912a')]
DHS = [0, 4, 8, 12, 16]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
means, means_ex = [], []
for dh in DHS:
    t = []
    for sid, _ in IDS:
        j = json.load(open(f'out/theo3/{sid}_dh{dh}.json'))
        e = S[sid]['exif']
        dn = (j['lat']-e['lat'])*111132.
        de = (j['lon']-e['lon'])*111320.*np.cos(np.radians(e['lat']))
        t.append(np.hypot(dn, de))
    means.append(np.mean(t))
    means_ex.append(np.mean([x for x, (s, _) in zip(t, IDS) if s != 'PQBC6867']))

for key, ax, lab in ((0, axes[0], 'total error (m)'),
                     (1, axes[1], 'along-sight (m)')):
    for sid, col in IDS:
        ys = []
        for dh in DHS:
            j = json.load(open(f'out/theo3/{sid}_dh{dh}.json'))
            e = S[sid]['exif']
            dn = (j['lat']-e['lat'])*111132.
            de = (j['lon']-e['lon'])*111320.*np.cos(np.radians(e['lat']))
            h = np.radians(e['heading_deg'])
            ys.append(np.hypot(dn, de) if key == 0
                      else dn*np.cos(h)+de*np.sin(h))
        ax.plot(DHS, ys, 'o-', color=col, lw=1.7, ms=5, label=sid)
    ax.axvline(9, color='#888', ls='--', lw=1, zorder=0)
    ax.set_xlabel('--crest-dh (m)')
    ax.set_ylabel(lab)
    ax.grid(alpha=0.25)
    if key == 1:
        ax.axhline(0, color='#bbb', lw=0.8, zorder=0)
axes[0].plot(DHS, means, 'k-', lw=2.6, label='mean (all 5)')
axes[0].plot(DHS, means_ex, 'k--', lw=1.6, label='mean (без PQBC)'.replace('без', 'without'))
axes[0].legend(fontsize=8)
axes[0].set_title('Theodolite set: the optimum sits near dh = 12, not 9')
axes[1].set_title('along-sight bias crosses zero near dh = 8-12')
fig.suptitle('E5ao — crest deficit is regional: Marmaris coast wants 9, '
             'this set wants ~12 (dashed line: dh = 9)', fontsize=11)
fig.tight_layout()
fig.savefig('out/theo3/dh_curve.png', dpi=115)
print('mean all5 :', [round(x) for x in means])
print('mean ex-PQ:', [round(x) for x in means_ex])
print('-> out/theo3/dh_curve.png')
