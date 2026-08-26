import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import extract, skyfix

photo, out = sys.argv[1], sys.argv[2]
img = extract.load_image(photo)
H, W = img.shape[:2]
skyfix.EXTRACTOR = 'ewasr'
rows, conf = skyfix.extract_boundary(img)
cls = skyfix._EWASR.predict(img)

sky = cls == 2; water = cls == 1; land = cls == 0
# per column: top of the water region, and bottom of the land region
wtop = np.full(W, np.nan); lbot = np.full(W, np.nan)
for x in range(W):
    w = np.where(water[:, x])[0]
    l = np.where(land[:, x])[0]
    if w.size: wtop[x] = w[0]
    if l.size: lbot[x] = l[-1]
print(f'su sınıfı hiç yok: {int(np.isnan(wtop).sum())}/{W} sütun')
print(f'su tepesi (satır): p10 {np.nanpercentile(wtop,10):.0f} '
      f'p50 {np.nanpercentile(wtop,50):.0f} p90 {np.nanpercentile(wtop,90):.0f}  (H={H})')
print(f'kara tabanı      : p10 {np.nanpercentile(lbot,10):.0f} '
      f'p50 {np.nanpercentile(lbot,50):.0f} p90 {np.nanpercentile(lbot,90):.0f}')
gap = lbot - wtop
print(f'kara tabanı - su tepesi (örtüşme, + = kara suyun içine taşmış): '
      f'medyan {np.nanmedian(gap):+.0f} px, p90 {np.nanpercentile(gap,90):+.0f} px')
print(f'silüet satırı    : p50 {np.median(np.asarray(rows,float)):.0f}')

fig, ax = plt.subplots(figsize=(15, 15*H/W + 1))
ax.imshow(img)
ax.imshow(cls, alpha=0.30, interpolation='nearest',
          cmap=ListedColormap(['#d94f2b', '#2f7fd9', '#f2e34a']))
x = np.arange(W)
ax.plot(x, np.asarray(rows, float), '-', lw=1.6, color='#39ff14',
        label='sky/terrain boundary (used)')
ax.plot(x, wtop, '-', lw=1.6, color='#00e5ff', label='top of WATER class')
ax.plot(x, lbot, '--', lw=1.2, color='#ff2d55', label='bottom of LAND class')
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
ax.legend(loc='lower left', fontsize=9, framealpha=0.85)
ax.set_title('eWaSR: where does it put the waterline?', fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=115)
print('->', out)
