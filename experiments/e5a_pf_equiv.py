#!/usr/bin/env python3
"""PeakFinder-equivalent panels for the Theodolite sightings.

peakfinder.com is not reachable from this container, so instead of its
screenshots this renders the same viewpoint from our own DEM — the
photograph on top, the synthesised view below, terrain shaded by
range so the depth structure PeakFinder colours is visible — and
prints the PeakFinder URL for the identical viewpoint so the two can
be compared on a machine that can reach the service.

Viewpoint parameters come from the sighting's own EXIF: position and
altitude from GPS, azimuth from GPSImgDirection, field of view from
the 35 mm equivalent focal length.

Run:  python3 e5a_pf_equiv.py ID [ID ...]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import skyline as S, extract

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'e5a')
PHOTOS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      'celestial-navigation', 'theodolite')
DEM = os.path.expanduser('~/.horizonator/DEMs_SRTM1')


def pf_url(lat, lon, ele, azi, fov, name=''):
    q = (f'https://www.peakfinder.com/?lat={lat:.5f}&lng={lon:.5f}'
         f'&ele={int(round(ele))}&azi={azi:.1f}&fov={int(round(fov))}')
    return q + (f'&name={name}' if name else '')


def panel(sid, s):
    e, a = s['exif'], s['attitude']
    lat, lon = e['lat'], e['lon']
    z = max(e.get('alt_m') or 5.0, 2.0)
    fov, hdg = a['fov_deg'], a['heading_deg']
    img = extract.load_image(os.path.join(PHOTOS, s['raw']))

    az = hdg + np.linspace(-fov / 2, fov / 2, 600)
    cm = S.CMarcher(DEM, (lat - .6, lat + .6), (lon - .8, lon + .8),
                    d_min=1000.)
    el, rng = cm.skyline(lat, lon, z, az)
    # every visible crest, not just the highest: what PeakFinder draws
    # is the depth structure, and a single silhouette curve hides the
    # island that stands in front of the far coast
    dem_obj = S.Dem(DEM)
    lay_el, lay_r = S.visible_layers(dem_obj, lat, lon, z, az,
                                     n_layers=4)
    dip = S.horizon_dip_rad(z)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                             gridspec_kw=dict(height_ratios=[1, 1]))
    axes[0].imshow(img)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].set_title(f'{sid} — photograph   ({lat:.4f}, {lon:.4f})  '
                      f'{z:.0f} m  az {hdg:.1f} deg  fov {fov:.1f} deg',
                      fontsize=10)

    ax = axes[1]
    good = np.isfinite(el)
    norm = Normalize(0, max(np.nanmax(rng[good]) / 1000.0, 1))
    cmap = plt.get_cmap('viridis_r')
    lo = min(np.nanmin(el[good]) * 1e3, -dip * 1e3) - 2
    # far layers first, near ones painted over them
    for li in range(lay_el.shape[0] - 1, -1, -1):
        ee, rr = lay_el[li], lay_r[li]
        m = np.isfinite(ee)
        if not m.any():
            continue
        for k in range(az.size - 1):
            if not (m[k] and m[k + 1]):
                continue
            ax.fill_between(az[k:k + 2], lo, ee[k:k + 2] * 1e3,
                            color=cmap(norm(rr[k] / 1000.0)), lw=0)
        ax.plot(az[m], ee[m] * 1e3, color='#1c1c1e', lw=0.7)
        med = np.nanmedian(rr) / 1000.0
        j = int(np.nanargmax(np.where(m, ee, -np.inf)))
        ax.annotate(f'{med:.0f} km', (az[j], ee[j] * 1e3),
                    textcoords='offset points', xytext=(0, 4),
                    fontsize=7, ha='center', color='#1c1c1e')
    ax.axhline(-dip * 1e3, color='#0a84ff', ls='--', lw=1,
               label='sea horizon')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, pad=0.01)
    cb.set_label('range to the terrain (km)', fontsize=8)
    ax.set_xlim(az[0], az[-1])
    ax.set_ylim(lo, max(np.nanmax(el[good]) * 1e3 + 3, 5))
    ax.set_xlabel('azimuth (deg true)')
    ax.set_ylabel('elevation (mrad)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('synthesised from the DEM at the same viewpoint, '
                 'shaded by range', fontsize=10)

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'{sid}_pf.png')
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    print(f'{sid}: {p}')
    print(f'   PeakFinder: {pf_url(lat, lon, z, hdg, fov, sid)}')
    return p


if __name__ == '__main__':
    idx = {x['id']: x for x in json.load(open(os.path.join(
        HERE, 'out', 'theodolite', 'index.json')))['sightings']}
    for sid in (sys.argv[1:] or ['OREJ1026', 'KWHC9160', 'RSRW2787']):
        if sid in idx:
            panel(sid, idx[sid])
        else:
            print(f'{sid}: not in the curated index')
