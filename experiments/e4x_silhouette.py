#!/usr/bin/env python3
"""E4x: look at what the matcher is actually being fed.

E4v ended with a measurement, not an explanation: on the failing field
sightings the observation fits BETTER somewhere else than at the place
the camera actually stood (rms 3.01 vs 1.60 mrad and so on). That says
the observation disagrees with the DEM at truth, and there are only
two ways that happens — the extracted silhouette is not the terrain
silhouette, or the terrain model is wrong. This draws both so the
difference is visible rather than inferred.

Per sighting, three panels:

  1  the photograph with the extracted skyline drawn on it, plus the
     sea-horizon line the E4q-2 detector finds. Where the two coincide,
     the "skyline" being matched is the sea horizon — which looks
     identical from every position and therefore carries no position
     information, while the DEM may well predict visible terrain there.
  2  observed elevation profile against the DEM profile AT THE TRUE
     POSITION, over the photo's azimuth window
  3  the same against the DEM profile at the position the solver chose

Run:  python3 e4x_silhouette.py ID [ID ...]   (writes out/e4x/<id>.png)
"""

import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extract
import skyline as S
import skyfix as SF

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'e4x')
PHOTOS = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                      'celestial-navigation', 'theodolite')
INDEX = os.path.join(HERE, 'out', 'theodolite', 'index.json')
RESULTS = os.path.join(HERE, 'out', 'e4v_results_all.json')
DEM = os.path.expanduser('~/.horizonator/DEMs_SRTM3')


def panels(sid, sight, res):
    e, a = sight['exif'], sight['attitude']
    img = extract.load_image(os.path.join(PHOTOS, sight['raw']))
    H, W, _ = img.shape
    f_px = (W / 2) / np.tan(np.radians(a['fov_deg']) / 2)
    z = max(e.get('alt_m') or 5.0, 2.0)

    rows, conf = extract.skyline_seam(img)
    dip = S.horizon_dip_rad(z)
    lvl = extract.sea_horizon_attitude_radon(img, f_px, dip)
    pitch = lvl['pitch_deg'] if lvl else (a.get('pitch_deg') or 0.0)
    roll = lvl['roll_deg'] if lvl else (a.get('roll_deg') or 0.0)

    el_obs, w, diag = SF.observation(img, a['fov_deg'], a['heading_deg'],
                                     roll, pitch)
    m = w > 0
    az = SF.AZ[m]

    cm = S.CMarcher(DEM, (e['lat'] - 0.6, e['lat'] + 0.6),
                    (e['lon'] - 0.8, e['lon'] + 0.8), d_min=1000.0)
    el_true, _ = cm.skyline(e['lat'], e['lon'], z, SF.AZ)
    fl = fo = None
    if res:
        mlat, mlon = S.meters_per_degree(e['lat'])
        fl = e['lat'] + res['dn'] / mlat
        fo = e['lon'] + res['de'] / mlon
        el_fix, _ = cm.skyline(fl, fo, z, SF.AZ)
    else:
        el_fix = None

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(img)
    ax.plot(np.arange(W), rows, color='#ff3b30', lw=1.2,
            label='extracted skyline (what the matcher is fed)')
    if lvl:
        u = np.arange(W) - (W - 1) / 2
        v = (np.tan(-dip - np.radians(pitch)) * np.hypot(u, f_px)
             - np.radians(roll) * u)
        ax.plot(np.arange(W), (H - 1) / 2 - v, color='#34c759', lw=1.2,
                ls='--', label='sea horizon (E4q-2 detector)')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='lower left', fontsize=8, framealpha=0.85)
    ax.set_title(f"{sid}   fov {a['fov_deg']:.1f} deg   heading "
                 f"{a['heading_deg']:.1f} deg   z {z:.1f} m"
                 + (f"   fix error {res['err_m']:.0f} m" if res else ''),
                 fontsize=10)

    for k, (elc, lab, col) in enumerate((
            (el_true, 'DEM at the TRUE position', '#0a84ff'),
            (el_fix, 'DEM at the fix the solver chose', '#ff9f0a'))):
        axp = fig.add_subplot(2, 2, 3 + k)
        axp.plot(az, el_obs[m] * 1e3, color='#ff3b30', lw=1.0,
                 label='observed')
        if elc is not None:
            axp.plot(az, elc[m] * 1e3, color=col, lw=1.0, label=lab)
        axp.axhline(-dip * 1e3, color='#34c759', ls='--', lw=0.9,
                    label='sea horizon (-dip)')
        axp.set_xlabel('azimuth (deg true)')
        if k == 0:
            axp.set_ylabel('elevation (mrad)')
        axp.legend(fontsize=7)
        axp.grid(alpha=0.25)
        if elc is not None:
            r = (el_obs[m] - elc[m]) * 1e3
            axp.set_title(f'rms {np.sqrt((r ** 2).mean()):.2f} mrad',
                          fontsize=9)

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'{sid}.png')
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)

    # how much of the observation is just the sea horizon?
    sea = np.abs(el_obs[m] + dip) * 1e3 < 1.0
    print(f'{sid}: {100 * sea.mean():.0f}% of the matched azimuths sit '
          f'within 1 mrad of the sea horizon; DEM predicts terrain above '
          f'the horizon in {100 * (el_true[m] > -dip + 1e-3).mean():.0f}%'
          f' of them -> {p}')
    return p


if __name__ == '__main__':
    with open(INDEX) as fh:
        idx = {s['id']: s for s in json.load(fh)['sightings']}
    res = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            res = {r['id']: r for r in json.load(fh)}
    ids = sys.argv[1:] or ['LPFA0425', 'SRYK4301', 'MYQR7719']
    for sid in ids:
        if sid not in idx:
            print(f'{sid}: not in the curated index')
            continue
        panels(sid, idx[sid], res.get(sid))
