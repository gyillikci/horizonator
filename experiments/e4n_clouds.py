#!/usr/bin/env python3
"""E4n: clouds vs the extractor and the trust gates.

E2 named clouds the dominant environmental risk, and the front-end has
never been stress-tested against them. This paints three degradations
onto the E4c sea composites, at increasing severity, and runs the full
skyfix pipeline (attitude priors, default gates) on each:

  stratus f   an overcast deck whose base hides the top fraction f of
              the skyline's relief span (the extractor then traces the
              CLOUD BASE as if it were terrain — the dangerous case)
  cumulus n   n bright blobs sitting on / breaking the ridge line
  haze a      contrast washed toward the sky color by factor a

The question is not whether the fix survives (past some severity it
cannot) but WHICH failure mode appears: a wrong fix that passes the
gates, or an honest INCONCLUSIVE. Writes out/e4n_results.json.

Run:   python3 e4n_clouds.py     (no GL needed; reuses out/synth/*.jpg)
"""

import os
import sys
import json
import subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
import skyline as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
IMGDIR = os.path.join(OUT, 'synth')
CLDDIR = os.path.join(OUT, 'clouds')
os.makedirs(CLDDIR, exist_ok=True)
DIR3 = os.environ.get('HORIZONATOR_DEMS',
                      os.path.expanduser('~/.horizonator/DEMs_SRTM3'))
AZ = np.arange(-180.0, 180.0, 0.1) + 0.05
F35 = 24
FOV = np.degrees(2 * np.arctan(0.8 * 21.63 / F35))

CASES = [  # (name, lat, lon, z, heading, pitch, roll) — E4c sea cases
    ('strait1', 36.9500, 27.2500, 5.0, 180.0, 0.5, 0.0),
    ('strait2', 36.9622, 27.2384, 5.0, 25.0, -0.5, 0.5),
    ('offshore', 36.6050, 26.8590, 5.0, 60.0, 0.0, 0.0),
]
rng = np.random.default_rng(20260823)
CLOUD = np.array([0.82, 0.83, 0.86])


def el_to_row(el_deg, pitch, H, W):
    f = (W / 2) / np.tan(np.radians(FOV) / 2)
    return (H - 1) / 2 - np.tan(np.radians(el_deg - pitch)) * f


def stratus(img, base_row):
    H, W, _ = img.shape
    out = img.copy()
    rr = np.arange(H)[:, None]
    edge = base_row + 6.0 * np.sin(np.arange(W) / 37.0) \
        + np.cumsum(rng.normal(0, 0.4, W))
    m = rr < edge[None, :]
    tex = 0.05 * rng.standard_normal((H, W, 1))
    out[m] = (CLOUD[None, :] + tex[m]).clip(0, 1)
    return out


def cumulus(img, n, skyline_row):
    H, W, _ = img.shape
    out = img.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(n):
        cx = rng.uniform(0.05, 0.95) * W
        cy = skyline_row[int(cx)] + rng.uniform(-30, 8)
        a, b = rng.uniform(40, 110), rng.uniform(14, 30)
        m = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 < 1
        out[m] = (CLOUD * rng.uniform(0.95, 1.1)).clip(0, 1)
    return out


def haze(img, alpha, sky_probe_row=30):
    sky = img[sky_probe_row].mean(axis=0)
    return (img * (1 - alpha) + sky[None, None] * alpha).clip(0, 1)


def solve(path, case, center):
    name, lat, lon, z, heading, pitch, roll = case
    p = subprocess.run(
        [sys.executable, os.path.join(HERE, 'skyfix.py'), path,
         '--center', center, '--box', '5000', '--fov', str(FOV),
         '--heading', str(heading), '--z', str(z),
         '--pitch', str(pitch), '--roll', str(roll), '--dmin', '1000'],
        capture_output=True, text=True)
    if p.returncode not in (0, 2):
        return dict(status='crash', err_m=-1.0, margin=0.0, rms=0.0,
                    reasons=[p.stderr[-200:]])
    t = p.stdout
    j = json.loads(t[t.index('{'):t.rindex('}') + 1])
    mlat, mlon = S.meters_per_degree(lat)
    err = float(np.hypot((j['lat'] - lat) * mlat, (j['lon'] - lon) * mlon))
    return dict(status=j['status'], err_m=err,
                margin=j['basin_margin'], rms=j['rms_mrad'],
                reasons=j['reasons'])


if __name__ == '__main__':
    cm = {}
    results = {}
    for case in CASES:
        name, lat, lon, z, heading, pitch, roll = case
        img = np.asarray(Image.open(os.path.join(IMGDIR, name + '.jpg'))
                         .convert('RGB'), dtype=np.float32) / 255.0
        H, W, _ = img.shape
        if name not in cm:
            cm[name] = S.CMarcher(DIR3, (lat - .6, lat + .6),
                                  (lon - .8, lon + .8), d_min=1000.)
        el, _ = cm[name].skyline(lat, lon, z, AZ)
        rel = (AZ - heading + 180.0) % 360.0 - 180.0
        infov = np.abs(rel) < FOV / 2
        el_deg = np.degrees(el[infov])
        lo, hi = el_deg.min(), el_deg.max()
        # skyline row per column for cumulus placement
        f = (W / 2) / np.tan(np.radians(FOV) / 2)
        u = (np.arange(W) - (W - 1) / 2)
        azc = heading + np.degrees(np.arctan2(u, f))
        relc = (azc - heading + 180.0) % 360.0 - 180.0
        eli = np.interp((relc + heading - heading), rel[infov], el_deg)
        sk_row = el_to_row(eli, pitch, H, W)

        off = rng.uniform(-1200, 1200, 2)
        mlat, mlon = S.meters_per_degree(lat)
        center = f'{lat + off[0] / mlat:.6f},{lon + off[1] / mlon:.6f}'

        variants = [('clean', img)]
        for fclip in (0.25, 0.5, 0.75):
            base_el = hi - fclip * (hi - lo)
            variants.append((f'stratus{fclip}',
                             stratus(img, el_to_row(base_el, pitch, H, W))))
        for n in (4, 10, 20):
            variants.append((f'cumulus{n}', cumulus(img, n, sk_row)))
        for a in (0.5, 0.75):
            variants.append((f'haze{a}', haze(img, a)))

        results[name] = {}
        for vname, vimg in variants:
            path = os.path.join(CLDDIR, f'{name}_{vname}.jpg')
            Image.fromarray((vimg * 255).astype(np.uint8)).save(
                path, quality=92)
            r = solve(path, case, center)
            results[name][vname] = r
            print(f"{name:9s} {vname:12s}: {r['status']:12s} "
                  f"err {r['err_m']:6.0f} m  margin {r.get('margin', 0):6.2f}"
                  f"  rms {r.get('rms', 0):5.1f}", flush=True)
    with open(os.path.join(OUT, 'e4n_results.json'), 'w') as fj:
        json.dump(results, fj, indent=1)

    wrong_ok = sum(1 for c in results.values() for v in c.values()
                   if v['status'] == 'ok' and (v['err_m'] or 0) >= 500)
    incon = sum(1 for c in results.values() for v in c.values()
                if v['status'] == 'inconclusive')
    good = sum(1 for c in results.values() for v in c.values()
               if v['status'] == 'ok' and (v['err_m'] or 1e9) < 500)
    print(f'\n{good} good fixes, {incon} honest inconclusives, '
          f'{wrong_ok} WRONG fixes passing the gates '
          f'(of {sum(len(c) for c in results.values())} runs)')
