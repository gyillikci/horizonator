#!/usr/bin/env python3
"""E4g: sea-horizon auto-levelling A/B on the E4c sea cases.

The Grelsson-style levelling stage implemented in closed form
(extract.sea_horizon_attitude): the sea horizon's dip below level is
exactly known from the camera height, so the horizon line in the image is
a drift-free pitch/roll reference. Per E4c sea case, skyfix runs twice on
the same composite JPEG and the same search box:

  A (baseline)    --pitch/--roll from the case table (perfect IMU prior),
                  elevation-offset window +-10 mrad  == the E4c setup
  B (auto-level)  NO attitude prior given; pitch/roll estimated from the
                  sea horizon, offset window tightened to +-2 mrad

Reuses the composites in out/synth/ (from e4c_synth.py). Results to
out/e4g_results.json.

Run:   python3 e4g_autolevel.py     (no GL needed if out/synth/ exists)
"""

import os
import sys
import json
import subprocess
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skyline as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
IMGDIR = os.path.join(OUT, 'synth')
BOX = 5000.0

# the four sea cases of e4c_synth.CASES (name, lat, lon, z, heading,
# pitch, roll); the land cases are out of scope for horizon levelling
CASES = [
    ('strait1',  36.9500, 27.2500,  5.0, 180.0,  0.5,  0.0),
    ('strait2',  36.9622, 27.2384,  5.0,  25.0, -0.5,  0.5),
    ('strait3',  36.9411, 27.2661,  5.0, 300.0,  1.0, -0.5),
    ('offshore', 36.6050, 26.8590,  5.0,  60.0,  0.0,  0.0),
]

rng = np.random.default_rng(20260817)


def run_case(name, lat, lon, pitch, roll, center, variant):
    args = [sys.executable, os.path.join(HERE, 'skyfix.py'),
            os.path.join(IMGDIR, name + '.jpg'),
            '--center', center, '--box', str(BOX), '--dmin', '1000',
            '--out', os.path.join(IMGDIR, f'{name}_{variant}')]
    if variant == 'prior':
        args += ['--pitch', str(pitch), '--roll', str(roll)]
    else:
        # these are SYNTHETIC composites whose sea is painted darker
        # than physics allows at the horizon (brightness step 0.17-0.30
        # vs 0.03 measured on real maritime photos in E4q), so the
        # continuity water check needs a widened window here — real
        # imagery uses the 0.20 default
        args += ['--auto-level', '--max-step', '0.35']
    t0 = time.time()
    p = subprocess.run(args, capture_output=True, text=True)
    dt = time.time() - t0
    if p.returncode not in (0, 2):
        print(f'{name}/{variant} FAILED:', p.stderr[-400:])
        return None
    R = json.load(open(os.path.join(IMGDIR, f'{name}_{variant}.json')))
    mlat, mlon = S.meters_per_degree(lat)
    R['err_m'] = float(np.hypot((R['lat'] - lat) * mlat,
                                (R['lon'] - lon) * mlon))
    R['t_s'] = dt
    R['stdout'] = p.stdout
    return R


results = []
for name, lat, lon, z, heading, pitch, roll in CASES:
    if not os.path.exists(os.path.join(IMGDIR, name + '.jpg')):
        sys.exit(f'{name}.jpg missing: run  xvfb-run -a python3 '
                 f'e4c_synth.py  first')
    mlat, mlon = S.meters_per_degree(lat)
    off = rng.uniform(-1500, 1500, 2)
    center = f'{lat + off[0] / mlat:.6f},{lon + off[1] / mlon:.6f}'
    A = run_case(name, lat, lon, pitch, roll, center, 'prior')
    B = run_case(name, lat, lon, pitch, roll, center, 'auto')
    if not (A and B):
        continue
    al = B.get('auto_level')
    att = B.get('attitude')
    if att is None:
        # no sea horizon found: the levelling stage declined this scene
        # (land-dominated composites legitimately have none). Record the
        # refusal instead of crashing — availability is a result too.
        print(f'{name}: auto-level found no sea horizon, skipping')
        results.append(dict(name=name, truth=dict(pitch=pitch, roll=roll),
                            no_horizon=True,
                            prior=dict(err_m=A['err_m'],
                                       rms_mrad=A['rms_mrad'])))
        continue
    dp = att['pitch_deg'] - pitch
    dr = att['roll_deg'] - roll
    results.append(dict(
        name=name, truth=dict(pitch=pitch, roll=roll),
        prior=dict(err_m=A['err_m'], rms_mrad=A['rms_mrad'],
                   sigma=[A['sigma_n_m'], A['sigma_e_m']],
                   el_off_mrad=A['el_offset_mrad'], status=A['status'],
                   t_s=A['t_s']),
        auto=dict(err_m=B['err_m'], rms_mrad=B['rms_mrad'],
                  sigma=[B['sigma_n_m'], B['sigma_e_m']],
                  el_off_mrad=B['el_offset_mrad'], status=B['status'],
                  t_s=B['t_s'], source=att['source'],
                  pitch_deg=att['pitch_deg'], roll_deg=att['roll_deg'],
                  pitch_err_deg=dp, roll_err_deg=dr,
                  sea_frac=(al or {}).get('frac'),
                  rms_px=(al or {}).get('rms_px'))))
    print(f"{name:9s} A(prior): {A['err_m']:6.1f} m "
          f"sig({A['sigma_n_m']:.0f},{A['sigma_e_m']:.0f})  "
          f"B(auto): {B['err_m']:6.1f} m "
          f"sig({B['sigma_n_m']:.0f},{B['sigma_e_m']:.0f})  "
          f"att src {att['source']}, dpitch {dp:+.3f} deg, "
          f"droll {dr:+.3f} deg", flush=True)

with open(os.path.join(OUT, 'e4g_results.json'), 'w') as f:
    json.dump(results, f, indent=1)
levelled = [r for r in results if not r.get('no_horizon')]
ea = np.array([r['prior']['err_m'] for r in results])
print(f'\n{len(results)}/{len(CASES)} cases, sea horizon found on '
      f'{len(levelled)}: prior median {np.median(ea):.1f} m '
      f'max {ea.max():.1f} m')
if levelled:
    eb = np.array([r['auto']['err_m'] for r in levelled])
    print(f'  auto-level median {np.median(eb):.1f} m '
          f'max {eb.max():.1f} m')
pe = [abs(r['auto']['pitch_err_deg']) for r in levelled
      if r['auto']['source'] == 'sea-horizon']
if pe:
    print(f'sea-horizon attitude recovered on {len(pe)}/{len(results)}: '
          f'|pitch err| max {max(pe) * 1000:.1f} mdeg')
