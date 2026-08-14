#!/usr/bin/env python3
"""E0: validate the (curvature-patched) renderer for a sea-level observer.

Checks, per the study doc (doc/skyline-matching-study.md, section 6):

  1. The apparent sea horizon dips below the horizontal by sqrt(2h/Reff) and
     lies at distance sqrt(2*Reff*h).
  2. The GL renderer agrees with an independent NumPy ray-marcher that
     implements the same curvature+refraction model.
  3. The GL skyline matches the *curved*-earth ray-marcher much better than a
     flat-earth ray-marcher: terrain beyond the horizon is actually hidden.

Run headlessly:   xvfb-run -a python3 e0_validate.py

Requires SRTM3 .hgt tiles N36E026..N37E027 (see README.md) — the test area is
the Bodrum/Kos region of the Aegean.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import skyline as S

DIR_DEMS = os.environ.get('HORIZONATOR_DEMS',
                          os.path.expanduser('~/.horizonator/DEMs_SRTM3'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)

# open water SW of Kos: no land for >40km in the southern sector
SEA = dict(lat=36.55, lon=26.75)
# mid-strait between the Bodrum peninsula and Kos: terrain all around
STRAIT = dict(lat=36.95, lon=27.25)

npass = nfail = 0
def check(name, ok, detail=''):
    global npass, nfail
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    if ok: npass += 1
    else:  nfail += 1


print('== E0.1: sea-horizon dip and distance ==')
dem = S.Dem(DIR_DEMS)
heights = np.array([2.0, 5.0, 10.0, 20.0])

# high-resolution GL view of an open-sea sector (0.01 deg/pixel)
gl_hires = S.GlSkyline(SEA['lat'], SEA['lon'], width=3000, height=600,
                       render_radius_m=45000., dir_dems=DIR_DEMS)
marcher = S.RayMarcher(dem, d_max=40000., d_step=30., d_min=150.)
az_sector = np.arange(185., 215., 0.5)

dip_gl, dip_rm, dist_rm = [], [], []
for h in heights:
    _, el, _ = gl_hires.skyline(SEA['lat'], SEA['lon'], h,
                                az_deg0=185., az_deg1=215.)
    dip_gl.append(-np.nanmedian(el))
    el2, r2 = marcher.skyline(SEA['lat'], SEA['lon'], h, az_sector)
    dip_rm.append(-np.median(el2))
    dist_rm.append(np.median(r2))
dip_gl, dip_rm, dist_rm = map(np.array, (dip_gl, dip_rm, dist_rm))
dip_true = S.horizon_dip_rad(heights)
dist_true = S.horizon_distance_m(heights)

for i, h in enumerate(heights):
    check(f'dip(h={h:.0f}m) GL', abs(dip_gl[i] - dip_true[i]) < 0.35e-3,
          f'measured {dip_gl[i]*1e3:.2f} mrad, analytic {dip_true[i]*1e3:.2f} mrad')
    check(f'dip(h={h:.0f}m) ray-marcher', abs(dip_rm[i] - dip_true[i]) < 0.1e-3,
          f'measured {dip_rm[i]*1e3:.2f} mrad, analytic {dip_true[i]*1e3:.2f} mrad')
    check(f'horizon distance(h={h:.0f}m)', abs(dist_rm[i] - dist_true[i]) < 1500,
          f'measured {dist_rm[i]/1e3:.1f} km, analytic {dist_true[i]/1e3:.1f} km')

print('== E0.2: GL renderer vs independent ray-marcher, terrain skyline ==')
gl = S.GlSkyline(STRAIT['lat'], STRAIT['lon'], width=3600, height=400,
                 render_radius_m=45000., dir_dems=DIR_DEMS)
z = 5.0
az, el_gl, r_gl = gl.skyline(STRAIT['lat'], STRAIT['lon'], z)
el_gl = S.seahorizon_fill(el_gl, z)

el_rm, r_rm = marcher.skyline(STRAIT['lat'], STRAIT['lon'], z, az)
d = el_gl - el_rm
rms = float(np.sqrt(np.mean(d * d)))
med = float(np.median(np.abs(d)))
check('GL vs ray-marcher RMS < 3 mrad', rms < 3e-3, f'RMS {rms*1e3:.2f} mrad')
check('GL vs ray-marcher median < 1.5 mrad', med < 1.5e-3,
      f'median {med*1e3:.2f} mrad')

print('== E0.3: curvature actually applied (curved beats flat) ==')
class FlatMarcher(S.RayMarcher):
    def skyline(self, lat, lon, z, az_deg):
        Reff_save = S.REFF
        try:
            S.REFF = 1e18  # flat earth
            return super().skyline(lat, lon, z, az_deg)
        finally:
            S.REFF = Reff_save

flat = FlatMarcher(dem, d_max=40000., d_step=30., d_min=150.)
el_flat, _ = flat.skyline(STRAIT['lat'], STRAIT['lon'], z, az)
rms_flat = float(np.sqrt(np.mean((el_gl - el_flat) ** 2)))
# evaluate where curvature actually matters: distant terrain, where the flat
# and curved models disagree by > 1 mrad
m = (el_flat - el_rm) > 1e-3
err_curved = float(np.mean(np.abs(el_gl - el_rm)[m]))
err_flat   = float(np.mean(np.abs(el_gl - el_flat)[m]))
check('GL matches curved marcher better than flat (distant terrain)',
      err_curved < 0.5 * err_flat,
      f'on {m.sum()} bins: curved {err_curved*1e3:.2f} mrad '
      f'vs flat {err_flat*1e3:.2f} mrad (all-bin RMS: '
      f'{rms*1e3:.2f} vs {rms_flat*1e3:.2f})')
# curvature magnitude sanity: at ~20km the drop is ~1.4 mrad
check('curved-vs-flat effect is mrad-scale',
      np.max(np.abs(el_flat - el_rm)) > 1e-3,
      f'max effect {np.max(np.abs(el_flat-el_rm))*1e3:.2f} mrad')

print(f'\n{npass} passed, {nfail} failed')

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_GL, C_RM, C_FLAT = '#0072b2', '#d55e00', '#909090'   # Okabe-Ito blue/vermillion
fig, axs = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[1, 1.4])
fig.suptitle('E0: renderer validation, sea-level observer (Bodrum–Kos strait)',
             fontsize=12)

ax = axs[0]
hh = np.linspace(1, 25, 100)
ax.plot(hh, S.horizon_dip_rad(hh) * 1e3, color='#333333', lw=1.5,
        label=r'analytic  $\sqrt{2h/R_{\rm eff}}$')
ax.plot(heights, dip_gl * 1e3, 'o', color=C_GL, ms=8, label='GL renderer')
ax.plot(heights, dip_rm * 1e3, 's', color=C_RM, ms=7, mfc='none',
        label='ray-marcher')
ax.set_xlabel('viewer height above sea level (m)')
ax.set_ylabel('sea-horizon dip (mrad)')
ax.legend(frameon=False)
ax.grid(alpha=0.25, lw=0.5)

ax = axs[1]
ax.plot(az, el_flat * 1e3, color=C_FLAT, lw=1.0, ls='--',
        label='ray-marcher, flat earth (unpatched model)')
ax.plot(az, el_rm * 1e3, color=C_RM, lw=1.2, label='ray-marcher, curved+refraction')
ax.plot(az, el_gl * 1e3, color=C_GL, lw=1.2, alpha=0.75,
        label='GL renderer (patched)')
ax.set_xlabel('azimuth (deg, 0=N)')
ax.set_ylabel('skyline elevation (mrad)')
ax.set_xlim(-180, 180)
ax.legend(frameon=False, loc='upper left')
ax.grid(alpha=0.25, lw=0.5)
ax.set_title(f'360° skyline at ({STRAIT["lat"]}, {STRAIT["lon"]}), z=5 m — '
             f'GL vs curved RMS {rms*1e3:.2f} mrad; vs flat {rms_flat*1e3:.2f} mrad',
             fontsize=10)

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'e0_validation.png'), dpi=110)
print(f'wrote {os.path.join(OUT, "e0_validation.png")}')

sys.exit(1 if nfail else 0)
