"""The waterline's range information for a boat camera 2.5 m above the sea: how far does the
predicted water edge move when the boat moves 500 m along its line of sight?"""
import glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import skyline as S

dem = S.Dem(os.path.expanduser('~/.horizonator/DEMs_COP30'))
R = 6371000.0


def waterline(lat, lon, z, az, d_max=25000.0, step=25.0):
    mlat, mlon = S.meters_per_degree(lat)
    d = np.arange(step, d_max, step)
    a = np.radians(az)[:, None]
    h = dem.sample((lat + d * np.cos(a) / mlat).ravel(), (lon + d * np.sin(a) / mlon).ravel()).reshape(a.shape[0], d.size)
    wet = h <= 0.5
    hor = math.sqrt(2 * R * z)
    el = np.full(a.shape[0], -math.sqrt(2 * z / R))
    for i in range(a.shape[0]):
        land = np.flatnonzero(~wet[i])
        ds = d[land[0]] if land.size else np.inf
        if ds < hor:
            el[i] = math.atan2(-z, ds) - ds / (2 * R)
    return el


sens, shore = [], []
for f in sorted(glob.glob('out/aegean/cruise_*_0[0-4]0.json')):
    p = json.load(open(f))
    az = p['heading_deg'] + np.linspace(-35, 35, 141)
    mlat, mlon = S.meters_per_degree(p['lat'])
    h = math.radians(p['heading_deg'])
    e0 = waterline(p['lat'], p['lon'], 2.5, az)
    e1 = waterline(p['lat'] + 500 * math.cos(h) / mlat, p['lon'] + 500 * math.sin(h) / mlon, 2.5, az)
    sens.append(np.median(np.abs(e1 - e0)) * 1e3)
    shore.append(np.mean(e0 < -math.sqrt(2 * 2.5 / R) - 1e-6))
print('%d frames: 500 m along the line of sight moves the predicted water edge a median %.2f mrad (p90 %.2f); '
      'a shore stands in front of the horizon on %.0f %% of the azimuths' % (len(sens), np.median(sens), np.percentile(sens, 90), 100 * np.mean(shore)))
print("scale: the horizon dip at 2.5 m is %.2f mrad; SegFormer's sea line sits ~1.6 px = %.2f mrad from the true horizon" % (
    math.sqrt(2 * 2.5 / R) * 1e3, 1.6 / 914 * 1e3))
