"""Levelling variants on recorded cruise frames: how many sea lines each accepts and how far
the pitch and roll it gives are from the truth. 'imu' adds a consistency gate against a
simulated IMU (truth + N(0, 0.3 deg) per axis): a boat knows its pitch and roll that well."""
import glob, json, math, os, sys, zlib
import numpy as np
EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP)
import e5cj_seg_level as J, extract, skyline as S

d = os.path.abspath(sys.argv[1])
V = {'strict 1.5/3 px': dict(tol_px=1.5, below_px=3.0), 'relaxed 2.5/5 px': dict(tol_px=2.5, below_px=5.0)}
res = {k: [] for k in list(V) + ['relaxed + IMU gate 1.5 deg']}
for jf in sorted(glob.glob(os.path.join(d, 'cruise_*.json'))):
    p = json.load(open(jf)); t = p['tag']
    p['roll_deg'] = (p['roll_deg'] + 180) % 360 - 180
    sl = os.path.join(d, t + '_sealine.npz')
    if not os.path.exists(sl):
        continue
    sea = np.load(sl)['rows']; W, H = p['width'], p['height']
    f = (W / 2) / math.tan(math.radians(p['hfov_deg']) / 2)
    dip = S.horizon_dip_rad(max(p['z_msl'], 0.5), k=0.0)
    rng = np.random.default_rng(zlib.crc32(t.encode()))
    imu = (p['pitch_deg'] + rng.normal(0, 0.3), p['roll_deg'] + rng.normal(0, 0.3))
    for k, kw in V.items():
        lv = extract.sea_horizon_attitude(np.nan_to_num(sea), np.isfinite(sea) * 1.0, (H, W), f, dip, **kw)
        e = (lv['pitch_deg'] - p['pitch_deg'], lv['roll_deg'] - p['roll_deg']) if lv else None
        res[k].append(e)
        if k.startswith('relaxed'):
            ok = lv and abs(lv['pitch_deg'] - imu[0]) <= 1.5 and abs(lv['roll_deg'] - imu[1]) <= 1.5
            res['relaxed + IMU gate 1.5 deg'].append(e if ok else None)
for k, v in res.items():
    a = np.array([x for x in v if x is not None])
    bad = int((np.abs(a) > 1.0).any(axis=1).sum()) if a.size else 0
    print('%-28s accepted %2d/%d  |dpitch| median %.2f max %.2f  |droll| median %.2f max %.2f  off by >1 deg: %d' % (
        k, len(a), len(v), np.median(np.abs(a[:, 0])), np.abs(a[:, 0]).max(), np.median(np.abs(a[:, 1])), np.abs(a[:, 1]).max(), bad))
