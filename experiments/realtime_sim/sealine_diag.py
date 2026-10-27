"""Why does the levelling refuse a cruise frame's sea line? Per frame: SegFormer's raw and
cleaned sea line against the true sea horizon (exact camera at the recorded pose), and the
levelling's verdict with its gates; a sheet per frame."""
import glob, json, math, os, sys
import numpy as np
EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP)
sys.argv = sys.argv[:2]
import e5ck_oracle as E, e5cj_seg_level as J, extract, skyline as S
from segformer_bridge import SegFormer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = os.path.abspath(sys.argv[1])
sf = SegFormer()
out = os.path.join(d, 'sealine_diag'); os.makedirs(out, exist_ok=True)
rows = []
for jf in sorted(glob.glob(os.path.join(d, 'cruise_*.json'))):
    p = json.load(open(jf)); t = p['tag']
    p['roll_deg'] = (p['roll_deg'] + 180) % 360 - 180
    img = extract.load_image(os.path.join(d, t + '_photo.png')); H, W = img.shape[:2]
    cls = sf.predict(img)
    sky, sea_raw = J.raw_lines(cls)
    sea, fit = J.clean_sea_line(sea_raw)
    f = (W / 2) / math.tan(math.radians(p['hfov_deg']) / 2)
    dip = S.horizon_dip_rad(max(p['z_msl'], 0.5), k=0.0)
    az = p['heading_deg'] + np.linspace(-p['hfov_deg'] / 2, p['hfov_deg'] / 2, 400)
    hz = E.project_exact(az, np.full(az.size, -dip), p['heading_deg'], p['pitch_deg'], p['roll_deg'], p['hfov_deg'], W, H)
    lv = extract.sea_horizon_attitude(np.nan_to_num(sea), np.isfinite(sea) * 1.0, (H, W), f, dip) if fit else None
    m = np.isfinite(sea_raw) & np.isfinite(hz)
    raw_vs_hz = float(np.median(np.abs(sea_raw[m] - hz[m]))) if m.any() else float('nan')
    water = float((cls == 1).mean()); skyf = float((cls == 2).mean())
    rows.append(dict(tag=t, accepted=bool(lv), fit=bool(fit), kept=(fit or {}).get('kept'), raw_cols=float(np.isfinite(sea_raw).mean()),
                     raw_vs_horizon_px=raw_vs_hz, water=water, sky=skyf,
                     lev_err=(lv['pitch_deg'] - p['pitch_deg'], lv['roll_deg'] - p['roll_deg']) if lv else None))
    fig, ax = plt.subplots(figsize=(10, 5.6)); ax.imshow(img)
    x = np.arange(W)
    ax.plot(x, sky, color='#ffd23f', lw=1, label='skyline (SegFormer)')
    ax.plot(x, sea_raw, color='#00e5ff', lw=0.8, alpha=0.5, label='sea line raw')
    ax.plot(x, sea, color='#ff2d2d', lw=1.4, label='sea line cleaned')
    ax.plot(x, hz, color='#ffffff', lw=1, ls='--', label='true sea horizon')
    ax.set_ylim(min(H, np.nanmax(np.r_[hz, sea_raw[np.isfinite(sea_raw)]]) + 80), max(0, np.nanmin(sky) - 80))
    ax.set_title('%s  levelling %s' % (t, 'ACCEPTED' if lv else 'REFUSED')); ax.axis('off'); ax.legend(loc='lower left', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, t + '.png'), dpi=80); plt.close(fig)
for r in rows:
    print('%-11s %-8s fit %-5s kept %4s raw cols %.2f |raw-horizon| %6.1f px  water %.2f sky %.2f %s' % (
        r['tag'], 'ACCEPT' if r['accepted'] else 'refuse', r['fit'], '%.2f' % r['kept'] if r['kept'] else '-', r['raw_cols'],
        r['raw_vs_horizon_px'], r['water'], r['sky'], 'dpitch %+.2f droll %+.2f' % r['lev_err'] if r['lev_err'] else ''))
json.dump(rows, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
