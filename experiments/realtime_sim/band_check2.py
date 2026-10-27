import glob, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solve_cruise as SC
import e5cj_seg_level as J, e5ck_oracle as E, extract, skyline as S
from segformer_bridge import SegFormer
d = os.path.abspath(sys.argv[1]); SC.IMU_SIG = 0.3; SC.BAND_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 1 / 3
sf = SegFormer()
sky_sd, sea_full, sea_band, n = [], [], [], 0
for t in sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(d, 'cruise_*.json')))[::3]:
    p = SC.frame(d, t); img = extract.load_image(os.path.join(d, t + '_photo.png')); H, W = img.shape[:2]
    _, _, imu = SC.knows(p, t, 1000.0, 5.0, 'a')
    full = sf.predict(img); band, _ = SC.segment_band(sf, img, imu, p['hfov_deg'], p['z_msl'])
    s1, w1 = J.raw_lines(full); s2, w2 = J.raw_lines(band)
    m = np.isfinite(s1) & np.isfinite(s2); sky_sd.append(np.median(s2[m] - s1[m]))
    az = p['heading_deg'] + np.degrees(np.arctan((np.arange(W) - (W - 1) / 2) / ((W / 2) / math.tan(math.radians(p['hfov_deg']) / 2))))
    hz = E.project_exact(az, np.full(W, -S.horizon_dip_rad(max(p['z_msl'], .5), k=0.0)), p['heading_deg'], p['pitch_deg'], p['roll_deg'], p['hfov_deg'], W, H)
    for w, acc in ((w1, sea_full), (w2, sea_band)):
        c, _ = J.clean_sea_line(w)
        mm = np.isfinite(c) & np.isfinite(hz)
        acc.append(np.median(np.abs(c[mm] - hz[mm])) if mm.sum() > 50 else np.nan)
    n += 1
print('%d frames: band skyline minus full, per-frame median: median %+.2f px, IQR %+.2f..%+.2f' % (n, np.median(sky_sd), *np.percentile(sky_sd, [25, 75])))
print('cleaned sea line vs the true horizon, per-frame median |d|: full %.2f px, band %.2f px (frames %d / %d)' % (
    np.nanmedian(sea_full), np.nanmedian(sea_band), np.isfinite(sea_full).sum(), np.isfinite(sea_band).sum()))
