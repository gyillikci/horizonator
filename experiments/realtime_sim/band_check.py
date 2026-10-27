"""Band against full-frame SegFormer on recorded cruise frames: time per frame, how often the
band had to grow, and how far its skyline and sea line are from the full frame's."""
import glob, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import solve_cruise as SC
import e5cj_seg_level as J
import extract
from segformer_bridge import SegFormer
import torch

d = os.path.abspath(sys.argv[1])
SC.IMU_SIG = 0.3
sf = SegFormer()
tags = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(d, 'cruise_*.json')))
tf, tb, grown, dsky, dsea, frac_rows = [], [], 0, [], [], []
for t in tags:
    p = SC.frame(d, t)
    img = extract.load_image(os.path.join(d, t + '_photo.png'))
    _, _, imu = SC.knows(p, t, 1000.0, 5.0, 'a')
    torch.cuda.synchronize(); t0 = time.perf_counter()
    full = sf.predict(img); torch.cuda.synchronize(); t1 = time.perf_counter()
    band, (r0, r1, g) = SC.segment_band(sf, img, imu, p['hfov_deg'], p['z_msl']); torch.cuda.synchronize(); t2 = time.perf_counter()
    tf.append(t1 - t0); tb.append(t2 - t1); grown += g > 0; frac_rows.append((r1 - r0) / img.shape[0])
    s1, w1 = J.raw_lines(full); s2, w2 = J.raw_lines(band)
    m = np.isfinite(s1) & np.isfinite(s2); dsky.append(np.abs(s1[m] - s2[m]))
    c1, _ = J.clean_sea_line(w1); c2, _ = J.clean_sea_line(w2)
    m = np.isfinite(c1) & np.isfinite(c2)
    if m.any():
        dsea.append(np.abs(c1[m] - c2[m]))
a, b = np.concatenate(dsky), np.concatenate(dsea)
print('%d frames: full %.3f s, band %.3f s median per frame (x%.1f); band rows %.0f %% of the frame (median), grown on %d' % (
    len(tags), np.median(tf), np.median(tb), np.median(tf) / np.median(tb), 100 * np.median(frac_rows), grown))
print('skyline band vs full: mean |d| %.2f px, p95 %.1f, >3 px %.1f %% of columns' % (a.mean(), np.percentile(a, 95), 100 * (a > 3).mean()))
print('sea line band vs full: mean |d| %.2f px, p95 %.1f, >3 px %.1f %% of columns' % (b.mean(), np.percentile(b, 95), 100 * (b > 3).mean()))
