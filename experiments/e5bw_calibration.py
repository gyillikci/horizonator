#!/usr/bin/env python3
"""E5bw calibration: does a column's sigma predict its real error?

For every field frame and each confidence-carrying front end, the
extracted boundary is mapped to (azimuth, elevation) exactly as skyfix
does it, the DEM skyline is ray-cast from the TRUE position (the EXIF
fix), the heading offset and elevation offset are fitted there, and
each column's residual is compared with the sigma the front end
assigned it. The residual is not pure extraction error - it carries
the DEM's own error (canopy, buildings, near field) - so a correlation
here is a lower bound on how well sigma tracks extraction error.

    python3 e5bw_calibration.py
"""

import json
import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import skyfix as F  # noqa: E402
import skyline as S  # noqa: E402
import extract  # noqa: E402
from assets import celnav  # noqa: E402

EXT = os.path.join(HERE, 'out', 'extract')


def main():
    meta = json.load(open(os.path.join(HERE, 'data', 'campaign_frames.json')))
    dem = os.path.expanduser('~/.horizonator/DEMs_COP30')
    out = {}
    pooled = {m: ([], []) for m in ('ewasr_w', 'fused_w')}
    for fr in meta['frames']:
        img = extract.load_image(celnav(fr['path']))
        cm = S.CMarcher(dem, (fr['lat'] - .6, fr['lat'] + .6),
                        (fr['lon'] - .8, fr['lon'] + .8), d_min=1000.0)
        el_syn, r_syn = cm.skyline(fr['lat'], fr['lon'], fr['z'], F.AZ)
        el_syn = el_syn + np.where(np.isfinite(r_syn) & (r_syn > 0),
                                   9.0 / np.maximum(r_syn, 1000.0), 0.0)
        for m in pooled:
            z = np.load(os.path.join(EXT, '%s_%s.npz' % (fr['tag'], m)))
            F.BOUNDARY_ROWS, F.BOUNDARY_CONF = z['rows'], z['conf']
            el_obs, w, d = F.observation(img, fr['fov'], fr['heading'], 0.0)
            c, sh, beta = F.fast_photo_cost(
                el_obs, w, np.nan_to_num(el_syn), np.arange(-60, 61, 2))
            # per COLUMN: synthetic elevation at the column's azimuth
            az = fr['heading'] + d['az_rel'] + sh * 0.1
            es = np.interp((az + 180) % 360 - 180, F.AZ, np.nan_to_num(el_syn))
            res = np.abs(d['el_pt'] + beta - es) * 1e3          # mrad
            sig = z['sigma']
            ok = np.isfinite(sig) & (d['conf'] > 0) & np.isfinite(res)
            if ok.sum() < 50:
                continue
            fpx = (img.shape[1] / 2) / math.tan(math.radians(fr['fov']) / 2)
            sig_mrad = sig[ok] / fpx * 1e3
            rho = spearmanr(sig_mrad, res[ok]).correlation
            out.setdefault(fr['tag'], {})[m] = float(rho)
            pooled[m][0].extend(sig_mrad.tolist())
            pooled[m][1].extend(res[ok].tolist())
    print('per-frame Spearman(sigma, |residual vs DEM at truth|):')
    summary = {}
    for m, (sg, rs) in pooled.items():
        rhos = np.array([v[m] for v in out.values() if m in v])
        sg, rs = np.array(sg), np.array(rs)
        q = np.quantile(sg, [1 / 3, 2 / 3])
        t = [rs[sg <= q[0]], rs[(sg > q[0]) & (sg <= q[1])], rs[sg > q[1]]]
        summary[m] = dict(frames=int(rhos.size),
                          rho_median=float(np.median(rhos)),
                          rho_positive=int((rhos > 0).sum()),
                          tercile_median_residual_mrad=[float(np.median(x))
                                                        for x in t])
        print('  %-8s %d frames, median rho %+.2f, positive on %d; pooled '
              'median |residual| by sigma tercile: %.1f / %.1f / %.1f mrad'
              % (m, rhos.size, np.median(rhos), (rhos > 0).sum(),
                 *summary[m]['tercile_median_residual_mrad']))
    json.dump(dict(summary=summary, frames=out),
              open(os.path.join(EXT, 'calibration.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
