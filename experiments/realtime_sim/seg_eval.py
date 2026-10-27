#!/usr/bin/env python3
"""How good can SegFormer's skyline get? On renders that carry their exact skyline (the
segmentation pass of cesium_render/driver.mjs), SegFormer-B2 (ADE20K) at several input
sizes, with and without a horizontal-flip average of the logits, against the exact skyline;
errors in mrad (pixels over the focal length), so frame sizes compare.

    python3 seg_eval.py RENDER_DIR [RENDER_DIR ...] [--sizes 512,768,1024,0] [--tta]
      size 0 = the frame's own height (no downscale)
"""
import argparse, glob, json, math, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP); sys.path.insert(0, os.path.join(EXP, 'cesium_render'))
import e5cj_seg_level as J  # noqa: E402
import extract  # noqa: E402
from compare import seg_lines  # noqa: E402
from segformer_bridge import SegFormer  # noqa: E402
from PIL import Image  # noqa: E402


def predict(sf, rgb01, short, tta):
    """segformer_bridge.predict at a chosen input size, optionally averaging the logits of the
    frame and its mirror image."""
    H, W = rgb01.shape[:2]
    sf.proc.size = {'shortest_edge': int(short or min(H, W))}
    im = Image.fromarray((rgb01 * 255).astype(np.uint8))
    x = sf.proc(images=im, return_tensors='pt')['pixel_values'].to(sf.device)
    cd = torch.backends.cudnn
    with torch.no_grad(), cd.flags(enabled=cd.enabled, benchmark=cd.benchmark, deterministic=cd.deterministic, allow_tf32=False):
        lg = sf.model(pixel_values=x).logits
        if tta:
            lg = 0.5 * (lg + torch.flip(sf.model(pixel_values=torch.flip(x, dims=[3])).logits, dims=[3]))
        n = max(1, int(2.5e8 // (4 * H * W)))
        best = ade = None
        for c0 in range(0, lg.shape[1], n):
            up = torch.nn.functional.interpolate(lg[:, c0:c0 + n], size=(H, W), mode='bilinear', align_corners=False)[0]
            v, i = up.max(0)
            if best is None:
                best, ade = v, i
            else:
                better = v > best
                best = torch.where(better, v, best); ade = torch.where(better, i + c0, ade)
        out = sf.lut[ade].cpu().numpy()
    sf.proc.size = {'shortest_edge': sf.short_side}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--sizes', default='512,768,1024,0')
    ap.add_argument('--tta', action='store_true', help='also each size with the flip average')
    ap.add_argument('--save', default=None, help='write SegFormer lines of this variant (e.g. 1024tta) as <tag>_<variant>_segpost.npz')
    a = ap.parse_args()
    sf = SegFormer()
    sizes = [int(s) for s in a.sizes.split(',')]
    variants = [(s, False) for s in sizes] + ([(s, True) for s in sizes] if a.tta else [])
    for d in a.dirs:
        tags = sorted(os.path.basename(f)[:-len('_seg.png')] for f in glob.glob(os.path.join(d, 'cruise_*_seg.png')))
        res = {v: dict(err=[], bias=[], t=[]) for v in variants}
        f_px = None
        for t in tags:
            meta = json.load(open(os.path.join(d, t + '.json')))
            p = meta['pose']
            exact, _, _ = seg_lines(os.path.join(d, t + '_seg.png'))
            img = extract.load_image(os.path.join(d, t + '_photo.png'), max_w=10 ** 6)     # full resolution
            H, W = img.shape[:2]
            f_px = (W / 2) / math.tan(math.radians(p['hfov_deg']) / 2)
            for v in variants:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                cls = predict(sf, img, v[0], v[1])
                torch.cuda.synchronize(); res[v]['t'].append(time.perf_counter() - t0)
                sky, sea = J.raw_lines(cls)
                m = np.isfinite(sky) & np.isfinite(exact)
                e = (sky[m] - exact[m]) / f_px * 1e3                 # mrad, + = SegFormer lower
                res[v]['err'].append(np.abs(e)); res[v]['bias'].append(np.median(e))
                tag = '%d%s' % (v[0], 'tta' if v[1] else '')
                if a.save == tag:
                    conf = np.isfinite(sky) * 1.0
                    k = max(1, round(J.EDGE_FRAC * W)); conf[:k] = conf[-k:] = 0.0
                    np.savez(os.path.join(d, '%s_%s_segpost.npz' % (t, tag)), rows=np.nan_to_num(sky), conf=conf)
                    s2, fit = J.clean_sea_line(sea)
                    if fit:
                        np.savez(os.path.join(d, '%s_%s_sealine.npz' % (t, tag)), rows=s2)
        print('\n%s: %d frames, %dx%d, f %.0f px (1 px = %.2f mrad)' % (d, len(tags), W, H, f_px, 1e3 / f_px))
        # both lines are whole rows: the mean and the share of columns 1 and 2+ px off say more
        # than a median of integers
        px = 1e3 / f_px
        print('  %-14s %10s %9s %8s %8s %10s %8s' % ('input', 'mean mrad', 'p90', '>=1 px', '>=2 px', 'bias p50', 's/frame'))
        for v in variants:
            e = np.concatenate(res[v]['err'])
            print('  %-14s %10.2f %9.2f %7.1f%% %7.1f%% %+10.2f %8.3f' % (
                ('%d' % v[0] if v[0] else 'native') + (' +flip' if v[1] else ''), e.mean(), np.percentile(e, 90),
                100 * (e > 0.5 * px).mean(), 100 * (e > 1.5 * px).mean(), np.median(res[v]['bias']), np.median(res[v]['t'])))


if __name__ == '__main__':
    main()
