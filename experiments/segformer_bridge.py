#!/usr/bin/env python3
"""SegFormer bridge: ADE20K scene parsing as a sky / water / land map.

Same contract as ewasr_bridge.EWasr: predict(rgb01) returns HxW uint8
with 0 land (obstacle), 1 water, 2 sky, so the boundary code in
e5cf_faint_ridge.ewasr_boundary and skyfix's extractors can use it
unchanged. ADE20K gives a richer vocabulary than eWaSR's three classes,
but it is not a maritime model: it was trained on indoor and urban
scenes, so the sky/water split on a coast is its own evaluation.

Weights: nvidia/segformer-b2-finetuned-ade-512-512 (Hugging Face,
Apache 2.0), stored under $HORIZONATOR_ASSETS/hf/.
"""

import os

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, \
    SegformerImageProcessor

from assets import assets_dir

WATER = ('water', 'sea', 'river', 'lake')
LAND = ('mountain', 'hill', 'rock', 'earth', 'land', 'tree', 'plant',
        'building', 'skyscraper', 'waterfall')


class SegFormer:
    def __init__(self, path=None, short_side=512, device=None):
        path = path or os.path.join(assets_dir(), 'hf',
                                    'nvidia__segformer-b2-finetuned-ade-512-512')
        # on the GPU when there is one: a frame's forward pass is 30 ms
        # there against 1-5 s on the CPU
        self.device = device or ('cuda' if torch.cuda.is_available()
                                 else 'cpu')
        self.proc = SegformerImageProcessor.from_pretrained(path)
        self.short_side = short_side
        self.proc.size = {'shortest_edge': short_side}
        self.model = SegformerForSemanticSegmentation.from_pretrained(path)
        self.model.eval().to(self.device)
        names = {int(k): v for k, v in self.model.config.id2label.items()}
        self.sky = [k for k, v in names.items() if v == 'sky']
        self.water = [k for k, v in names.items() if v in WATER]
        self.land = [k for k, v in names.items() if v in LAND]
        lut = torch.zeros(len(names), dtype=torch.uint8)
        lut[self.water] = 1
        lut[self.sky] = 2
        self.lut = lut.to(self.device)

    def predict(self, rgb01):
        """rgb01: HxWx3 float in [0,1] -> HxW uint8 (0 land, 1 water, 2 sky)."""
        H, W = rgb01.shape[:2]
        im = Image.fromarray((rgb01 * 255).astype(np.uint8))
        x = self.proc(images=im, return_tensors='pt')['pixel_values']
        cd = torch.backends.cudnn
        with torch.no_grad():
            # fp32 convolutions: with TF32, PyTorch's GPU default, every
            # campaign frame's lines moved a pixel in a few columns against
            # the CPU's (one column 20 px); in fp32 43 of 44 frames match
            # the CPU exactly, at the same speed
            with cd.flags(enabled=cd.enabled, benchmark=cd.benchmark,
                          deterministic=cd.deterministic, allow_tf32=False):
                logits = self.model(pixel_values=x.to(self.device)).logits
            # the full-resolution argmax, a slice of classes at a time:
            # all 150 upsampled maps of a 1600x3479 frame are 3.3 GB. Ties
            # keep the lowest class index, as one argmax would
            n = max(1, int(2.5e8 // (4 * H * W)))
            best = ade = None
            for c0 in range(0, logits.shape[1], n):
                up = torch.nn.functional.interpolate(
                    logits[:, c0:c0 + n], size=(H, W), mode='bilinear',
                    align_corners=False)[0]
                v, i = up.max(0)
                if best is None:
                    best, ade = v, i
                else:
                    better = v > best
                    best = torch.where(better, v, best)
                    ade = torch.where(better, i + c0, ade)
            return self.lut[ade].cpu().numpy()

    def predict_band(self, rgb01, r0, r1):
        """predict() on rows r0:r1 only, at the scale the whole frame would get (a third of
        the frame is a third of the pixels, not the same 512 rows stretched). Rows above the
        band come back sky, rows below it land: the caller grows the band when its top edge
        is not sky (a mountain taller than the band) or its bottom holds no water."""
        H, W = rgb01.shape[:2]
        short = max(32, int(round(min(r1 - r0, W) * self.short_side / min(H, W))))
        size = self.proc.size
        self.proc.size = {'shortest_edge': short}
        try:
            band = self.predict(rgb01[r0:r1])
        finally:
            self.proc.size = size
        out = np.zeros((H, W), np.uint8)
        out[:r0] = 2
        out[r0:r1] = band
        return out
