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
    def __init__(self, path=None, short_side=512):
        path = path or os.path.join(assets_dir(), 'hf',
                                    'nvidia__segformer-b2-finetuned-ade-512-512')
        self.proc = SegformerImageProcessor.from_pretrained(path)
        self.proc.size = {'shortest_edge': short_side}
        self.model = SegformerForSemanticSegmentation.from_pretrained(path)
        self.model.eval()
        names = {int(k): v for k, v in self.model.config.id2label.items()}
        self.sky = [k for k, v in names.items() if v == 'sky']
        self.water = [k for k, v in names.items() if v in WATER]
        self.land = [k for k, v in names.items() if v in LAND]

    def predict(self, rgb01):
        """rgb01: HxWx3 float in [0,1] -> HxW uint8 (0 land, 1 water, 2 sky)."""
        H, W = rgb01.shape[:2]
        im = Image.fromarray((rgb01 * 255).astype(np.uint8))
        with torch.no_grad():
            logits = self.model(**self.proc(images=im,
                                            return_tensors='pt')).logits
            logits = torch.nn.functional.interpolate(
                logits, size=(H, W), mode='bilinear', align_corners=False)
        ade = logits[0].argmax(0).numpy()
        out = np.zeros((H, W), np.uint8)
        out[np.isin(ade, self.water)] = 1
        out[np.isin(ade, self.sky)] = 2
        return out
