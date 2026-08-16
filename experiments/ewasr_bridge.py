#!/usr/bin/env python3
"""eWaSR bridge: a pretrained maritime water/sky/obstacle segmenter.

eWaSR (Tersek, Zust, Kristan 2023 — the embedded-compute WaSR) ships
MaSTr1325-trained weights on GitHub Releases under Apache 2.0, so the
Grelsson-style segmentation front-end arrives pretrained and
license-clean, no dataset needed. Roles here (see study doc E4n/E4g):
cloud masking (an overcast deck classifies as SKY, where our boundary
extractor would trace the cloud base as terrain), sea-span selection
for --auto-level, and a second opinion against false horizons.

Setup (one-time; ~750 MB of wheels + 240 MB weights):

    pip install torch torchvision timm pytorch_lightning
    git clone --depth 1 https://github.com/tersekmatija/eWaSR
    curl -LO https://github.com/tersekmatija/eWaSR/releases/download/\
0.1.0/ewasr_resnet18.pth
    # two small patches to the clone:
    #  - wasr/metaformer.py: from timm.layers import to_2tuple
    #    (timm >= 1.0 moved it)
    #  - wasr/models.py: pretrained=True -> False everywhere (the
    #    ImageNet backbone download is dead weight; the checkpoint
    #    overwrites it)

Usage:
    from ewasr_bridge import EWasr
    seg = EWasr('/path/to/eWaSR', '/path/to/ewasr_resnet18.pth')
    classes = seg.predict(rgb01)   # HxW uint8: 0 obstacle 1 water 2 sky
"""

import sys

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


class EWasr:
    def __init__(self, repo_dir, weights_path,
                 model_name='ewasr_resnet18'):
        sys.path.insert(0, repo_dir)
        import torch
        import wasr.models as models
        from wasr.utils import load_weights
        self.torch = torch
        m = models.get_model(model_name, num_classes=3,
                             pretrained=False, mixer='CCCCSS',
                             enricher='SS', project=False)
        m.load_state_dict(load_weights(weights_path))
        m.eval()
        self.model = m

    def predict(self, rgb01, imu_mask=None):
        """rgb01: HxWx3 float in [0,1]. imu_mask: HxW bool, True above
        the horizon (from the IMU / auto-level attitude); zeros are
        accepted by the non-IMU model. Returns HxW uint8 classes at the
        input resolution: 0 obstacle/land, 1 water, 2 sky."""
        torch = self.torch
        H, W = rgb01.shape[:2]
        from PIL import Image
        im = Image.fromarray((rgb01 * 255).astype(np.uint8)) \
            .resize((512, 384), Image.BILINEAR)
        x = np.asarray(im, dtype=np.float32) / 255.0
        t = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)[None]
                             .astype(np.float32))
        if imu_mask is None:
            mk = torch.zeros(1, 384, 512)
        else:
            mk = torch.from_numpy(
                np.asarray(Image.fromarray(
                    imu_mask.astype(np.uint8) * 255).resize(
                        (512, 384), Image.NEAREST)) > 127)[None].float()
        with torch.no_grad():
            out = self.model({'image': t, 'imu_mask': mk})['out']
            out = torch.nn.functional.interpolate(
                out, size=(H, W), mode='bilinear', align_corners=False)
        return out.argmax(1)[0].numpy().astype(np.uint8)
