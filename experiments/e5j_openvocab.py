#!/usr/bin/env python3
"""E5j: open-vocabulary detector assay for the landmark channels.

The landmark web (E5h/E5i) needs an image-space detector for
turbines, pylons and masts. Rather than training per-class models,
assay an open-vocabulary detector (detect anything described by
text) zero-shot on the CH1 photo set with landmark prompts.

Model choice is egress-driven: YOLO-World v2's text encoder is
OpenAI CLIP hosted on Azure (blocked from this container), while
YOLOE's is MobileCLIP TorchScript on GitHub release assets
(reachable) — so YOLOE (yoloe-11s-seg) it is; detector weights for
both live on GitHub assets.

CH1 is Swiss alpine terrain: pylons, cable-car masts and summit
antennas do appear in mountain photos; turbines and lighthouses do
not. The assay measures (a) does the detector fire on real
infrastructure in real outdoor photos, (b) at what confidence, (c)
false-fire behaviour on empty ridgelines — the inputs needed to
decide whether zero-shot detection is deployable as the front end,
or a fine-tune (TTPLA etc., user-side downloads) is required.

Run:   python3 e5j_openvocab.py [n_photos]
       (writes out/e5j_results.json, annotated crops to out/e5j/)
"""

import os
import sys
import glob
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
CH1 = os.path.join(os.path.dirname(HERE), '..',
                   'celestial-navigation', 'CH1')
if not os.path.isdir(CH1):
    CH1 = os.path.expanduser('~/celestial-navigation/CH1')

PROMPTS = ['electricity pylon', 'transmission tower',
           'communication antenna mast', 'wind turbine', 'lighthouse']
CONF = 0.05


def main(n=40):
    from ultralytics import YOLOE
    model = YOLOE('yoloe-11s-seg.pt')
    model.set_classes(PROMPTS, model.get_text_pe(PROMPTS))
    photos = sorted(glob.glob(os.path.join(CH1, '*', '*.png')))
    photos = [p for p in photos if not p.endswith('-mask.png')][:n]
    os.makedirs(os.path.join(OUT, 'e5j'), exist_ok=True)
    rows = []
    for p in photos:
        name = os.path.basename(p)[:-4]
        res = model.predict(p, conf=CONF, verbose=False)[0]
        dets = []
        for b in res.boxes:
            dets.append(dict(cls=PROMPTS[int(b.cls)],
                             conf=float(b.conf),
                             xyxy=[float(x) for x in b.xyxy[0]]))
        rows.append(dict(photo=name, detections=dets))
        if dets:
            best = max(dets, key=lambda d: d['conf'])
            print(f"{name}: {len(dets)} det, best "
                  f"{best['cls']} {best['conf']:.2f}", flush=True)
            res.save(os.path.join(OUT, 'e5j', name + '_det.jpg'))
    n_with = sum(1 for r in rows if r['detections'])
    by_cls = {}
    for r in rows:
        for d in r['detections']:
            by_cls.setdefault(d['cls'], []).append(d['conf'])
    print(f'\n{len(rows)} photos, {n_with} with detections')
    for c, confs in sorted(by_cls.items()):
        print(f'  {c:28s} n {len(confs):3d}  max conf '
              f'{max(confs):.2f}  mean {np.mean(confs):.2f}')
    with open(os.path.join(OUT, 'e5j_results.json'), 'w') as f:
        json.dump(dict(prompts=PROMPTS, conf_min=CONF, photos=rows),
                  f, indent=1)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
