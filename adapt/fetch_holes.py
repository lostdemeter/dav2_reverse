#!/usr/bin/env python3
"""Fetch hole-filler scenes by COCO image_id + HF-oracle refs; APPENDS to
strata_real.npz (never overwrites). Run: python adapt/fetch_holes.py
Needs GPU for oracle (run after the co-search race frees it).
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np
from PIL import Image

OUT = ADAPT / 'fixtures' / 'strata_real.npz'
SIZE = 168
TARGETS = [92, 349, 359, 387, 389, 626, 681, 927, 1139, 1146, 1183, 1390,
           2445, 4139]


def square168(img):
    img = img.convert('RGB')
    w, h = img.size
    s = min(w, h)
    crop = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return np.array(crop.resize((SIZE, SIZE))).astype(np.float32) / 255.0


def main():
    from datasets import load_dataset
    from harden_anchors import oracle_refs
    z = np.load(OUT, allow_pickle=True)
    have = set(str(x) for x in z['ids'])
    want = [i for i in TARGETS if f"coco-train-{i}" not in have]
    print(f"have {len(have)}, want {len(want)} more", flush=True)
    if not want:
        print("nothing to fetch")
        return
    ds = load_dataset('detection-datasets/coco', split='train', streaming=True)
    got = {}
    for ex in ds:
        if ex['image_id'] in want:
            got[ex['image_id']] = square168(ex['image'])
            print(f"  got id={ex['image_id']} ({len(got)}/{len(want)})", flush=True)
            if len(got) == len(want):
                break
    ids = [f"coco-train-{i}" for i in want]
    rgbs = [got[i] for i in want]
    refs = oracle_refs(rgbs)
    rgb_all = np.concatenate([z['rgb'], np.stack([(r * 255).astype(np.uint8) for r in rgbs])])
    ref_all = np.concatenate([z['ref'], np.stack(refs)])
    ids_all = np.concatenate([z['ids'], np.array(ids)])
    np.savez_compressed(OUT, rgb=rgb_all, ref=ref_all, ids=ids_all,
                        source=np.array(["detection-datasets/coco(train-streamed)"] * len(ids_all)))
    print(f"appended {len(ids)} scenes -> {OUT} ({len(ids_all)} total)")


if __name__ == '__main__':
    main()
