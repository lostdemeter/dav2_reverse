#!/usr/bin/env python3
"""Fetch fit-pool scenes (training data for closed-form fits).

SEPARATE from strata_real.npz by design: appending to strata_real
would shift strata medians (moving target). Fit data lives in
adapt/fixtures/fit_pool.npz. Run: python adapt/fetch_fitpool.py --n 150
"""
import argparse
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np

OUT = ADAPT / 'fixtures' / 'fit_pool.npz'
SIZE = 168


def square168(img):
    from PIL import Image as _I  # noqa
    img = img.convert('RGB')
    w, h = img.size
    s = min(w, h)
    crop = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return np.array(crop.resize((SIZE, SIZE))).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=150)
    ap.add_argument('--stride', type=int, default=7)
    ap.add_argument('--span', type=int, default=3000)
    args = ap.parse_args()

    from datasets import load_dataset
    from harden_anchors import oracle_refs

    have_ids = set()
    if OUT.exists():
        z = np.load(OUT, allow_pickle=True)
        have_ids = set(str(x) for x in z['ids'])
        print(f"pool has {len(have_ids)} already", flush=True)
    ds = load_dataset('detection-datasets/coco', split='train', streaming=True)
    rgbs, ids = [], []
    for i, ex in enumerate(ds):
        if i >= args.span:
            break
        if i % args.stride != 0:
            continue
        cid = f"coco-train-{ex['image_id']}"
        if cid in have_ids:
            continue
        rgbs.append(square168(ex['image']))
        ids.append(cid)
        if len(rgbs) >= args.n:
            break
    print(f"collected {len(rgbs)}, oracling...", flush=True)
    refs = oracle_refs(rgbs)
    if OUT.exists():
        z = np.load(OUT, allow_pickle=True)
        rgb_all = np.concatenate(
            [z['rgb'], np.stack([(r * 255).astype(np.uint8) for r in rgbs])])
        ref_all = np.concatenate([z['ref'], np.stack(refs)])
        ids_all = np.concatenate([z['ids'], np.array(ids)])
    else:
        rgb_all = np.stack([(r * 255).astype(np.uint8) for r in rgbs])
        ref_all = np.stack(refs)
        ids_all = np.array(ids)
    np.savez_compressed(OUT, rgb=rgb_all, ref=ref_all, ids=ids_all,
                        source=np.array(["coco-train"] * len(ids_all)))
    print(f"pool now {len(ids_all)} scenes -> {OUT}")


if __name__ == '__main__':
    main()
