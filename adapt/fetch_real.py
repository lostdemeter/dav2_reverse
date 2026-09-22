#!/usr/bin/env python3
"""Fetch diverse real RGB scenes (COCO via HF) + HF-oracle refs for strata fill.

Writes adapt/fixtures/strata_real.npz (gitignored): rgb uint8 (N,168,168,3),
ref float32 (N,168,168), ids + source provenance. Strided streaming for
diversity (consecutive COCO frames are near-duplicates in stat space).

Run: python adapt/fetch_real.py [--n 40 --stride 25 --span 1000]
Needs HF + CUDA recommended. Reuses harden_anchors.oracle_refs (same
protocol as build_fixtures: DAV2-Small oracle, 168px).
"""
import argparse
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


def square168(img):
    img = img.convert('RGB')
    w, h = img.size
    s = min(w, h)
    crop = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return np.array(crop.resize((SIZE, SIZE))).astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--stride', type=int, default=25)
    ap.add_argument('--span', type=int, default=1000)
    args = ap.parse_args()

    from datasets import load_dataset
    from harden_anchors import oracle_refs

    ds = load_dataset('detection-datasets/coco', split='train', streaming=True)
    rgbs, ids = [], []
    for i, ex in enumerate(ds):
        if i >= args.span:
            break
        if i % args.stride != 0:
            continue
        rgb = square168(ex['image'])
        rgbs.append(rgb)
        ids.append(f"coco-train-{ex['image_id']}")
        print(f"  kept stream[{i}] id={ex['image_id']} ({len(rgbs)}/{args.n})", flush=True)
        if len(rgbs) >= args.n:
            break
    print(f"collected {len(rgbs)} scenes, oracling...", flush=True)
    refs = oracle_refs(rgbs)
    for cid, ref in zip(ids, refs):
        print(f"  {cid} ref range [{ref.min():.3f},{ref.max():.3f}]", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        rgb=np.stack([(r * 255).astype(np.uint8) for r in rgbs]),
        ref=np.stack([np.asarray(r, dtype=np.float32) for r in refs]),
        ids=np.array(ids),
        source=np.array(["detection-datasets/coco(train-streamed)"] * len(rgbs)))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == '__main__':
    main()
