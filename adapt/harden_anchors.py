#!/usr/bin/env python3
"""Mine hard retention anchors near the threshold cliff (round two).

Scores fresh scenes with seed widths vs narrowed widths (integer tree
head) and keeps scenes where the seed passes but the narrow config
degrades most. These become retention anchors so future searches must
hold the cliff, not just the easy cases. Run once:
    python adapt/harden_anchors.py [--scenes N] [--keep K]

Writes adapt/fixtures/hard_anchors.npz (gitignored). Needs baked
weights + HF (oracle) + CUDA recommended.
"""
import sys
import time
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))
sys.path.insert(0, str(ADAPT / 'third_party'))

import numpy as np

SEED_CFG = {"frac_cap": 13312, "exp_span": 16, "dmax": 4096, "accum": "tree"}
NARROW_CFG = {"frac_cap": 2048, "exp_span": 8, "dmax": 4096, "accum": "tree"}
OUT = ADAPT / 'fixtures' / 'hard_anchors.npz'


def oracle_refs(scenes):
    """HF oracle depths at 168px (same protocol as build_fixtures)."""
    import torch
    from PIL import Image
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc = AutoImageProcessor.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf')
    model = AutoModelForDepthEstimation.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf').to(device).eval()
    refs = []
    for rgb in scenes:
        with torch.no_grad():
            pv = proc(images=Image.fromarray((rgb * 255).astype(np.uint8)),
                      return_tensors='pt')['pixel_values'].to(device)
            d = model(pixel_values=pv).predicted_depth.squeeze().cpu().numpy()
        if d.shape != (168, 168):
            import cv2
            d = cv2.resize(d, (168, 168), interpolation=cv2.INTER_LINEAR)
        refs.append(d.astype(np.float32))
    return refs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, default=40)
    ap.add_argument('--keep', type=int, default=4)
    args = ap.parse_args()

    import build_fixtures as bf
    from graduate import load_all, _integer_depth, _corr
    import geo_int as G

    print("mining hard anchors: seed vs narrowed integer-head scores...")
    scenes = [bf.make_scene(1000 + i) for i in range(args.scenes)]
    refs = oracle_refs(scenes)

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared, _fx, _probe = load_all(device)
    # integer-head eval needs the stride-4 refs; reuse graduate's grid
    import cv2
    grid = 518 // 4 + (1 if 518 % 4 else 0)
    scored = []
    for i, (rgb, ref168) in enumerate(zip(scenes, refs)):
        ref = cv2.resize(ref168.astype(np.float64), (grid, grid),
                         interpolation=cv2.INTER_LINEAR)
        row = {"id": i}
        for tag, cfg in (("seed", SEED_CFG), ("narrow", NARROW_CFG)):
            saved = (G.FRAC_CAP, G._FRAC_LUT, G.DMAX, G._EXP_LUT)
            try:
                from lut_pareto import build_frac_table, build_exp_table
                G.FRAC_CAP = cfg["frac_cap"]
                G._FRAC_LUT = build_frac_table(cfg["frac_cap"])
                G.DMAX = cfg["dmax"]
                G._EXP_LUT = build_exp_table(cfg["exp_span"])
                add = G.build_add_lut(dmax=cfg["dmax"])
                sub = G.build_sub_lut(dmax=cfg["dmax"])
                pred = _integer_depth(shared, cfg, rgb, add, sub)
                row[tag] = _corr(pred, ref)
            finally:
                G.FRAC_CAP, G._FRAC_LUT, G.DMAX, G._EXP_LUT = saved
        row["gap"] = row["seed"] - row["narrow"]
        scored.append(row)
        print(f"  scene {i}: seed={row['seed']:.5f} narrow={row['narrow']:.5f} "
              f"gap={row['gap']:+.5f}", flush=True)

    cands = [r for r in scored if r["seed"] >= 0.9995]
    cands.sort(key=lambda r: -r["gap"])
    kept = cands[:args.keep]
    print(f"keeping {len(kept)} anchors (seed>=0.9995, largest seed-narrow gap):")
    for r in kept:
        print(f"  scene {r['id']}: seed={r['seed']:.5f} narrow={r['narrow']:.5f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        rgb=np.stack([(scenes[r["id"]] * 255).astype(np.uint8) for r in kept]),
        ref=np.stack([refs[r["id"]] for r in kept]),
        ids=np.array([f"retention-h{i}" for i in range(len(kept))]))
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
