#!/usr/bin/env python3
"""Parametric synthetic scene generator (core-inputs synthesis loop).

Gradients + occluder discs + checker/noise texture over WIDE ranges
(wider than build_fixtures.make_scene: steeper gradients, more discs,
finer checkers, Perlin-ish noise, stronger tints). Every scene saves
as PNG for human viewing; pool saved as npz (RGB uint8, no refs needed
for identification — teacher activations only).
Run: python adapt/gen_synthpool.py --n 200 --out adapt/fixtures/synth_pool.npz
"""
import argparse
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np

SIZE = 168


def make_synth(rng, h=SIZE, w=SIZE):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # base gradient: random direction + steepness (wider than fixtures)
    ang = rng.uniform(0, 2 * np.pi)
    steep = rng.uniform(0.5, 1.5)
    proj = (xx * np.cos(ang) + yy * np.sin(ang))
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)
    scene = np.clip(proj * steep, 0, 1)
    # occluder discs (more, wider size range)
    for _ in range(int(rng.integers(0, 6))):
        cy, cx = rng.uniform(0, 1, 2) * np.array([h, w])
        r = rng.uniform(0.03, 0.30) * min(h, w)
        disc = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
        scene[disc] = rng.uniform(0.0, 0.6)
    # checker patches (finer grids than fixtures)
    if rng.random() < 0.7:
        n = int(rng.integers(6, 28))
        checker = (((yy // (h // n)) + (xx // (w // n))) % 2).astype(bool)
        region = yy > h * rng.uniform(0.2, 0.8)
        scene = np.where(checker & region, scene * 0.9 + 0.05, scene)
    # smooth noise (Perlin-ish via filtered white noise)
    if rng.random() < 0.7:
        from scipy.ndimage import gaussian_filter
        nz = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32),
                             rng.uniform(0.5, 4.0))
        nz = (nz - nz.min()) / (nz.max() - nz.min() + 1e-9)
        a = rng.uniform(0.1, 0.5)
        scene = np.clip(scene * (1 - a) + nz * a, 0, 1)
    # tint (stronger than fixtures)
    tint = rng.uniform(0.6, 1.0, 3).astype(np.float32)
    return np.clip(scene[..., None] * tint, 0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='adapt/fixtures/synth_pool.npz')
    ap.add_argument('--pngdir', default='captures/synth_core')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    outp = Path(args.out)
    pngdir = REPO / args.pngdir if not Path(args.pngdir).is_absolute() \
        else Path(args.pngdir)
    pngdir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    rgbs, stats = [], []
    from strata import profile
    for i in range(args.n):
        rgb = make_synth(rng)
        rgbs.append((rgb * 255).astype(np.uint8))
        Image.fromarray((rgb * 255).astype(np.uint8)).save(
            pngdir / f"synth-{i:03d}.png")
        stats.append(profile(rgb))
    np.savez_compressed(outp, rgb=np.stack(rgbs),
                        ids=np.array([f"synth-{i}" for i in range(args.n)]))
    print(f"wrote {outp} ({args.n} scenes) + PNGs -> {pngdir}", flush=True)
    # strata spread vs COCO (medians from strata.json)
    import json as _j
    from strata import assign
    medians = _j.load(open(ADAPT / 'runs' / 'strata.json'))['medians']
    from collections import Counter
    dist = Counter(assign(s, medians) for s in stats)
    print("synth strata dist:", dict(sorted(dist.items())), flush=True)
    # contact sheet (10 cols)
    ims = [Image.open(pngdir / f"synth-{i:03d}.png") for i in range(args.n)]
    cols = 10
    rows = (args.n + cols - 1) // cols
    sheet = Image.new('RGB', (cols * SIZE, rows * SIZE))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * SIZE, (i // cols) * SIZE))
    sheet.save(pngdir / '_contact.png')
    print(f"contact sheet -> {pngdir / '_contact.png'}", flush=True)


if __name__ == '__main__':
    main()
