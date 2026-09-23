#!/usr/bin/env python3
"""Analytic heightfield scenes: RGB + TRUE depth, zero oracle.

Height = tilted plane + Gaussian bumps (known objects/placements) +
steps; Lambertian shading with random light dir + albedo texture
(checker/fine noise). Knobs: range (height scale), lighting (light
vector), texture. Depth ref = normalized height (exact, not pseudo).
Saves pool npz (rgb uint8 + ref float32 + ids) + PNGs + contact sheet.
Run: python adapt/gen_heightfields.py --n 200
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


def make_heightfield(rng, h=SIZE, w=SIZE):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # base tilted plane (range knob via tilt + scale)
    ang = rng.uniform(0, 2 * np.pi)
    tilt = rng.uniform(0.1, 0.6)
    H = tilt * ((xx * np.cos(ang) + yy * np.sin(ang)) /
                np.sqrt(h * h + w * w) + 0.5)
    # steps (cliffs)
    for _ in range(int(rng.integers(0, 3))):
        ax = rng.uniform(0.2, 0.8) * w
        H[:, xx[0] > ax] += rng.uniform(0.1, 0.4)
    # Gaussian bumps (objects: known placement + height)
    for _ in range(int(rng.integers(1, 7))):
        cy, cx = rng.uniform(0.1, 0.9, 2) * np.array([h, w])
        sx, sy = rng.uniform(0.03, 0.18, 2) * min(h, w)
        amp = rng.uniform(0.15, 0.8) * rng.choice([-1.0, 1.0])
        H = H + amp * np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) +
                               ((yy - cy) ** 2) / (2 * sy * sy)))
    H = (H - H.min()) / (H.max() - H.min() + 1e-9)
    H = np.clip(H * rng.uniform(0.7, 1.3), 0, 1)  # range knob
    # normals via central differences
    dzdx = (np.roll(H, -1, 1) - np.roll(H, 1, 1)) / 2.0
    dzdy = (np.roll(H, -1, 0) - np.roll(H, 1, 0)) / 2.0
    n = np.stack([-dzdx, -dzdy, np.ones_like(H)], axis=-1)
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
    # light: random azimuth + elevation (lighting knob)
    az, el = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 1.2)
    L = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el),
                  np.sin(el)], dtype=np.float32)
    shade = np.clip((n * L).sum(-1), 0.05, 1.0).astype(np.float32)
    # albedo texture (texture knob): checker and/or fine noise
    alb = np.ones((h, w), dtype=np.float32)
    r = rng.random()
    if r < 0.4:
        n_ = int(rng.integers(6, 24))
        alb = np.where((((yy // (h // n_)) + (xx // (w // n_))) % 2) == 0,
                       1.0, rng.uniform(0.4, 0.8))
    elif r < 0.8:
        from scipy.ndimage import gaussian_filter
        nz = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32),
                             rng.uniform(0.5, 3.0))
        alb = 0.5 + 0.5 * (nz - nz.min()) / (nz.max() - nz.min() + 1e-9)
    tint = rng.uniform(0.7, 1.0, 3).astype(np.float32)
    rgb = np.clip(shade[..., None] * alb[..., None] * tint, 0, 1)
    return (rgb.astype(np.float32),
            (1.0 - H).astype(np.float32))  # nearer = brighter like depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='adapt/fixtures/hfield_pool.npz')
    ap.add_argument('--pngdir', default='captures/hfields')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pngdir = REPO / args.pngdir
    pngdir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    rgbs, refs = [], []
    for i in range(args.n):
        rgb, depth = make_heightfield(rng)
        rgbs.append((rgb * 255).astype(np.uint8))
        refs.append(depth)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(
            pngdir / f"hf-{i:03d}.png")
    np.savez_compressed(REPO / args.out, rgb=np.stack(rgbs),
                        ref=np.stack(refs),
                        ids=np.array([f"hf-{i}" for i in range(args.n)]))
    print(f"wrote {args.out} + PNGs -> {pngdir}", flush=True)
    from strata import profile, assign
    import json as _j
    medians = _j.load(open(ADAPT / 'runs' / 'strata.json'))['medians']
    from collections import Counter
    rgb01 = [r.astype(np.float32) / 255.0 for r in rgbs]
    print("hfield strata dist:",
          dict(sorted(Counter(assign(profile(r), medians)
                              for r in rgb01).items())), flush=True)
    cols = 10
    rows = (args.n + cols - 1) // cols
    sheet = Image.new('RGB', (cols * SIZE, rows * SIZE))
    for i in range(args.n):
        sheet.paste(Image.open(pngdir / f"hf-{i:03d}.png"),
                    ((i % cols) * SIZE, (i // cols) * SIZE))
    sheet.save(pngdir / '_contact.png')
    print(f"contact sheet -> {pngdir / '_contact.png'}", flush=True)


if __name__ == '__main__':
    main()
