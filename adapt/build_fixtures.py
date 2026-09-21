#!/usr/bin/env python3
"""One-time fixture build: synthetic scenes + HF oracle reference depths.

Writes adapt/fixtures/scenes.npz (gitignored). Roles: 6 exploration,
4 gate, 3 retention anchors, 4 sealed audit. All 168px. The oracle is the
HF model — hidden from the learner exactly as the foundry pattern asks.
"""
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
FIX = HERE / 'fixtures' / 'scenes.npz'
SIZE = 168


def make_scene(seed: int, h: int = SIZE, w: int = SIZE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    scene = yy / max(h - 1, 1)
    for _ in range(int(rng.integers(1, 4))):
        cy, cx = rng.uniform(0.2, 0.8, 2) * np.array([h, w])
        r = rng.uniform(0.08, 0.22) * min(h, w)
        disc = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
        scene[disc] = rng.uniform(0.05, 0.4)
    n = int(rng.integers(4, 10))
    checker = (((yy // (h // n)) + (xx // (w // n))) % 2).astype(bool)
    low = yy > h * rng.uniform(0.5, 0.75)
    scene = np.where(checker & low, scene * 0.92 + 0.04, scene)
    tint = rng.uniform(0.85, 1.0, 3).astype(np.float32)
    return np.clip(scene[..., None] * tint, 0, 1).astype(np.float32)


def anchors():
    h = w = SIZE
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = np.stack([yy / (h - 1)] * 3, axis=-1).astype(np.float32)
    cb = (((yy // 21) + (xx // 21)) % 2).astype(np.float32)
    checker = np.stack([cb] * 3, axis=-1)
    disc = np.full((h, w, 3), 0.7, dtype=np.float32)
    disc[(yy - h / 2) ** 2 + (xx - w / 2) ** 2 < (h * 0.2) ** 2] = 0.15
    return [grad, checker, disc]


def main():
    import torch
    from PIL import Image
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc = AutoImageProcessor.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf')
    model = AutoModelForDepthEstimation.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf').to(device).eval()

    def oracle(rgb):
        with torch.no_grad():
            pv = proc(images=Image.fromarray((rgb * 255).astype(np.uint8)),
                      return_tensors='pt')['pixel_values'].to(device)
            d = model(pixel_values=pv).predicted_depth.squeeze().cpu().numpy()
        if d.shape != (SIZE, SIZE):
            import cv2
            d = cv2.resize(d, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        return d.astype(np.float32)

    roles = {'exploration': [make_scene(100 + i) for i in range(6)],
             'gate': [make_scene(200 + i) for i in range(4)],
             'retention': anchors(),
             'audit': [make_scene(300 + i) for i in range(4)]}
    out = {}
    for role, scenes in roles.items():
        rgbs, refs = [], []
        for i, rgb in enumerate(scenes):
            ref = oracle(rgb)
            rgbs.append((rgb * 255).astype(np.uint8))
            refs.append(ref)
            print(f"  {role}[{i}] ref range [{ref.min():.3f},{ref.max():.3f}]", flush=True)
        out[f'{role}_rgb'] = np.stack(rgbs)
        out[f'{role}_ref'] = np.stack(refs)
    FIX.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIX, **out)
    print(f'wrote {FIX} ({FIX.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
