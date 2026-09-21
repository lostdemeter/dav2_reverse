"""
Parity test: fully-geometric pipeline vs HF baseline.

Uses the HF processor's pixel_values as shared input (isolates
backbone/neck/head replication from preprocessing), on:
  1. horizontal gradient 518x518
  2. checkerboard 518x518
Reports per-case correlation, relative MAE, max abs error, shapes.
Pass threshold: correlation > 0.999 (phi K=512 quantization).

Run:
    python test_geometric_parity.py
Needs: transformers (baseline only), baked weights/geometric_*.npz.
"""

import numpy as np
import torch

PASS_CORR = 0.999


def make_images():
    h = w = 518
    xs = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
    grad = np.stack([xs, xs, xs], axis=-1)
    n = 8
    sq = 518 // n
    cb = np.zeros((h, w), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            cb[i * sq:(i + 1) * sq, j * sq:(j + 1) * sq] = (i + j) % 2
    checker = np.stack([cb, cb, cb], axis=-1)
    return {'gradient': grad, 'checker': checker}


def main():
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    from PIL import Image
    from geo_depth import GeometricDepthAnythingV2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    print("loading HF baseline...")
    processor = AutoImageProcessor.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf')
    hf = AutoModelForDepthEstimation.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf').to(device).eval()

    print("loading geometric pipeline...")
    geo = GeometricDepthAnythingV2(device=device)

    ok_all = True
    for name, rgb in make_images().items():
        pil = Image.fromarray((rgb * 255).astype(np.uint8))
        inputs = processor(images=pil, return_tensors='pt')
        pv = inputs['pixel_values'].to(device)
        with torch.no_grad():
            base = hf(pixel_values=pv).predicted_depth.squeeze().cpu().numpy()
            g = geo.forward(pv).squeeze(0).cpu().numpy()
        # align shapes (both should be 518x518 already)
        if base.shape != g.shape:
            from scipy.ndimage import zoom
            zy, zx = base.shape[0] / g.shape[0], base.shape[1] / g.shape[1]
            g = zoom(g, (zy, zx), order=1)
        b = base.flatten().astype(np.float64)
        v = g.flatten().astype(np.float64)
        corr = float(np.corrcoef(b, v)[0, 1])
        mae = float(np.mean(np.abs(b - v)) / (np.mean(np.abs(b)) + 1e-9))
        mx = float(np.max(np.abs(b - v)))
        ok = corr > PASS_CORR
        ok_all &= ok
        print(f"[{name}] base{base.shape} geo{g.shape} "
              f"corr={corr:.6f} relMAE={mae:.4f} maxAbs={mx:.4f} "
              f"{'PASS' if ok else 'FAIL'}")

    print("ALL PASS" if ok_all else "FAILURE — inspect quantization or arch mismatch")
    return 0 if ok_all else 1


if __name__ == '__main__':
    raise SystemExit(main())
