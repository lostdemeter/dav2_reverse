"""
Demonstration: HuggingFace DAV2 -> fully-geometric DAV2.

No webcam needed. Builds a synthetic scene, runs the HF baseline
(`Depth-Anything-V2-Small-hf`) and our geometric pipeline
(backbone+neck+head, phi-encoded + LUT) on the SAME input, and saves
a side-by-side figure proving parity:

    docs/geometric_parity.png  (input | HF depth | geometric depth | |error|)

If baked weights are missing, they are built first automatically via
export_geometric_weights.py (one-time HF download, ~94MB).

Run:
    python demo_geometric.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

BASE = Path(__file__).parent
DOCS = BASE / 'docs'


def make_scene(h: int = 518, w: int = 518) -> np.ndarray:
    """Synthetic scene: vertical gradient + foreground disc + checker floor."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    background = yy / max(h - 1, 1)
    cy, cx, r = h * 0.45, w * 0.5, min(h, w) * 0.18
    disc = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
    scene = background.copy()
    scene[disc] = 0.15  # close object = small depth value
    n = 8
    sqh, sqw = h // n, w // n
    checker = (((yy // sqh) + (xx // sqw)) % 2).astype(bool)
    lowband = yy > h * 0.6
    scene = np.where(checker & lowband, scene * 0.92 + 0.04, scene)
    rgb = np.stack([scene, scene, scene], axis=-1)
    # tint for a nicer input panel
    tinted = np.stack([scene,
                       scene * 0.95 + 0.03,
                       scene * 0.85 + 0.08], axis=-1)
    return np.clip(tinted, 0, 1).astype(np.float32)


def colorize(depth: np.ndarray) -> np.ndarray:
    dmin, dmax = float(depth.min()), float(depth.max())
    norm = ((depth - dmin) / (dmax - dmin + 1e-12) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)


def ensure_weights():
    needed = ['geometric_backbone.npz', 'geometric_neck.npz', 'geometric_head.npz']
    missing = [n for n in needed if not (BASE / 'weights' / n).exists()]
    if missing:
        print(f"baking missing weights {missing} from HF (one-time)...")
        from export_geometric_weights import main as bake
        bake()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    ensure_weights()

    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    from geo_depth import GeometricDepthAnythingV2

    print("loading HF DAV2 (baseline)...")
    proc = AutoImageProcessor.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf')
    hf = AutoModelForDepthEstimation.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf').to(device).eval()

    print("loading geometric DAV2 (backbone+neck+head)...")
    geo = GeometricDepthAnythingV2(device=device)

    rgb = make_scene()
    inputs = proc(images=Image.fromarray((rgb * 255).astype(np.uint8)),
                  return_tensors='pt')
    pv = inputs['pixel_values'].to(device)
    print(f"shared input: {tuple(pv.shape)}")

    with torch.no_grad():
        t0 = time.perf_counter()
        base = hf(pixel_values=pv).predicted_depth.squeeze().cpu().numpy()
        t_hf = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        g = geo.forward(pv).squeeze(0).cpu().numpy()
        t_geo = (time.perf_counter() - t0) * 1000

    if base.shape != g.shape:
        g = cv2.resize(g, (base.shape[1], base.shape[0]))
    b = base.flatten().astype(np.float64)
    v = g.flatten().astype(np.float64)
    corr = float(np.corrcoef(b, v)[0, 1])
    err = np.abs(base - g)
    print(f"HF: {t_hf:.0f}ms  geometric: {t_geo:.0f}ms  corr={corr:.6f} "
          f"maxAbsErr={err.max():.4f}")

    h, w = base.shape
    rgb_u8 = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    rgb_u8 = cv2.resize(rgb_u8, (w, h))
    panels = [rgb_u8, colorize(base), colorize(g),
              colorize(err / (err.max() + 1e-12))]
    labels = ['input (synthetic)', f'HF DAV2 ({t_hf:.0f}ms)',
              f'geometric DAV2 ({t_geo:.0f}ms)', f'|error| corr={corr:.5f}']
    fig = np.hstack(panels)
    for i, lab in enumerate(labels):
        cv2.putText(fig, lab, (i * w + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    DOCS.mkdir(exist_ok=True)
    out = DOCS / 'geometric_parity.png'
    cv2.imwrite(str(out), fig)
    print(f"saved {out}")
    return 0 if corr > 0.999 else 1


if __name__ == '__main__':
    raise SystemExit(main())
