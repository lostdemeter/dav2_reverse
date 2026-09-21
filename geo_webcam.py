"""
Live webcam test for the FULLY-GEOMETRIC DAV2 pipeline.

Runs every frame through:
  GeometricDinov2Backbone -> GeometricNeck -> GeometricHead
(phi-encoded weights + LUT, baked in weights/geometric_*.npz —
 no `transformers` / HF download at inference).

Usage:
    python geo_webcam.py                  # interactive GUI (M/S/X like phi_depth.py)
    python geo_webcam.py --frames 5       # headless: capture 5 frames, save to captures/
    python geo_webcam.py --frames 5 --compare-hf   # also run HF baseline for parity

Controls (interactive): [M] colormap  [S] save  [X] quit
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from geo_depth import GeometricDepthAnythingV2

COLORMAPS = [
    (cv2.COLORMAP_MAGMA, 'magma'),
    (cv2.COLORMAP_VIRIDIS, 'viridis'),
    (cv2.COLORMAP_PLASMA, 'plasma'),
    (cv2.COLORMAP_INFERNO, 'inferno'),
    (cv2.COLORMAP_TURBO, 'turbo'),
]


def colorize(depth: np.ndarray, cmap) -> np.ndarray:
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax > dmin:
        norm = ((depth - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(depth, dtype=np.uint8)
    return cv2.applyColorMap(norm, cmap)


def main():
    ap = argparse.ArgumentParser(description='Fully-geometric DAV2 webcam test')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--frames', type=int, default=0,
                    help='0 = interactive GUI, N = headless capture N frames')
    ap.add_argument('--compare-hf', action='store_true',
                    help='also run HF baseline on captured frames for parity')
    ap.add_argument('--no-fp16', action='store_true')
    ap.add_argument('--cpu', action='store_true',
                    help='force CPU-only (no GPU needed, slower, smaller default size)')
    ap.add_argument('--size', type=int, default=None,
                    help='input long side (default 518, use 364/336 on CPU for speed)')
    args = ap.parse_args()

    if args.size is None:
        args.size = 364 if args.cpu else 518
    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    use_fp16 = (not args.no_fp16) and device.type == 'cuda'
    print(f"device: {device}  fp16: {use_fp16}")

    print("loading fully-geometric pipeline (backbone+neck+head, no HF)...")
    geo = GeometricDepthAnythingV2(device=device)
    if use_fp16:
        for mod in (geo.backbone, geo.neck, geo.head):
            for k, v in mod._w.items():
                mod._w[k] = v.half()
    geo.eval()
    print("ready.")

    hf = hf_proc = None
    if args.compare_hf:
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        from PIL import Image
        print("loading HF baseline for comparison...")
        hf_proc = AutoImageProcessor.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf')
        hf = AutoModelForDepthEstimation.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf').to(device).eval()
        if use_fp16:
            hf = hf.half()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"error: could not open camera {args.camera}")
        raise SystemExit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    outdir = Path('./captures')
    outdir.mkdir(exist_ok=True)

    def process(rgb_float):
        """RGB float [0,1] -> geometric depth float32."""
        from geo_depth import preprocess
        pv = preprocess(rgb_float, size=args.size)
        if use_fp16:
            pv = pv.half()
        with torch.no_grad():
            d = geo.forward(pv).squeeze(0).float().cpu().numpy()
        return d

    if args.frames > 0:
        # headless capture test
        corrs = []
        for i in range(args.frames):
            ret, frame = cap.read()
            if not ret:
                print("error: could not read frame")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t0 = time.perf_counter()
            depth = process(rgb)
            ms = (time.perf_counter() - t0) * 1000
            colored = colorize(depth, COLORMAPS[0][0])
            colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))
            cv2.imwrite(str(outdir / f'geo_webcam_{i}_combined.png'),
                        np.hstack([frame, colored]))
            line = f"frame {i}: geo {ms:.0f}ms ({1000 / ms:.1f} FPS) depth{depth.shape}"
            if hf is not None:
                from PIL import Image
                inputs = hf_proc(images=Image.fromarray((rgb * 255).astype(np.uint8)),
                                 return_tensors='pt')
                pv = inputs['pixel_values'].to(device)
                if use_fp16:
                    pv = pv.half()
                with torch.no_grad():
                    base = hf(pixel_values=pv).predicted_depth.squeeze().float().cpu().numpy()
                if base.shape != depth.shape:
                    import cv2 as _cv
                    depth_r = _cv.resize(depth, (base.shape[1], base.shape[0]))
                else:
                    depth_r = depth
                c = float(np.corrcoef(base.flatten().astype(np.float64),
                                      depth_r.flatten().astype(np.float64))[0, 1])
                corrs.append(c)
                line += f"  HF parity corr={c:.6f}"
            print(line, flush=True)
        if corrs:
            print(f"mean HF parity corr: {np.mean(corrs):.6f}")
        cap.release()
        print(f"saved to {outdir}/geo_webcam_*_combined.png")
        return

    # interactive GUI
    cmap_idx = 0
    win = []
    print("\ncontrols: [M] colormap  [S] save  [X] quit\n")
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth = process(rgb)
        colored = colorize(depth, COLORMAPS[cmap_idx][0])
        colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))
        dt = time.perf_counter() - t0
        win.append(dt)
        if len(win) > 30:
            win.pop(0)
        fps = len(win) / sum(win)
        cv2.putText(colored, f"FPS: {fps:.1f} | {COLORMAPS[cmap_idx][1]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(colored, "geo-DAV2 (backbone+neck+head)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('geo-DAV2', np.hstack([frame, colored]))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('x'), ord('X')):
            break
        elif key in (ord('m'), ord('M')):
            cmap_idx = (cmap_idx + 1) % len(COLORMAPS)
        elif key in (ord('s'), ord('S')):
            ts = int(time.time())
            cv2.imwrite(str(outdir / f'geo_webcam_{ts}_combined.png'),
                        np.hstack([frame, colored]))
            print(f"saved {outdir}/geo_webcam_{ts}_combined.png")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
