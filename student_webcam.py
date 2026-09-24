"""Live webcam test for the DISTILLED student backbone (bank-and-encode).

Runs every frame through:
  StudentBackbone (cosine r128 factors, L0-2 bottlenecks) -> neck -> head
vs the teacher (exact maps) side by side + optional HF parity.
This is the distillation program's artifact running live.

Usage:
    python student_webcam.py --frames 5        # headless + parity numbers
    python student_webcam.py --frames 5 --compare-hf
    python student_webcam.py                    # interactive GUI
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).parent
CKPT = HERE / 'adapt' / 'runs' / 'student_grad_best_r128_cos.pt'
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


def load_student(device, ckpt_path=None):
    import sys
    sys.path.insert(0, str(HERE / 'adapt'))
    from student_grad import StudentBackbone
    from run_search import load_shared
    from geo_neck import GeometricNeck
    shared = load_shared(device)
    student = StudentBackbone(shared['backbone'])
    ckpt_path = Path(ckpt_path) if ckpt_path else CKPT
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"checkpoint {ckpt_path.name} mean_corr={ckpt['mean_corr']:.5f}",
          flush=True)
    with torch.no_grad():
        for (li, m), tup in ckpt['factors'].items():
            A, B, b = student.factors[(li, m)]
            A.copy_(tup[0].to(device))
            B.copy_(tup[1].to(device))
            b.copy_(tup[2].to(device))
    neck = GeometricNeck(HERE / 'weights' / 'geometric_neck.npz',
                         device=device)
    return shared, student, neck


def main():
    ap = argparse.ArgumentParser(description='Distilled-student webcam test')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--frames', type=int, default=0,
                    help='0 = interactive GUI, N = headless capture N frames')
    ap.add_argument('--compare-hf', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--size', type=int, default=None)
    ap.add_argument('--ckpt', type=str, default=None,
                    help='student checkpoint (default: banked cosine r128)')
    args = ap.parse_args()

    if args.size is None:
        args.size = 364 if args.cpu else 518
    device = torch.device('cpu' if args.cpu else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"device: {device}")

    print("loading teacher pipeline + student backbone...")
    shared, student, neck = load_student(device, args.ckpt)
    teacher_bb = shared['backbone']
    head = shared['head']
    pre = shared['preprocess']
    print("ready.")

    hf = hf_proc = None
    if args.compare_hf:
        from transformers import (AutoModelForDepthEstimation,
                                  AutoImageProcessor)
        from PIL import Image
        print("loading HF baseline...")
        hf_proc = AutoImageProcessor.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf')
        hf = AutoModelForDepthEstimation.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf').to(device).eval()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"error: could not open camera {args.camera}")
        raise SystemExit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    outdir = Path('./captures')
    outdir.mkdir(exist_ok=True)

    def process(rgb_float):
        from geo_depth import preprocess
        pv = preprocess(rgb_float, size=args.size).to(device)
        with torch.no_grad():
            fmaps, ph, pw = student.forward_backbone(pv)
            fused = neck(fmaps)
            d_stud = head(fused, ph, pw).squeeze(0).float().cpu().numpy()
            fmaps_t, _, _ = teacher_bb.forward_stages(pv)
            fused_t = neck(fmaps_t)
            d_teach = head(fused_t, ph, pw).squeeze(0).float().cpu().numpy()
        return d_stud, d_teach

    if args.frames > 0:
        corrs, ms = [], []
        for i in range(args.frames):
            ret, frame = cap.read()
            if not ret:
                print("error: could not read frame")
                break
            rgb = cv2.cvtColor(frame,
                               cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t0 = time.perf_counter()
            d_stud, d_teach = process(rgb)
            ms.append((time.perf_counter() - t0) * 1000)
            c = float(np.corrcoef(d_stud.flatten().astype(np.float64),
                                  d_teach.flatten().astype(np.float64))[0, 1])
            corrs.append(c)
            scol = colorize(d_stud, COLORMAPS[0][0])
            tcol = colorize(d_teach, COLORMAPS[0][0])
            scol = cv2.resize(scol, (frame.shape[1], frame.shape[0]))
            tcol = cv2.resize(tcol, (frame.shape[1], frame.shape[0]))
            cv2.imwrite(str(outdir / f'student_webcam_{i}_combined.png'),
                        np.hstack([frame, tcol, scol]))
            line = (f"frame {i}: {ms[-1]:.0f}ms "
                    f"student-vs-teacher corr={c:.6f}")
            if hf is not None:
                from PIL import Image
                inp = hf_proc(images=Image.fromarray(
                    (rgb * 255).astype(np.uint8)), return_tensors='pt')
                with torch.no_grad():
                    base = hf(pixel_values=inp['pixel_values'].to(device)
                              ).predicted_depth.squeeze().float().cpu().numpy()
                import cv2 as _cv
                ds = _cv.resize(d_stud, (base.shape[1], base.shape[0]))
                ch = float(np.corrcoef(base.flatten().astype(np.float64),
                                       ds.flatten().astype(np.float64))[0, 1])
                line += f"  student-vs-HF corr={ch:.6f}"
            print(line, flush=True)
        print(f"mean student-vs-teacher corr: {np.mean(corrs):.6f} "
              f"({np.mean(ms):.0f}ms/frame)")
        cap.release()
        return

    cmap_idx = 0
    print("\ncontrols: [M] colormap  [S] save  [X] quit\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        d_stud, d_teach = process(rgb)
        scol = colorize(d_stud, COLORMAPS[cmap_idx][0])
        tcol = colorize(d_teach, COLORMAPS[cmap_idx][0])
        scol = cv2.resize(scol, (frame.shape[1], frame.shape[0]))
        tcol = cv2.resize(tcol, (frame.shape[1], frame.shape[0]))
        view = np.hstack([frame, tcol, scol])
        cv2.putText(view, "camera | teacher | distilled student",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.imshow('distilled student', view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('x'), ord('X')):
            break
        elif key in (ord('m'), ord('M')):
            cmap_idx = (cmap_idx + 1) % len(COLORMAPS)
        elif key in (ord('s'), ord('S')):
            ts = int(time.time())
            cv2.imwrite(str(outdir / f'student_webcam_{ts}_combined.png'), view)
            print(f"saved {outdir}/student_webcam_{ts}_combined.png")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
