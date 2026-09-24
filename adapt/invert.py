#!/usr/bin/env python3
"""Inversion microscope: reproduce a measured activation pattern live.

Target: L0-block activations of a chosen scene (default: a saved dark
webcam frame the student mishandles). Noise init -> Adam on pixels to
minimize relative MSE vs the measured pattern (+TV + jitter), through
the TEACHER (default) or --student. Live cv2 window shows target
scene | evolving reconstruction | loss curve; headless --frames N
saves a progress strip to captures/.

What "works" means (pre-registered 2026-09-24): teacher pattern match
>=0.99 (relative MSE <=1e-4-ish on normalized acts); the reconstruction
need not look like the scene (diagnostic, not art). Teacher-vs-student
inversion difference is the follow-up analysis, not this script.
Run: python adapt/invert.py [--student] [--steps 300]
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np

LI = 0
TV_W = 1e-3
LR = 0.1
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--student', action='store_true',
                    help='invert through the student (default: teacher)')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--frames', type=int, default=0,
                    help='0 = live GUI, N = headless, save strip every N steps')
    ap.add_argument('--scene', default='captures/student_webcam_0_combined.png',
                    help='combined PNG (raw frame = left third) or RGB file')
    ap.add_argument('--ckpt', default='adapt/runs/student_grad_best_r128_cos.pt')
    args = ap.parse_args()

    import torch
    import cv2
    from run_search import load_shared
    device = torch.device('cuda')
    shared = load_shared(device)
    bb = shared['backbone']
    pre = shared['preprocess']

    # target scene -> measured L0 pattern (teacher, no_grad, exact path)
    img = cv2.imread(args.scene)
    if img is None:
        raise SystemExit(f"cannot read {args.scene}")
    h, w = img.shape[:2]
    raw = img[:, :w // 3] if w > h else img  # combined PNG: raw = left third
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    from geo_depth import preprocess as _pp
    with torch.no_grad():
        cap = {}
        bb.forward_stages(_pp(rgb, size=518).to(device), capture=cap)
        T = {k: v.detach().clone()
             for k, v in cap[(LI, 'blk')].items()
             if k in ('H', 'Q', 'K', 'V', 'C', 'Oattn', 'N2', 'P1', 'G', 'Y')}
        scales = {k: v.double().var().clamp_min(1e-12).item()
                  for k, v in T.items()}
    print(f"target: {args.scene} ({rgb.shape[1]}x{rgb.shape[0]}), "
          f"{len(T)} tensors", flush=True)

    fwd = bb.forward_stages
    if args.student:
        from student_grad import StudentBackbone
        st = StudentBackbone(bb)
        ckpt = torch.load(REPO / args.ckpt, map_location=device,
                          weights_only=False)
        print(f"student ckpt mean={ckpt['mean_corr']:.5f}", flush=True)
        with torch.no_grad():
            for (li, m), tup in ckpt['factors'].items():
                A, B, b = st.factors[(li, m)]
                A.copy_(tup[0].to(device))
                B.copy_(tup[1].to(device))
                b.copy_(tup[2].to(device))

        def fwd(pv, capture=None):
            return st.forward_backbone(pv, capture=capture)

    torch.manual_seed(7)
    canvas = torch.randn(1, 3, 518, 518, device=device) * 0.5
    canvas.requires_grad_(True)
    opt = torch.optim.Adam([canvas], lr=LR)
    rng = np.random.default_rng(3)
    outdir = REPO / 'captures' / 'invert'
    outdir.mkdir(parents=True, exist_ok=True)
    hist = []
    show = args.frames == 0

    def denorm(t):
        o = t.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        return np.clip(o * STD[None, None, :] + MEAN[None, None, :], 0, 1)

    for s in range(args.steps):
        opt.zero_grad()
        dy, dx = int(rng.integers(-16, 17)), int(rng.integers(-16, 17))
        j = torch.roll(canvas, shifts=(dy, dx), dims=(2, 3))
        cap = {}
        fwd(j, capture=cap)
        B = cap[(LI, 'blk')]
        terms = [((B[k].float() - T[k]) ** 2).mean() / scales[k]
                 for k in T]
        fit = torch.stack(terms).mean()
        tv = ((j[:, :, 1:, :] - j[:, :, :-1, :]).abs().mean() +
              (j[:, :, :, 1:] - j[:, :, :, :-1]).abs().mean())
        (fit + TV_W * tv).backward()
        opt.step()
        with torch.no_grad():
            canvas.clamp_(-3.0, 3.0)
        if (s + 1) % 10 == 0 or s == 0:
            m = 1.0 - float(fit.detach())
            hist.append(m)
            print(f"step {s + 1}/{args.steps}: match={m:.5f}", flush=True)
        if show and (s + 1) % 5 == 0:
            rec = denorm(canvas)
            tgt = cv2.resize(rgb, (rec.shape[1], rec.shape[0]))
            curve = np.zeros((120, 300, 3), dtype=np.float32)
            if len(hist) > 1:
                ys = (np.array(hist) * 110).astype(int).clip(0, 119)
                xs = (np.arange(len(ys)) / max(len(ys) - 1, 1)
                      * 299).astype(int)
                for x, y in zip(xs, ys):
                    curve[119 - y, x] = (0.2, 1.0, 0.3)
            curve = cv2.resize(curve, (tgt.shape[1], 120))
            view = np.vstack([np.hstack([tgt, rec]),
                              np.hstack([curve, np.zeros_like(curve)])])
            cv2.imshow('inversion: target | reconstruction (X quits)',
                       (view * 255).astype(np.uint8))
            if cv2.waitKey(1) & 0xFF in (ord('x'), ord('X')):
                break
        if not show and (s + 1) % args.frames == 0:
            cv2.imwrite(str(outdir / f"prog_{s + 1:04d}.png"),
                        (denorm(canvas) * 255).astype(np.uint8))
    if show:
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(str(outdir / 'final.png'),
                    (denorm(canvas) * 255).astype(np.uint8))
        cv2.imwrite(str(outdir / 'target.png'),
                    (rgb * 255).astype(np.uint8))
        print(f"saved strip to {outdir}/prog_*.png", flush=True)
    print(f"INVERT: final match={hist[-1]:.5f} "
          f"({'PASS' if hist[-1] >= 0.99 else 'FAIL'}, bar 0.99)")


if __name__ == '__main__':
    main()
