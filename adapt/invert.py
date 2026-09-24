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


def block_tensors_diff(bb, x, li):
    """Differentiable L0-block tensors (capture detaches; this doesn't).

    Mirrors forward_stages' math exactly (verified below by maxabs
    vs capture on the target pattern); gradients flow to x.
    """
    import torch
    import torch.nn.functional as F
    from geo_backbone import HEADS, HEAD_DIM, HIDDEN
    g = bb._w.__getitem__ if hasattr(bb._w, '__getitem__') else None
    p = f'layer{li}.'
    W = bb._w
    H = F.layer_norm(x, (HIDDEN,), W[p + 'norm1.weight'],
                     W[p + 'norm1.bias'])
    Q = F.linear(H, W[p + 'q.weight'], W[p + 'q.bias'])
    K = F.linear(H, W[p + 'k.weight'], W[p + 'k.bias'])
    V = F.linear(H, W[p + 'v.weight'], W[p + 'v.bias'])
    B, N, _ = H.shape
    q = Q.view(B, -1, HEADS, HEAD_DIM).transpose(1, 2)
    k = K.view(B, -1, HEADS, HEAD_DIM).transpose(1, 2)
    v = V.view(B, -1, HEADS, HEAD_DIM).transpose(1, 2)
    attn = (q @ k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
    attn = attn.softmax(dim=-1)
    C = (attn @ v).transpose(1, 2).contiguous().view(B, -1, HIDDEN)
    Oattn = F.linear(C, W[p + 'proj.weight'], W[p + 'proj.bias'])
    M = x + Oattn * W[p + 'ls1']
    N2 = F.layer_norm(M, (HIDDEN,), W[p + 'norm2.weight'],
                      W[p + 'norm2.bias'])
    P1 = F.linear(N2, W[p + 'mlp1.weight'], W[p + 'mlp1.bias'])
    G = F.gelu(P1)
    Y = F.linear(G, W[p + 'mlp2.weight'], W[p + 'mlp2.bias'])
    return {'H': H, 'Q': Q, 'K': K, 'V': V, 'C': C, 'Oattn': Oattn,
            'M': M, 'N2': N2, 'P1': P1, 'G': G, 'Y': Y}


def student_block_tensors_diff(st, x, li):
    """Same, through student bottleneck factors (for --student)."""
    import torch
    import torch.nn.functional as F
    from geo_backbone import HEADS, HEAD_DIM, HIDDEN
    bb = st.teacher
    W = bb._w
    p = f'layer{li}.'

    def lin(h, name):
        key = (li, name)
        if key in st.factors:
            A, B, b = st.factors[key]
            return F.linear(F.linear(h, A, None), B, b)
        return F.linear(h, W[p + name],
                        W.get(p + name.replace('.weight', '.bias')))
    H = F.layer_norm(x, (HIDDEN,), W[p + 'norm1.weight'],
                     W[p + 'norm1.bias'])
    Q, K, V = lin(H, 'q.weight'), lin(H, 'k.weight'), lin(H, 'v.weight')
    B, N, _ = H.shape
    q = Q.view(B, -1, HEADS, HEAD_DIM).transpose(1, 2)
    k = K.view(B, -1, HEADS, HEAD_DIM).transpose(1, 2)
    v = V.view(B, -1, HEADS, HEAD_DIM).transpose(1, 2)
    attn = (q @ k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
    attn = attn.softmax(dim=-1)
    C = (attn @ v).transpose(1, 2).contiguous().view(B, -1, HIDDEN)
    Oattn = lin(C, 'proj.weight')
    M = x + Oattn * W[p + 'ls1']
    N2 = F.layer_norm(M, (HIDDEN,), W[p + 'norm2.weight'],
                      W[p + 'norm2.bias'])
    P1 = lin(N2, 'mlp1.weight')
    G = F.gelu(P1)
    Y = lin(G, 'mlp2.weight')
    return {'H': H, 'Q': Q, 'K': K, 'V': V, 'C': C, 'Oattn': Oattn,
            'M': M, 'N2': N2, 'P1': P1, 'G': G, 'Y': Y}


def embed_diff(bb, pv):
    """Differentiable patch-embed + CLS + pos (mirrors forward_stages).

    Verified by maxabs vs capture below; gradients flow to pv.
    """
    import torch
    import torch.nn.functional as F
    from geo_backbone import HIDDEN, PATCH
    _g = bb._w.__getitem__
    B, _, H, W = pv.shape
    x = pv.to(next(iter(bb._w.values())).device)
    ph, pw = H // PATCH, W // PATCH
    x = F.conv2d(x, _g('patch_proj.weight'), _g('patch_proj.bias'),
                 stride=PATCH)
    x = x.flatten(2).transpose(1, 2)
    cls_t = _g('cls_token')
    if cls_t.dim() == 1:
        cls_t = cls_t.view(1, 1, -1)
    elif cls_t.dim() == 2:
        cls_t = cls_t.unsqueeze(0) if cls_t.shape[0] == 1 \
            else cls_t.view(1, 1, -1)
    x = torch.cat([cls_t.expand(B, -1, -1), x], dim=1)
    pos = _g('pos_embed')
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
    if pos.shape[1] != x.shape[1]:
        cls_pos = pos[:, :1, :]
        patch_pos = pos[:, 1:, :].reshape(1, 37, 37, HIDDEN)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(ph, pw),
                                  mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, HIDDEN)
        pos = torch.cat([cls_pos, patch_pos], dim=1)
    return x + pos


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
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--clip', type=float, default=1.0,
                    help='gradient clip norm (0 = off)')
    ap.add_argument('--tv', type=float, default=TV_W)
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
    tgt_pv = _pp(rgb, size=518)
    with torch.no_grad():
        cap = {}
        bb.forward_stages(tgt_pv.to(device), capture=cap)
        T = {k: v.detach().clone()
             for k, v in cap[(LI, 'blk')].items()
             if k in ('H', 'Q', 'K', 'V', 'C', 'Oattn', 'N2', 'P1', 'G', 'Y')}
        scales = {k: v.double().var().clamp_min(1e-12).item()
                  for k, v in T.items()}
    print(f"target: {args.scene} ({rgb.shape[1]}x{rgb.shape[0]}), "
          f"{len(T)} tensors", flush=True)

    fwd = bb.forward_stages
    use_student = args.student
    st = None
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

    def block_diff(x):
        if use_student:
            return student_block_tensors_diff(st, x, LI)
        return block_tensors_diff(bb, x, LI)

    # verify: differentiable path matches capture exactly on target pv
    with torch.no_grad():
        cap = {}
        bb.forward_stages(tgt_pv.to(device), capture=cap)
        ref = {k: cap[(LI, 'blk')][k] for k in T}
        got = block_diff(embed_diff(bb, tgt_pv.to(device)))
        ma = max((got[k].double() - ref[k].double()).abs().max().item()
                 for k in T)
        print(f"diff-path verify maxabs vs capture: {ma:.2e} "
              f"({'OK' if ma < 1e-4 else 'MISMATCH'})", flush=True)

    torch.manual_seed(7)
    canvas = (torch.randn_like(tgt_pv) * 0.5).to(device)
    canvas.requires_grad_(True)
    opt = torch.optim.Adam([canvas], lr=args.lr)
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
        # NOTE: no jitter — the per-token target is position-locked,
        # so translation augmentation fights the fit (v1 stalled at
        # 0.48 with jitter; jitter suits translation-tolerant
        # channel-mean objectives, not pattern matching).
        j = canvas
        Bd = block_diff(embed_diff(bb, j))
        terms = [((Bd[k].float() - T[k]) ** 2).mean() / scales[k]
                 for k in T]
        fit = torch.stack(terms).mean()
        tv = ((j[:, :, 1:, :] - j[:, :, :-1, :]).abs().mean() +
              (j[:, :, :, 1:] - j[:, :, :, :-1]).abs().mean())
        (fit + args.tv * tv).backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_([canvas], args.clip)
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
