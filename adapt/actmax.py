#!/usr/bin/env python3
"""Activation-maximization pilot: what stimuli do early neurons ask for?

Gradient ascent on 518px pixels (ImageNet-mean init) maximizing L0
post-GELU channels (high-variance picks) + one L0 attention-output
channel. Priors: random-roll jitter + TV + L2. Reports selectivity
(target vs other channels; synth-vs-best-of-10-reals drive) and saves
stimuli PNGs to adapt/runs/actmax/.
Pre-registered 2026-09-23: edge/texture structure, selectivity >5x.
Run: python adapt/actmax.py [--steps N]
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np

LI = 0
NSTEPS = 200
LR = 0.05
TV_W = 1e-4
L2_W = 1e-5
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pre0(shared, rgb):
    return shared['preprocess'](rgb)


def pick_channels(shared, device, n_mlp=3):
    """High-variance L0 post-GELU channels on 3 real scenes."""
    import torch
    bb, pre = shared['backbone'], shared['preprocess']
    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    acc = None
    with torch.no_grad():
        for i in range(3):
            cap = {}
            bb.forward_stages(
                pre(zr['rgb'][i].astype(np.float32) / 255.0).to(device),
                capture=cap)
            G = cap[(LI, 'blk')]['G'].squeeze(0).double().cpu().numpy()
            v = G.var(axis=0)
            acc = v if acc is None else acc + v
    order = np.argsort(acc)[::-1]
    return [int(c) for c in order[:n_mlp]]


def init_is_tensor(x):
    import torch
    return isinstance(x, torch.Tensor)


def synth_channel(shared, device, kind, ch, steps, init=None, lr=LR):
    """Maximize mean activation of (kind, ch) over non-CLS tokens."""
    import torch
    bb = shared['backbone']
    torch.manual_seed(0)
    if init is None:
        img = torch.zeros(1, 3, 518, 518, device=device)  # = mean after norm
    else:
        img = init.clone().detach().to(device).requires_grad_(True)
    if not init_is_tensor(init):
        img.requires_grad_(True)
    opt = torch.optim.Adam([img], lr=lr)
    rng = np.random.default_rng(1)
    for s in range(steps):
        opt.zero_grad()
        # jitter: random roll (translation prior)
        dy, dx = int(rng.integers(-16, 17)), int(rng.integers(-16, 17))
        jimg = torch.roll(img, shifts=(dy, dx), dims=(2, 3))
        cap = {}
        bb.forward_stages(jimg, capture=cap)
        B = cap[(LI, 'blk')]
        act = B['G'][:, 1:, ch].mean() if kind == 'mlp' else \
            B['Oattn'][:, 1:, ch].mean()
        tv = ((jimg[:, :, 1:, :] - jimg[:, :, :-1, :]).abs().mean() +
              (jimg[:, :, :, 1:] - jimg[:, :, :, :-1]).abs().mean())
        l2 = (jimg ** 2).mean()
        (-act + TV_W * tv + L2_W * l2).backward()
        opt.step()
    with torch.no_grad():
        cap = {}
        bb.forward_stages(img, capture=cap)
        B = cap[(LI, 'blk')]
        final = (B['G'][:, 1:, ch].mean().item() if kind == 'mlp' else
                 B['Oattn'][:, 1:, ch].mean().item())
    # denormalize to RGB for saving
    out = img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    out = np.clip(out * STD[None, None, :] + MEAN[None, None, :], 0, 1)
    return out, final


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=NSTEPS)
    ap.add_argument('--init', default='mean',
                    help="'mean' or 'real:IDX' (COCO scene start)")
    ap.add_argument('--lr', type=float, default=LR)
    args = ap.parse_args()
    import torch
    from run_search import load_shared
    device = torch.device('cuda')
    shared = load_shared(device)

    targets = [('mlp', c) for c in pick_channels(shared, device)] + \
              [('attn', 0)]
    print(f"targets: {targets} init={args.init} lr={args.lr}", flush=True)
    init_pv = None
    tag = "mean"
    if args.init.startswith('real:'):
        idx = int(args.init.split(':')[1])
        zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
        init_pv = pre0(shared,
                       zr['rgb'][idx].astype(np.float32) / 255.0).to(device)
        tag = f"real{idx}"
    outdir = ADAPT / 'runs' / 'actmax'
    outdir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    stims = {}
    for kind, ch in targets:
        rgb, final = synth_channel(shared, device, kind, ch, args.steps,
                                   init=init_pv, lr=args.lr)
        Image.fromarray((rgb * 255).astype(np.uint8)).save(
            outdir / f"{kind}{ch}_{tag}.png")
        stims[(kind, ch)] = (rgb, final)
        print(f"{kind}{ch}: final act={final:.4f}", flush=True)

    # selectivity: each stimulus through, all target channels read
    import torch as _t
    bb, pre = shared['backbone'], shared['preprocess']
    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    reals = [zr['rgb'][i].astype(np.float32) / 255.0 for i in range(10)]

    def read(rgb01=None, pv=None):
        with _t.no_grad():
            cap = {}
            bb.forward_stages(pre(rgb01).to(device) if pv is None else pv,
                              capture=cap)
            B = cap[(LI, 'blk')]
            return {('mlp', c): B['G'][:, 1:, c].mean().item()
                    for _, c in targets if _ == 'mlp'} | \
                   {('attn', 0): B['Oattn'][:, 1:, 0].mean().item()}

    base = {k: np.mean([read(r)[k] for r in reals]) for k in targets}
    best_real = {k: max(read(r)[k] for r in reals) for k in targets}
    print("selectivity (stim rows x channel cols):", flush=True)
    hdr = "stim\\" + " ".join(f"{k[0]}{k[1]}" for k in targets)
    print(hdr, flush=True)
    for (kind, ch), (rgb, final) in stims.items():
        got = read(rgb)
        row = " ".join(f"{got[k] / max(base[k], 1e-9):7.1f}x"
                       for k in targets)
        print(f"{kind}{ch} {row}  (vs-best-real "
              + " ".join(f"{got[k] / max(best_real[k], 1e-9):.1f}x"
                         for k in targets) + ")", flush=True)


if __name__ == '__main__':
    main()
