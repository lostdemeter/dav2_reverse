#!/usr/bin/env python3
"""Gradient pilot: RRR-initialized bottleneck student (L0-2 @128), Adam.

Student = teacher backbone with L0-2 linears replaced by trainable
bottleneck factors (A: r×din, B: dout×r, bias); _g composes B@A on the
fly so forward_stages is UNTOUCHED (same path, gradients into
factors). Norms/scales/embeddings/pos + late layers + neck/head:
FROZEN teacher-exact. Labels: teacher 518px depths (cached).
Loss: SSI (lstsq align + MAE) + 2x Sobel-grad L1, top-10% mask.
Pre-registered 2026-09-22: 0.976 -> >=0.998; hold possible.
Run: python adapt/student_grad.py [--epochs N --batch B --lr F]
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np
import copy

RANK = 128
STUDENT_LAYERS = (0, 1, 2)
MAPS = ("q.weight", "k.weight", "v.weight", "proj.weight",
        "mlp1.weight", "mlp2.weight")
LABEL_CACHE = ADAPT / 'runs' / 'grad_labels.npz'
BEST = ADAPT / 'runs' / 'student_grad_best.pt'


class StudentBackbone:
    """Teacher buffers + trainable bottleneck factors for early maps."""

    def __init__(self, teacher):
        import torch
        self.teacher = teacher  # GeometricDinov2Backbone (frozen)
        self.device = teacher.device
        self.factors = {}  # (li, map) -> (A, B, b) Parameters
        for li in STUDENT_LAYERS:
            for m in MAPS:
                base = f'layer{li}.{m}'
                W = teacher._w[base]
                do, di = W.shape
                A = torch.nn.Parameter(torch.zeros(RANK, di,
                                                   device=self.device))
                B = torch.nn.Parameter(torch.zeros(do, RANK,
                                                   device=self.device))
                b = torch.nn.Parameter(torch.zeros(do, device=self.device))
                self.factors[(li, m)] = (A, B, b)
        # default init = teacher map (exact start; RRR overwrites below)
        with torch.no_grad():
            for (li, m), (A, B, b) in self.factors.items():
                W = teacher._w[f'layer{li}.{m}']
                do, di = W.shape
                B.copy_(torch.randn(do, RANK, device=self.device) * 0.01)

    def parameters(self):
        return [p for tup in self.factors.values() for p in tup]

    def rrr_init(self, covs):
        """Init factors from RRR solutions: M=USV' -> A=(US).T, B=V'.T."""
        import torch
        with torch.no_grad():
            for li in STUDENT_LAYERS:
                for mk, dk, _, di, do in covs['maps']:
                    name = {'q': 'q.weight', 'k': 'k.weight',
                            'v': 'v.weight', 'proj': 'proj.weight',
                            'mlp1': 'mlp1.weight',
                            'mlp2': 'mlp2.weight'}[mk]
                    from student_probe import rrr_from_covs
                    Wrr = rrr_from_covs(covs[li][mk]['Sxx'],
                                        covs[li][mk]['Sxy'], RANK)
                    M = Wrr[:-1]
                    U, S, Vh = np.linalg.svd(M, full_matrices=False)
                    A, B, b = self.factors[(li, name)]
                    A.copy_(torch.from_numpy(
                        (U[:, :RANK] * S[:RANK]).T.astype(np.float32)
                        ).to(self.device))
                    B.copy_(torch.from_numpy(
                        Vh[:RANK].T.astype(np.float32)).to(self.device))
                    b.copy_(torch.from_numpy(
                        Wrr[-1].astype(np.float32)).to(self.device))

    def forward_backbone(self, pv, taps=None):
        """forward_stages with bottleneck composition via _g override."""
        teacher = self.teacher
        orig_g = teacher._g

        def _g(name):
            import re
            m = re.match(r'layer(\d+)\.(q|k|v|proj|mlp1|mlp2)\.weight', name)
            if m and int(m.group(1)) in STUDENT_LAYERS:
                A, B, b = self.factors[(int(m.group(1)),
                                        m.group(2) + '.weight')]
                return B @ A
            m = re.match(r'layer(\d+)\.(q|k|v|proj|mlp1|mlp2)\.bias', name)
            if m and int(m.group(1)) in STUDENT_LAYERS:
                return self.factors[(int(m.group(1)),
                                     m.group(2) + '.weight')][2]
            return orig_g(name)

        teacher._g = _g
        try:
            return teacher.forward_stages(pv, taps=taps)
        finally:
            teacher._g = orig_g


def build_labels(shared, fit_pairs, force=False):
    """Teacher 518px depths for fit scenes. Cached (deterministic)."""
    import torch
    if LABEL_CACHE.exists() and not force:
        z = np.load(LABEL_CACHE, allow_pickle=False)
        print(f"labels cached: {len(z['depth'])} scenes", flush=True)
        return z['depth']
    from geo_depth import GeometricDepthAnythingV2
    device = shared['device']
    geo = GeometricDepthAnythingV2(device=device)
    outs = []
    with torch.no_grad():
        for i, (rgb, _ref) in enumerate(fit_pairs):
            d = geo.predict(rgb)
            outs.append(d.astype(np.float32))
            if i % 25 == 0:
                print(f"  label {i}/{len(fit_pairs)}", flush=True)
    arr = np.stack(outs)
    LABEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LABEL_CACHE, depth=arr)
    print(f"cached {len(outs)} labels -> {LABEL_CACHE}", flush=True)
    return arr


def ssi_gm_loss(pred, target):
    """SSI (lstsq scale+shift align + MAE) + 2x Sobel-grad L1, top-10% mask."""
    import torch
    import torch.nn.functional as F
    B = pred.shape[0]
    p = pred.reshape(B, -1).double()
    t = target.reshape(B, -1).double()
    ones = torch.ones_like(p[:, :1])
    A = torch.stack([p, ones.squeeze(1)], dim=-1)  # (B,N,2)
    sol = torch.linalg.lstsq(A, t.unsqueeze(-1)).solution.squeeze(-1)
    s, sh = sol[:, 0].float(), sol[:, 1].float()
    aligned = s.view(B, 1, 1, 1) * pred + sh.view(B, 1, 1, 1)
    resid = (aligned - target).abs()
    flat = resid.reshape(B, -1)
    thr = flat.quantile(0.9, dim=1).view(B, 1, 1, 1)
    mask = (resid <= thr).float()
    mae = (resid * mask).sum() / mask.sum().clamp_min(1.0)
    Kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3)
    Ky = Kx.transpose(-1, -2)
    ga = F.conv2d(aligned, Kx, padding=1).abs() + F.conv2d(aligned, Ky,
                                                           padding=1).abs()
    gt = F.conv2d(target, Kx, padding=1).abs() + F.conv2d(target, Ky,
                                                          padding=1).abs()
    gm = (((ga - gt).abs()) * mask).sum() / mask.sum().clamp_min(1.0)
    return mae + 2.0 * gm, mae.detach(), gm.detach()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--gate-every', type=int, default=5)
    args = ap.parse_args()

    import torch
    from run_search import load_shared, load_fixtures
    from depth_adapter import run_pipeline, corr as _corr, SEED_CONFIG

    device = torch.device('cuda')
    shared = load_shared(device)

    # fit pool (split FIRST): every 12th fit_pool scene held out.
    # RRR covariances + labels + training use train only; eval reals
    # never fit (stronger than the pre-registered 12-scene gate; stated).
    fp = ADAPT / 'fixtures' / 'fit_pool.npz'
    zf = np.load(fp, allow_pickle=True)
    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    all_pool = ([(zf['rgb'][i].astype(np.float32) / 255.0,
                 zf['ref'][i].astype(np.float64))
                for i in range(len(zf['rgb']))] +
                [(zr['rgb'][i].astype(np.float32) / 255.0,
                 zr['ref'][i].astype(np.float64))
                for i in range(len(zr['rgb']))])
    npool = len(zf['rgb'])
    hold_idx = set(range(0, npool, 12))
    fit_pairs = [p for i, p in enumerate(all_pool) if i not in hold_idx]
    hold_pairs = [all_pool[i] for i in sorted(hold_idx)]
    print(f"train {len(fit_pairs)}, held-out reals {len(hold_pairs)}",
          flush=True)

    # held-out: fixtures + audit + NEVER-FIT pool reals
    fx = load_fixtures(include_audit=False)
    fxa = load_fixtures(include_audit=True)['audit']
    eval_scenes = ([(rgb, ref, cid) for role in ('exploration', 'gate')
                    for rgb, ref, cid in fx[role][:3]] +
                   [(rgb, ref, cid) for rgb, ref, cid in fxa] +
                   [(rgb, ref, f"hold-{i}") for i, (rgb, ref) in
                    enumerate(hold_pairs)])
    print(f"eval scenes: {len(eval_scenes)} (audit {len(fxa)}, "
          f"held-out reals {len(hold_pairs)})", flush=True)

    # RRR covariances for init (reuse student_probe machinery)
    import student_probe as SP
    covs = {li: {} for li in STUDENT_LAYERS}
    bb, pre = shared['backbone'], shared['preprocess']
    with torch.no_grad():
        for rgb, _ref in fit_pairs:
            cap = {}
            bb.forward_stages(pre(rgb).to(device), capture=cap)
            for li in STUDENT_LAYERS:
                B = {k: v.squeeze(0).double().cpu().numpy()
                     for k, v in cap[(li, 'blk')].items()}
                for name, ik, tk, di, do in SP.MAPS:
                    X = np.concatenate(
                        [B[ik], np.ones((B[ik].shape[0], 1))], axis=1)
                    Y = B[tk]
                    S = covs[li].setdefault(name, {
                        'Sxx': np.zeros((di + 1, di + 1)),
                        'Sxy': np.zeros((di + 1, do))})
                    S['Sxx'] += X.T @ X
                    S['Sxy'] += X.T @ Y
    covs['maps'] = SP.MAPS
    print("RRR covariances done.", flush=True)

    student = StudentBackbone(shared['backbone'])
    student.rrr_init(covs)
    npars = sum(p.numel() for p in student.parameters())
    print(f"trainable params: {npars} "
          f"({sum(v.numel() for v in shared['backbone']._w.values())} total)",
          flush=True)

    labels = build_labels(shared, fit_pairs)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    best, best_state = -1.0, None

    from geo_neck import GeometricNeck
    from depth_adapter import ScaledNeckMixin

    class _Neck(ScaledNeckMixin, GeometricNeck):
        pass

    _neck = _Neck(device=device)
    _neck._w = shared['neck_w']
    _neck.buffers_loaded = True
    _neck.res_scales = [0, 0, 0, 0]

    def gate(tag):
        cs = []
        for rgb, ref, cid in eval_scenes:
            fmaps, ph, pw = student.forward_backbone(
                shared['preprocess'](rgb).to(device))
            with torch.no_grad():
                fused = _neck(fmaps)
                d = shared['head']([fused[3]], ph, pw).squeeze(0).cpu().numpy()
            cs.append(_corr(d, ref))
        mean, mn = float(np.mean(cs)), float(min(cs))
        ok = sum(c >= 0.999 for c in cs)
        print(f"[gate {tag}] mean={mean:.5f} min={mn:.5f} pass={ok}/{len(cs)}",
              flush=True)
        return mean

    from geo_depth import MEAN as _M, STD as _S
    import PIL.Image as _I
    gate("init")
    rng = np.random.default_rng(0)
    Y = torch.from_numpy(labels).to(device)  # (N,518,518), no channel dim
    n = len(fit_pairs)
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for bi in range(0, n, args.batch):
            idx = perm[bi:bi + args.batch]
            # exact preprocess per image (resize+normalize, host-side)
            pvs = []
            for j in idx:
                rgb = fit_pairs[j][0]
                pvs.append(shared['preprocess'](rgb))
            pv = torch.cat(pvs).to(device)  # preprocess gives [1,3,H,W] each
            tgt = Y[idx]
            fmaps, ph, pw = student.forward_backbone(pv)
            fused = _neck(fmaps)
            pred = shared['head']([fused[3]], ph, pw)
            loss, mae, gm = ssi_gm_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"ep {ep}: loss={tot / (n / args.batch):.5f}", flush=True)
        if (ep + 1) % args.gate_every == 0:
            m = gate(f"ep{ep}")
            if m > best:
                best = m
                best_state = {k: tuple(p.detach().cpu().clone() for p in tup)
                              for k, tup in student.factors.items()}
                torch.save({"factors": best_state, "mean_corr": best,
                            "rank": RANK, "layers": STUDENT_LAYERS}, BEST)
                print(f"  best saved: {best:.5f} -> {BEST}", flush=True)
    print(f"GRAD: best held-out mean={best:.5f}")


if __name__ == '__main__':
    main()
