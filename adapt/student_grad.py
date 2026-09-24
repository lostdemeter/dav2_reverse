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

RANK = int(__import__('os').environ.get('STUDENT_RANK', '128'))
STUDENT_LAYERS = (0, 1, 2)
RANDOM_INIT = __import__('os').environ.get('STUDENT_RANDOM_INIT', '') == '1'
# Full-width mode: direct (W,b) Parameters per map, teacher-exact init.
# No bottleneck, no RRR — isolates narrowing-vs-training causally.
FULLWIDTH = __import__('os').environ.get('STUDENT_FULLWIDTH', '') == '1'
# Deep supervision: student L0-2 layer-outs vs teacher layer-outs
# (online teacher forward, no_grad), variance-normalized MSE x lambda.
# 0 = off (legacy depth-only loss).
DEEPSUP = float(__import__('os').environ.get('STUDENT_DEEPSUP', '0.0'))
# Low-light augmentation (dark-room gap fix): photometric-only
# transforms preserving geometry, so clean teacher labels stay valid.
# Env STUDENT_AUGLOW=1 enables in the gradient phase (RRR init untouched).
AUGLOW = __import__('os').environ.get('STUDENT_AUGLOW', '') == '1'
AUGLOW_P = float(__import__('os').environ.get('STUDENT_AUGLOW_P', '0.5'))
# v3 (correct-labels): precomputed dark scenes as FIRST-CLASS fit pairs
# with teacher-on-dark labels (v1/v2's clean-label pairing was WRONG:
# teacher(dark)-vs-teacher(clean) is 0.75-0.78, so clean labels
# mis-supervise dark inputs). STUDENT_DARKPOOL=N darkens the first N
# fit scenes once (seeded) and appends them; labels flow through
# build_labels (staleness rebuild) and RRR covariances automatically.
DARKPOOL_N = int(__import__('os').environ.get('STUDENT_DARKPOOL', '0'))
# Dark fraction of the final fit list (interference scales with
# fraction; v3 ran 0.26). Subsamples the dark variants to hit it.
DARKFRAC = float(__import__('os').environ.get('STUDENT_DARKFRAC', '0.0'))
# Gentle variant (v2): milder photometrics after v1's destructive
# interference (dark-eval 0.93->0.70: off-manifold dark batches vs
# clean RRR basin). v2 also augments the RRR covariances (basin
# includes dark) — the actual mechanism fix.
AUGLOW_GENTLE = __import__('os').environ.get('STUDENT_AUGLOW_GENTLE', '') == '1'
# Late layers unfrozen at small LR (absorbability lever). Env override
# for sweeps without flag plumbing: STUDENT_UNFREEZE="9,10,11".
LATE_LAYERS = tuple(int(x) for x in
                    __import__('os').environ.get('STUDENT_UNFREEZE', '').split(',')
                    if x.strip() != '')
LATE_LR = float(__import__('os').environ.get('STUDENT_LATE_LR', '1e-6'))
MAPS = ("q.weight", "k.weight", "v.weight", "proj.weight",
        "mlp1.weight", "mlp2.weight")
LABEL_CACHE = ADAPT / 'runs' / 'grad_labels.npz'
BEST_STEM = (f'student_grad_best_r{RANK}'
               f'{"_ul" + "-".join(map(str, LATE_LAYERS)) if LATE_LAYERS else ""}'
               f'{"_rand" if RANDOM_INIT else ""}'
               f'{f"_ds{DEEPSUP:g}" if DEEPSUP > 0 else ""}'
               f'{"_fw" if FULLWIDTH else ""}'
               f'{"_auglow" if AUGLOW else ""}'
               f'{f"_darkpool{DARKPOOL_N}" if DARKPOOL_N > 0 else ""}'
               f'{f"_f{DARKFRAC:g}" if DARKFRAC > 0 else ""}')
BEST = ADAPT / 'runs' / (BEST_STEM + '.pt')  # may gain _cos suffix in main()


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
        # Late full maps (absorbability lever): cloned teacher tensors
        # as Parameters, trained at LATE_LR. Empty = frozen (legacy).
        self.late = {}
        with torch.no_grad():
            for li in LATE_LAYERS:
                for m in MAPS:
                    W = teacher._w[f'layer{li}.{m}'].detach().clone()
                    bkey = f'layer{li}.{m.replace(".weight", ".bias")}'
                    b = (teacher._w[bkey].detach().clone()
                         if bkey in teacher._w else None)
                    Wp = torch.nn.Parameter(W)
                    bp = torch.nn.Parameter(b) if b is not None else None
                    self.late[(li, m)] = (Wp, bp)
        # Full-width direct maps (causal-isolation mode): teacher-exact.
        self.full = {}
        if FULLWIDTH:
            with torch.no_grad():
                for li in STUDENT_LAYERS:
                    for m in MAPS:
                        W = teacher._w[f'layer{li}.{m}'].detach().clone()
                        bkey = f'layer{li}.{m.replace(".weight", ".bias")}'
                        b = (teacher._w[bkey].detach().clone()
                             if bkey in teacher._w else None)
                        self.full[(li, m)] = (
                            torch.nn.Parameter(W),
                            torch.nn.Parameter(b) if b is not None else None)
            print("FULLWIDTH mode (teacher-exact init, no bottleneck)",
                  flush=True)

    def parameters(self):
        ps = [p for tup in self.factors.values() for p in tup]
        if FULLWIDTH:
            ps = [p for tup in self.full.values() for p in tup
                  if p is not None]
        return ps

    def late_parameters(self):
        return [p for tup in self.late.values() for p in tup
                if p is not None]

    def rrr_init(self, covs):
        """Init factors from RRR solutions (skipped when RANDOM_INIT
        or FULLWIDTH — the latter is teacher-exact by construction)."""
        import torch
        if FULLWIDTH:
            print("FULLWIDTH: teacher-exact init, RRR skipped", flush=True)
            return
        if RANDOM_INIT:
            with torch.no_grad():
                for (li, m), (A, B, b) in self.factors.items():
                    W = self.teacher._w[f'layer{li}.{m}']
                    do, di = W.shape
                    # proper small-random (NOT the dead A=0 default):
                    # |B@A| entries ~ N(0, 0.02^2*sqrt(r))
                    A.copy_(torch.randn(RANK, di, device=self.device) * 0.02)
                    B.copy_(torch.randn(do, RANK, device=self.device) * 0.02)
                    b.zero_()
            print("RANDOM init (basin-trap test, no RRR)", flush=True)
            return
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

    def forward_backbone(self, pv, taps=None, capture=None):
        """forward_stages with bottleneck composition via _g override."""
        teacher = self.teacher
        orig_g = teacher._g

        def _g(name):
            import re
            m = re.match(r'layer(\d+)\.(q|k|v|proj|mlp1|mlp2)\.weight', name)
            if m and int(m.group(1)) in STUDENT_LAYERS:
                if FULLWIDTH:
                    return self.full[(int(m.group(1)),
                                      m.group(2) + '.weight')][0]
                A, B, b = self.factors[(int(m.group(1)),
                                        m.group(2) + '.weight')]
                return B @ A
            if m and int(m.group(1)) in LATE_LAYERS:
                return self.late[(int(m.group(1)),
                                  m.group(2) + '.weight')][0]
            m = re.match(r'layer(\d+)\.(q|k|v|proj|mlp1|mlp2)\.bias', name)
            if m and int(m.group(1)) in STUDENT_LAYERS:
                if FULLWIDTH:
                    fb = self.full[(int(m.group(1)),
                                    m.group(2) + '.weight')][1]
                    if fb is not None:
                        return fb
                else:
                    return self.factors[(int(m.group(1)),
                                         m.group(2) + '.weight')][2]
            if m and int(m.group(1)) in LATE_LAYERS:
                late_b = self.late[(int(m.group(1)),
                                    m.group(2) + '.weight')][1]
                if late_b is not None:
                    return late_b
            return orig_g(name)

        teacher._g = _g
        try:
            return teacher.forward_stages(pv, taps=taps, capture=capture)
        finally:
            teacher._g = orig_g


def build_labels(shared, fit_pairs, force=False):
    """Teacher 518px depths for fit scenes. Cached (deterministic)."""
    import torch
    if LABEL_CACHE.exists() and not force:
        z = np.load(LABEL_CACHE, allow_pickle=False)
        if len(z['depth']) == len(fit_pairs):
            print(f"labels cached: {len(z['depth'])} scenes", flush=True)
            return z['depth']
        print(f"label cache stale ({len(z['depth'])} vs {len(fit_pairs)}), "
              f"rebuilding...", flush=True)
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


def lowlight_augment(rgb, rng, gentle=False):
    """Photometric dark-room transform (geometry-preserving).

    v1 (strong): darken xU(0.25,0.6) + shadow clip + noise + tint.
    v2 gentle: darken xU(0.5,0.8), minimal clip/noise/tint.
    Mirrors the webcam failure regime.
    """
    if gentle:
        f, clip_lo, clip_hi = float(rng.uniform(0.5, 0.8)), 0.0, 0.03
        ns_lo, ns_hi, t_lo, t_hi = 0.002, 0.008, 0.9, 1.1
    else:
        f, clip_lo, clip_hi = float(rng.uniform(0.25, 0.6)), 0.02, 0.08
        ns_lo, ns_hi, t_lo, t_hi = 0.005, 0.02, 0.8, 1.2
    out = np.clip(np.asanyarray(rgb, dtype=np.float32) * f, 0, 1)
    clip = float(rng.uniform(clip_lo, clip_hi))
    if clip > 0:
        out = np.where(out < clip, 0.0, (out - clip) / (1 - clip))
    out = np.clip(out + rng.normal(
        0, float(rng.uniform(ns_lo, ns_hi)), out.shape).astype(np.float32),
        0, 1)
    tint = rng.uniform(t_lo, t_hi, 3).astype(np.float32)
    return np.clip(out * tint, 0, 1).astype(np.float32)


def ssi_gm_loss(pred, target):
    """SSI (lstsq scale+shift align + MAE) + 2x Sobel-grad L1, top-10% mask."""
    import torch
    import torch.nn.functional as F
    B = pred.shape[0]
    p = pred.reshape(B, -1).double()
    t = target.reshape(B, -1).double()
    # 2x2 normal equations (lstsq driver materializes full U on 268k rows
    # -> TBs; closed form is exact for 2 unknowns).
    n = p.shape[1]
    sp, st = p.sum(1), t.sum(1)
    spp, spt = (p * p).sum(1), (p * t).sum(1)
    den = (n * spp - sp * sp).clamp_min(1e-12)
    s = ((n * spt - sp * st) / den).float()
    sh = ((st - s.double() * sp) / n).float()
    P4 = pred.unsqueeze(1)
    T4 = target.unsqueeze(1)
    aligned = s.view(B, 1, 1, 1) * P4 + sh.view(B, 1, 1, 1)
    resid = (aligned - T4).abs()
    flat = resid.reshape(B, -1)
    thr = flat.quantile(0.9, dim=1).view(B, 1, 1, 1)
    mask = (resid <= thr).float()
    mae = (resid * mask).sum() / mask.sum().clamp_min(1.0)
    Kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3)
    Ky = Kx.transpose(-1, -2)
    ga = F.conv2d(aligned, Kx, padding=1).abs() + F.conv2d(aligned, Ky,
                                                           padding=1).abs()
    gt = F.conv2d(T4, Kx, padding=1).abs() + F.conv2d(T4, Ky,
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
    ap.add_argument('--schedule', choices=('const', 'cosine'), default='const',
                    help='LR schedule: constant or cosine annealing to 0')
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
    if DARKPOOL_N > 0:
        # v3: fixed dark variants (seeded) as first-class pairs. Labels
        # are computed ON the dark images by build_labels below.
        drank = np.random.default_rng(123)
        darks = []
        for rgb, _ref in fit_pairs[:DARKPOOL_N]:
            darks.append((lowlight_augment(
                rgb, drank, gentle=AUGLOW_GENTLE), None))
        fit_pairs = fit_pairs + darks
        if DARKFRAC > 0:
            n_clean = len(fit_pairs) - len(darks)
            k = min(len(darks), int(DARKFRAC * n_clean / (1 - DARKFRAC)))
            darks = darks[:k]
            fit_pairs = fit_pairs[:n_clean] + darks
        print(f"darkpool: +{len(darks)} fixed dark pairs "
              f"(gentle={AUGLOW_GENTLE}, "
              f"frac={len(darks) / len(fit_pairs):.2f})", flush=True)
    # targeted coverage (failmap 2026-09-22): fit is 100% real, student
    # fails synthetics it never saw. Mix exploration+retention synth
    # into fit; gate+audit stay eval-only.
    _fx = load_fixtures(include_audit=False)
    for _role in ('exploration', 'retention'):
        for _rgb, _ref, _cid in _fx[_role]:
            fit_pairs.append((_rgb, _ref))
    print(f"train {len(fit_pairs)} (incl. 9 synth), held-out reals "
          f"{len(hold_pairs)}", flush=True)

    # held-out: gate + audit (never fit) + NEVER-FIT pool reals.
    # exploration/retention synth are IN fit now (failmap coverage fix).
    fx = load_fixtures(include_audit=False)
    fxa = load_fixtures(include_audit=True)['audit']
    eval_scenes = ([(rgb, ref, cid) for rgb, ref, cid in fx['gate']] +
                   [(rgb, ref, cid) for rgb, ref, cid in fxa] +
                   [(rgb, ref, f"hold-{i}") for i, (rgb, ref) in
                    enumerate(hold_pairs)])
    print(f"eval scenes: {len(eval_scenes)} (audit {len(fxa)}, "
          f"held-out reals {len(hold_pairs)})", flush=True)

    # RRR covariances for init (reuse student_probe machinery).
    # With AUGLOW the basin must include dark: augment a fraction of
    # covariance scenes identically (mechanism fix for v1's destructive
    # interference — clean-only basin vs dark gradient batches).
    import student_probe as SP
    covs = {li: {} for li in STUDENT_LAYERS}
    bb, pre = shared['backbone'], shared['preprocess']
    cov_rng = np.random.default_rng(11)
    with torch.no_grad():
        for rgb, _ref in fit_pairs:
            if AUGLOW and cov_rng.random() < AUGLOW_P:
                rgb = lowlight_augment(rgb, cov_rng,
                                       gentle=AUGLOW_GENTLE)
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
    groups = [{"params": student.parameters(), "lr": args.lr}]
    if student.late_parameters():
        groups.append({"params": student.late_parameters(), "lr": LATE_LR})
        print(f"unfreeze-late {list(LATE_LAYERS)} @lr={LATE_LR} "
              f"({sum(p.numel() for p in student.late_parameters())} params)",
              flush=True)
    opt = torch.optim.Adam(groups)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
             if args.schedule == 'cosine' else None)
    global BEST
    if args.schedule == 'cosine':
        BEST = ADAPT / 'runs' / (BEST_STEM + '_cos.pt')
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
        if AUGLOW or DARKPOOL_N > 0:
            # dark-eval (report-only): student-vs-teacher on darkened
            # eval scenes — the webcam failure regime.
            dark_rng = np.random.default_rng(1234)
            cd = []
            for rgb, ref, cid in eval_scenes[:12]:
                d = lowlight_augment(rgb, dark_rng)
                cfg = copy.deepcopy(dict(SEED_CONFIG))
                t = run_pipeline(shared, cfg, [d])[0]
                fmaps, ph, pw = student.forward_backbone(
                    shared['preprocess'](d).to(device))
                with torch.no_grad():
                    s = shared['head'](
                        _neck(fmaps), ph, pw).squeeze(0).cpu().numpy()
                cd.append(_corr(s, t))
            print(f"[dark-eval {tag}] student-vs-teacher "
                  f"mean={np.mean(cd):.5f} min={min(cd):.5f}", flush=True)
        return mean

    from geo_depth import MEAN as _M, STD as _S
    import PIL.Image as _I
    gate("init")
    rng = np.random.default_rng(0)
    aug_rng = np.random.default_rng(7)
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
                if AUGLOW and aug_rng.random() < AUGLOW_P:
                    rgb = lowlight_augment(rgb, aug_rng,
                                           gentle=AUGLOW_GENTLE)
                pvs.append(shared['preprocess'](rgb))
            pv = torch.cat(pvs).to(device)  # preprocess gives [1,3,H,W] each
            tgt = Y[idx]
            if DEEPSUP > 0:
                cap_s = {}
                fmaps, ph, pw = student.forward_backbone(pv, capture=cap_s)
                with torch.no_grad():
                    cap_t = {}
                    shared['backbone'].forward_stages(pv, capture=cap_t)
                ds_terms = []
                for li in STUDENT_LAYERS:
                    s_out = cap_s[li][1].float()
                    t_out = cap_t[li][1].float()
                    ds_terms.append(
                        ((s_out - t_out) ** 2).mean() /
                        (t_out.var().clamp_min(1e-12)))
                ds_loss = torch.stack(ds_terms).mean()
            else:
                fmaps, ph, pw = student.forward_backbone(pv)
                ds_loss = None
            fused = _neck(fmaps)
            pred = shared['head']([fused[3]], ph, pw)
            loss, mae, gm = ssi_gm_loss(pred, tgt)
            if ds_loss is not None:
                loss = loss + DEEPSUP * ds_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"ep {ep}: loss={tot / (n / args.batch):.5f}", flush=True)
        if sched is not None:
            sched.step()
        if (ep + 1) % args.gate_every == 0:
            m = gate(f"ep{ep}")
            if m > best:
                best = m
                best_state = {k: tuple(p.detach().cpu().clone() for p in tup)
                              for k, tup in student.factors.items()}
                late_state = {k: tuple(p.detach().cpu().clone() for p in tup
                                       if p is not None)
                              for k, tup in student.late.items()}
                full_state = {k: tuple(p.detach().cpu().clone() for p in tup
                                       if p is not None)
                              for k, tup in student.full.items()}
                torch.save({"factors": best_state, "late": late_state,
                            "full": full_state,
                            "mean_corr": best, "rank": RANK,
                            "layers": STUDENT_LAYERS,
                            "late_layers": LATE_LAYERS}, BEST)
                print(f"  best saved: {best:.5f} -> {BEST}", flush=True)
    print(f"GRAD: best held-out mean={best:.5f}")


if __name__ == '__main__':
    main()
