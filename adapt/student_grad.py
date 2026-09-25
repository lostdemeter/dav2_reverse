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
# Map-class mode (shape-thesis test): dense trainable maps for the
# named classes (e.g. STUDENT_MAPS="mlp2.weight"), init from banked
# composed values, everything else frozen. If a single map class
# moves audit-1, the missing information lives in its subspace and
# the delta gets characterized (rank? novel directions?) against
# phi_lattice primitives. Flat -> scope closed.
MAPSEL = tuple(x.strip() for x in
               __import__('os').environ.get('STUDENT_MAPS', '').split(',')
               if x.strip() != '')
# Joint-bias mode (chords-not-notes test): train ALL early biases
# jointly from the banked Phase-E start, A/B frozen. Greedy singles
# failed (Phase G 0/324); joint optimization may preserve floors
# where isolated moves cannot. Init ckpt via STUDENT_BIASONLY_INIT.
BIASONLY = __import__('os').environ.get('STUDENT_BIASONLY', '') == '1'
BIASONLY_INIT = __import__('os').environ.get(
    'STUDENT_BIASONLY_INIT',
    str(ADAPT / 'runs' / 'student_grad_best_r192_ds0.1_gms1p0_0p5_0p25_synthpool100_edgew_detailw_structmask_cos.pt'))
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
# Phase B (synth-gradient coverage): append N parametric-synth scenes
# (E0-heavy pool) as first-class pairs with teacher-on-synth labels.
# Same correct-labels machinery as DARKPOOL. E0 needs scenes in fit,
# not sharper loss (A1 worsened exploration-2 0.971->0.919).
SYNTHPOOL_N = int(__import__('os').environ.get('STUDENT_SYNTHPOOL', '0'))
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
# Multi-scale gradient-matching scales (Phase A): edge/texture
# structure is scale-dependent. "1.0" = legacy single-scale.
GMSCALES = tuple(float(x) for x in
                 __import__('os').environ.get('STUDENT_GMSCALES', '1.0').split(','))
GMSCALES_TAG = "_".join(str(x).replace('.', 'p') for x in GMSCALES)
# Phase A2 (edge-weighted GM): per-image gradient-loss weight from
# the input's edge statistic (AxB located the tension: edge scenes
# want fine GM, smooth scenes want it gone). STUDENT_EDGEW=1 enables;
# weights precomputed once (deterministic), MAE stays unweighted.
EDGEW = __import__('os').environ.get('STUDENT_EDGEW', '') == '1'
# Phase E (detail-weighted loss): the webcam falloff diagnosis —
# thin structures vanish (bottleneck low-pass + Sobel-L1 blind to
# 2px cables + top-10% mask drops exactly the detail-error pixels).
# STUDENT_DETAILW=1: per-pixel GM weight UP where the teacher has
# fine structure (symmetric completion of edge-weighting, which
# only turned GM DOWN on smooth scenes). STUDENT_STRUCTMASK=1:
# never mask high-teacher-gradient pixels (keeps noise rejection
# without deleting the detail signal).
DETAILW = __import__('os').environ.get('STUDENT_DETAILW', '') == '1'
STRUCTMASK = __import__('os').environ.get('STUDENT_STRUCTMASK', '') == '1'
LABEL_CACHE = ADAPT / 'runs' / 'grad_labels.npz'
BEST_STEM = (f'student_grad_best_r{RANK}'
               f'{"_ul" + "-".join(map(str, LATE_LAYERS)) if LATE_LAYERS else ""}'
               f'{"_rand" if RANDOM_INIT else ""}'
               f'{f"_ds{DEEPSUP:g}" if DEEPSUP > 0 else ""}'
               f'{"_fw" if FULLWIDTH else ""}'
               f'{"_auglow" if AUGLOW else ""}'
               f'{f"_darkpool{DARKPOOL_N}" if DARKPOOL_N > 0 else ""}'
               f'{f"_f{DARKFRAC:g}" if DARKFRAC > 0 else ""}'
               f'{f"_gms{GMSCALES_TAG}" if GMSCALES != (1.0,) else ""}'
               f'{f"_synthpool{SYNTHPOOL_N}" if SYNTHPOOL_N > 0 else ""}'
               f'{"_edgew" if EDGEW else ""}'
               f'{"_detailw" if DETAILW else ""}'
               f'{"_structmask" if STRUCTMASK else ""}'
               f'{"_biasonly" if BIASONLY else ""}'
               f'{f"_mapsel-{'-'.join(m.split(chr(46))[0] for m in MAPSEL)}" if MAPSEL else ""}')
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

        # Map-class dense maps (shape-thesis test): full-rank
        # trainable copies for the named classes, init from banked
        # composed values in init_banked. Empty = off.
        self.mapsel = {}
        if MAPSEL:
            for li in STUDENT_LAYERS:
                for m in MAPSEL:
                    W = teacher._w[f'layer{li}.{m}'].detach().clone()
                    bkey = f'layer{li}.{m.replace(".weight", ".bias")}'
                    b = (teacher._w[bkey].detach().clone()
                         if bkey in teacher._w else None)
                    self.mapsel[(li, m)] = (
                        torch.nn.Parameter(W),
                        torch.nn.Parameter(b) if b is not None else None)
            print(f"mapsel classes {list(MAPSEL)} "
                  f"({sum(p.numel() for p in self.mapsel_parameters())} "
                  f"params)", flush=True)

    def mapsel_parameters(self):
        return [p for tup in self.mapsel.values() for p in tup
                if p is not None]

    def parameters(self):
        ps = [p for tup in self.factors.values() for p in tup]
        if FULLWIDTH:
            ps = [p for tup in self.full.values() for p in tup
                  if p is not None]
        if BIASONLY:
            # biases only, A/B frozen (set requires_grad in init_banked)
            ps = [tup[2] for tup in self.factors.values()]
        if MAPSEL:
            ps = self.mapsel_parameters()
        return ps

    def late_parameters(self):
        return [p for tup in self.late.values() for p in tup
                if p is not None]

    def init_banked(self, path=None):
        """Load a banked checkpoint's factors (for BIASONLY start);
        also composes dense mapsel maps when MAPSEL is set."""
        import torch
        ckpt = torch.load(path or BIASONLY_INIT, map_location=self.device,
                          weights_only=False)
        print(f"banked init: {path or BIASONLY_INIT} "
              f"mean_corr={ckpt.get('mean_corr', float('nan')):.5f}",
              flush=True)
        with torch.no_grad():
            for (li, m), tup in ckpt['factors'].items():
                A, B, b = self.factors[(li, m)]
                A.copy_(tup[0].to(self.device))
                B.copy_(tup[1].to(self.device))
                b.copy_(tup[2].to(self.device))
                if BIASONLY:
                    A.requires_grad_(False)
                    B.requires_grad_(False)
            if MAPSEL:
                for li in STUDENT_LAYERS:
                    for m in MAPSEL:
                        A, B, b = self.factors[(li, m)]
                        Wp, bp = self.mapsel[(li, m)]
                        Wp.copy_(B @ A)
                        bp.copy_(b)

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

    def flin(self, h, li, m):
        """Factored linear for bottleneck maps: B@(A@h) + b.

        Same math as F.linear(h, B@A, b), fewer MACs when
        r*(di+do) < di*do. Falls back to dense for non-student,
        late-unfrozen, and full-width maps.
        """
        import torch.nn.functional as F
        key = (li, m)
        if key in self.factors:
            A, B, b = self.factors[key]
            return F.linear(F.linear(h, A, None), B, b)
        return F.linear(h, self.teacher._w[f'layer{li}.{m}'],
                        self.teacher._w.get(
                            f'layer{li}.{m.replace(".weight", ".bias")}'))

    def forward_factored(self, pv, taps=None, capture=None):
        """Mirror of forward_stages with factored bottleneck matmuls.

        Honest duplication (documented): any change to the teacher
        path must be mirrored here and re-verified (maxabs + depth
        parity). Restrictions: no dropped/bgains/zero_heads and no
        late/fullwidth overrides (asserted) — inference-only path
        for the bottleneck student.
        """
        import torch
        import torch.nn.functional as F
        from geo_backbone import (HIDDEN, LAYERS, HEADS, HEAD_DIM, PATCH,
                                  OUT_STAGES)
        assert not LATE_LAYERS and not FULLWIDTH, \
            "factored supports bottleneck student only"
        teacher = self.teacher
        want = tuple(taps) if taps is not None else OUT_STAGES
        B, _, H, W = pv.shape
        x = pv.to(self.device)
        ph, pw = H // PATCH, W // PATCH
        _g = teacher._g
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
        x = x + pos
        stages = {}
        for li in range(LAYERS):
            p = f'layer{li}.'
            D = HIDDEN
            layer_in = x if capture is not None else None
            h = F.layer_norm(x, (D,), _g(p + 'norm1.weight'),
                             _g(p + 'norm1.bias'))
            if li in STUDENT_LAYERS:
                q, k_, v = (self.flin(h, li, m) for m in
                            ('q.weight', 'k.weight', 'v.weight'))
            else:
                q = F.linear(h, _g(p + 'q.weight'), _g(p + 'q.bias'))
                k_ = F.linear(h, _g(p + 'k.weight'), _g(p + 'k.bias'))
                v = F.linear(h, _g(p + 'v.weight'), _g(p + 'v.bias'))
            if capture is not None:
                _cap_h, _cap_q, _cap_k, _cap_v = h, q, k_, v
            q = q.view(B, -1, HEADS, HEAD_DIM).transpose(1, 2)
            k_ = k_.view(B, -1, HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(B, -1, HEADS, HEAD_DIM).transpose(1, 2)
            attn = (q @ k_.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
            attn = attn.softmax(dim=-1)
            o = (attn @ v).transpose(1, 2).contiguous().view(B, -1, D)
            if capture is not None:
                _cap_c = o
            o = self.flin(o, li, 'proj.weight') if li in STUDENT_LAYERS \
                else F.linear(o, _g(p + 'proj.weight'), _g(p + 'proj.bias'))
            if capture is not None:
                _cap_o = o
            x = x + o * _g(p + 'ls1')
            h2 = F.layer_norm(x, (D,), _g(p + 'norm2.weight'),
                              _g(p + 'norm2.bias'))
            m_in = x
            if capture is not None:
                _cap_n2 = h2
            if li in STUDENT_LAYERS:
                h2 = self.flin(h2, li, 'mlp1.weight')
            else:
                h2 = F.linear(h2, _g(p + 'mlp1.weight'), _g(p + 'mlp1.bias'))
            pre = h2
            h2 = F.gelu(h2)
            post = h2
            if li in STUDENT_LAYERS:
                h2 = self.flin(h2, li, 'mlp2.weight')
            else:
                h2 = F.linear(h2, _g(p + 'mlp2.weight'), _g(p + 'mlp2.bias'))
            if capture is not None:
                _cap_y = h2
            x = x + h2 * _g(p + 'ls2')
            if (li + 1) in want:
                stages[li + 1] = x
            if capture is not None:
                capture[li] = (layer_in.detach().clone(), x.detach().clone())
                capture[(li, 'blk')] = {
                    'H': _cap_h.detach().clone(),
                    'Q': _cap_q.detach().clone(),
                    'K': _cap_k.detach().clone(),
                    'V': _cap_v.detach().clone(),
                    'C': _cap_c.detach().clone(),
                    'Oattn': _cap_o.detach().clone(),
                    'M': m_in.detach().clone(),
                    'N2': _cap_n2.detach().clone(),
                    'P1': pre.detach().clone(),
                    'G': post.detach().clone(),
                    'Y': _cap_y.detach().clone()}
        norm_w = teacher._w.get('final_norm.weight')
        norm_b = teacher._w.get('final_norm.bias')
        fmaps = []
        for s in want:
            h = stages[s]
            if norm_w is not None:
                h = F.layer_norm(h, (HIDDEN,), norm_w, norm_b)
            patch = h[:, 1:, :]
            fmaps.append(patch.reshape(B, ph, pw, HIDDEN)
                         .permute(0, 3, 1, 2).contiguous())
        return fmaps, ph, pw

    def forward_backbone(self, pv, taps=None, capture=None):
        """forward_stages with bottleneck composition via _g override."""
        teacher = self.teacher
        orig_g = teacher._g

        def _g(name):
            import re
            m = re.match(r'layer(\d+)\.(q|k|v|proj|mlp1|mlp2)\.weight', name)
            if m and (int(m.group(1)), m.group(2) + '.weight') in self.mapsel:
                return self.mapsel[(int(m.group(1)),
                                    m.group(2) + '.weight')][0]
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
            if m and (int(m.group(1)), m.group(2) + '.weight') in self.mapsel:
                mb = self.mapsel[(int(m.group(1)),
                                  m.group(2) + '.weight')][1]
                if mb is not None:
                    return mb
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


def ssi_gm_loss(pred, target, gm_w=None):
    """SSI (lstsq scale+shift align + MAE) + multi-scale Sobel-grad L1.

    GMSCALES env (default "1.0" = legacy single-scale): gradient term
    averaged over downsampled copies (edge/texture structure is
    scale-dependent — our strata split on it). Top-10% residual mask
    (downsampled per scale) throughout, unless STRUCTMASK keeps
    structural pixels (Phase E). Per-image GM weights via gm_w
    (Phase A2 edge-weighting); per-pixel detail boost via DETAILW
    (Phase E): weight UP where the teacher has fine structure.
    """
    import torch
    import torch.nn.functional as F
    scales = GMSCALES
    B = pred.shape[0]
    p = pred.reshape(B, -1).double()
    t = target.reshape(B, -1).double()
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
    beta = float(__import__('os').environ.get('STUDENT_DETAILW_BETA', '1.0'))
    gm_terms = []
    for sc in scales:
        if sc == 1.0:
            A, Tt, M = aligned, T4, mask
        else:
            A = F.interpolate(aligned, scale_factor=sc, mode='bilinear',
                              align_corners=True)
            Tt = F.interpolate(T4, scale_factor=sc, mode='bilinear',
                               align_corners=True)
            M = F.interpolate(mask, scale_factor=sc, mode='nearest')
        ga = F.conv2d(A, Kx, padding=1).abs() + F.conv2d(A, Ky, padding=1).abs()
        gt = F.conv2d(Tt, Kx, padding=1).abs() + F.conv2d(Tt, Ky,
                                                          padding=1).abs()
        if STRUCTMASK:
            # never mask structural pixels: keep where teacher gradient
            # is above its per-image median (scale-local)
            gflat = gt.reshape(B, -1)
            gthr = gflat.quantile(0.5, dim=1).view(B, 1, 1, 1)
            M = torch.maximum(M, (gt >= gthr).float())
        gmap = ((ga - gt).abs()) * M
        pw = None
        if DETAILW:
            # boost where the teacher has fine structure: 1 + beta *
            # (teacher grad / its per-image mean), scale-local; clamped
            # so near-flat images (tiny denominator) can't explode
            gmean = gt.reshape(B, -1).mean(dim=1).view(B, 1, 1, 1)
            pw = (1.0 + beta * gt / gmean.clamp_min(1e-12)).clamp_max(4.0)
            gmap = gmap * pw
        if gm_w is not None:
            Wm = gm_w.view(B, *([1] * (gmap.dim() - 1))).to(gmap.dtype)
            gm_terms.append((gmap * Wm).sum() / (M * Wm).sum().clamp_min(1.0))
        else:
            gm_terms.append(gmap.sum() / M.sum().clamp_min(1.0))
    gm = torch.stack(gm_terms).mean()
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
    if SYNTHPOOL_N > 0:
        # Phase B: parametric-synth scenes (E0-heavy) as first-class
        # pairs with teacher-on-synth labels (same correct-labels
        # machinery as DARKPOOL). E0 needs scenes in fit (A1
        # worsened exploration-2 by sharpening loss without them).
        _zs = np.load(ADAPT / 'fixtures' / 'synth_pool.npz',
                      allow_pickle=True)
        for i in range(min(SYNTHPOOL_N, len(_zs['rgb']))):
            fit_pairs.append((_zs['rgb'][i].astype(np.float32) / 255.0,
                              None))
        print(f"synthpool: +{min(SYNTHPOOL_N, len(_zs['rgb']))} "
              f"parametric scenes", flush=True)
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
    if BIASONLY or MAPSEL:
        # banked start (MAPSEL composes dense maps inside init_banked)
        student.init_banked()
    else:
        student.rrr_init(covs)
    # Trace mode (sequentiality probe): per-scene corrs + per-layer/map
    # factor drift from init, recorded at every gate eval.
    TRACE = __import__('os').environ.get('STUDENT_TRACE', '') == '1'
    trace_rows = []
    init_snap = None
    if TRACE:
        with torch.no_grad():
            init_snap = {
                k: tuple(p.detach().cpu().clone() for p in tup)
                for k, tup in student.factors.items()}
        print("trace: init snapshot taken", flush=True)
    npars = sum(p.numel() for p in student.parameters())
    print(f"trainable params: {npars} "
          f"({sum(v.numel() for v in shared['backbone']._w.values())} total)",
          flush=True)

    labels = build_labels(shared, fit_pairs)
    edge_w = None
    if EDGEW:
        # per-image GM weights from input edge stat (deterministic).
        from strata import profile as _prof
        es = np.array([_prof(rgb)["edge"] for rgb, _ in fit_pairs],
                      dtype=np.float64)
        med = float(np.median(es))
        edge_w = torch.from_numpy(
            np.clip(es / max(med, 1e-12), 0.1, 2.0).astype(np.float32))
        print(f"edgew: median edge={med:.4f}, "
              f"w range [{edge_w.min():.2f},{edge_w.max():.2f}]", flush=True)
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
    if TRACE:
        # never clobber a real best with an instrumented rerun
        BEST = BEST.with_name(BEST.stem + '_trace.pt')
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
        recs = []
        medians = None
        if TRACE:
            from strata import profile as _tprof, assign as _tassign
            import json as _tjson
            medians = _tjson.load(
                open(ADAPT / 'runs' / 'strata.json'))['medians']
        for rgb, ref, cid in eval_scenes:
            fmaps, ph, pw = student.forward_backbone(
                shared['preprocess'](rgb).to(device))
            with torch.no_grad():
                fused = _neck(fmaps)
                d = shared['head']([fused[3]], ph, pw).squeeze(0).cpu().numpy()
            c = _corr(d, ref)
            cs.append(c)
            if TRACE:
                recs.append((cid, _tassign(_tprof(rgb), medians),
                             round(float(c), 6)))
            else:
                recs.append((cid, round(float(c), 6)))
        mean, mn = float(np.mean(cs)), float(min(cs))
        ok = sum(c >= 0.999 for c in cs)
        print(f"[gate {tag}] mean={mean:.5f} min={mn:.5f} pass={ok}/{len(cs)}",
              flush=True)
        if TRACE:
            # per-layer + per-map factor drift from init snapshot
            drift_layer, drift_map = {}, {}
            with torch.no_grad():
                for (li, m), tup in student.factors.items():
                    init = init_snap[(li, m)]
                    num, den = 0.0, 0.0
                    for p, q in zip(tup, init):
                        d = (p.detach().double().cpu().numpy()
                             - q.double().cpu().numpy())
                        num += float((d ** 2).sum())
                        den += float((q.double().cpu().numpy() ** 2).sum())
                    drift_map[f"{li}/{m}"] = round(
                        float(np.sqrt(num / max(den, 1e-30))), 6)
                    a, b = drift_layer.get(li, (0.0, 0.0))
                    drift_layer[li] = (a + num, b + den)
            drift_layer = {str(li): round(float(np.sqrt(a / max(b, 1e-30))), 6)
                           for li, (a, b) in drift_layer.items()}
            trace_rows.append({"tag": tag, "mean": round(mean, 6),
                               "scenes": recs, "drift_layer": drift_layer,
                               "drift_map": drift_map})
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
            loss, mae, gm = ssi_gm_loss(
                pred, tgt, edge_w[idx].to(device)
                if edge_w is not None else None)
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
    if TRACE:
        import json as _json
        tpath = ADAPT / 'runs' / 'trace.jsonl'
        with open(tpath, 'w') as f:
            for row in trace_rows:
                f.write(_json.dumps(row, default=str) + "\n")
        print(f"trace: wrote {len(trace_rows)} rows -> {tpath}", flush=True)


if __name__ == '__main__':
    main()
