"""Depth domain adapter: learn DAV2 pieces via foundry search (experimental).

Implements the vendored ExperimentAdapter protocol over a small JSON DSL:

    {"readout": 3, "res_scales": [0,0,0,0], "head": "geo-conv",
     "taps": [3,6,9,12], "tap_gains": [0,0,0,0]}

taps/gains reach into the BACKBONE: which ViT layers feed the neck's four
reassemble stages, and at what phi-power gain (free as an exponent add in
the integer datapath). Torch/geo imports are lazy so DSL/proposal logic
stays importable and testable without GPU or weights (see test_smoke.py).
"""

import hashlib
import json
import pickle
import sys
from pathlib import Path

_ADAPT = Path(__file__).parent
sys.path.insert(0, str(_ADAPT / 'third_party'))  # explicit vendored-core path
from experimenter import (  # noqa: E402
    TrialSpec, CaseResult, Scorecard, Measurement,
)

CORR_PASS = 0.999
READOUTS = (0, 1, 2, 3)
SCALE_LO, SCALE_HI = -2, 2
HEADS = ("geo-conv", "direct", "analytic")
# 1-indexed backbone layers per neck position; strict increase enforced
# so pyramid roles (x4up/x2up/identity/x2down) stay sane.
TAP_BANDS = ((2, 3, 4), (5, 6, 7), (8, 9, 10), (11, 12))
GAIN_LO, GAIN_HI = -1, 1
# Per-block output gains (attn0,mlp0,attn1,mlp1,...): the learnable backbone
# weights. NOT 22M params — 24 residual-mixing scalars as phi exponents.
# Uniform weight scaling is absorbed by LayerNorm; these output gains survive
# it and are free exponent adds in the integer datapath.
BGAIN_LO, BGAIN_HI = -2, 2
N_BGAINS = 24

SEED_CONFIG = {"readout": 3, "res_scales": [0, 0, 0, 0], "head": "geo-conv",
               "taps": [3, 6, 9, 12], "tap_gains": [0, 0, 0, 0],
               "block_gains": [0] * N_BGAINS, "dropped": []}

# Float32 parameter bytes removed per dropped block (exact shapes):
#   attn: norm1(768) + q/k/v(3x(384x384+384)) + proj(384x384+384) = 592,128
#   mlp:  norm2(768) + fc1(1536x384+1536) + fc2(384x1536+384) = 1,182,336
ATTN_BLOCK_PARAMS = 768 + 3 * (384 * 384 + 384) + (384 * 384 + 384)
MLP_BLOCK_PARAMS = 768 + (1536 * 384 + 1536) + (384 * 1536 + 384)
assert ATTN_BLOCK_PARAMS == 592128 and MLP_BLOCK_PARAMS == 1182336
BACKBONE_PARAMS = 22056192  # measured sum over baked backbone buffers
NECK_PARAMS = 2700768       # measured (reassemble + fusion)
HEAD_PARAMS = 27745         # conv1 + conv2 + conv3 (+biases)


def validate_config(cfg):
    """Return canonical config or raise ValueError (counts as build_failed)."""
    if not isinstance(cfg, dict):
        raise ValueError("config must be an object")
    keys = set(cfg)
    want_keys = {"readout", "res_scales", "head", "taps", "tap_gains",
                 "block_gains", "dropped"}
    if keys != want_keys:
        raise ValueError(f"config keys must be exactly {sorted(want_keys)}, got {sorted(keys)}")
    readout = cfg["readout"]
    if type(readout) is not int or readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}")
    scales = cfg["res_scales"]
    if (not isinstance(scales, list) or len(scales) != 4
            or any(type(v) is not int or not SCALE_LO <= v <= SCALE_HI for v in scales)):
        raise ValueError(f"res_scales must be 4 ints in [{SCALE_LO},{SCALE_HI}]")
    if cfg["head"] not in HEADS:
        raise ValueError(f"head must be one of {HEADS}")
    taps = cfg["taps"]
    if (not isinstance(taps, list) or len(taps) != 4
            or any(type(v) is not int for v in taps)):
        raise ValueError("taps must be 4 layer ints")
    for i, v in enumerate(taps):
        if v not in TAP_BANDS[i]:
            raise ValueError(f"taps[{i}]={v} outside band {TAP_BANDS[i]}")
    if not (taps[0] < taps[1] < taps[2] < taps[3]):
        raise ValueError(f"taps must strictly increase, got {taps}")
    gains = cfg["tap_gains"]
    if (not isinstance(gains, list) or len(gains) != 4
            or any(type(v) is not int or not GAIN_LO <= v <= GAIN_HI for v in gains)):
        raise ValueError(f"tap_gains must be 4 ints in [{GAIN_LO},{GAIN_HI}]")
    bg = cfg["block_gains"]
    if (not isinstance(bg, list) or len(bg) != N_BGAINS
            or any(type(v) is not int or not BGAIN_LO <= v <= BGAIN_HI for v in bg)):
        raise ValueError(f"block_gains must be {N_BGAINS} ints in [{BGAIN_LO},{BGAIN_HI}]")
    dropped = cfg["dropped"]
    if (not isinstance(dropped, list)
            or any(type(v) is not int or not 0 <= v < N_BGAINS for v in dropped)
            or len(set(dropped)) != len(dropped)):
        raise ValueError(f"dropped must be a unique list of block ids in [0,{N_BGAINS})")
    dropped = sorted(dropped)
    return {"readout": readout, "res_scales": list(scales), "head": cfg["head"],
            "taps": list(taps), "tap_gains": list(gains), "block_gains": list(bg),
            "dropped": dropped}


def neighbor_configs(cfg):
    """Single moves from cfg: (config, rationale, priority). Lower = first."""
    out = []
    if cfg["head"] == "geo-conv":
        c = dict(cfg, head="direct")
        out.append((c, "direct 64->1 head avoids the full conv stack", 10))
        c = dict(cfg, head="analytic")
        out.append((c, "analytic zero-weight head (edge/texture/perspective, PHI powers)", 15))
    else:
        c = dict(cfg, head="geo-conv")
        out.append((c, "full geometric head restores conv context", 10))
    for r in READOUTS:
        if r != cfg["readout"]:
            c = dict(cfg, readout=r)
            out.append((c, f"readout stage {cfg['readout']}->{r} changes scale emphasis", 20))
    for i, v in enumerate(cfg["taps"]):
        for d in (-1, 1):
            nv = v + d
            if nv in TAP_BANDS[i]:
                taps = list(cfg["taps"])
                taps[i] = nv
                if taps[0] < taps[1] < taps[2] < taps[3]:
                    c = dict(cfg, taps=taps)
                    out.append((c, f"backbone tap {i}: layer {v}->{nv}", 22))
    for i, v in enumerate(cfg["tap_gains"]):
        for d in (-1, 1):
            nv = v + d
            if GAIN_LO <= nv <= GAIN_HI:
                gains = list(cfg["tap_gains"])
                gains[i] = nv
                c = dict(cfg, tap_gains=gains)
                out.append((c, f"tap {i} gain xphi^{nv} (free in integer datapath)", 26))
    for i, v in enumerate(cfg["block_gains"]):
        for d in (-1, 1):
            nv = v + d
            if BGAIN_LO <= nv <= BGAIN_HI:
                bg = list(cfg["block_gains"])
                bg[i] = nv
                c = dict(cfg, block_gains=bg)
                kind = "attn" if i % 2 == 0 else "mlp"
                out.append((c, f"backbone L{i // 2} {kind} out xphi^{nv}", 28))
    have = set(cfg["dropped"])
    for b in range(N_BGAINS):
        if b not in have:
            c = dict(cfg, dropped=sorted(have | {b}))
            kind = "attn" if b % 2 == 0 else "mlp"
            out.append((c, f"DROP backbone L{b // 2} {kind} (residual-only)", 9))
    for b in sorted(have):
        c = dict(cfg, dropped=sorted(have - {b}))
        kind = "attn" if b % 2 == 0 else "mlp"
        out.append((c, f"restore backbone L{b // 2} {kind}", 11))
    for i, v in enumerate(cfg["res_scales"]):
        for d in (-1, 1):
            nv = v + d
            if SCALE_LO <= nv <= SCALE_HI:
                scales = list(cfg["res_scales"])
                scales[i] = nv
                c = dict(cfg, res_scales=scales)
                out.append((c, f"fusion stage {i} residual xphi^{nv}", 30 + i))
    return sorted(out, key=lambda t: t[2])


class ScaledNeckMixin:
    """Phi-scale fusion residual branches: hidden += phi^e * res1(residual)."""

    def _fusion_layer(self, hidden, residual, size, li):
        import torch.nn.functional as F
        from geo_lut import PHI
        p = f'fusion{li}.'
        e = self.res_scales[li]
        if residual is not None:
            if hidden.shape != residual.shape:
                residual = F.interpolate(residual, size=(hidden.shape[2], hidden.shape[3]),
                                         mode='bilinear', align_corners=False)
            r = self._residual(residual, p + 'res1')
            if e:
                r = r * float(PHI ** e)
            hidden = hidden + r
        hidden = self._residual(hidden, p + 'res2')
        if size is None:
            hidden = F.interpolate(hidden, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            hidden = F.interpolate(hidden, size=size, mode='bilinear', align_corners=True)
        hidden = F.conv2d(hidden, self._g(p + 'proj.weight'), self._g(p + 'proj.bias'))
        return hidden


def _backbone_tapped(backbone, preprocess, device, rgb, taps, gains, bgains=None,
                     dropped=None):
    """Backbone stages at tap layers with phi-power gains. Returns fmaps, ph, pw."""
    import torch
    from geo_lut import PHI
    pv = preprocess(rgb).to(device)
    with torch.no_grad():
        fmaps, ph, pw = backbone.forward_stages(pv, taps=taps, bgains=bgains,
                                                dropped=dropped)
    out = []
    for fm, g in zip(fmaps, gains):
        out.append(fm * float(PHI ** g) if g else fm)
    return out, ph, pw


def _model_bytes(cfg, direct):
    """Honest artifact size: remaining backbone float32 params + neck + head
    + fitted head bytes (if any). Pickle framing is EXCLUDED deliberately:
    small-int list values change pickle length by bytes (noise next to MB
    block sizes — the same noise NOTES flagged in run_search deltas).
    A dropped block genuinely shrinks this; ties+smaller can promote."""
    remaining = BACKBONE_PARAMS
    for b in cfg["dropped"]:
        remaining -= ATTN_BLOCK_PARAMS if b % 2 == 0 else MLP_BLOCK_PARAMS
    fit_bytes = 0 if direct is None else len(direct["w"].tobytes()) + 8
    return remaining * 4 + (NECK_PARAMS + HEAD_PARAMS) * 4 + fit_bytes


def run_pipeline(shared, config, rgb_list, fit_explore=None):
    """Run built pipeline on RGB scenes. Returns list of depth arrays.

    shared: dict with backbone/neck/head modules + device + preprocess.
    fit_explore: (features, depths) to fit the direct head, or None.
    Heavy imports stay inside (keeps DSL logic torch-free).
    """
    import numpy as np
    import torch
    from geo_neck import GeometricNeck

    backbone, device = shared['backbone'], shared['device']
    preprocess = shared['preprocess']

    class Neck(ScaledNeckMixin, GeometricNeck):
        pass

    neck = Neck(device=device)
    neck._w = shared['neck_w']
    neck.buffers_loaded = True
    neck.res_scales = list(config['res_scales'])

    depths = []
    for rgb in rgb_list:
        if config['head'] == 'analytic':
            depths.append(_analytic_predict(rgb))
            continue
        fmaps, ph, pw = _backbone_tapped(
            backbone, preprocess, device, rgb,
            config['taps'], config['tap_gains'],
            config['block_gains'], config['dropped'])
        with torch.no_grad():
            fused = neck(fmaps)
            h = fused[config['readout']]
            if config['head'] == 'geo-conv':
                d = shared['head']([h], ph, pw)
            else:
                d = _direct_predict(shared, fit_explore, h, ph, pw)
            depths.append(d.squeeze(0).cpu().numpy())
    return depths


def _analytic_predict(rgb):
    """Zero-learned-weight geometric head (experimental_decoder tradition).

    Edge/texture/color/perspective cues with PHI^0,-1,-2,-3 weights.
    No fitted parameters — the no-learning limit probe.
    """
    import numpy as np
    import sys
    sys.path.insert(0, str(_ADAPT.parent))
    from geo_head import AnalyticHead
    gray = np.asanyarray(rgb, dtype=np.float64)[..., :3].mean(axis=-1)
    return AnalyticHead()(gray).astype(np.float64)


def _direct_predict(shared, fit, h, ph, pw):
    """64->1 least-squares head fitted on exploration data (fit_weights idea)."""
    import torch
    import torch.nn.functional as F
    w, b, Ht, Wt = fit['w'], fit['b'], fit['Ht'], fit['Wt']
    x = F.interpolate(h, (Ht, Wt), mode='bilinear', align_corners=True)
    B, C, H, W = x.shape
    flat = x.permute(0, 2, 3, 1).reshape(-1, C).float()
    d = flat @ torch.from_numpy(w).to(flat.device).float() + float(b)
    d = d.reshape(B, H, W)
    full = F.interpolate(d.unsqueeze(1), (int(ph * 14), int(pw * 14)),
                         mode='bilinear', align_corners=True)
    return torch.relu(full).squeeze(1)


def fit_direct_head(shared, config, explore):
    """Least squares 64->1 on exploration fused features. Returns fit dict."""
    import numpy as np
    import torch
    from geo_neck import GeometricNeck

    backbone, device = shared['backbone'], shared['device']
    preprocess = shared['preprocess']

    class Neck(ScaledNeckMixin, GeometricNeck):
        pass

    neck = Neck(device=device)
    neck._w = shared['neck_w']
    neck.buffers_loaded = True
    neck.res_scales = list(config['res_scales'])

    feats, tgts = [], []
    for rgb, ref, _cid in explore:
        fmaps, ph, pw = _backbone_tapped(
            backbone, preprocess, device, rgb,
            config['taps'], config['tap_gains'],
            config['block_gains'], config['dropped'])
        with torch.no_grad():
            fused = neck(fmaps)
            h = fused[config['readout']]
        Ht, Wt = h.shape[2], h.shape[3]
        f = h.squeeze(0).permute(1, 2, 0).reshape(-1, 64).cpu().numpy().astype(np.float64)
        t = np.array(ref, dtype=np.float64).reshape(-1)
        # match feature res to reference res
        if f.shape[0] != t.shape[0]:
            import cv2
            t = cv2.resize(t.reshape(ref.shape), (Wt, Ht),
                           interpolation=cv2.INTER_LINEAR).reshape(-1)
        feats.append(f)
        tgts.append(t)
    F_ = np.concatenate(feats)
    T_ = np.concatenate(tgts)
    A = np.concatenate([F_, np.ones((F_.shape[0], 1))], axis=1)
    sol, *_ = np.linalg.lstsq(A, T_, rcond=None)
    return {'w': sol[:-1].astype(np.float64), 'b': float(sol[-1]), 'Ht': Ht, 'Wt': Wt}


def corr(a, b):
    """Pearson corr; degenerate (constant) inputs score -1.0, never NaN.

    The core serializes diagnostics as strict JSON (no NaN/inf), so a
    degenerate prediction must be a clean failure, not an exception.
    """
    import numpy as np
    a = np.asanyarray(a, dtype=np.float64).flatten()
    b = np.asanyarray(b, dtype=np.float64).flatten()
    if a.shape != b.shape:
        import cv2
        n = int(np.sqrt(b.shape[0]))
        b = cv2.resize(b.reshape(n, n), (int(np.sqrt(a.shape[0])),) * 2,
                       interpolation=cv2.INTER_LINEAR).flatten()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return -1.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else -1.0


class DepthAdapter:
    """ExperimentAdapter over the depth DSL. Gate/audit data never enters build.

    size_mode: 'pickle' (legacy: config pickle bytes; keeps run_search
    behavior identical) or 'params' (honest model bytes incl. dropped
    blocks; for efficiency-gated drop search).
    """

    def __init__(self, shared, fixtures, size_mode='pickle'):
        self.shared = shared
        self.fixtures = fixtures  # role -> list[(rgb, ref, case_id)], NO audit key
        assert size_mode in ('pickle', 'params')
        self.size_mode = size_mode

    def seed(self):
        return TrialSpec("exact-replication", dict(SEED_CONFIG),
                         "incumbent: unmodified geometric pipeline")

    def propose(self, incumbent, measurement, history, tried):
        base = validate_config(incumbent.config)
        tried = set(tried)
        for cfg, rationale, _prio in neighbor_configs(base):
            bg = ''.join('m' if v == -2 else 'n' if v == -1 else str(v)
                         if v in (0, 1) else 'p' for v in cfg['block_gains'])
            name = (f"r{cfg['readout']}-t{''.join(map(str, cfg['taps']))}"
                    f"-g{''.join(map(str, cfg['tap_gains']))}"
                    f"-b{bg}-x{''.join(map(str, sorted(cfg['dropped'])))}"
                    f"-s{''.join(map(str, cfg['res_scales']))}-{cfg['head']}")
            spec = TrialSpec(name, cfg, rationale)
            if spec.identifier not in tried:
                return spec
        return None

    def build(self, proposal):
        cfg = validate_config(proposal.config)
        artifact = {"config": cfg, "direct": None}
        if cfg["head"] == "direct":
            fit = fit_direct_head(self.shared, cfg, self.fixtures["exploration"])
            artifact["direct"] = {"w": fit["w"], "b": fit["b"],
                                  "Ht": fit["Ht"], "Wt": fit["Wt"]}
        return artifact

    def evaluate(self, artifact):
        cfg = validate_config(artifact["config"])
        fit = artifact["direct"]
        cases = {}
        for role in ("exploration", "gate", "retention"):
            results = []
            for rgb, ref, case_id in self.fixtures[role]:
                pred = run_pipeline(self.shared, cfg, [rgb],
                                    fit_explore=fit)[0]
                c = corr(pred, ref)
                results.append(CaseResult(case_id, True, c >= CORR_PASS, f"corr={c:.5f}"))
            cases[role] = Scorecard(f"dav2-{role}-168", tuple(results))
        blob = pickle.dumps({"config": cfg,
                             "direct": None if fit is None else
                             {"w": fit["w"].tobytes(), "b": fit["b"],
                              "Ht": fit["Ht"], "Wt": fit["Wt"]}}, protocol=5)
        nbytes = _model_bytes(cfg, fit) if self.size_mode == 'params' else len(blob)
        diags = {role: {"mean_corr": sum(float(r.signature.split('=')[1])
                                          for r in cases[role].cases) / len(cases[role].cases)}
                 for role in cases}
        return Measurement(cases["exploration"], cases["gate"], cases["retention"],
                           nbytes, diags)

    def fingerprint(self, artifact):
        cfg = validate_config(artifact["config"])
        h = hashlib.sha256()
        h.update(json.dumps(cfg, sort_keys=True).encode())
        if artifact["direct"] is not None:
            h.update(artifact["direct"]["w"].tobytes())
            h.update(str(artifact["direct"]["b"]).encode())
        return h.hexdigest()
