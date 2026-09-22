#!/usr/bin/env python3
"""Neck v4 search: fusion-stage drops + stream zeros + channel halves.

Own DSL {stage_drop:[], stream_zero:[], chan_half:[]} over the seed
arch config. Byte model: per-stage neck params measured EXACT from npz
shapes (signs+exps arrays halved) x4 fp32 — same discipline as
ATTN/MLP block bytes. Evaluation: 13 fixture scenes + 8 COCO real
panel (the arena upgrade), CORR_PASS 0.999, EfficiencyRule, margin
reading via adapt/margin.py conventions (per-case corrs recorded).

Pre-registered 2026-09-22: stage drops fail; stream-zero i=3 nearest
to tie; channel halves fail. Any hold+bytes-win goes to the emitter.
Run: python adapt/neck_search.py [--trials N]
"""
import copy
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))
sys.path.insert(0, str(ADAPT / 'third_party'))

import numpy as np

from depth_adapter import (DepthAdapter, validate_config, SEED_CONFIG,
                           ScaledNeckMixin, _model_bytes, CORR_PASS)
from experimenter import TrialSpec, CaseResult, Scorecard, Measurement
from efficiency import EfficiencyRule

STAGES = (0, 1, 2, 3)

SEED_NECK = {"stage_drop": [], "stream_zero": [], "chan_half": []}


def stage_bytes():
    """Per-stage fp32 bytes (re+cv+fusion), measured from npz shapes.

    Arrays come in signs+exps pairs: true params = sum(sizes)/2.
    """
    import re
    z = np.load(REPO / 'weights' / 'geometric_neck.npz', allow_pickle=False)
    groups = {}
    for k in z.files:
        if k.endswith('.shape'):
            continue
        base = k[:-6] if k.endswith('.signs') else k[:-5]
        key = re.match(r'(re\d|cv\d|fusion\d)', base).group(1)
        g = key[:3] if key.startswith(('re', 'cv')) else key
        groups.setdefault(g, []).append(k)
    out = {}
    for i in range(4):
        n = sum(z[k].size for g in (f're{i}', f'cv{i}', f'fusion{i}')
                for k in groups[g])
        out[i] = n * 2  # true params x 4B fp32 = sum(sizes) x 2B
    return out


STAGE_BYTES = stage_bytes()
NECK_TOTAL = sum(STAGE_BYTES.values())


def validate_neck(cfg):
    if not isinstance(cfg, dict) or set(cfg) != set(SEED_NECK):
        raise ValueError(f"neck config keys must be exactly {sorted(SEED_NECK)}")
    out = {}
    for k in ("stage_drop", "stream_zero", "chan_half"):
        v = cfg[k]
        if (not isinstance(v, list) or any(type(x) is not int or x not in STAGES for x in v)
                or len(set(v)) != len(v)):
            raise ValueError(f"{k} must be a unique list of stage ids 0..3")
        out[k] = sorted(v)
    if set(out["stage_drop"]) & set(out["stream_zero"]):
        raise ValueError("a stage cannot be both dropped and zeroed")
    return out


def neck_bytes(ncfg):
    """Remaining neck fp32 bytes. Dropped/zeroed stage frontend (re+cv)
    always saves; fusion saves only on drop. chan_half saves half the
    stage's conv bytes (narrow-net accounting; functional test is the
    zero-info probe — documented)."""
    import re
    z = np.load(REPO / 'weights' / 'geometric_neck.npz', allow_pickle=False)
    groups = {}
    for k in z.files:
        if k.endswith('.shape'):
            continue
        base = k[:-6] if k.endswith('.signs') else k[:-5]
        key = re.match(r'(re\d|cv\d|fusion\d)', base).group(1)
        g = key[:3] if key.startswith(('re', 'cv')) else key
        groups.setdefault(g, []).append(k)

    def gbytes(*gs):
        return sum(z[k].size for g in gs for k in groups[g]) * 2

    keep = NECK_TOTAL
    for i in ncfg["stage_drop"]:
        keep -= gbytes(f're{i}', f'cv{i}', f'fusion{i}')
    for i in ncfg["stream_zero"]:
        keep -= gbytes(f're{i}', f'cv{i}')
    for i in ncfg["chan_half"]:
        if i not in ncfg["stage_drop"] and i not in ncfg["stream_zero"]:
            keep -= gbytes(f'cv{i}', f'fusion{i}') // 2
    return keep


def neighbor_neck(cfg):
    out = []
    for key in ("stage_drop", "stream_zero", "chan_half"):
        have = set(cfg[key])
        for i in STAGES:
            if i not in have:
                c = copy.deepcopy(cfg)
                c[key] = sorted(have | {i})
                try:
                    out.append((validate_neck(c), f"neck {key} +{i}", 10))
                except ValueError:
                    pass
        for i in sorted(have):
            c = copy.deepcopy(cfg)
            c[key] = sorted(have - {i})
            out.append((validate_neck(c), f"neck {key} -{i} restore", 12))
    return sorted(out, key=lambda t: t[2])


def run_neck_pipeline(shared, arch_cfg, ncfg, rgb_list, fit_explore=None):
    """Like run_pipeline + neck mods. Readout maps to last fused stage."""
    import torch
    import torch.nn.functional as F
    from geo_neck import GeometricNeck
    from depth_adapter import (_backbone_tapped, _direct_predict,
                               _analytic_predict)

    backbone, device = shared['backbone'], shared['device']
    preprocess = shared['preprocess']

    class Neck(ScaledNeckMixin, GeometricNeck):
        pass

    neck = Neck(device=device)
    neck._w = shared['neck_w']
    neck.buffers_loaded = True
    neck.res_scales = list(arch_cfg['res_scales'])

    depths = []
    for rgb in rgb_list:
        if arch_cfg['head'] == 'analytic':
            depths.append(_analytic_predict(rgb))
            continue
        fmaps, ph, pw = _backbone_tapped(
            backbone, preprocess, device, rgb,
            arch_cfg['taps'], arch_cfg['tap_gains'],
            arch_cfg['block_gains'], arch_cfg['dropped'])
        with torch.no_grad():
            # reassemble (mirror GeometricNeck.forward, with mods)
            feats = []
            for i, fm in enumerate(fmaps):
                x = F.conv2d(fm, neck._g(f're{i}.proj.weight'),
                             neck._g(f're{i}.proj.bias'))
                if i == 0:
                    x = F.conv_transpose2d(
                        x, neck._g('re0.deconv.weight'),
                        neck._g('re0.deconv.bias'), stride=4)
                elif i == 1:
                    x = F.conv_transpose2d(
                        x, neck._g('re1.deconv.weight'),
                        neck._g('re1.deconv.bias'), stride=2)
                elif i == 3:
                    x = F.conv2d(x, neck._g('re3.down.weight'),
                                 neck._g('re3.down.bias'), stride=2, padding=1)
                x = F.conv2d(x, neck._g(f'cv{i}.weight'), None, padding=1)
                if i in ncfg["stream_zero"]:
                    x = torch.zeros_like(x)
                if i in ncfg["chan_half"]:
                    x = torch.cat([x[:, :32], torch.zeros_like(x[:, 32:])],
                                  dim=1)
                feats.append(x)
            # fusion with stage drops (bridge sizes by interp)
            rev = feats[::-1]
            fused_list, fused = [], None
            for idx in range(4):
                if idx in ncfg["stage_drop"]:
                    if fused is not None and idx < 3:
                        want = rev[idx + 1].shape[2:]
                        if fused.shape[2:] != want:
                            fused = F.interpolate(
                                fused, size=want, mode='bilinear',
                                align_corners=True)
                    continue
                hs = rev[idx]
                if fused is None:
                    size = rev[idx + 1].shape[2:] if idx < 3 else None
                    fused = neck._fusion_layer(hs, None, size, idx)
                else:
                    size = rev[idx + 1].shape[2:] if idx < 3 else None
                    fused = neck._fusion_layer(fused, hs, size, idx)
                fused_list.append(fused)
            h = fused_list[min(arch_cfg['readout'], len(fused_list) - 1)]
            if arch_cfg['head'] == 'geo-conv':
                d = shared['head']([h], ph, pw)
            else:
                d = _direct_predict(shared, fit_explore, h, ph, pw)
            depths.append(d.squeeze(0).cpu().numpy())
    return depths


class NeckAdapter(DepthAdapter):
    """DepthAdapter + neck DSL. Artifact carries neck mods; bytes honest."""

    def __init__(self, shared, fixtures, real_panel=()):
        super().__init__(shared, fixtures)
        self.real_panel = list(real_panel)

    def seed(self):
        cfg = dict(SEED_CONFIG)
        return TrialSpec("neck-seed", {"arch": cfg, "neck": dict(SEED_NECK)},
                         "incumbent: unmodified neck")

    def propose(self, incumbent, measurement, history, tried):
        base = validate_neck(incumbent.config["neck"])
        tried = set(tried)
        for cfg, rationale, _prio in neighbor_neck(base):
            name = (f"drop{''.join(map(str, cfg['stage_drop'])) or '-'}"
                    f"_zero{''.join(map(str, cfg['stream_zero'])) or '-'}"
                    f"_half{''.join(map(str, cfg['chan_half'])) or '-'}")
            spec = TrialSpec(name, {"arch": dict(incumbent.config["arch"]),
                                    "neck": cfg}, rationale)
            if spec.identifier not in tried:
                return spec
        return None

    def build(self, proposal):
        arch = validate_config(proposal.config["arch"])
        neck = validate_neck(proposal.config["neck"])
        artifact = {"config": {"arch": arch, "neck": neck}, "direct": None}
        if arch["head"] == "direct":
            from depth_adapter import fit_direct_head
            fit = fit_direct_head(self.shared, arch, self.fixtures["exploration"])
            artifact["direct"] = {"w": fit["w"], "b": fit["b"],
                                  "Ht": fit["Ht"], "Wt": fit["Wt"]}
        return artifact

    def evaluate(self, artifact):
        import pickle
        arch = validate_config(artifact["config"]["arch"])
        ncfg = validate_neck(artifact["config"]["neck"])
        fit = artifact["direct"]
        cases = {}
        for role in ("exploration", "gate", "retention"):
            results = []
            for rgb, ref, case_id in self.fixtures[role]:
                from depth_adapter import corr
                pred = run_neck_pipeline(self.shared, arch, ncfg, [rgb],
                                         fit_explore=fit)[0]
                c = corr(pred, ref)
                results.append(CaseResult(case_id, True, c >= CORR_PASS,
                                         f"corr={c:.5f}"))
            cases[role] = Scorecard(f"dav2-neck-{role}", tuple(results))
        # real panel, same bar
        from depth_adapter import corr as _corr
        real_corrs = []
        for rgb, ref, cid in self.real_panel:
            pred = run_neck_pipeline(self.shared, arch, ncfg, [rgb],
                                     fit_explore=fit)[0]
            real_corrs.append(_corr(pred, ref))
        nb = (_model_bytes(arch, fit) - NECK_TOTAL + neck_bytes(ncfg))
        diags = {role: {"mean_corr": sum(float(r.signature.split('=')[1])
                                         for r in cases[role].cases) / len(cases[role].cases)}
                 for role in cases}
        diags["real"] = {"mean_corr": float(np.mean(real_corrs)) if real_corrs else 1.0,
                         "min_corr": float(min(real_corrs)) if real_corrs else 1.0,
                         "pass": sum(1 for c in real_corrs if c >= CORR_PASS),
                         "total": len(real_corrs)}
        return Measurement(cases["exploration"], cases["gate"], cases["retention"],
                           nb, diags)

    def fingerprint(self, artifact):
        import hashlib
        import json
        h = hashlib.sha256()
        h.update(json.dumps(artifact["config"], sort_keys=True).encode())
        return h.hexdigest()


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=24)
    ap.add_argument('--real', type=int, default=8)
    args = ap.parse_args()

    import torch
    from run_search import load_shared, load_fixtures
    from baseline import check_baseline, fmt_counts, parity_gate
    from experimenter import Experimenter

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device} neck_total={NECK_TOTAL / 1e6:.1f}MB "
          f"stages={[STAGE_BYTES[i] / 1e6 for i in range(4)]}", flush=True)
    shared = load_shared(device)
    fx = load_fixtures(include_audit=False)
    fixtures = {r: fx[r] for r in ('exploration', 'gate', 'retention')}
    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    rids = [str(x) for x in zr['ids']]
    real = [(zr['rgb'][i].astype(np.float32) / 255.0,
             zr['ref'][i].astype(np.float64), rids[i])
            for i in range(min(args.real, len(zr['rgb'])))]
    adapter = NeckAdapter(shared, fixtures, real)
    check_baseline(adapter, parity_gate(CORR_PASS), label="neck-seed")

    exp = Experimenter(adapter, max_trials=args.trials,
                       rule=EfficiencyRule(
                           max_model_bytes=200 * 1024 * 1024))
    t0 = time.perf_counter()
    while not exp.state()["done"]:
        exp.step()
        st = exp.state()
        if st["stage"] in ("build", "evaluate", "decide"):
            continue
        h = st["history"][-1] if st["history"] else None
        if h:
            m = h.get("measurement") or {}
            dg = (m.get("diagnostics") or {})
            real_d = dg.get("real", {})
            print(f"trial {len(st['history'])}: {h['proposal']['name']} "
                  f"-> {h['decision']['action']} "
                  f"real={real_d.get('pass')}/{real_d.get('total')} "
                  f"abs={fmt_counts(m)}", flush=True)
    dt = time.perf_counter() - t0
    st = exp.state()
    print(f"NECK: {st['stop_reason']} {dt:.0f}s")
    (ADAPT / 'runs' / 'neck_search.json').write_text(
        __import__('json').dumps(
            {"stop": st["stop_reason"], "history": st["history"]},
            indent=0, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
