#!/usr/bin/env python3
"""Graduation experiment: search LUT widths (+accum) for minimal tables holding parity.

Known-answer test: seed full tables {13312,16,4096,tree}, expect discovery
of {8192,8,4096} (~576 kB). Integer-head evaluation on fixture scenes
(tree: explicit python loop = njit-bit-exact; fixed: bridge loop), plus a
deterministic softmax probe (makes exp_span observable) and sealed audit.
Promotion via EfficiencyRule (ties + fewer bytes).

Run: python adapt/graduate.py [--trials N]
Torch/geo imports stay inside functions (DSL + rules import torch-free).
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))
sys.path.insert(0, str(ADAPT / 'third_party'))

from experimenter import TrialSpec, CaseResult, Scorecard, Measurement, Experimenter  # noqa: E402
from efficiency import EfficiencyRule  # noqa: E402

FRACS = (13312, 8192, 4096, 2048)
EXPS = (16, 8, 4)
DMAXS = (4096, 1024)
ACCUMS = ("tree", "fixed")
SEED = {"frac_cap": 13312, "exp_span": 16, "dmax": 4096, "accum": "tree"}
CORR_PASS = 0.999
STRIDE = 4  # integer-head eval grid stride on 518px features (130^2 pts)


def validate_widths(cfg):
    if not isinstance(cfg, dict) or set(cfg) != {"frac_cap", "exp_span", "dmax", "accum"}:
        raise ValueError("width config needs exactly frac_cap/exp_span/dmax/accum")
    if cfg["frac_cap"] not in FRACS:
        raise ValueError(f"frac_cap must be one of {FRACS}")
    if cfg["exp_span"] not in EXPS:
        raise ValueError(f"exp_span must be one of {EXPS}")
    if cfg["dmax"] not in DMAXS:
        raise ValueError(f"dmax must be one of {DMAXS}")
    if cfg["accum"] not in ACCUMS:
        raise ValueError(f"accum must be one of {ACCUMS}")
    return dict(cfg)


def table_bytes(cfg):
    """Shippable int32 LUT bytes for a width config (125B head const excluded)."""
    return ((cfg["dmax"] + 1) * 4 * 2 + (cfg["frac_cap"] + 1) * 4
            + (cfg["exp_span"] * (1 << 14) + 1) * 4)


def neighbor_widths(cfg):
    """Single moves ordered by expected byte saving. (config, rationale)."""
    out = []
    f, e, d, a = cfg["frac_cap"], cfg["exp_span"], cfg["dmax"], cfg["accum"]
    for v in EXPS:
        if v != e:
            c = dict(cfg, exp_span=v)
            out.append((c, f"exp span {e}->{v} ({table_bytes(c) - table_bytes(cfg):+}B)",
                        table_bytes(cfg) - table_bytes(c)))
    for v in FRACS:
        if v != f:
            c = dict(cfg, frac_cap=v)
            out.append((c, f"frac cap {f}->{v} ({table_bytes(c) - table_bytes(cfg):+}B)",
                        table_bytes(cfg) - table_bytes(c)))
    for v in DMAXS:
        if v != d:
            c = dict(cfg, dmax=v)
            out.append((c, f"dmax {d}->{v} ({table_bytes(c) - table_bytes(cfg):+}B)",
                        table_bytes(cfg) - table_bytes(c)))
    for v in ACCUMS:
        if v != a:
            c = dict(cfg, accum=v)
            out.append((c, f"accumulator {a}->{v} (0B, parity comparison)",
                        0))
    out.sort(key=lambda t: -t[2])
    return [(c, r) for c, r, _ in out]


class WidthAdapter:
    """ExperimentAdapter over LUT widths. Integer-path evaluation only."""

    def __init__(self, shared, fixtures, probe):
        self.shared = shared
        self.fixtures = fixtures
        self.probe = probe  # (logits, ref_probs) deterministic softmax probe

    def seed(self):
        return TrialSpec("full-tables", dict(SEED),
                         "incumbent: full LUT widths, tree accumulation")

    def propose(self, incumbent, measurement, history, tried):
        base = validate_widths(incumbent.config)
        tried = set(tried)
        for cfg, rationale in neighbor_widths(base):
            spec = TrialSpec(f"f{cfg['frac_cap']}-e{cfg['exp_span']}-"
                             f"d{cfg['dmax']}-{cfg['accum']}", cfg, rationale)
            if spec.identifier not in tried:
                return spec
        return None

    def build(self, proposal):
        return {"config": validate_widths(proposal.config)}

    def evaluate(self, artifact):
        import geo_int as G
        cfg = validate_widths(artifact["config"])
        from lut_pareto import build_frac_table, build_exp_table
        saved = (G.FRAC_CAP, G._FRAC_LUT, G.DMAX, G._EXP_LUT)
        try:
            G.FRAC_CAP = cfg["frac_cap"]
            G._FRAC_LUT = build_frac_table(cfg["frac_cap"])
            G.DMAX = cfg["dmax"]
            G._EXP_LUT = build_exp_table(cfg["exp_span"])
            add = G.build_add_lut(cfg["dmax"])
            sub = G.build_sub_lut(cfg["dmax"])
            results = {}
            for role in ("exploration", "gate", "retention"):
                cases = []
                for rgb, ref, cid in self.fixtures[role]:
                    pred = _integer_depth(self.shared, cfg, rgb, add, sub)
                    c = _corr(pred, ref)
                    cases.append(CaseResult(cid, True, c >= CORR_PASS, f"corr={c:.5f}"))
                results[role] = Scorecard(f"dav2int-{role}-s4", tuple(cases))
            # deterministic softmax probe (makes exp_span observable)
            logits, ref_p = self.probe
            s, e, z = G.int_encode_array(logits)
            num, den = G.int_softmax_fixed(s, e, z)
            got = (num.astype(float) / den).astype(float)
            cp = float(__import__('numpy').corrcoef(got.flatten(), ref_p.flatten())[0, 1])
            exp_cases = list(results["exploration"].cases) + [
                CaseResult("softmax-probe", True, cp >= 0.9999, f"corr={cp:.5f}")]
            results["exploration"] = Scorecard("dav2int-exploration-s4", tuple(exp_cases))
            diags = {r: {"mean_corr": sum(float(c.signature.split('=')[1])
                                          for c in results[r].cases) / len(results[r].cases)}
                     for r in results}
            return Measurement(results["exploration"], results["gate"],
                               results["retention"], table_bytes(cfg), diags)
        finally:
            G.FRAC_CAP, G._FRAC_LUT, G.DMAX, G._EXP_LUT = saved

    def fingerprint(self, artifact):
        cfg = validate_widths(artifact["config"])
        return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _integer_depth(shared, cfg, rgb, add, sub):
    """Integer head on stride-4 518px features. Returns 130^2 depth grid."""
    import numpy as np
    import torch
    import geo_int as G
    head = shared['int_head']
    pv = shared['preprocess'](rgb).to(shared['device'])
    with torch.no_grad():
        fmaps, ph, pw = shared['backbone'].forward_stages(pv)
        fused = shared['neck'](fmaps)
        _ = shared['head'](fused, ph, pw)
        feat = shared['head'].last_features.squeeze(0).permute(1, 2, 0)
        feat = feat[::STRIDE, ::STRIDE, :].reshape(-1, 32).cpu().numpy()
    fs, fe = G.IntegerPhiHead.encode_features(feat)
    if cfg["accum"] == "tree":
        n = fs.shape[0]
        out = np.empty(n)
        for i in range(n):
            s, e = head.int_predict_pixel(fs[i], fe[i])
            out[i] = G._decode_exp(e, 0) * s if e != 0 else 0.0
    else:
        out = np.empty(fs.shape[0])
        for i in range(fs.shape[0]):
            out[i] = G._head_fixed_pixel(head, fs[i], fe[i])
    H = Wd = 518 // STRIDE + (1 if 518 % STRIDE else 0)
    return out.reshape(H, Wd)


def _corr(a, b):
    import numpy as np
    a = np.asanyarray(a, dtype=np.float64).flatten()
    b = np.asanyarray(b, dtype=np.float64).flatten()
    if a.shape != b.shape:
        import cv2
        n = int(round(b.shape[0] ** 0.5))
        side = int(round(a.shape[0] ** 0.5))
        b = cv2.resize(b.reshape(n, n), (side, side),
                       interpolation=cv2.INTER_LINEAR).flatten()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return -1.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else -1.0


def load_all(device):
    """Shared float pipeline + fixtures (refs resized to eval grid) + probe."""
    import numpy as np
    import torch
    import build_fixtures as bf
    from geo_backbone import GeometricDinov2Backbone
    from geo_neck import GeometricNeck
    from geo_head import GeometricHead
    from geo_depth import preprocess
    from geo_int import IntegerPhiHead
    from pathlib import Path as _P
    W = REPO / 'weights'
    backbone = GeometricDinov2Backbone(W / 'geometric_backbone.npz', device=device)
    backbone.eval()
    neck = GeometricNeck(W / 'geometric_neck.npz', device=device)
    head = GeometricHead(W / 'geometric_head.npz', device=device)
    int_head = IntegerPhiHead(REPO / 'weights' / 'phi_weights_compact.bin')
    shared = {'backbone': backbone, 'neck': neck, 'head': head,
              'int_head': int_head, 'device': device,
              'preprocess': lambda rgb: preprocess(rgb, size=518)}
    if not bf.FIX.exists():
        print("building fixtures (one-time HF oracle)...")
        bf.main()
    z = np.load(bf.FIX, allow_pickle=False)
    import cv2
    grid = 518 // STRIDE + (1 if 518 % STRIDE else 0)
    fixtures = {}
    for role in ('exploration', 'gate', 'retention', 'audit'):
        items = []
        for i in range(len(z[f'{role}_rgb'])):
            rgb = z[f'{role}_rgb'][i].astype(np.float32) / 255.0
            ref = cv2.resize(z[f'{role}_ref'][i].astype(np.float64), (grid, grid),
                             interpolation=cv2.INTER_LINEAR)
            items.append((rgb, ref, f"{role}-{i}"))
        fixtures[role] = items
    rng = np.random.default_rng(7)
    logits = (rng.standard_normal(290) * 1.5 - 2.0).astype(np.float64)
    logits[rng.choice(290, 4, replace=False)] += 6.0
    import torch.nn.functional as F
    ref_p = F.softmax(torch.from_numpy(logits).float(), dim=0).numpy().astype(float)
    return shared, fixtures, (logits, ref_p)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=14)
    args = ap.parse_args()
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    shared, fixtures, probe = load_all(device)
    search_fx = {r: fixtures[r] for r in ('exploration', 'gate', 'retention')}
    lab = Experimenter(WidthAdapter(shared, search_fx, probe),
                       max_trials=args.trials, rule=EfficiencyRule())
    t0 = time.perf_counter()
    last = 0
    while not lab.done:
        st = lab.step()
        if len(st['history']) > last:
            last = len(st['history'])
            h = st['history'][-1]
            print(f"trial {h['trial']}: {h['proposal']['name']} -> "
                  f"{h['decision']['action']} {h['decision']['reasons']} "
                  f"{json.dumps(h['decision'].get('deltas', {}))}", flush=True)
    dt = time.perf_counter() - t0
    final = lab.model()
    print(f"\nstop: {lab.state()['stop_reason']} sealed: {lab.state()['sealed']}")
    print(f"incumbent: {json.dumps(final['config'])} bytes={table_bytes(final['config'])}")
    print(f"search time: {dt:.0f}s")
    res = []
    for rgb, ref, cid in fixtures['audit']:
        import geo_int as G
        from lut_pareto import build_frac_table, build_exp_table
        saved = (G.FRAC_CAP, G._FRAC_LUT, G.DMAX, G._EXP_LUT)
        try:
            G.FRAC_CAP = final['config']['frac_cap']
            G._FRAC_LUT = build_frac_table(final['config']['frac_cap'])
            G.DMAX = final['config']['dmax']
            G._EXP_LUT = build_exp_table(final['config']['exp_span'])
            add = G.build_add_lut(final['config']['dmax'])
            sub = G.build_sub_lut(final['config']['dmax'])
            pred = _integer_depth(shared, final['config'], rgb, add, sub)
            res.append((cid, _corr(pred, ref)))
        finally:
            G.FRAC_CAP, G._FRAC_LUT, G.DMAX, G._EXP_LUT = saved
    print("audit (post-seal, report-only):")
    for cid, c in res:
        print(f"  {cid}: corr={c:.5f}")
    runs = ADAPT / 'runs'
    runs.mkdir(exist_ok=True)
    (runs / 'graduation.json').write_text(json.dumps(
        {"incumbent": final, "audit": res, "seconds": dt}, indent=1, default=str))
    print("wrote adapt/runs/graduation.json")


if __name__ == '__main__':
    main()
