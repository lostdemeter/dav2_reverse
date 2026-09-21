#!/usr/bin/env python3
"""Run the depth search: seed -> propose/build/evaluate/promote -> seal -> audit.

The adapter NEVER sees audit fixtures (they open once, post-seal, for the
report only). Run: python adapt/run_search.py [--trials N]
"""
import json
import sys
import time
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))                  # geo_* modules
sys.path.insert(0, str(ADAPT))                 # depth_adapter, build_fixtures
sys.path.insert(0, str(ADAPT / 'third_party'))  # vendored core

from experimenter import Experimenter, PromotionRule  # noqa: E402
import build_fixtures as bf  # noqa: E402
from depth_adapter import DepthAdapter, run_pipeline, corr  # noqa: E402


def load_shared(device):
    import numpy as np
    import torch
    from geo_backbone import GeometricDinov2Backbone
    from geo_head import GeometricHead
    from geo_depth import preprocess
    from geo_lut import PHI, K, BIAS
    W = REPO / 'weights'
    print("loading geometric pipeline (shared, read-only for search)...")
    backbone = GeometricDinov2Backbone(W / 'geometric_backbone.npz', device=device)
    backbone.eval()
    z = np.load(W / 'geometric_neck.npz', allow_pickle=False)
    neck_w = {}
    for k in z.files:
        if k.endswith('.signs'):
            base = k[:-len('.signs')]
            neck_w[base] = torch.from_numpy(
                (z[k].astype(np.float32)
                 * np.float32(PHI) ** ((z[base + '.exps'].astype(np.float32) - BIAS) / K))
            ).to(device)
    head = GeometricHead(W / 'geometric_head.npz', device=device)
    # NOTE: fixtures' HF references were rendered through the HF processor
    # (518px), so the search runs at 518 too — same pixels in, comparable
    # depths out. Smaller/faster sizes are future work (needs matched refs).
    return {'backbone': backbone, 'neck_w': neck_w, 'head': head,
            'device': device, 'preprocess': lambda rgb: preprocess(rgb, size=518)}


def load_fixtures(include_audit=False):
    import numpy as np
    if not bf.FIX.exists():
        print("building fixtures (one-time HF oracle)...")
        bf.main()
    z = np.load(bf.FIX, allow_pickle=False)
    roles = ['exploration', 'gate', 'retention'] + (['audit'] if include_audit else [])
    out = {}
    for role in roles:
        rgbs = z[f'{role}_rgb'].astype(np.float32) / 255.0
        refs = z[f'{role}_ref'].astype(np.float64)
        out[role] = [(rgbs[i], refs[i], f"{role}-{i}") for i in range(len(rgbs))]
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=10)
    args = ap.parse_args()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    shared = load_shared(device)
    fixtures = load_fixtures(include_audit=False)
    adapter = DepthAdapter(shared, fixtures)
    lab = Experimenter(adapter, max_trials=args.trials, rule=PromotionRule())
    t0 = time.perf_counter()
    last_hist = 0
    while not lab.done:
        st = lab.step()
        if len(st['history']) > last_hist:
            last_hist = len(st['history'])
            h = st['history'][-1]
            print(f"trial {h['trial']}: {h['proposal']['name']} -> "
                  f"{h['decision']['action']} {h['decision']['reasons']} "
                  f"{json.dumps(h['decision'].get('deltas', {}))}", flush=True)
    dt = time.perf_counter() - t0
    final = lab.model()
    print(f"\nstop: {lab.state()['stop_reason']}  sealed: {lab.state()['sealed']}")
    print(f"incumbent: {json.dumps(final['config'])}" if isinstance(final, dict) else final)
    print(f"search time: {dt:.0f}s")

    # sealed audit: open once, report only
    audit = load_fixtures(include_audit=True)['audit']
    res = []
    fit = final.get('direct') if isinstance(final, dict) else None
    cfg = final['config'] if isinstance(final, dict) else final.config
    for rgb, ref, cid in audit:
        pred = run_pipeline(shared, cfg, [rgb], fit_explore=fit)[0]
        c = corr(pred, ref)
        res.append((cid, c))
    print("audit (post-seal, report-only):")
    for cid, c in res:
        print(f"  {cid}: corr={c:.5f}")
    runs = ADAPT / 'runs'
    runs.mkdir(exist_ok=True)
    out = runs / 'latest.json'
    out.write_text(json.dumps({"incumbent": final, "audit": res, "seconds": dt},
                              indent=1, default=str))
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
