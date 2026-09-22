#!/usr/bin/env python3
"""Structured sparsity probe: zero whole output-rows (neurons) by row-L2.

Row fractions {5%, 10%, 25%} on 2D backbone weights; norms/biases/scales
untouched. Same 6 probe scenes (3 synth + 3 real), seed pipeline.
Pre-registered 2026-09-22: 5% ~0.99 marginal, 25% collapse; 5%-at-parity
goes to the gate immediately.
Run: python adapt/sparsity_probe.py
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np
import copy


def main():
    import torch
    from run_search import load_shared, load_fixtures
    from depth_adapter import run_pipeline, corr, SEED_CONFIG

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared = load_shared(device)
    backbone = shared['backbone']

    fx = load_fixtures()['exploration'][:3]
    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    rids = [str(x) for x in zr['ids']]
    reals = [(zr['rgb'][i].astype(np.float32) / 255.0,
              zr['ref'][i].astype(np.float64), rids[i]) for i in (0, 7, 19)]
    scenes = fx + reals

    orig = {k: v.clone() for k, v in backbone._w.items()}

    def evaluate(tag):
        cs = []
        for rgb, ref, cid in scenes:
            cfg = copy.deepcopy(dict(SEED_CONFIG))
            pred = run_pipeline(shared, cfg, [rgb])[0]
            cs.append(corr(pred, ref))
        n_ok = sum(c >= 0.999 for c in cs)
        print(f"  [{tag}] mean={np.mean(cs):.5f} min={min(cs):.5f} "
              f"pass={n_ok}/{len(cs)} " +
              " ".join(f"{c:.5f}" for c in cs), flush=True)
        return cs

    print("baseline:", flush=True)
    evaluate("full")

    for frac in (0.05, 0.10, 0.25):
        nrows = 0
        for k, v in orig.items():
            if v.ndim != 2:
                continue
            W = v.cpu().numpy().astype(np.float64)
            row_l2 = np.sqrt((W ** 2).sum(axis=1))
            thr = np.quantile(row_l2, frac)
            mask = (row_l2 > thr).astype(np.float32)[:, None]
            nrows += int((mask[:, 0] == 0).sum())
            backbone._w[k].copy_(
                torch.from_numpy((W * mask).astype(np.float32)).to(device))
        tag = f"sparsity-{frac:.0%} ({nrows} rows zeroed)"
        print(tag + ":", flush=True)
        evaluate(tag)
        for k, v in orig.items():
            backbone._w[k].copy_(v)


if __name__ == '__main__':
    main()
