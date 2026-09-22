#!/usr/bin/env python3
"""Low-rank pathway probe: per-matrix truncated SVD, float factors first.

Patches rank-r reconstructions (float32, NO phi rounding — thesis upper
bound) into backbone buffers, runs seed pipeline on 3 synth + 3 real
scenes. Per-matrix rank fractions {1/2, 1/4, 1/8} of min(dim); 2D
weights only (norms/biases/layer-scales untouched).

Pre-registered 2026-09-22: rank-1/2 holds >=0.999 on >=4/6; rank-1/8
collapses; attn proj/qkv more sensitive than MLP.
Run: python adapt/lowrank_probe.py
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

    # per-family sensitivity split: attn (q/k/v/proj) vs mlp (mlp1/mlp2)
    import re
    for frac in (0.5, 0.25, 0.125):
        for family in ('all', 'attn', 'mlp'):
            nb = 0
            for k, v in orig.items():
                if v.ndim != 2:
                    continue
                is_attn = any(s in k for s in ('q.weight', 'k.weight',
                                              'v.weight', 'proj.weight'))
                is_mlp = 'mlp' in k
                if family == 'attn' and not is_attn:
                    continue
                if family == 'mlp' and not is_mlp:
                    continue
                W = v.double().cpu().numpy()
                r = max(1, int(min(W.shape) * frac))
                U, S, Vh = np.linalg.svd(W, full_matrices=False)
                Wr = (U[:, :r] * S[:r]) @ Vh[:r]
                backbone._w[k].copy_(
                    torch.from_numpy(Wr.astype(np.float32)).to(device))
                nb += r * (W.shape[0] + W.shape[1])
            nel = sum(v.numel() for k, v in orig.items() if v.ndim == 2)
            tag = f"rank-{frac} family={family} ({nb / 1e6:.1f}M factor params)"
            print(tag + ":", flush=True)
            evaluate(tag)
            for k, v in orig.items():
                backbone._w[k].copy_(v)


if __name__ == '__main__':
    main()
