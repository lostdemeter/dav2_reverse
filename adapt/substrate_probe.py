#!/usr/bin/env python3
"""Substrate experiment: exact-vs-mush verdict comparison, same harness.

Mush = per-layer C=2048 codebook backbone (measured float-pipeline
0.9983 — principled coarsening, not arbitrary noise). Width DSL +
integer head harness UNCHANGED (backbone is the substrate, widths the
variable). Bar recalibrated to MUSH-seed capability (same rule as
CORR_PASS_INT); absolute bars would trivially fail everything, which
proves only tightness.

Decisive trials (from graduation history): exp8 (promoted 14/14),
dmax1024 (rejected, retention loss), frac2048 (correctly
unpromotable). Verdict pattern compared at respective bars.

Pre-registered 2026-09-22: mush-seed lands 0.997-0.9985; pattern
reproduces -> contracts substrate-invariant at recalibrated bars.
Run: python adapt/substrate_probe.py
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np
import copy

TRIALS = {
    "seed": {"frac_cap": 13312, "exp_span": 16, "dmax": 4096, "accum": "tree"},
    "exp8": {"frac_cap": 13312, "exp_span": 8, "dmax": 4096, "accum": "tree"},
    "dmax1024": {"frac_cap": 13312, "exp_span": 16, "dmax": 1024, "accum": "tree"},
    "frac2048": {"frac_cap": 2048, "exp_span": 16, "dmax": 4096, "accum": "tree"},
}


def eval_config(adapter, cfg):
    m = adapter.evaluate({"config": dict(cfg)})
    out = {}
    for role in ("exploration", "gate", "retention"):
        card = getattr(m, role)
        cs = [float(c.signature.split('=')[1]) for c in card.cases]
        ok = [c.correct for c in card.cases]
        out[role] = {"mean": float(np.mean(cs)), "min": float(min(cs)),
                     "pass": f"{sum(ok)}/{len(ok)}"}
    return out


def main():
    import torch
    import graduate as W
    from graduate import WidthAdapter
    from run_search import load_fixtures

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared_w, fx_w, probe = W.load_all(device)
    fx = {r: fx_w[r] for r in ('exploration', 'gate', 'retention')}
    adapter = WidthAdapter(shared_w, fx, probe)

    print("== exact substrate ==", flush=True)
    exact = {t: eval_config(adapter, c) for t, c in TRIALS.items()}
    for t, roles in exact.items():
        print(f"  {t}: " + " ".join(
            f"{r}={v['pass']}/{v['mean']:.5f}" for r, v in roles.items()),
            flush=True)

    # mush: per-layer C=2048 codebook patched into backbone float buffers
    import codebook_probe as CB
    from geo_backbone import GeometricDinov2Backbone  # noqa (backbone class ref)
    bb = shared_w['backbone']
    orig = {k: v.clone() for k, v in bb._w.items()}
    zb = np.load(REPO / 'weights' / 'geometric_backbone.npz', allow_pickle=False)
    PHI = float((1 + 5 ** 0.5) / 2)
    books, keys, _ = CB.build_codebook(zb, 2048, 'per-layer', None)
    for base in keys:
        cent = books[base]
        midpoints = (cent[:-1] + cent[1:]) / 2
        s = zb[base + '.signs']
        e = zb[base + '.exps'].astype(np.int64)
        mu = ((e - CB.BIAS) / CB.K_PHI).ravel().astype(np.float64)
        qmu = cent[np.searchsorted(midpoints, mu)]
        qval = s.ravel().astype(np.float64) * (PHI ** qmu)
        if base in bb._w:
            bb._w[base].copy_(torch.from_numpy(
                qval.reshape(s.shape).astype(np.float32)).to(device))
    try:
        print("== mush substrate (per-layer C=2048 backbone) ==", flush=True)
        mush = {t: eval_config(adapter, c) for t, c in TRIALS.items()}
        for t, roles in mush.items():
            print(f"  {t}: " + " ".join(
                f"{r}={v['pass']}/{v['mean']:.5f}" for r, v in roles.items()),
                flush=True)
    finally:
        for k, v in orig.items():
            bb._w[k].copy_(v)

    print("== verdict comparison ==", flush=True)
    for t in TRIALS:
        e = exact[t]
        m = mush[t]
        print(f"  {t}: exact(" + ",".join(v['pass'] for v in e.values()) +
              f") mush(" + ",".join(v['pass'] for v in m.values()) + ")",
              flush=True)


if __name__ == '__main__':
    main()
