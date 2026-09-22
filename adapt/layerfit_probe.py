#!/usr/bin/env python3
"""Linear-layer pilot: ridge 384->384 fit of teacher layer I/O on tokens.

Fit on 10 real images' tokens (13.7k pairs), test on 5 held-out reals +
3 synthetics, layers {0,5,11}. Closed-form only (numpy lstsq). Teacher
I/O captured inside forward_stages (same code path, no lookalike).
Pre-registered 2026-09-22: fails everywhere (<0.95); L0 worst.
Run: python adapt/layerfit_probe.py
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np

LAYERS = (0, 5, 11)
N_FIT = 10


def corr(a, b):
    a = np.asanyarray(a, dtype=np.float64).flatten()
    b = np.asanyarray(b, dtype=np.float64).flatten()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return -1.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else -1.0


def main():
    import torch
    from run_search import load_shared, load_fixtures
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared = load_shared(device)
    bb, pre = shared['backbone'], shared['preprocess']

    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    fit_rgb = [zr['rgb'][i].astype(np.float32) / 255.0 for i in range(N_FIT)]
    test_rgb = [zr['rgb'][i].astype(np.float32) / 255.0
                for i in range(N_FIT, N_FIT + 5)]
    fx = load_fixtures(include_audit=False)['exploration'][:3]
    syn_rgb = [rgb for rgb, _, _ in fx]

    # per-image capture (tokens incl. CLS col 0; kept — student sees same)
    def image_pairs(rgb_list):
        pairs = {li: ([], []) for li in LAYERS}
        with torch.no_grad():
            for rgb in rgb_list:
                cap = {}
                bb.forward_stages(pre(rgb).to(device), capture=cap)
                for li in LAYERS:
                    xin, xout = cap[li]
                    pairs[li][0].append(
                        xin.squeeze(0).cpu().numpy().astype(np.float64))
                    pairs[li][1].append(
                        xout.squeeze(0).cpu().numpy().astype(np.float64))
        return {li: (np.concatenate(v[0]), np.concatenate(v[1]))
                for li, v in pairs.items()}

    print("collecting fit tokens...", flush=True)
    fit = image_pairs(fit_rgb)
    print("collecting test tokens...", flush=True)
    test = image_pairs(test_rgb)
    syn = image_pairs(syn_rgb)

    for li in LAYERS:
        Xtr, Ytr = fit[li]
        # ridge via lstsq on augmented [X, 1] (closed-form, CPU)
        A = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
        sol, res, rank, sv = np.linalg.lstsq(A, Ytr, rcond=None)
        print(f"L{li}: train {Xtr.shape}, rank={rank}, "
              f"top-sv={sv[0]:.1f} sv381={sv[380] if len(sv) > 380 else float('nan'):.3f}",
              flush=True)
        for tag, D in (("heldout-real", test[li]), ("synth", syn[li])):
            Xte, Yte = D
            Ate = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)
            pred = Ate @ sol
            c = corr(pred, Yte)
            rel = (float(np.mean((pred - Yte) ** 2)) /
                   float(np.mean(Yte ** 2)))
            print(f"  L{li} {tag}: corr={c:.5f} relMSE={rel:.4f} "
                  f"{'PASS?' if c >= 0.999 else 'fail'}", flush=True)


if __name__ == '__main__':
    main()
