#!/usr/bin/env python3
"""Covariance convergence: how many scenes identify the teacher's maps?

Stream RRR covariances over fit_pool (fixed order); snapshot W_N at
N={5,10,20,40,80,160,320}; report per-map relative drift vs final +
held-out token-output corr (2 reals + 1 synth). Pre-registered
2026-09-23: elbow at 20-40 scenes, mlp2 slowest.
Run: python adapt/covconverge.py
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np

LAYERS = (0, 1, 2)
RANK = 128
LAMBDA = 1e-6
SNAPS = (5, 10, 20, 40, 80, 160, 320)


def rrr(Sxx, Sxy, r):
    d1 = Sxx.shape[0]
    lam = LAMBDA * float(np.trace(Sxx)) / d1
    W_ols = np.linalg.solve(Sxx + lam * np.eye(d1), Sxy)
    M = W_ols.T @ Sxx @ W_ols
    vals, vecs = np.linalg.eigh(M)
    Vr = vecs[:, -r:] if r < M.shape[0] else vecs
    return W_ols @ Vr @ Vr.T


def main():
    import torch
    import student_probe as SP
    from run_search import load_shared, load_fixtures
    device = torch.device('cuda')
    shared = load_shared(device)
    bb, pre = shared['backbone'], shared['preprocess']

    zf = np.load(ADAPT / 'fixtures' / 'fit_pool.npz', allow_pickle=True)
    order = (np.random.default_rng(0).permutation(len(zf['rgb'])))[:max(SNAPS)]
    fx = load_fixtures(include_audit=False)
    probe = ([zf['rgb'][i].astype(np.float32) / 255.0
              for i in sorted(set(range(0, len(zf['rgb']), 12)))[:2]] +
             [fx['exploration'][2][0]])

    def blk_of(rgb):
        with torch.no_grad():
            cap = {}
            bb.forward_stages(pre(rgb).to(device), capture=cap)
            return {li: {k: v.squeeze(0).double().cpu().numpy()
                         for k, v in cap[(li, 'blk')].items()}
                    for li in LAYERS}

    print("streaming covariances...", flush=True)
    acc = {li: {} for li in LAYERS}
    snaps = {}
    for n, i in enumerate(order, 1):
        rgb = zf['rgb'][i].astype(np.float32) / 255.0
        for li, B in blk_of(rgb).items():
            for name, ik, tk, di, do in SP.MAPS:
                X = np.concatenate(
                    [B[ik], np.ones((B[ik].shape[0], 1))], axis=1)
                S = acc[li].setdefault(name, {
                    'Sxx': np.zeros((di + 1, di + 1)),
                    'Sxy': np.zeros((di + 1, do))})
                S['Sxx'] += X.T @ X
                S['Sxy'] += X.T @ B[tk]
        if n in SNAPS:
            snaps[n] = {li: {nm: (S['Sxx'].copy(), S['Sxy'].copy())
                             for nm, S in acc[li].items()} for li in LAYERS}
            print(f"  snapshot N={n}", flush=True)
    Nmax = max(SNAPS)
    Wfin = {li: {nm: rrr(S[0], S[1], RANK) for nm, S in snaps[Nmax][li].items()}
            for li in LAYERS}

    print("probe tokens...", flush=True)
    P = {li: [] for li in LAYERS}
    for rgb in probe:
        for li, B in blk_of(rgb).items():
            P[li].append(B)
    for li in LAYERS:
        for name, ik, tk, di, do in SP.MAPS:
            drifts, corrs = [], []
            for n in SNAPS:
                Sxx, Sxy = snaps[n][li][name]
                W = rrr(Sxx, Sxy, RANK)
                drifts.append(float(np.linalg.norm(W - Wfin[li][name]) /
                                    np.linalg.norm(Wfin[li][name])))
                cs = []
                for B in P[li]:
                    X = np.concatenate(
                        [B[ik], np.ones((B[ik].shape[0], 1))], axis=1)
                    pr, tg = (X @ W).flatten(), B[tk].flatten()
                    c = float(np.corrcoef(pr, tg)[0, 1])
                    cs.append(c if np.isfinite(c) else -1.0)
                corrs.append(float(np.mean(cs)))
            print(f"L{li} {name}: drift=" +
                  " ".join(f"{n}:{d:.4f}" for n, d in zip(SNAPS, drifts)),
                  flush=True)
            print(f"         tokcorr=" +
                  " ".join(f"{n}:{c:.5f}" for n, c in zip(SNAPS, corrs)),
                  flush=True)


if __name__ == '__main__':
    main()
