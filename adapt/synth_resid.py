#!/usr/bin/env python3
"""Synth-residual probe: do RRR bases span synth activations?

RRR-128 maps fit on real-dominated covariances (train split); relative
reconstruction residual ||Y-XW||/||Y|| on held-out REAL vs SYNTHETIC
tokens, per map, layers 0-2. Pre-registered 2026-09-23: synth >> real.
Run: python adapt/synth_resid.py
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
    npool = len(zf['rgb'])
    hold = set(range(0, npool, 12))
    train_idx = [i for i in range(npool) if i not in hold][:200]
    real_test = [zf['rgb'][i].astype(np.float32) / 255.0
                 for i in sorted(hold)[:12]]
    fx = load_fixtures(include_audit=False)
    synth = ([rgb for rgb, _, _ in fx['exploration']] +
             [rgb for rgb, _, _ in fx['retention']])

    def collect(rgb_list):
        per = {li: [] for li in LAYERS}
        with torch.no_grad():
            for rgb in rgb_list:
                cap = {}
                bb.forward_stages(pre(rgb).to(device), capture=cap)
                for li in LAYERS:
                    per[li].append(
                        {k: v.squeeze(0).double().cpu().numpy()
                         for k, v in cap[(li, 'blk')].items()})
        return per

    print("collecting train...", flush=True)
    train = collect([zf['rgb'][i].astype(np.float32) / 255.0
                     for i in train_idx])
    print("collecting test...", flush=True)
    rtest = collect(real_test)
    stest = collect(synth)

    for li in LAYERS:
        # fit RRR on train
        acc = {}
        for img in train[li]:
            pass
        for blk in train[li]:
            for name, ik, tk, di, do in SP.MAPS:
                X = np.concatenate(
                    [blk[ik], np.ones((blk[ik].shape[0], 1))], axis=1)
                S = acc.setdefault(name, {
                    'Sxx': np.zeros((di + 1, di + 1)),
                    'Sxy': np.zeros((di + 1, do))})
                S['Sxx'] += X.T @ X
                S['Sxy'] += X.T @ blk[tk]
        W = {n: rrr(S['Sxx'], S['Sxy'], RANK) for n, S in acc.items()}
        for tag, D in (("real", rtest[li]), ("synth", stest[li])):
            line = []
            for name, ik, tk, di, do in SP.MAPS:
                num, den = 0.0, 0.0
                for blk in D:
                    X = np.concatenate(
                        [blk[ik], np.ones((blk[ik].shape[0], 1))], axis=1)
                    R = blk[tk] - X @ W[name]
                    num += float((R ** 2).sum())
                    den += float((blk[tk] ** 2).sum())
                line.append(f"{name}={np.sqrt(num / den):.4f}")
            print(f"L{li} {tag}: " + " ".join(line), flush=True)


if __name__ == '__main__':
    main()
