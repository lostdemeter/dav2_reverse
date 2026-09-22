#!/usr/bin/env python3
"""Blockfit pilot: RRR per linear map + exact nonlinearity forms.

Streamed covariances (10 fit reals) -> RRR {192,96,48} per map
(q/k/v/proj/mlp1/mlp2) -> ALL narrowed maps composed SIMULTANEOUSLY
with exact softmax/GELU/LN forms (+teacher ls/norms) -> block-out and
layer-out corr on 5 held-out reals + 3 synthetics. Activation-rank
diagnostic first. Pre-registered 2026-09-22.
Run: python adapt/blockfit_probe.py
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np

LAYERS = (0, 5, 11)
RANKS = (192, 96, 48)
N_FIT = 10
LAMBDA = 1e-6
# (map name, input key, target key, in_dim, out_dim)
MAPS = (("q", "H", "Q", 384, 384), ("k", "H", "K", 384, 384),
        ("v", "H", "V", 384, 384), ("proj", "C", "Oattn", 384, 384),
        ("mlp1", "N2", "P1", 384, 1536), ("mlp2", "G", "Y", 1536, 384))


def corr(a, b):
    a = np.asanyarray(a, dtype=np.float64).flatten()
    b = np.asanyarray(b, dtype=np.float64).flatten()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return -1.0
    c = float(np.corrcoef(a, b)[0, 1])
    return c if np.isfinite(c) else -1.0


def rrr_from_covs(Sxx, Sxy, r):
    """Reduced-rank regression W (d+1 x p) from covariances."""
    d1 = Sxx.shape[0]
    lam = LAMBDA * float(np.trace(Sxx)) / d1
    W_ols = np.linalg.solve(Sxx + lam * np.eye(d1), Sxy)
    M = W_ols.T @ Sxx @ W_ols
    vals, vecs = np.linalg.eigh(M)
    Vr = vecs[:, -r:] if r < M.shape[0] else vecs
    return W_ols @ Vr @ Vr.T


def eff_rank(Syy, qs=(0.99, 0.999)):
    tot = float(np.trace(Syy))
    vals = np.linalg.eigvalsh(Syy)[::-1]
    cum = np.cumsum(vals) / tot
    return {q: int(np.searchsorted(cum, q)) + 1 for q in qs}


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

    # stream covariances per target layer
    covs = {li: {} for li in LAYERS}
    with torch.no_grad():
        for rgb in fit_rgb:
            cap = {}
            bb.forward_stages(pre(rgb).to(device), capture=cap)
            for li in LAYERS:
                B = cap[(li, 'blk')]
                T = {k: B[k].squeeze(0).double().cpu().numpy() for k in
                     ('H', 'Q', 'K', 'V', 'C', 'Oattn', 'N2', 'P1', 'G', 'Y')}
                for name, ik, tk, di, do in MAPS:
                    X = np.concatenate(
                        [T[ik], np.ones((T[ik].shape[0], 1))], axis=1)
                    Y = T[tk]
                    S = covs[li].setdefault(name, {
                        'Sxx': np.zeros((di + 1, di + 1)),
                        'Sxy': np.zeros((di + 1, do)),
                        'Syy': np.zeros((do, do)), 'n': 0})
                    S['Sxx'] += X.T @ X
                    S['Sxy'] += X.T @ Y
                    S['Syy'] += Y.T @ Y
                    S['n'] += X.shape[0]
    print("covariances streamed.", flush=True)
    for li in LAYERS:
        for name, _, _, _, _ in MAPS:
            S = covs[li][name]
            r = eff_rank(S['Syy'])
            print(f"L{li} {name}: effrank99={r[0.99]}/{S['Syy'].shape[0]} "
                  f"99.9={r[0.999]} n={S['n']}", flush=True)

    # RRR solves per map/rank (+OLS reference)
    Ws = {li: {} for li in LAYERS}
    for li in LAYERS:
        for name, _, _, di, do in MAPS:
            S = covs[li][name]
            full = min(di + 1, do)
            Ws[li][name] = {r: rrr_from_covs(S['Sxx'], S['Sxy'], min(r, full))
                            for r in RANKS + (full,)}

    # composed eval (exact LN/softmax/GELU forms, teacher ls/norms)
    import torch.nn.functional as Fn
    W = {k: bb._w[k].double().cpu().numpy() for k in bb._w}
    eval_scenes = ([(rgb, f"test-{i}") for i, rgb in enumerate(test_rgb)] +
                   [(rgb, f"syn-{i}") for i, rgb in enumerate(syn_rgb)])
    for li in LAYERS:
        ls1, ls2 = W[f'layer{li}.ls1'], W[f'layer{li}.ls2']
        n1w, n1b = W[f'layer{li}.norm1.weight'], W[f'layer{li}.norm1.bias']
        n2w, n2b = W[f'layer{li}.norm2.weight'], W[f'layer{li}.norm2.bias']
        for r in RANKS + ('ols',):
            acc = {'attn': [], 'mlp': [], 'layer': []}
            with torch.no_grad():
                for rgb, cid in eval_scenes:
                    cap = {}
                    bb.forward_stages(pre(rgb).to(device), capture=cap)
                    B = {k: v.squeeze(0).double().cpu().numpy()
                         for k, v in cap[(li, 'blk')].items()}
                    Xin = cap[li][0].squeeze(0).double().cpu().numpy()
                    Xout = cap[li][1].squeeze(0).double().cpu().numpy()
                    Wm = {n: Ws[li][n][r if r != 'ols' else
                                         max(Ws[li][n].keys())]
                          for n in ('q', 'k', 'v', 'proj', 'mlp1', 'mlp2')}

                    def lin(X, W_):
                        return np.concatenate(
                            [X, np.ones((X.shape[0], 1))], axis=1) @ W_

                    def ln(X, w, b):
                        mu = X.mean(axis=1, keepdims=True)
                        va = ((X - mu) ** 2).mean(axis=1, keepdims=True)
                        return (X - mu) / np.sqrt(va + 1e-6) * w + b

                    def gelu(X):
                        import torch as _t
                        return _t.nn.functional.gelu(
                            _t.from_numpy(X)).numpy()

                    # --- composed attention from layer input ---
                    H = ln(Xin, n1w, n1b)
                    Q, K, V = lin(H, Wm['q']), lin(H, Wm['k']), lin(H, Wm['v'])
                    outs = []
                    for h in range(6):
                        q = Q[:, h * 64:(h + 1) * 64]
                        k = K[:, h * 64:(h + 1) * 64]
                        v = V[:, h * 64:(h + 1) * 64]
                        sc = q @ k.T / 8.0
                        sc = np.exp(sc - sc.max(axis=1, keepdims=True))
                        outs.append(sc / sc.sum(axis=1, keepdims=True) @ v)
                    Oa = lin(np.concatenate(outs, axis=1), Wm['proj'])
                    acc['attn'].append(corr(Oa, B['Oattn']))
                    # --- MLP isolated (teacher M/N2 in) for attribution ---
                    Yiso = lin(gelu(lin(B['N2'], Wm['mlp1'])), Wm['mlp2'])
                    acc['mlp'].append(corr(B['M'] + Yiso * ls2, Xout))
                    # --- full layer composed (student M' — compounding) ---
                    Mp = Xin + Oa * ls1
                    N2 = ln(Mp, n2w, n2b)
                    Yp = lin(gelu(lin(N2, Wm['mlp1'])), Wm['mlp2'])
                    acc['layer'].append(corr(Mp + Yp * ls2, Xout))
            line = " ".join(f"{k}={np.mean(v):.5f}" for k, v in acc.items())
            print(f"L{li} rank={r}: {line}", flush=True)


if __name__ == '__main__':
    main()
