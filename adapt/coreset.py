#!/usr/bin/env python3
"""Greedy coreset: minimal real-scene subset identifying the maps.

Per-scene covariances cached (pool 150); greedy forward selection to
K=25 on mean held-out tokcorr (OLS-scored; held-out = overfit guard);
RRR-128 verify; random-25 x3 control; strata of selected reported.
Pre-registered 2026-09-23: greedy wins, elbow <=15, diversity emerges.
Run: python adapt/coreset.py [--pool N --steps K]
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
COV_CACHE = ADAPT / 'runs' / 'coreset_covs.npz'


def ols(Sxx, Sxy):
    d1 = Sxx.shape[0]
    lam = LAMBDA * float(np.trace(Sxx)) / d1
    return np.linalg.solve(Sxx + lam * np.eye(d1), Sxy)


def rrr(Sxx, Sxy, r):
    W_ols = ols(Sxx, Sxy)
    M = W_ols.T @ Sxx @ W_ols
    vals, vecs = np.linalg.eigh(M)
    Vr = vecs[:, -r:] if r < M.shape[0] else vecs
    return W_ols @ Vr @ Vr.T


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=150)
    ap.add_argument('--steps', type=int, default=25)
    args = ap.parse_args()
    import torch
    import student_probe as SP
    from run_search import load_shared, load_fixtures
    device = torch.device('cuda')
    shared = load_shared(device)
    bb, pre = shared['backbone'], shared['preprocess']

    zf = np.load(ADAPT / 'fixtures' / 'fit_pool.npz', allow_pickle=True)
    pool_idx = list(range(min(args.pool, len(zf['rgb']))))
    pool_ids = [str(zf['ids'][i]) for i in pool_idx]

    def blk_of(rgb):
        with torch.no_grad():
            cap = {}
            bb.forward_stages(pre(rgb).to(device), capture=cap)
            return {li: {k: v.squeeze(0).double().cpu().numpy()
                         for k, v in cap[(li, 'blk')].items()}
                    for li in LAYERS}

    if COV_CACHE.exists():
        z = np.load(COV_CACHE, allow_pickle=False)
        if int(z['n']) >= len(pool_idx) and z['ids'].tolist() == pool_ids:
            print("cov cache hit", flush=True)
            covs = {li: {nm: (z[f'{li}_{nm}_Sxx'], z[f'{li}_{nm}_Sxy'])
                         for nm, _, _, _, _ in SP.MAPS} for li in LAYERS}
            per = None
        else:
            per = True
    else:
        per = True
    if per is not None:
        print("caching per-scene covariances...", flush=True)
        per = []
        for n, i in enumerate(pool_idx):
            rgb = zf['rgb'][i].astype(np.float32) / 255.0
            B = blk_of(rgb)
            per.append({li: {nm: None for nm, _, _, _, _ in SP.MAPS}
                        for li in LAYERS})
            for li in LAYERS:
                for nm, ik, tk, di, do in SP.MAPS:
                    X = np.concatenate(
                        [B[li][ik], np.ones((B[li][ik].shape[0], 1))], axis=1)
                    per[-1][li][nm] = (X.T @ X, X.T @ B[li][tk])
            if (n + 1) % 25 == 0:
                print(f"  {n + 1}/{len(pool_idx)}", flush=True)
        out = {'n': np.array([len(pool_idx)]),
               'ids': np.array(pool_ids)}
        for li in LAYERS:
            for nm, _, _, _, _ in SP.MAPS:
                out[f'{li}_{nm}_Sxx'] = np.stack([per[i][li][nm][0]
                                                  for i in range(len(per))])
                out[f'{li}_{nm}_Sxy'] = np.stack([per[i][li][nm][1]
                                                  for i in range(len(per))])
        np.savez_compressed(COV_CACHE, **out)
        print(f"cached -> {COV_CACHE}", flush=True)
        covs = None
    if covs is None:
        z = np.load(COV_CACHE, allow_pickle=False)
        Sxx = {li: {nm: z[f'{li}_{nm}_Sxx'] for nm, _, _, _, _ in SP.MAPS}
               for li in LAYERS}
        Sxy = {li: {nm: z[f'{li}_{nm}_Sxy'] for nm, _, _, _, _ in SP.MAPS}
               for li in LAYERS}
    else:
        Sxx = {li: {nm: np.stack([covs[li][nm][0][i]
                                  for i in range(len(pool_idx))])
                    for nm, _, _, _, _ in SP.MAPS} for li in LAYERS}
        Sxy = {li: {nm: np.stack([covs[li][nm][1][i]
                                  for i in range(len(pool_idx))])
                    for nm, _, _, _, _ in SP.MAPS} for li in LAYERS}

    # probe tokens: 2 held-out reals (outside pool) + 1 synth
    npool = len(zf['rgb'])
    h1 = zf['rgb'][npool - 2].astype(np.float32) / 255.0
    h2 = zf['rgb'][npool - 1].astype(np.float32) / 255.0
    fx = load_fixtures(include_audit=False)
    probe_rgb = [h1, h2, fx['exploration'][2][0]]
    print("probe tokens...", flush=True)
    P = {li: [] for li in LAYERS}
    for rgb in probe_rgb:
        for li, B in blk_of(rgb).items():
            P[li].append(B)

    def score(idxs, solver):
        tot, cnt = 0.0, 0
        for li in LAYERS:
            for nm, ik, tk, di, do in SP.MAPS:
                W = solver(Sxx[li][nm][idxs].sum(axis=0),
                           Sxy[li][nm][idxs].sum(axis=0))
                for B in P[li]:
                    X = np.concatenate(
                        [B[ik], np.ones((B[ik].shape[0], 1))], axis=1)
                    pr, tg = (X @ W).flatten(), B[tk].flatten()
                    c = float(np.corrcoef(pr, tg)[0, 1])
                    tot += c if np.isfinite(c) else -1.0
                    cnt += 1
        return tot / cnt

    rng = np.random.default_rng(0)
    chosen, gains, cur = [], [], []
    avail = list(range(len(pool_idx)))
    for step in range(args.steps):
        best, best_c, best_g = None, -2.0, 0.0
        base = score(cur, ols) if cur else -1.0
        for c in avail:
            s = score(cur + [c], ols)
            if s > best_c:
                best, best_c = c, s
        best_g = best_c - base
        chosen.append(best)
        avail.remove(best)
        cur = list(chosen)
        gains.append(best_g)
        print(f"step {len(chosen)}: scene {pool_ids[best]} "
              f"score={best_c:.5f} gain={best_g:+.5f}", flush=True)
    print("selected:", [pool_ids[c] for c in chosen], flush=True)

    # RRR-128 verify + random controls
    def score_rrr(idxs):
        return score(idxs, lambda A, B: rrr(A, B, RANK))

    g = score_rrr(chosen)
    rs = []
    for t in range(3):
        r = sorted(rng.choice(len(pool_idx), len(chosen),
                              replace=False).tolist())
        rs.append(score_rrr(r))
    print(f"greedy RRR={g:.5f} random RRR=" +
          " ".join(f"{r:.5f}" for r in rs), flush=True)
    (ADAPT / 'runs' / 'coreset.json').write_text(__import__('json').dumps(
        {"chosen": [pool_ids[c] for c in chosen], "gains": gains,
         "greedy_rrr": g, "random_rrr": rs}, indent=1))
    # strata of selected
    import json as _j
    from strata import profile, assign
    medians = _j.load(open(ADAPT / 'runs' / 'strata.json'))['medians']
    sts = [assign(profile(zf['rgb'][pool_idx[c]].astype(np.float32) / 255.0),
                  medians) for c in chosen]
    print("strata:", " ".join(f"{i}:{s}" for i, s in enumerate(sts)),
          flush=True)


if __name__ == '__main__':
    main()
