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
_fp = Path(__file__).parent / 'fixtures' / 'fit_pool.npz'
_zfp = np.load(_fp, allow_pickle=True)
PROBE_RGB = ([_zfp['rgb'][len(_zfp['rgb']) - 1 - i].astype(np.float32) / 255.0
              for i in range(5)])
del _zfp


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
    ap.add_argument('--pool-file', default='adapt/fixtures/fit_pool.npz',
                    help='candidate pool npz (rgb uint8 + ids); probe stays '
                         'fixed (fit_pool tail + fixture synth) for '
                         'comparability across pools')
    ap.add_argument('--cov-suffix', default='',
                    help='cov-cache filename suffix per pool (default: none)')
    args = ap.parse_args()
    import torch
    import student_probe as SP
    from run_search import load_shared, load_fixtures
    device = torch.device('cuda')
    shared = load_shared(device)
    bb, pre = shared['backbone'], shared['preprocess']

    zf = np.load(REPO / args.pool_file if not
                 Path(args.pool_file).is_absolute() else args.pool_file,
                 allow_pickle=True)
    pool_idx = list(range(min(args.pool, len(zf['rgb']))))
    pool_ids = [str(zf['ids'][i]) for i in pool_idx]
    print(f"pool: {args.pool_file} ({len(pool_idx)} scenes)", flush=True)
    cov_cache = (ADAPT / 'runs' /
                 f'coreset_covs{args.cov_suffix or ""}.npz')
    COV = cov_cache

    def blk_of(rgb):
        with torch.no_grad():
            cap = {}
            bb.forward_stages(pre(rgb).to(device), capture=cap)
            return {li: {k: v.squeeze(0).double().cpu().numpy()
                         for k, v in cap[(li, 'blk')].items()}
                    for li in LAYERS}

    if COV.exists():
        z = np.load(COV, allow_pickle=False)
        if int(np.asarray(z['n']).ravel()[0]) >= len(pool_idx) and z['ids'].tolist() == pool_ids:
            print("cov cache hit", flush=True)
            covs = {li: {nm: (z[f'{li}_{nm}_Sxx'], z[f'{li}_{nm}_Sxy'])
                         for nm, _, _, _, _ in SP.MAPS} for li in LAYERS}
            per = None
        else:
            per = True
    else:
        per = True
    if per is not None:
        print("caching per-scene covariances (float32)...", flush=True)
        per32 = []
        for n, i in enumerate(pool_idx):
            rgb = zf['rgb'][i].astype(np.float32) / 255.0
            B = blk_of(rgb)
            rec = {}
            for li in LAYERS:
                for nm, ik, tk, di, do in SP.MAPS:
                    X = np.concatenate(
                        [B[li][ik], np.ones((B[li][ik].shape[0], 1))], axis=1)
                    # float32 storage (selection precision; exact
                    # float64 re-fit for the chosen set at verify)
                    rec[f'{li}_{nm}'] = (
                        (X.T @ X).astype(np.float32),
                        (X.T @ B[li][tk]).astype(np.float32))
            per32.append(rec)
            del B
            if (n + 1) % 25 == 0:
                print(f"  {n + 1}/{len(pool_idx)}", flush=True)
        out = {'n': np.array([len(pool_idx)]),
               'ids': np.array(pool_ids)}
        keys = list(per32[0].keys())
        for k in keys:
            out[k + '_Sxx'] = np.stack([r[k][0] for r in per32])
            out[k + '_Sxy'] = np.stack([r[k][1] for r in per32])
        del per32
        np.savez_compressed(COV, **out)
        print(f"cached -> {COV}", flush=True)
        import gc as _gc
        _gc.collect()
    z = np.load(COV, allow_pickle=False)
    Sxx = {li: {nm: z[f'{li}_{nm}_Sxx'].astype(np.float64)
                for nm, _, _, _, _ in SP.MAPS} for li in LAYERS}
    Sxy = {li: {nm: z[f'{li}_{nm}_Sxy'].astype(np.float64)
                for nm, _, _, _, _ in SP.MAPS} for li in LAYERS}
    del z
    import gc as _gc
    _gc.collect()

    # probe tokens: FIXED across pools (fit_pool tail + fixture synth)
    # so synth-pool numbers compare directly with real-pool numbers.
    fx = load_fixtures(include_audit=False)
    probe_rgb = list(PROBE_RGB) + [fx['exploration'][2][0],
                                   fx['exploration'][4][0]]
    print("probe tokens...", flush=True)
    P = {li: [] for li in LAYERS}
    for rgb in probe_rgb:
        for li, B in blk_of(rgb).items():
            P[li].append(B)

    def score(idxs, solver, worst=False):
        """Mean tokcorr (worst=True: min over map/scene cells — the
        core criterion; v1's mean saturated by step 3)."""
        cells = []
        for li in LAYERS:
            for nm, ik, tk, di, do in SP.MAPS:
                W = solver(Sxx[li][nm][idxs].sum(axis=0),
                           Sxy[li][nm][idxs].sum(axis=0))
                for B in P[li]:
                    X = np.concatenate(
                        [B[ik], np.ones((B[ik].shape[0], 1))], axis=1)
                    pr, tg = (X @ W).flatten(), B[tk].flatten()
                    c = float(np.corrcoef(pr, tg)[0, 1])
                    cells.append(c if np.isfinite(c) else -1.0)
        return min(cells) if worst else sum(cells) / len(cells)

    rng = np.random.default_rng(0)
    chosen, gains = [], []
    avail = list(range(len(pool_idx)))
    prev = None
    for step in range(args.steps):
        best, best_c = None, -2.0
        for c in avail:
            s = score(chosen + [c], ols, worst=True)
            if s > best_c:
                best, best_c = c, s
        gain = best_c - (prev if prev is not None else best_c)
        prev = best_c
        chosen.append(best)
        avail.remove(best)
        gains.append(gain)
        print(f"step {len(chosen)}: scene {pool_ids[best]} "
              f"worst={best_c:.5f} gain={gain:+.5f}", flush=True)
    print("selected:", [pool_ids[c] for c in chosen], flush=True)

    # RRR-128 verify + random controls (worst-case + mean, both)
    def score_rrr(idxs):
        return score(idxs, lambda A, B: rrr(A, B, RANK))

    def score_rrr_worst(idxs):
        return score(idxs, lambda A, B: rrr(A, B, RANK), worst=True)

    g, gw = score_rrr(chosen), score_rrr_worst(chosen)
    rs, rsw = [], []
    for t in range(3):
        r = sorted(rng.choice(len(pool_idx), len(chosen),
                              replace=False).tolist())
        rs.append(score_rrr(r))
        rsw.append(score_rrr_worst(r))
    print(f"greedy RRR mean={g:.5f} worst={gw:.5f} | random mean=" +
          " ".join(f"{r:.5f}" for r in rs) + " worst=" +
          " ".join(f"{r:.5f}" for r in rsw), flush=True)
    (ADAPT / 'runs' / f'coreset{args.cov_suffix or ""}.json').write_text(__import__('json').dumps(
        {"chosen": [pool_ids[c] for c in chosen], "gains": gains,
         "greedy_rrr": g, "greedy_worst": gw,
         "random_rrr": rs, "random_worst": rsw}, indent=1))
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
