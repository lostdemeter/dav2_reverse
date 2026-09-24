#!/usr/bin/env python3
"""Poison autopsy: which synth scenes wreck fits, and what are they?

1. Reproduce synth-N60d0 indices (rng(0) order: pools real/synth/hf
   x N={7,15,30,60} x 2 draws).
2. Stats screen: profile + activation norms vs d1/pool.
3. Drop-one-out tokcorr screening (CPU) to rank suspects.
4. Confirm: refit minus suspects + depth-gate vs with-poison.
Run: python adapt/autopsy.py
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np
import json
import copy

LAYERS = (0, 1, 2)
RANK = 128
LAMBDA = 1e-6
NS = (7, 15, 30, 60)
DRAWS = 2
BUF = {"q": "q.weight", "k": "k.weight", "v": "v.weight",
       "proj": "proj.weight", "mlp1": "mlp1.weight", "mlp2": "mlp2.weight"}


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
    from depth_adapter import run_pipeline, corr as _corr, SEED_CONFIG
    device = torch.device('cuda')
    shared = load_shared(device)
    bb, pre = shared['backbone'], shared['preprocess']

    # 1. reproduce draw order
    rng = np.random.default_rng(0)
    draws = {}
    for pname, npool in (('real', 150), ('synth', 200), ('hf', 200)):
        draws[pname] = {}
        for N in NS:
            for d in range(DRAWS):
                draws[pname][(N, d)] = sorted(
                    rng.choice(npool, N, replace=False).tolist())
    d0 = draws['synth'][(60, 0)]
    d1 = draws['synth'][(60, 1)]
    print(f"d0[:8]={d0[:8]} d1[:8]={d1[:8]} overlap={len(set(d0) & set(d1))}",
          flush=True)

    z = np.load(ADAPT / 'runs' / 'coreset_covs_synth.npz', allow_pickle=False)
    ids = [str(x) for x in z['ids']]
    zp = np.load(ADAPT / 'fixtures' / 'synth_pool.npz', allow_pickle=True)

    def blk_of(rgb):
        with torch.no_grad():
            cap = {}
            bb.forward_stages(pre(rgb).to(device), capture=cap)
            return {li: {k: v.squeeze(0).double().cpu().numpy()
                         for k, v in cap[(li, 'blk')].items()}
                    for li in LAYERS}

    # fixed probe (same 7 scenes as greedy verify)
    from coreset import PROBE_RGB
    fx = load_fixtures(include_audit=False)
    probe_rgb = (list(PROBE_RGB) + [fx['exploration'][2][0],
                                    fx['exploration'][4][0]])
    P = {li: [] for li in LAYERS}
    for rgb in probe_rgb:
        for li, B in blk_of(rgb).items():
            P[li].append(B)

    def tokcorr(idxs, solverank=RANK):
        cells = []
        for li in LAYERS:
            for nm, ik, tk, di, do in SP.MAPS:
                W = rrr(z[f'{li}_{nm}_Sxx'][idxs].sum(axis=0).astype(np.float64),
                        z[f'{li}_{nm}_Sxy'][idxs].sum(axis=0).astype(np.float64),
                        solverank)
                for B in P[li]:
                    X = np.concatenate(
                        [B[ik], np.ones((B[ik].shape[0], 1))], axis=1)
                    pr, tg = (X @ W).flatten(), B[tk].flatten()
                    c = float(np.corrcoef(pr, tg)[0, 1])
                    cells.append(c if np.isfinite(c) else -1.0)
        return float(np.mean(cells)), float(min(cells))

    # 2. stats screen
    from strata import profile, assign
    medians = json.load(open(ADAPT / 'runs' / 'strata.json'))['medians']
    print("d0 stats (id, stratum, edge, texture, lumspread):", flush=True)
    for i in d0:
        rgb = zp['rgb'][i].astype(np.float32) / 255.0
        p = profile(rgb)
        print(f"  {ids[i]} {assign(p, medians)} "
              f"e={p['edge']:.3f} t={p['texture']:.3f} l={p['lumspread']:.3f}",
              flush=True)

    # 3. drop-one-out screening
    base_mean, base_min = tokcorr(d0)
    print(f"d0 full: mean={base_mean:.5f} worst={base_min:.5f}", flush=True)
    gains = []
    for j, i in enumerate(d0):
        rest = [x for x in d0 if x != i]
        m, w = tokcorr(rest)
        gains.append((m - base_mean, w - base_min, ids[i]))
    gains.sort(reverse=True)
    print("top-8 suspects by mean-gain on removal:", flush=True)
    for g, gw, cid in gains[:8]:
        print(f"  {cid} dmean={g:+.5f} dmin={gw:+.5f}", flush=True)
    (ADAPT / 'runs' / 'autopsy.json').write_text(json.dumps(
        {"d0": [ids[i] for i in d0], "d1": [ids[i] for i in d1],
         "base": [base_mean, base_min],
         "gains": [(c, round(g, 6), round(gw, 6)) for g, gw, c in gains]},
        indent=1))

    # 4. confirm: refit minus top-3 suspects + depth-gate
    from run_search import load_fixtures as _lf
    fx2 = _lf(include_audit=False)
    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    scenes = ([(rgb, ref, cid) for role in ('exploration', 'gate')
               for rgb, ref, cid in fx2[role][:3]] +
              [(zr['rgb'][i].astype(np.float32) / 255.0,
                zr['ref'][i].astype(np.float64), f"eval-{i}")
               for i in range(48, 54)])
    orig = {k: v.clone() for k, v in bb._w.items()}

    def install(idxs):
        for li in LAYERS:
            for nm, ik, tk, di, do in SP.MAPS:
                W = rrr(z[f'{li}_{nm}_Sxx'][idxs].sum(axis=0).astype(np.float64),
                        z[f'{li}_{nm}_Sxy'][idxs].sum(axis=0).astype(np.float64),
                        RANK)
                bb._w[f'layer{li}.{BUF[nm]}'].copy_(torch.from_numpy(
                    W[:-1].astype(np.float32).T.reshape(
                        orig[f'layer{li}.{BUF[nm]}'].shape)).to(device))
                bk = f'layer{li}.{BUF[nm].replace(".weight", ".bias")}'
                if bk in bb._w:
                    bb._w[bk].copy_(torch.from_numpy(
                        W[-1].astype(np.float32)).to(device))

    def gate(tag, idxs):
        install(idxs)
        cs = []
        for rgb, ref, cid in scenes:
            cfg = copy.deepcopy(dict(SEED_CONFIG))
            pred = run_pipeline(shared, cfg, [rgb])[0]
            cs.append(_corr(pred, ref))
        print(f"[{tag}] mean={np.mean(cs):.5f} min={min(cs):.5f} "
              f"pass={sum(c >= 0.999 for c in cs)}/{len(cs)}", flush=True)
        for k, v in orig.items():
            bb._w[k].copy_(v)
        return float(np.mean(cs))

    suspects = [ids.index(c) for c, _, _ in gains[:3]]
    gate("d0-with-poison", d0)
    gate("d0-minus-top3", [x for x in d0 if x not in suspects])
    gate("d1-clean", d1)


if __name__ == '__main__':
    main()
