#!/usr/bin/env python3
"""Post-hoc verify: first-N greedy picks per pool (RRR-128, fixed probe).

Pools: real (coreset.json) / synth + hf (parsed from killed-run logs).
Compares greedy-N vs random-N x3 (mean + worst tokcorr). Then phase 2
is depth-gating RRR students fit on these sets (phase2_gate.py).
Run: python adapt/verify_sets.py [--n 7]
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np
import re
import json

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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=7)
    args = ap.parse_args()
    import torch
    import student_probe as SP
    from run_search import load_shared, load_fixtures
    device = torch.device('cuda')
    shared = load_shared(device)
    bb, pre = shared['backbone'], shared['preprocess']

    pools = {
        'real': (ADAPT / 'runs' / 'coreset_covs.npz',
                 json.load(open(ADAPT / 'runs' / 'coreset.json'))['chosen']),
        'synth': (ADAPT / 'runs' / 'coreset_covs_synth.npz', None),
        'hf': (ADAPT / 'runs' / 'coreset_covs_hf.npz', None),
    }
    logmap = {'synth': '/tmp/opencode/coresynth.log',
              'hf': '/tmp/opencode/corehf.log'}
    pat = {'synth': re.compile(r'^(synth-\d+)$'),
           'hf': re.compile(r'^(hf-\d+)$')}
    out = {}
    for pname, (covpath, chosen) in pools.items():
        z = np.load(covpath, allow_pickle=False)
        ids = [str(x) for x in z['ids']]
        if chosen is None:
            seen, ordered = set(), []
            for line in open(logmap[pname], errors='replace'):
                m = re.search(r'scene\s+(\S+)', line)
                if m:
                    cid = m.group(1)
                    mm = pat[pname].match(cid)
                    if mm and cid in ids and cid not in seen:
                        seen.add(cid)
                        ordered.append(cid)
            chosen = ordered
        sel = chosen[:args.n]
        idx = [ids.index(c) for c in sel]
        Sxx = {li: {nm: z[f'{li}_{nm}_Sxx'][idx].sum(axis=0).astype(np.float64)
                    for nm, _, _, _, _ in SP.MAPS} for li in LAYERS}
        Sxy = {li: {nm: z[f'{li}_{nm}_Sxy'][idx].sum(axis=0).astype(np.float64)
                    for nm, _, _, _, _ in SP.MAPS} for li in LAYERS}
        out[pname] = {"ids": ids, "chosen": sel, "Sxx": Sxx, "Sxy": Sxy,
                      "covpath": str(covpath)}
        print(f"{pname}: greedy-{len(sel)} {sel}", flush=True)

    # fixed probe tokens (same 7 scenes as greedy runs)
    from coreset import PROBE_RGB  # noqa (fixed probe set)
    fx = load_fixtures(include_audit=False)
    probe_rgb = list(PROBE_RGB) + [fx['exploration'][2][0],
                                   fx['exploration'][4][0]]

    def blk_of(rgb):
        with torch.no_grad():
            cap = {}
            bb.forward_stages(pre(rgb).to(device), capture=cap)
            return {li: {k: v.squeeze(0).double().cpu().numpy()
                         for k, v in cap[(li, 'blk')].items()}
                    for li in LAYERS}

    P = {li: [] for li in LAYERS}
    for rgb in probe_rgb:
        for li, B in blk_of(rgb).items():
            P[li].append(B)

    def evaluate(Sxx, Sxy):
        cells = []
        for li in LAYERS:
            for nm, ik, tk, di, do in SP.MAPS:
                W = rrr(Sxx[li][nm], Sxy[li][nm], RANK)
                for B in P[li]:
                    X = np.concatenate(
                        [B[ik], np.ones((B[ik].shape[0], 1))], axis=1)
                    pr, tg = (X @ W).flatten(), B[tk].flatten()
                    c = float(np.corrcoef(pr, tg)[0, 1])
                    cells.append(c if np.isfinite(c) else -1.0)
        return float(np.mean(cells)), float(min(cells))

    rng = np.random.default_rng(0)
    rep = {"n": args.n}
    for pname, D in out.items():
        g, gw = evaluate(D["Sxx"], D["Sxy"])
        npool = len(D["ids"])
        rs, rsw = [], []
        for t in range(3):
            ri = sorted(rng.choice(npool, len(D["chosen"]),
                                   replace=False).tolist())
            z = np.load(D["covpath"], allow_pickle=False)
            qxx = {li: {nm: z[f'{li}_{nm}_Sxx'][ri].sum(axis=0)
                        .astype(np.float64)
                        for nm, _, _, _, _ in SP.MAPS} for li in LAYERS}
            qxy = {li: {nm: z[f'{li}_{nm}_Sxy'][ri].sum(axis=0)
                        .astype(np.float64)
                        for nm, _, _, _, _ in SP.MAPS} for li in LAYERS}
            del z
            r, rw = evaluate(qxx, qxy)
            rs.append(r)
            rsw.append(rw)
        rep[pname] = {"greedy": [g, gw], "random": [rs, rsw],
                      "chosen": D["chosen"]}
        print(f"{pname}: greedy mean={g:.5f} worst={gw:.5f} | random mean=" +
              " ".join(f"{r:.5f}" for r in rs) + " worst=" +
              " ".join(f"{r:.5f}" for r in rsw), flush=True)
    (ADAPT / 'runs' / 'verify_sets.json').write_text(json.dumps(rep, indent=1))
    print("wrote adapt/runs/verify_sets.json", flush=True)


if __name__ == '__main__':
    main()
