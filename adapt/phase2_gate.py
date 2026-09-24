#!/usr/bin/env python3
"""Phase 2: depth-gate RRR students fit on greedy sets (real/synth/hf).

RRR-128 maps fit on verify_sets greedy-N covariances -> dense install
-> full pipeline vs HF-oracle refs, 12 scenes (6 fixture + 6 real).
Pre-registered: real-N >= synth/hf; hf >= synth; small-N, below bar.
Run: python adapt/phase2_gate.py
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
    bb = shared['backbone']

    rep = json.load(open(ADAPT / 'runs' / 'verify_sets.json'))
    n = rep["n"]
    covpath = {
        'real': ADAPT / 'runs' / 'coreset_covs.npz',
        'synth': ADAPT / 'runs' / 'coreset_covs_synth.npz',
        'hf': ADAPT / 'runs' / 'coreset_covs_hf.npz',
    }
    fx = load_fixtures(include_audit=False)
    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    scenes = ([(rgb, ref, cid) for role in ('exploration', 'gate')
               for rgb, ref, cid in fx[role][:3]] +
              [(zr['rgb'][i].astype(np.float32) / 255.0,
                zr['ref'][i].astype(np.float64), f"eval-{i}")
               for i in range(48, 54)])
    print(f"{len(scenes)} eval scenes", flush=True)
    orig = {k: v.clone() for k, v in bb._w.items()}
    results = {}
    for pname in ('real', 'synth', 'hf'):
        z = np.load(covpath[pname], allow_pickle=False)
        ids = [str(x) for x in z['ids']]
        sel = rep[pname]['chosen'][:n]
        idx = [ids.index(c) for c in sel]
        for li in LAYERS:
            for nm, ik, tk, di, do in SP.MAPS:
                W = rrr(z[f'{li}_{nm}_Sxx'][idx].sum(axis=0).astype(np.float64),
                        z[f'{li}_{nm}_Sxy'][idx].sum(axis=0).astype(np.float64),
                        RANK)
                bb._w[f'layer{li}.{BUF[nm]}'].copy_(torch.from_numpy(
                    W[:-1].astype(np.float32).T.reshape(
                        orig[f'layer{li}.{BUF[nm]}'].shape)).to(device))
                bkey = f'layer{li}.{BUF[nm].replace(".weight", ".bias")}'
                if bkey in bb._w:
                    bb._w[bkey].copy_(torch.from_numpy(
                        W[-1].astype(np.float32)).to(device))
        del z
        cs = []
        for rgb, ref, cid in scenes:
            cfg = copy.deepcopy(dict(SEED_CONFIG))
            pred = run_pipeline(shared, cfg, [rgb])[0]
            cs.append(_corr(pred, ref))
        results[pname] = [round(float(c), 5) for c in cs]
        print(f"[{pname}-{n}] mean={np.mean(cs):.5f} min={min(cs):.5f} "
              f"pass={sum(c >= 0.999 for c in cs)}/{len(cs)}", flush=True)
        for k, v in orig.items():
            bb._w[k].copy_(v)
    (ADAPT / 'runs' / 'phase2.json').write_text(json.dumps(results, indent=1))
    print("wrote adapt/runs/phase2.json", flush=True)


if __name__ == '__main__':
    main()
