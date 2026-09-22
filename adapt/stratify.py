#!/usr/bin/env python3
"""Weight-direct stratification of the backbone (no behavior, no oracle).

Step 1 (--report): characterize each of the 12 ViT layers from baked
weights ALONE (phi-level stats, isotropy, effective rank, layer-scales),
cluster into roles WITHOUT position information, print the role map.
Step 2 (--probe): run gain/drop probes on GATE scenes (fresh data the
roles never saw) and rank-compare against pre-registered predictions.

Deliberate omissions (stated, not hidden): attention span needs
activations, so it is NOT a feature — roles come from weights alone.
Position is excluded from clustering (contiguity afterward is evidence,
not input).
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np

N_LAYERS = 12
K_PHI, BIAS = 512, 32768


def load_layer_weights():
    """Baked phi (signs, exps) + decoded float per matrix per layer."""
    z = np.load(REPO / 'weights' / 'geometric_backbone.npz', allow_pickle=False)

    def dec(base):
        s = z[base + '.signs'].astype(np.float32)
        e = z[base + '.exps'].astype(np.float32)
        return s, e, (s * np.float32((1 + 5 ** 0.5) / 2) ** ((e - BIAS) / K_PHI))

    mats = {}
    for li in range(N_LAYERS):
        p = f'layer{li}.'
        for name in ('q.weight', 'k.weight', 'v.weight', 'proj.weight',
                     'mlp1.weight', 'mlp2.weight', 'norm1.weight', 'norm1.bias',
                     'norm2.weight', 'norm2.bias', 'ls1', 'ls2'):
            try:
                mats[(li, name)] = dec(p + name)
            except KeyError:
                pass
    return mats


def iso_rank(float_w):
    """Isotropy (1.0 = rotation-like) + rank ratio from SVD. 2D weights only."""
    if float_w.ndim != 2:
        return None, None
    svals = np.linalg.svd(float_w.astype(np.float64), compute_uv=False)
    fro = float(np.sqrt((svals ** 2).sum()))
    d = min(float_w.shape)
    iso = float(svals[0] / (fro / np.sqrt(d))) if fro > 0 else 0.0
    rank_ratio = float((fro ** 2 / svals[0] ** 2) / d) if svals[0] > 0 else 0.0
    return iso, rank_ratio


def characterize():
    """Per-layer feature dict. Returns {li: {feature: value}}."""
    mats = load_layer_weights()
    feats = {}
    for li in range(N_LAYERS):
        f = {}
        isos, ranks, levels = [], [], []
        for name in ('q.weight', 'k.weight', 'v.weight', 'proj.weight'):
            s, e, fw = mats[(li, name)]
            iso, rr = iso_rank(fw)
            isos.append(iso)
            ranks.append(rr)
            levels.extend([float(v) for v in np.ravel((e - BIAS) / K_PHI)])
        misos, mranks = [], []
        for name in ('mlp1.weight', 'mlp2.weight'):
            s, e, fw = mats[(li, name)]
            iso, rr = iso_rank(fw)
            misos.append(iso)
            mranks.append(rr)
            levels.extend([float(v) for v in np.ravel((e - BIAS) / K_PHI)])
        ls1 = mats[(li, 'ls1')][2].astype(np.float64)
        ls2 = mats[(li, 'ls2')][2].astype(np.float64)
        f['attn_iso'] = float(np.mean(isos))
        f['attn_rank'] = float(np.mean(ranks))
        f['mlp_iso'] = float(np.mean(misos))
        f['mlp_rank'] = float(np.mean(mranks))
        f['ls1'] = float(ls1.mean())
        f['ls2'] = float(ls2.mean())
        f['phi_mean'] = float(np.mean(levels))
        f['phi_std'] = float(np.std(levels))
        feats[li] = f
    return feats


FEAT_ORDER = ('attn_iso', 'attn_rank', 'mlp_iso', 'mlp_rank',
              'ls1', 'ls2', 'phi_mean', 'phi_std')


def cluster_roles(feats, k=3):
    """Deterministic agglomerative clustering (single linkage) on
    standardized features. Returns {li: role}. Position excluded."""
    X = np.array([[feats[li][f] for f in FEAT_ORDER] for li in range(N_LAYERS)])
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    Z = (X - mu) / sd
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=-1))
    clusters = [{i} for i in range(N_LAYERS)]
    while len(clusters) > k:
        best, bv = None, 1e18
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = min(D[i, j] for i in clusters[a] for j in clusters[b])
                if d < bv:
                    best, bv = (a, b), d
        a, b = best
        clusters[a] = clusters[a] | clusters[b]
        del clusters[b]
    roles = {}
    for ri, c in enumerate(sorted(clusters, key=min)):
        for li in c:
            roles[li] = ri
    return roles


def scaling_score(feats, li):
    """Pre-registered fragility proxy: scaling-ness + drive strength.

    Low isotropy (scaling, does work) + large layer-scales (drives the
    residual) = predicted fragile. Rotation-like routers predicted robust.
    Higher score = predicted MORE fragile. Fixed before any probe runs.
    """
    f = feats[li]
    return ((f['attn_iso'] + f['mlp_iso']) * -1.0
            + abs(f['ls1']) * 50.0 + abs(f['ls2']) * 50.0)


def report():
    feats = characterize()
    print("layer | " + " ".join(f"{f[:9]:>9}" for f in FEAT_ORDER))
    for li in range(N_LAYERS):
        print(f"L{li:2d}   | " + " ".join(f"{feats[li][f]:9.3f}" for f in FEAT_ORDER))
    roles = cluster_roles(feats)
    print("\nroles:", {li: roles[li] for li in range(N_LAYERS)})
    runs = 1 + sum(1 for li in range(1, N_LAYERS) if roles[li] != roles[li - 1])
    print(f"contiguity: {runs} runs over 12 layers "
          f"({'contiguous' if runs == len(set(roles.values())) else 'NON-contiguous'})")
    pred = sorted(range(N_LAYERS), key=lambda li: -scaling_score(feats, li))
    print("predicted fragility order (most->least):", pred)
    return feats, roles, pred


def probe():
    """Gain (+1) and drop probes per block on GATE scenes; rank-compare."""
    import torch
    from run_search import load_shared, load_fixtures
    from depth_adapter import run_pipeline, corr, SEED_CONFIG
    import copy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared = load_shared(device)
    fx = load_fixtures()
    gate = fx['gate']
    feats, roles, pred = report()
    print("\n--- probes on GATE scenes (roles fixed above) ---")
    rows = []
    for b in range(24):
        for mode in ('gain', 'drop'):
            cfg = copy.deepcopy(dict(SEED_CONFIG))
            if mode == 'gain':
                bg = [0] * 24
                bg[b] = 1
                cfg['block_gains'] = bg
            else:
                cfg['dropped'] = [b]
            cs = []
            for rgb, ref, cid in gate:
                d = run_pipeline(shared, cfg, [rgb])[0]
                cs.append(corr(d, ref))
            rows.append((np.mean(cs), min(cs), b, mode))
            print(f"block {b:2d} {mode:4s}: mean={np.mean(cs):.5f} "
                  f"min={min(cs):.5f}", flush=True)
    # Per-BLOCK predictions: expand layer orders (attn before mlp within
    # a layer — arbitrary but fixed before seeing results).
    t1_robust_first = []
    for li in reversed(pred):  # pred is fragile-first over layers
        t1_robust_first += [2 * li, 2 * li + 1]
    for mode in ('gain', 'drop'):
        actual = sorted([r for r in rows if r[3] == mode],
                        key=lambda r: (r[0], r[1]))
        actual_order = [r[2] for r in actual]  # robust-first
        # Spearman on block ranks
        from scipy.stats import spearmanr
        pr = [t1_robust_first.index(b) for b in actual_order]
        ar = list(range(24))
        rho, p = spearmanr(pr, ar)
        print(f"{mode}: Spearman(predicted-robust-first, actual-robust-first) "
              f"rho={rho:.3f} p={p:.3g}")
    # T2 rival (positional fan-out; origin: exploration data — stated in
    # NOTES): fragile-first is [block 0..23] i.e. robust-first is reversed.
    # Per-block (not per-layer): block index itself orders depth.
    for mode in ('gain', 'drop'):
        actual = sorted([r for r in rows if r[3] == mode],
                        key=lambda r: (r[0], r[1]))
        actual_order = [r[2] for r in actual]  # robust-first
        t2_robust_first = list(reversed(range(24)))
        from scipy.stats import spearmanr
        pr = [t2_robust_first.index(b) for b in actual_order]
        ar = list(range(24))
        rho, p = spearmanr(pr, ar)
        print(f"{mode}: T2-positional rho={rho:.3f} p={p:.3g}")
    return rows


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='store_true')
    args = ap.parse_args()
    if args.probe:
        probe()
    else:
        report()
