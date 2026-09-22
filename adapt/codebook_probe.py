#!/usr/bin/env python3
"""Codebook landscape probe: joint (sign, phi-level) codebook on backbone.

Quantizes all 22M backbone phi weights to C centroids (weighted 1D
k-means on signed phi-levels, centroids snapped back to lattice),
patches decoded float buffers, runs the REAL seed pipeline on 3
synthetic + 3 real scenes, reports corr vs HF refs + analytic bytes.

Byte model (documented, analytic): bit-packed magnitude indices
N*ceil(log2(C))/8 + N/8 sign sidecar + C*2B centroids (int16 lattice
levels) vs phi baseline N*3B (sign u8 + exp u16) vs float32 N*4B.

Run: python adapt/codebook_probe.py [--levels 4096 2048 512 128]
P4 (pre-registered 2026-09-22): per-layer/global 2048 holds parity
with a byte win; flat global 512 fails retention.
"""
import argparse
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np

K_PHI, BIAS = 512, 32768


def weighted_kmeans_1d(values, weights, n_clusters, iters=25, seed=0):
    """Deterministic weighted 1D k-means (quantile init + Lloyd via
    interval assignment — O(n log C) per iter, no distance matrix)."""
    order = np.argsort(values)
    vs, ws = values[order].astype(np.float64), weights[order].astype(np.float64)
    cum = np.cumsum(ws)
    total = cum[-1]
    edges = (np.arange(n_clusters) + 0.5) / n_clusters * total
    centroids = vs[np.clip(np.searchsorted(cum, edges), 0, len(vs) - 1)].copy()
    assign = np.zeros(len(vs), dtype=np.int64)
    for it in range(iters):
        midpoints = (centroids[:-1] + centroids[1:]) / 2
        new_assign = np.searchsorted(midpoints, vs)
        if it > 0 and np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(n_clusters):
            m = assign == c
            if m.any():
                centroids[c] = np.sum(vs[m] * ws[m]) / np.sum(ws[m])
        centroids.sort()
    return centroids, assign, order


def build_codebook(backbone_z, n_levels, scope='global', guard=None):
    """Magnitude codebook over mu=(exp-BIAS)/K (signs exact sidecar).

    scope='global': one codebook for all 22M weights.
    scope='per-layer': one codebook per npz base (P4's prediction).
    guard=float|None: |mu|>guard weights stay EXACT (outlier residuals
      at 4B each) — only the core is quantized.
    Returns (dict base->sorted centroids, keys, n_guard)."""
    vals, counts = [], []
    keys = [k[:-len('.signs')] for k in backbone_z.files if k.endswith('.signs')]
    books, n_guard = {}, 0
    if scope == 'global':
        vals, counts = [], []
        for base in keys:
            e = backbone_z[base + '.exps'].ravel().astype(np.int64)
            mu = (e - BIAS) / K_PHI
            if guard is not None:
                n_guard += int((np.abs(mu) > guard).sum())
                mu = mu[np.abs(mu) <= guard]
            u, c = np.unique(mu, return_counts=True)
            vals.append(u)
            counts.append(c)
        vals = np.concatenate(vals)
        wts = np.concatenate(counts).astype(np.float64)
        centroids, _, _ = weighted_kmeans_1d(vals, wts, n_levels)
        lattice = np.sort(np.round(centroids * K_PHI) / K_PHI)
        books = {base: lattice for base in keys}
    else:
        for base in keys:
            e = backbone_z[base + '.exps'].ravel().astype(np.int64)
            mu = (e - BIAS) / K_PHI
            if guard is not None:
                n_guard += int((np.abs(mu) > guard).sum())
                mu = mu[np.abs(mu) <= guard]
            u, c = np.unique(mu, return_counts=True)
            centroids, _, _ = weighted_kmeans_1d(
                u, c.astype(np.float64), min(n_levels, len(u)))
            books[base] = np.sort(np.round(centroids * K_PHI) / K_PHI)
    return books, keys, n_guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', type=int, nargs='+', default=[4096, 2048, 512, 128])
    ap.add_argument('--scope', nargs='+', default=['global'],
                    choices=['global', 'per-layer'])
    ap.add_argument('--guard', type=float, nargs='*', default=[],
                    help='extra outlier-|mu| thresholds kept exact (None always included)')
    args = ap.parse_args()
    args.guard = [None] + list(args.guard)

    import torch
    from run_search import load_shared
    from depth_adapter import run_pipeline, corr, SEED_CONFIG
    import copy

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared = load_shared(device)
    backbone = shared['backbone']

    # probe scenes: 3 synthetic exploration + 3 real
    from run_search import load_fixtures
    fx = load_fixtures()['exploration'][:3]
    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    rids = [str(x) for x in zr['ids']]
    reals = [(zr['rgb'][i].astype(np.float32) / 255.0,
              zr['ref'][i].astype(np.float64), rids[i]) for i in (0, 7, 19)]
    scenes = fx + reals
    print(f"probe scenes: {[c for _, _, c in scenes]}", flush=True)

    # snapshot original float buffers
    orig = {k: v.clone() for k, v in backbone._w.items()}
    N = sum(v.numel() for v in orig.values())
    print(f"backbone params: {N} "
          f"(phi {N*3/1e6:.1f}MB, fp32 {N*4/1e6:.1f}MB)", flush=True)

    def evaluate(tag):
        cs = []
        for rgb, ref, cid in scenes:
            cfg = copy.deepcopy(dict(SEED_CONFIG))
            pred = run_pipeline(shared, cfg, [rgb])[0]
            c = corr(pred, ref)
            cs.append(c)
        n_ok = sum(c >= 0.999 for c in cs)
        print(f"  [{tag}] mean={np.mean(cs):.5f} min={min(cs):.5f} "
              f"pass={n_ok}/{len(cs)} " +
              " ".join(f"{c:.5f}" for c in cs), flush=True)
        return cs

    print("baseline (full 17.9k levels):", flush=True)
    evaluate("full")

    zb = np.load(REPO / 'weights' / 'geometric_backbone.npz', allow_pickle=False)
    PHI = float((1 + 5 ** 0.5) / 2)
    import itertools
    grid = list(itertools.product(args.scope, args.levels, args.guard))
    for scope, C, guard in grid:
        books, keys, n_guard = build_codebook(zb, C, scope, guard)
        nbits = int(np.ceil(np.log2(C)))
        nbooks = 1 if scope == 'global' else len(books)
        nbytes = (N * nbits / 8 + N / 8 + nbooks * C * 2
                  + n_guard * 4)  # indices + signs + centroids + residuals
        for base in keys:
            cent = books[base]
            midpoints = (cent[:-1] + cent[1:]) / 2
            s = zb[base + '.signs']
            e = zb[base + '.exps'].astype(np.int64)
            mu = ((e - BIAS) / K_PHI).ravel().astype(np.float64)
            if guard is not None:
                keep = np.abs(mu) > guard
            qmu = cent[np.searchsorted(midpoints, mu)]
            if guard is not None:
                qmu[keep] = mu[keep]  # outliers exact
            qval = (s.ravel().astype(np.float64)) * (PHI ** qmu)
            if base in backbone._w:
                backbone._w[base].copy_(torch.from_numpy(
                    qval.reshape(s.shape).astype(np.float32)).to(device))
        tag = f"{scope} C={C} guard={guard} ({nbits}b, {nbytes/1e6:.1f}MB, " \
            f"resid={n_guard})"
        print(tag + ":", flush=True)
        evaluate(tag)
        for k, v in orig.items():
            backbone._w[k].copy_(v)


if __name__ == '__main__':
    main()
