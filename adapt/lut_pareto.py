#!/usr/bin/env python3
"""LUT-width Pareto: how small can the tables get before parity breaks?

Sweeps FRAC_CAP x EXP span x ADD/SUB DMAX on head-fixed + re0-conv +
softmax tests, reporting corr + shippable LUT bytes (int32 accounting).
Informs what a minimal practical DAV2 needs. Run: python adapt/lut_pareto.py
"""
import sys
import time
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))

import numpy as np


def build_frac_table(cap, f=18, k=512):
    import math
    phi = (1 + 5 ** 0.5) / 2
    return np.array([round(phi ** (-d / k) * 2 ** f) for d in range(cap + 1)],
                    dtype=np.int64)


def build_exp_table(span, f=14, g=24):
    import math
    n = span * (1 << f) + 1
    return np.array([round((1 << g) * math.exp(-d / (1 << f))) for d in range(n)],
                    dtype=np.int64)


def main():
    import geo_int as G
    from geo_int import IntegerPhiHead
    rng = np.random.default_rng(41)
    feat = (rng.standard_normal((2000, 32)) * 4).astype(np.float64)
    head = IntegerPhiHead(REPO / 'weights' / 'phi_weights_compact.bin')
    ref = head.float_predict(feat)
    fs, fe = IntegerPhiHead.encode_features(feat)

    # small re0-style 1x1 case: random 384->48 on 8x8 map
    fmap = (rng.standard_normal((384, 8, 8)) * 2).astype(np.float64)
    w = (rng.standard_normal((48, 384)) * 0.2).astype(np.float64)
    mfs, mfe, _mfz = G.int_encode_array(fmap)
    mws, mwe, _mwz = G.int_encode_array(w)
    ref_c = np.einsum('ci,ihw->chw', w, fmap)
    # softmax rows
    slog = (rng.standard_normal((20, 90)) * 1.5 - 2.0).astype(np.float64)
    import torch.nn.functional as F
    import torch
    ref_s = F.softmax(torch.from_numpy(slog).float(), dim=-1).numpy()

    # save + restore globals around the sweep
    keep = (G.FRAC_CAP, G._FRAC_LUT, G.DMAX)
    print(f"  {'FRAC':>6} {'EXP':>4} {'DMAX':>5} | {'head':>10} {'conv1x1':>10} "
          f"{'softmax':>10} | {'LUT kB':>7}")
    try:
        for cap in (13312, 8192, 4096, 2048):
            for span in (16, 8, 4):
                for dmax in (4096, 1024):
                    G.FRAC_CAP = cap
                    G._FRAC_LUT = build_frac_table(cap)
                    G.DMAX = dmax
                    add = G.build_add_lut(dmax)
                    sub = G.build_sub_lut(dmax)
                    G._EXP_LUT = build_exp_table(span)
                    # head fixed-dot
                    t0 = time.perf_counter()
                    got = np.empty(2000)
                    for i in range(2000):
                        got[i] = G._head_fixed_pixel(head, fs[i], fe[i])
                    ch = float(np.corrcoef(ref.astype(float), got)[0, 1])
                    # conv: fixed accumulate per output, from_fixed back
                    P = 8 * 8
                    cc = np.empty((48, P))
                    for co in range(48):
                        ts = (mfs.reshape(384, P).astype(np.int16)
                              * mws[co].astype(np.int16)[:, None]).astype(np.int8)
                        te = np.clip(mfe.reshape(384, P).astype(np.int32)
                                     + mwe[co].astype(np.int32)[:, None] - G.BIAS,
                                     0, G.MAX_EXP).astype(np.int32)
                        q, m = G.to_fixed_group(ts, te)
                        tot = q.sum(axis=0, dtype=np.int64)
                        s2, e2, _z = G.from_fixed_vec(tot, m)
                        lut = np.float32(G.PHI) ** ((np.arange(
                            65536, dtype=np.float32) - G.BIAS) / G.K)
                        cc[co] = s2.astype(np.float32) * lut[np.clip(e2, 0, 65535)]
                    cc = cc.reshape(48, 8, 8)
                    ck = float(np.corrcoef(
                        cc.flatten().astype(float), ref_c.flatten().astype(float))[0, 1])
                    # softmax via patched tables
                    ss, se, sz = G.int_encode_array(slog)
                    probs = np.empty_like(slog)
                    for r in range(20):
                        num, den = G.int_softmax_fixed(ss[r], se[r], sz[r])
                        probs[r] = num.astype(float) / den
                    cs = float(np.corrcoef(probs.flatten().astype(float),
                                           ref_s.flatten().astype(float))[0, 1])
                    kb = ((dmax + 1) * 4 * 2 + (cap + 1) * 4
                          + (span * (1 << 14) + 1) * 4) / 1024
                    print(f"  {cap:>6} {span:>4} {dmax:>5} | {ch:>10.6f} "
                          f"{ck:>10.6f} {cs:>10.6f} | {kb:>7.1f}", flush=True)
    finally:
        G.FRAC_CAP, G._FRAC_LUT, G.DMAX = keep
    print("done (minima: head>0.999, conv>0.999, softmax>0.9999)")


if __name__ == '__main__':
    main()
