#!/usr/bin/env python3
"""Failure map: best student checkpoint, per-scene corr by stratum."""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np
import json


def main():
    import torch
    from run_search import load_shared, load_fixtures
    from depth_adapter import corr as _corr
    from student_grad import StudentBackbone, STUDENT_LAYERS, RANK
    from geo_neck import GeometricNeck
    from depth_adapter import ScaledNeckMixin

    device = torch.device('cuda')
    shared = load_shared(device)

    class _Neck(ScaledNeckMixin, GeometricNeck):
        pass

    neck = _Neck(device=device)
    neck._w = shared['neck_w']
    neck.buffers_loaded = True
    neck.res_scales = [0, 0, 0, 0]

    student = StudentBackbone(shared['backbone'])
    ckpt = torch.load(ADAPT / 'runs' / f'student_grad_best_r{RANK}.pt',
                      map_location=device, weights_only=False)
    print(f"checkpoint mean_corr={ckpt['mean_corr']:.5f}", flush=True)
    with torch.no_grad():
        for (li, m), tup in ckpt['factors'].items():
            A, B, b = student.factors[(li, m)]
            A.copy_(tup[0].to(device))
            B.copy_(tup[1].to(device))
            b.copy_(tup[2].to(device))

    fx = load_fixtures(include_audit=False)
    fxa = load_fixtures(include_audit=True)['audit']
    scenes = ([(rgb, ref, cid) for role in ('exploration', 'gate', 'retention')
               for rgb, ref, cid in fx[role]] +
              [(rgb, ref, cid) for rgb, ref, cid in fxa])
    zf = np.load(ADAPT / 'fixtures' / 'fit_pool.npz', allow_pickle=True)
    npool = len(zf['rgb'])
    for i in sorted(set(range(0, npool, 12)))[:14]:
        scenes.append((zf['rgb'][i].astype(np.float32) / 255.0,
                       zf['ref'][i].astype(np.float64), f"hold-{i}"))

    from strata import profile, assign
    medians = json.load(open(ADAPT / 'runs' / 'strata.json'))['medians']
    rows = []
    for rgb, ref, cid in scenes:
        fmaps, ph, pw = student.forward_backbone(
            shared['preprocess'](rgb).to(device))
        with torch.no_grad():
            fused = neck(fmaps)
            d = shared['head']([fused[3]], ph, pw).squeeze(0).cpu().numpy()
        c = _corr(d, ref)
        st = assign(profile(rgb), medians)
        rows.append((cid, st, c))
    print(f"{len(rows)} scenes", flush=True)
    by_st = {}
    for cid, st, c in rows:
        by_st.setdefault(st, []).append((cid, c))
    for st in sorted(by_st):
        vals = by_st[st]
        mean = float(np.mean([c for _, c in vals]))
        print(f"{st} n={len(vals)} mean={mean:.5f} "
              + " ".join(f"{cid}={c:.4f}" for cid, c in vals), flush=True)


if __name__ == '__main__':
    main()
