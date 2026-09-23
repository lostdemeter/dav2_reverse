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
    from geo_neck import GeometricNeck
    from depth_adapter import ScaledNeckMixin

    import glob as _glob
    _cands = sorted(_glob.glob(str(ADAPT / 'runs' / 'student_grad_best_r*.pt')),
                    key=lambda p: Path(p).stat().st_mtime)
    _ckpt_path = Path(_cands[-1])
    # env must be set BEFORE importing student_grad (RANK/LATE read at import)
    import re as _re
    _m = _re.search(r'_r(\d+)', _ckpt_path.name)
    if _m:
        import os as _os
        _os.environ['STUDENT_RANK'] = _m.group(1)
    _m2 = _re.search(r'_ul([\d-]+)', _ckpt_path.name)
    if _m2:
        import os as _os
        _os.environ['STUDENT_UNFREEZE'] = _m2.group(1).replace('-', ',')
    if '_fw' in _ckpt_path.name:
        import os as _os
        _os.environ['STUDENT_FULLWIDTH'] = '1'
    from student_grad import StudentBackbone
    device = torch.device('cuda')
    shared = load_shared(device)

    class _Neck(ScaledNeckMixin, GeometricNeck):
        pass

    neck = _Neck(device=device)
    neck._w = shared['neck_w']
    neck.buffers_loaded = True
    neck.res_scales = [0, 0, 0, 0]

    student = StudentBackbone(shared['backbone'])
    ckpt = torch.load(_ckpt_path, map_location=device, weights_only=False)
    print(f"checkpoint {_ckpt_path.name} mean_corr={ckpt['mean_corr']:.5f}",
          flush=True)
    with torch.no_grad():
        for (li, m), tup in ckpt['factors'].items():
            A, B, b = student.factors[(li, m)]
            A.copy_(tup[0].to(device))
            B.copy_(tup[1].to(device))
            b.copy_(tup[2].to(device))
        for (li, m), tup in ckpt.get('late', {}).items():
            Wp, bp = student.late.get((li, m), (None, None))
            if Wp is None:
                print(f"  WARNING: ckpt late {(li, m)} not in student "
                      f"(env STUDENT_UNFREEZE mismatch?)", flush=True)
                continue
            Wp.copy_(tup[0].to(device))
            if bp is not None and tup[1] is not None:
                bp.copy_(tup[1].to(device))
        for (li, m), tup in ckpt.get('full', {}).items():
            Wp, bp = student.full.get((li, m), (None, None))
            if Wp is None:
                print(f"  WARNING: ckpt full {(li, m)} not in student",
                      flush=True)
                continue
            Wp.copy_(tup[0].to(device))
            if bp is not None and tup[1] is not None:
                bp.copy_(tup[1].to(device))

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
