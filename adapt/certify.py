#!/usr/bin/env python3
"""Phase D certificate: margin-gated parity per stratum, student vs seed.

Student (given ckpt) and teacher (seed config) run the same 31 scenes
(fixtures + audit + 14 holdout reals); margin m calibrated from SEED
range (adapt/margin.py rule); per-stratum tie fractions reported.
Verdict per stratum: TIE (all within margin) / GAP (else, with size).
Run: python adapt/certify.py [--ckpt PATH]
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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='adapt/runs/student_grad_best_r192_ds0.1_gms1p0_0p5_0p25_synthpool100_edgew_cos.pt')
    args = ap.parse_args()
    import torch
    import re as _re
    _m = _re.search(r'_r(\d+)', Path(args.ckpt).name)
    if _m:
        import os as _os
        _os.environ['STUDENT_RANK'] = _m.group(1)
    from run_search import load_shared, load_fixtures
    from depth_adapter import run_pipeline, corr as _corr, SEED_CONFIG
    from student_grad import StudentBackbone
    from geo_neck import GeometricNeck
    from depth_adapter import ScaledNeckMixin
    from strata import profile, assign
    from margin import calibrate_margin, margin_pass

    device = torch.device('cuda')
    shared = load_shared(device)

    class _Neck(ScaledNeckMixin, GeometricNeck):
        pass

    neck = _Neck(device=device)
    neck._w = shared['neck_w']
    neck.buffers_loaded = True
    neck.res_scales = [0, 0, 0, 0]

    student = StudentBackbone(shared['backbone'])
    ckpt = torch.load(REPO / args.ckpt, map_location=device,
                      weights_only=False)
    print(f"student {Path(args.ckpt).name} "
          f"mean_corr={ckpt['mean_corr']:.5f}", flush=True)
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
    for i in sorted(set(range(0, len(zf['rgb']), 12)))[:14]:
        scenes.append((zf['rgb'][i].astype(np.float32) / 255.0,
                       zf['ref'][i].astype(np.float64), f"hold-{i}"))
    medians = json.load(open(ADAPT / 'runs' / 'strata.json'))['medians']

    seed_c, stud_c, rows = {}, {}, []
    for rgb, ref, cid in scenes:
        cfg = copy.deepcopy(dict(SEED_CONFIG))
        t = run_pipeline(shared, cfg, [rgb])[0]
        cs = _corr(t, ref)
        fmaps, ph, pw = student.forward_backbone(
            shared['preprocess'](rgb).to(device))
        with torch.no_grad():
            s = shared['head'](
                neck(fmaps), ph, pw).squeeze(0).cpu().numpy()
        ct = _corr(s, t)  # student vs TEACHER (distillation fidelity)
        cr = _corr(s, ref)  # student vs ref (absolute quality)
        st = assign(profile(rgb), medians)
        rows.append((cid, st, round(cs, 5), round(ct, 5), round(cr, 5)))
    m = calibrate_margin([c for _, _, c, _, _ in rows])
    print(f"calibrated margin m={m:.6f} (seed range over {len(rows)} scenes)",
          flush=True)
    by_st = {}
    for cid, st, cs, ct, cr in rows:
        by_st.setdefault(st, []).append((cid, cs, ct, cr))
    print("stratum n seed-min studR-min tie-frac | (tie iff "
          "student-vs-ref within margin of teacher-vs-ref)", flush=True)
    ties, total = 0, 0
    for st in sorted(by_st):
        vals = by_st[st]
        # certificate (margin.py rule): student quality within margin
        # of TEACHER quality per scene — the distillation parity test.
        tf = [margin_pass(cr, cs, m) for _, cs, _, cr in vals]
        ties += sum(tf)
        total += len(vals)
        print(f"{st} n={len(vals)} "
              f"seedmin={min(c for _, c, _, _ in vals):.5f} "
              f"studRmin={min(c for _, _, _, c in vals):.5f} "
              f"tie={sum(tf)}/{len(vals)}", flush=True)
    print(f"CERTIFICATE: {ties}/{total} scenes at teacher quality "
          f"(margin {m:.6f})", flush=True)
    (ADAPT / 'runs' / 'certificate.json').write_text(json.dumps(
        {"ckpt": Path(args.ckpt).name, "margin": m,
         "rows": rows, "ties": [ties, total]}, indent=1))
    print("wrote adapt/runs/certificate.json", flush=True)


if __name__ == '__main__':
    main()
