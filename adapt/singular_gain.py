#!/usr/bin/env python3
"""Phase F2: singular detail-gain knob on early maps.

SVD the composed student maps; scale TRAILING singular values
(bottom half) by x{1.25,1.5,2.0} in L0 (closest to pixels);
re-factorize to rank-192; evaluate base vs edited on the same
54 scenes with per-scene deltas (self-contained comparison).
Predicts: thin-detail (E1T1*) scenes improve slightly; overdone
(x2.0) -> noise everywhere (min drops).

Requires STUDENT_RANK=192. Run:
  STUDENT_RANK=192 python adapt/singular_gain.py
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

LAYERS = (0,)  # L0 only: closest to pixels, detail bandwidth lives here
RANK = 192
GAINS = (1.25, 1.5, 2.0)
BASE_CKPT = 'student_grad_best_r192_ds0.1_gms1p0_0p5_0p25_synthpool100_edgew_detailw_structmask_cos.pt'
SHORT2BUF = {'q': 'q.weight', 'k': 'k.weight', 'v': 'v.weight',
             'proj': 'proj.weight', 'mlp1': 'mlp1.weight',
             'mlp2': 'mlp2.weight'}


def main():
    import torch
    from run_search import load_shared, load_fixtures
    from geo_neck import GeometricNeck
    from depth_adapter import ScaledNeckMixin, corr as _corr
    import student_probe as SP

    device = torch.device('cuda')
    shared = load_shared(device)

    class _Neck(ScaledNeckMixin, GeometricNeck):
        pass

    neck = _Neck(device=device)
    neck._w = shared['neck_w']
    neck.buffers_loaded = True
    neck.res_scales = [0, 0, 0, 0]

    from student_grad import StudentBackbone
    student = StudentBackbone(shared['backbone'])
    have_rank = next(iter(student.factors.values()))[0].shape[0]
    assert have_rank == RANK, f"need STUDENT_RANK=192, got {have_rank}"
    base = torch.load(ADAPT / 'runs' / BASE_CKPT, map_location=device,
                      weights_only=False)
    print(f"base {BASE_CKPT} mean_corr={base['mean_corr']:.5f}", flush=True)

    def load_base():
        with torch.no_grad():
            for (li, m), tup in base['factors'].items():
                A, B, b = student.factors[(li, m)]
                A.copy_(tup[0].to(device))
                B.copy_(tup[1].to(device))
                b.copy_(tup[2].to(device))

    load_base()

    # eval scenes: gate + audit + holdouts (same 54-gate as training)
    fx = load_fixtures(include_audit=False)
    fxa = load_fixtures(include_audit=True)['audit']
    zf = np.load(ADAPT / 'fixtures' / 'fit_pool.npz', allow_pickle=True)
    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    npool = len(zf['rgb'])
    hold_idx = set(range(0, npool, 12))
    all_pool = ([(zf['rgb'][i].astype(np.float32) / 255.0,
                  zf['ref'][i].astype(np.float64))
                 for i in range(len(zf['rgb']))] +
                [(zr['rgb'][i].astype(np.float32) / 255.0,
                  zr['ref'][i].astype(np.float64))
                 for i in range(len(zr['rgb']))])
    hold_pairs = [all_pool[i] for i in sorted(hold_idx)]
    eval_scenes = ([(rgb, ref, cid) for rgb, ref, cid in fx['gate']] +
                   [(rgb, ref, cid) for rgb, ref, cid in fxa] +
                   [(rgb, ref, f"hold-{i}") for i, (rgb, ref) in
                    enumerate(hold_pairs)])

    def run_depth(rgb):
        fmaps, ph, pw = student.forward_backbone(
            shared['preprocess'](rgb).to(device))
        with torch.no_grad():
            fused = neck(fmaps)
            return shared['head']([fused[3]], ph, pw).squeeze(0).cpu().numpy()

    def gate(tag):
        cs = []
        with torch.no_grad():
            for rgb, ref, cid in eval_scenes:
                cs.append((cid, _corr(run_depth(rgb), ref)))
        mean = float(np.mean([c for _, c in cs]))
        print(f"[{tag}] mean={mean:.5f} min={min(c for _, c in cs):.5f}",
              flush=True)
        return dict(cs)

    base_cs = gate("base")
    results = {"base_mean": round(float(np.mean(list(base_cs.values()))), 5),
               "gains": {}}
    for g in GAINS:
        load_base()  # reset: trials independent
        with torch.no_grad():
            for li in LAYERS:
                for nm, _, _, _, _ in SP.MAPS:
                    A, B, b = (t.detach().cpu().numpy().astype(np.float64)
                               for t in student.factors[(li, SHORT2BUF[nm])])
                    W = B @ A
                    U, Sv, Vh = np.linalg.svd(W, full_matrices=False)
                    Sv[RANK // 2:] *= g  # boost trailing (detail?) dirs
                    Ap = (Sv[:RANK, None] * Vh[:RANK, :])
                    Bp = U[:, :RANK]
                    At, Bt, _ = student.factors[(li, SHORT2BUF[nm])]
                    At.copy_(torch.from_numpy(Ap.astype(np.float32))
                             .to(device))
                    Bt.copy_(torch.from_numpy(Bp.astype(np.float32))
                             .to(device))
        cs = gate(f"gain={g}")
        deltas = {cid: round(cs[cid] - base_cs[cid], 5) for cid in cs}
        up = sorted(deltas.items(), key=lambda kv: -kv[1])[:5]
        dn = sorted(deltas.items(), key=lambda kv: kv[1])[:5]
        print(f"  top-5 improved: {up}", flush=True)
        print(f"  top-5 worsened: {dn}", flush=True)
        results["gains"][str(g)] = {
            "mean": round(float(np.mean(list(cs.values()))), 5),
            "deltas": deltas}
        ck = {"factors": {(li, m): tuple(p.detach().cpu().clone()
                                         for p in tup)
                          for (li, m), tup in student.factors.items()},
              "meta": {"base": BASE_CKPT, "gain": g,
                       "layers": list(LAYERS)}}
        torch.save(ck, ADAPT / 'runs' / f'surgical_f2_gain{g}.pt')
    (ADAPT / 'runs' / 'surgical_f2.json').write_text(
        json.dumps(results, indent=1))
    print("wrote adapt/runs/surgical_f2.json", flush=True)


if __name__ == '__main__':
    main()
