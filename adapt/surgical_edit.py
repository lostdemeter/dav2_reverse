#!/usr/bin/env python3
"""Phase F1: residual-corrective surgical edit on the worst scene.

Target: audit-1 (0.9887). For each L0-2 map: residual R = teacher -
student block output on audit-1 tokens; constrained least-squares
delta D = (Ct + l*Cf + uI)^-1 (Xt'R) where Cf pins 60 fit scenes
(from cached per-scene covariances); refit W+DW, re-factorize to
rank-192 via SVD; reset between lambda trials. Gates: audit-1 +
full 54-gate per lambda. Saves per-lambda checkpoints (never
overwrites the banked best).

Requires STUDENT_RANK=192 (asserted). Run:
  STUDENT_RANK=192 python adapt/surgical_edit.py
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
RANK = 192
LAMBDAS = (0.01, 0.1, 1.0, 10.0)
N_FIT_PIN = 60
BASE_CKPT = 'student_grad_best_r192_ds0.1_gms1p0_0p5_0p25_synthpool100_edgew_detailw_structmask_cos.pt'

SHORT2BUF = {'q': 'q.weight', 'k': 'k.weight', 'v': 'v.weight',
             'proj': 'proj.weight', 'mlp1': 'mlp1.weight',
             'mlp2': 'mlp2.weight'}


def main():
    import torch
    from run_search import load_shared, load_fixtures
    from geo_neck import GeometricNeck
    from depth_adapter import ScaledNeckMixin, corr as _corr, SEED_CONFIG
    from run_search import load_shared as _ls  # noqa (same fn, clarity)
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
    audit1 = fxa[1]
    print(f"target audit-1 cid={audit1[2]}", flush=True)

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
                cs.append(_corr(run_depth(rgb), ref))
        a1 = _corr(run_depth(audit1[0]), audit1[1])
        mean = float(np.mean(cs))
        print(f"[{tag}] audit-1={a1:.5f} gate-mean={mean:.5f} "
              f"min={min(cs):.5f}", flush=True)
        return a1, mean

    a1_0, mean_0 = gate("base")
    print(f"base reproduces audit-1={a1_0:.5f} (expect ~0.9887)", flush=True)

    # captures on audit-1 (teacher + student, same code path)
    pv = shared['preprocess'](audit1[0]).to(device)
    with torch.no_grad():
        cap_t, cap_s = {}, {}
        shared['backbone'].forward_stages(pv, capture=cap_t)
        student.forward_backbone(pv, capture=cap_s)
    T = {li: {k: v.squeeze(0).double().cpu().numpy()
              for k, v in cap_t[(li, 'blk')].items()} for li in LAYERS}
    S = {li: {k: v.squeeze(0).double().cpu().numpy()
              for k, v in cap_s[(li, 'blk')].items()} for li in LAYERS}

    # fit pinning covariances (first 60 real-pool scenes, per-scene avg)
    zc = np.load(ADAPT / 'runs' / 'coreset_covs.npz', allow_pickle=False)
    Cf = {li: {nm: zc[f'{li}_{nm}_Sxx'][:N_FIT_PIN].sum(axis=0).astype(
        np.float64) / N_FIT_PIN for nm, _, _, _, _ in SP.MAPS}
        for li in LAYERS}

    results = {"base": [round(a1_0, 5), round(mean_0, 5)], "lambdas": {}}
    for lam in LAMBDAS:
        load_base()  # reset: trials independent
        with torch.no_grad():
            for li in LAYERS:
                for nm, ik, tk, di, do in SP.MAPS:
                    Xs = np.concatenate(
                        [S[li][ik], np.ones((S[li][ik].shape[0], 1))], axis=1)
                    Xt = np.ascontiguousarray(Xs)
                    Rt = T[li][tk] - S[li][tk]
                    Ct = Xt.T @ Xt
                    ct = Xt.T @ Rt
                    C = Ct + lam * Cf[li][nm]
                    mu = 1e-6 * float(np.trace(C)) / C.shape[0]
                    D = np.linalg.solve(
                        C + mu * np.eye(C.shape[0]), ct)  # (di+1, do)
                    A, B, b = (t.detach().cpu().numpy().astype(np.float64)
                               for t in student.factors[(li, SHORT2BUF[nm])])
                    Wnew = B @ A + D[:-1, :].T
                    bnew = b + D[-1, :]
                    U, Sv, Vh = np.linalg.svd(Wnew, full_matrices=False)
                    # Wnew is (do,di); want B'(do,r) A'(r,di) with B'A'=Wnew_r
                    Ap = (Sv[:RANK, None] * Vh[:RANK, :])
                    Bp = U[:, :RANK]
                    rec = Bp @ Ap
                    rel = (float(np.linalg.norm(rec - Wnew)) /
                           max(float(np.linalg.norm(Wnew)), 1e-12))
                    assert rel < 1e-4, f"refactor fail L{li}/{nm}: rel={rel}"
                    At, Bt, bt = student.factors[(li, SHORT2BUF[nm])]
                    At.copy_(torch.from_numpy(Ap.astype(np.float32))
                             .to(device))
                    Bt.copy_(torch.from_numpy(Bp.astype(np.float32))
                             .to(device))
                    bt.copy_(torch.from_numpy(bnew.astype(np.float32))
                             .to(device))
        a1, mean = gate(f"lam={lam}")
        out = {"audit1": round(a1, 5), "gate_mean": round(mean, 5)}
        results["lambdas"][str(lam)] = out
        ck = {"factors": {(li, m): tuple(p.detach().cpu().clone()
                                         for p in tup)
                          for (li, m), tup in student.factors.items()},
              "meta": {"base": BASE_CKPT, "lambda": lam, **out}}
        torch.save(ck, ADAPT / 'runs' / f'surgical_f1_lam{lam}.pt')
        print(f"  saved adapt/runs/surgical_f1_lam{lam}.pt", flush=True)
    (ADAPT / 'runs' / 'surgical_f1.json').write_text(
        json.dumps(results, indent=1))
    print(json.dumps(results, indent=1), flush=True)


if __name__ == '__main__':
    main()
