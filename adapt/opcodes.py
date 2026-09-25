#!/usr/bin/env python3
"""Phase G pilot: bias-level opcodes (ADD_IMM) on the student.

Treat the student as a compiled program; biases are per-channel
immediates. For audit-1 (worst scene): attribute block-output
residuals to top channels per map, try multiplicative bias steps
x(1+/-d), d in {0.001, 0.005, 0.01} (~1 phi-lattice step and up),
keep iff 54-gate mean rises AND no scene drops below its base floor.
Single sweep (no greedy chaining yet).

Requires STUDENT_RANK=192. Run:
  STUDENT_RANK=192 python adapt/opcodes.py
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
TOP_CH = 3
DELTAS = (0.001, 0.005, 0.01)
BASE_CKPT = 'student_grad_best_r192_ds0.1_gms1p0_0p5_0p25_synthpool100_edgew_detailw_structmask_cos.pt'
SHORT2BUF = {'q': 'q.weight', 'k': 'k.weight', 'v': 'v.weight',
             'proj': 'proj.weight', 'mlp1': 'mlp1.weight',
             'mlp2': 'mlp2.weight'}
# block-output key per map (what the map writes)
OUTKEY = {'q': 'Q', 'k': 'K', 'v': 'V', 'proj': 'Oattn',
          'mlp1': 'P1', 'mlp2': 'Y'}


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

    def run_depth(rgb):
        fmaps, ph, pw = student.forward_backbone(
            shared['preprocess'](rgb).to(device))
        with torch.no_grad():
            fused = neck(fmaps)
            return shared['head']([fused[3]], ph, pw).squeeze(0).cpu().numpy()

    def full_gate():
        with torch.no_grad():
            return {cid: _corr(run_depth(rgb), ref)
                    for rgb, ref, cid in eval_scenes}

    base_cs = full_gate()
    base_mean = float(np.mean(list(base_cs.values())))
    print(f"[base] mean={base_mean:.5f} "
          f"audit-1={base_cs[audit1[2]]:.5f}", flush=True)

    # attribute: per-channel |teacher-student| block residuals on audit-1
    pv = shared['preprocess'](audit1[0]).to(device)
    with torch.no_grad():
        cap_t, cap_s = {}, {}
        shared['backbone'].forward_stages(pv, capture=cap_t)
        student.forward_backbone(pv, capture=cap_s)
    cands = []  # (resid, li, short, channel)
    for li in LAYERS:
        T = {k: v.squeeze(0).double().cpu().numpy()
             for k, v in cap_t[(li, 'blk')].items()}
        S = {k: v.squeeze(0).double().cpu().numpy()
             for k, v in cap_s[(li, 'blk')].items()}
        for nm, _, _, _, _ in SP.MAPS:
            r = np.abs(T[OUTKEY[nm]] - S[OUTKEY[nm]]).mean(axis=0)
            for ch in np.argsort(r)[::-1][:TOP_CH]:
                cands.append((float(r[ch]), li, nm, int(ch)))
    cands.sort(reverse=True)
    print(f"{len(cands)} candidates; top-5: "
          + ", ".join(f"L{li}/{nm}c{ch}={r:.4f}"
                      for r, li, nm, ch in cands[:5]), flush=True)

    results = {"base_mean": round(base_mean, 5),
               "base_audit1": round(base_cs[audit1[2]], 5), "trials": []}
    for _, li, nm, ch in cands:
        buf = SHORT2BUF[nm]
        for sgn in (1.0, -1.0):
            for d in DELTAS:
                load_base()  # reset: trials independent
                with torch.no_grad():
                    _, _, b = student.factors[(li, buf)]
                    b[ch] *= (1.0 + sgn * d)
                cs = full_gate()
                mean = float(np.mean(list(cs.values())))
                floors = [cid for cid in cs if cs[cid] < base_cs[cid] - 1e-9]
                keep = mean > base_mean and not floors
                results["trials"].append(
                    {"op": f"ADD_IMM L{li}/{nm}[{ch}] x{1.0 + sgn * d:.4f}",
                     "mean": round(mean, 5),
                     "audit1": round(cs[audit1[2]], 5),
                     "floors": floors, "keep": keep})
                if keep:
                    print(f"  KEEP op L{li}/{nm}[{ch}] x{1.0 + sgn * d:.4f} "
                          f"mean={mean:.5f} audit1={cs[audit1[2]]:.5f}",
                          flush=True)
    load_base()  # leave model untouched
    keeps = [t for t in results["trials"] if t["keep"]]
    print(f"done: {len(keeps)}/{len(results['trials'])} kept", flush=True)
    (ADAPT / 'runs' / 'opcodes_g.json').write_text(
        json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
