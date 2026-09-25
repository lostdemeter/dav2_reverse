#!/usr/bin/env python3
"""Phase H3: loud-vs-quiet channel ablation (holographic gatekeeper).

Theory (phi_lattice §4.7, §6.7): dead channels carry cancellation
info via destructive interference; removing the dark fringe
collapses output. Standard view: damage ∝ energy removed.
Test on the Phase-E student L0-2 MLP GeLU outputs: zero top-k vs
bottom-k channels by energy (k in {32,128}, energy ranked on 6
calibration scenes), compare 54-gate damage.
Holographic prediction: quiet-k damage >= loud-k (or comparable
despite ~0 energy). Null: loud >> quiet -> analogy strained, stop.

Ablation via factor columns (A[:,c]=0 kills GeLU channel c exactly:
W=B@A loses column c). Requires STUDENT_RANK=192. Run:
  STUDENT_RANK=192 python adapt/holographic_ablate.py
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np
import json

LAYERS = (0, 1, 2)
RANK = 192
KS = (32, 128)
N_CALIB = 6
BASE_CKPT = 'student_grad_best_r192_ds0.1_gms1p0_0p5_0p25_synthpool100_edgew_detailw_structmask_cos.pt'


def main():
    import torch
    from run_search import load_shared, load_fixtures
    from geo_neck import GeometricNeck
    from depth_adapter import ScaledNeckMixin, corr as _corr

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

    def run_depth(rgb):
        fmaps, ph, pw = student.forward_backbone(
            shared['preprocess'](rgb).to(device))
        with torch.no_grad():
            fused = neck(fmaps)
            return shared['head']([fused[3]], ph, pw).squeeze(0).cpu().numpy()

    def gate(tag):
        with torch.no_grad():
            cs = [_corr(run_depth(rgb), ref) for rgb, ref, _ in eval_scenes]
        mean = float(np.mean(cs))
        print(f"[{tag}] mean={mean:.5f} min={min(cs):.5f}", flush=True)
        return mean

    base_mean = gate("base")

    # energy ranking on calibration scenes (first N_CALIB fit scenes)
    zc = np.load(ADAPT / 'fixtures' / 'fit_pool.npz', allow_pickle=True)
    energy = {}
    with torch.no_grad():
        for i in range(N_CALIB):
            rgb = zc['rgb'][i].astype(np.float32) / 255.0
            cap = {}
            student.forward_backbone(
                shared['preprocess'](rgb).to(device), capture=cap)
            for li in LAYERS:
                G = cap[(li, 'blk')]['G'].squeeze(0).double().cpu().numpy()
                e = energy.setdefault(li, np.zeros(G.shape[1]))
                e += (G ** 2).sum(axis=0)
    for li in LAYERS:
        e = energy[li]
        print(f"L{li} GeLU energy: top-32 share="
              f"{e[np.argsort(e)[::-1][:32]].sum() / e.sum():.3f}, "
              f"bottom-128 share="
              f"{e[np.argsort(e)[:128]].sum() / e.sum():.4f}", flush=True)

    results = {"base_mean": round(base_mean, 5), "ablations": {}}
    order = {li: np.argsort(energy[li]) for li in LAYERS}  # quiet-first
    for k in KS:
        for side in ("quiet", "loud"):
            load_base()  # reset: trials independent
            with torch.no_grad():
                for li in LAYERS:
                    A, _, _ = student.factors[(li, 'mlp1.weight')]
                    ch = (order[li][:k] if side == "quiet"
                          else order[li][::-1][:k])
                    A[:, torch.tensor(ch.tolist(),
                                      device=device)] = 0.0
            mean = gate(f"{side}-{k}")
            results["ablations"][f"{side}-{k}"] = round(mean, 5)
    load_base()  # leave model untouched
    (ADAPT / 'runs' / 'holographic_h3.json').write_text(
        json.dumps(results, indent=1))
    print(json.dumps(results, indent=1), flush=True)


if __name__ == '__main__':
    main()
