#!/usr/bin/env python3
"""Bank the distilled student as a drop-in geometric backbone npz.

Composes bottleneck factors (B@A + b) from a banked checkpoint,
phi-encodes the 18 L0-2 maps, copies all other arrays from the
teacher bake. Output loads in GeometricDinov2Backbone UNCHANGED
(true drop-in). Writes weights/student_backbone_<tag>.npz (gitignored,
rebuildable) + adapt/runs/student_banked.json receipt (committed).

The 0.99688 combined champion was lost to a checkpoint clobber;
banked candidate is cosine-only r128 (0.99624, best surviving
bottleneck mean). Run: python adapt/bank_student.py
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

BANK_CKPT = ADAPT / 'runs' / 'student_grad_best_r128_cos.pt'
BANK_TAG = 'r128-cos0.99624'
LAYERS = (0, 1, 2)
MAPS = ("q.weight", "k.weight", "v.weight", "proj.weight",
        "mlp1.weight", "mlp2.weight")


def main():
    import torch
    from geo_lut import phi_encode_numpy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(BANK_CKPT, map_location='cpu', weights_only=False)
    print(f"banking {BANK_CKPT.name} mean_corr={ckpt['mean_corr']:.5f}",
          flush=True)
    z = np.load(REPO / 'weights' / 'geometric_backbone.npz',
                allow_pickle=False)
    out = {k: z[k] for k in z.files}
    n_student = 0
    for li in LAYERS:
        for m in MAPS:
            A, B, b = (t.numpy().astype(np.float64)
                       for t in ckpt['factors'][(li, m)])
            W = (B @ A).astype(np.float64)  # (do, di), buffer order
            s, e = phi_encode_numpy(W)
            base = f'layer{li}.{m}'
            out[base + '.signs'] = s
            out[base + '.exps'] = e
            n_student += W.size
            bkey = base.replace('.weight', '.bias')
            if bkey + '.signs' in out:
                sb, eb = phi_encode_numpy(
                    b.astype(np.float64).reshape(out[bkey + '.shape']))
                out[bkey + '.signs'] = sb
                out[bkey + '.exps'] = eb
                n_student += b.size
    dst = REPO / 'weights' / f'student_backbone_{BANK_TAG}.npz'
    np.savez_compressed(dst, **out)
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB, "
          f"{n_student} student params phi-encoded)", flush=True)

    # float parity: phi-decoded student backbone vs teacher pipeline
    from run_search import load_shared, load_fixtures
    from depth_adapter import run_pipeline, corr as _corr, SEED_CONFIG
    from geo_backbone import GeometricDinov2Backbone
    shared = load_shared(torch.device(device))
    stud_bb = GeometricDinov2Backbone(dst, device=device)
    stud_bb.eval()
    shared_s = dict(shared)
    shared_s['backbone'] = stud_bb
    fx = load_fixtures(include_audit=False)
    scenes = ([(rgb, ref, cid) for role in ('exploration', 'gate')
               for rgb, ref, cid in fx[role][:3]])
    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    scenes += [(zr['rgb'][i].astype(np.float32) / 255.0,
                zr['ref'][i].astype(np.float64), f"eval-{i}")
               for i in range(6)]
    cs = []
    for rgb, ref, cid in scenes:
        cfg = copy.deepcopy(dict(SEED_CONFIG))
        pred = run_pipeline(shared_s, cfg, [rgb])[0]
        c = _corr(pred, ref)
        cs.append(c)
        print(f"  {cid}: {c:.5f}", flush=True)
    print(f"[banked-phi] mean={np.mean(cs):.5f} min={min(cs):.5f} "
          f"pass={sum(c >= 0.999 for c in cs)}/{len(cs)}", flush=True)
    (ADAPT / 'runs' / 'student_banked.json').write_text(json.dumps(
        {"tag": BANK_TAG, "ckpt": BANK_CKPT.name,
         "ckpt_mean": round(float(ckpt['mean_corr']), 5),
         "npz": dst.name, "phi_parity": [round(float(c), 5) for c in cs]},
        indent=1))
    print("wrote adapt/runs/student_banked.json", flush=True)


if __name__ == '__main__':
    main()
