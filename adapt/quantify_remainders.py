#!/usr/bin/env python3
"""Quantify remainder gaps with bootstrap CIs across checkpoints.

For each remainder_registry entry, measure the teacher-vs-student
shortfall on its scenes across 3 checkpoints
(r128_cos, r128_edgew=detailw_structmask, r192_phaseC), then
bootstrap (10k resamples over scenes) the mean gap. Writes bounds
back into remainder_registry.json (gap_mean, gap_ci95, n_scenes,
checkpoints). Deterministic gates -> variation comes from scenes
x checkpoints, honestly reported as such.
Run: python adapt/quantify_remainders.py
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

CKPTS = [
    # (filename, rank) — separate invocations per rank (module-level
    # RANK binds factor shapes at import; mixing ranks needs a worker
    # per rank, not one process).
    ('student_grad_best_r128_cos.pt', 128),
    ('student_grad_best_r128_ds0.1_gms1p0_0p5_0p25_synthpool100_'
     'edgew_cos.pt', 128),
    ('student_grad_best_r192_ds0.1_gms1p0_0p5_0p25_synthpool100_'
     'edgew_cos.pt', 192),
]
N_BOOT = 10000


def main():
    import torch
    import re as _re
    import os as _os
    _only = _os.environ.get('QUANT_RANK_ONLY')
    _rm = _re.search(r'_r(\d+)', CKPTS[0][0])
    _os.environ['STUDENT_RANK'] = _only or _rm.group(1)
    from run_search import load_shared, load_fixtures
    from depth_adapter import run_pipeline, corr as _corr, SEED_CONFIG
    from student_grad import StudentBackbone
    from geo_neck import GeometricNeck
    from depth_adapter import ScaledNeckMixin
    from student_grad import lowlight_augment

    device = torch.device('cuda')
    shared = load_shared(device)

    class _Neck(ScaledNeckMixin, GeometricNeck):
        pass

    neck = _Neck(device=device)
    neck._w = shared['neck_w']
    neck.buffers_loaded = True
    neck.res_scales = [0, 0, 0, 0]
    fx = load_fixtures(include_audit=False)
    fxa = load_fixtures(include_audit=True)['audit']
    by_cid = {}
    for role in ('exploration', 'gate', 'retention'):
        for rgb, ref, cid in fx[role]:
            by_cid[cid] = (rgb, ref)
    for rgb, ref, cid in fxa:
        by_cid[cid] = (rgb, ref)

    # remainder scene sets (mirror the registry definitions)
    e0_cids = ['exploration-2', 'exploration-4', 'gate-0', 'audit-1',
               'audit-2', 'retention-2']
    cert = json.load(open(ADAPT / 'runs' / 'certificate.json'))
    tied = set()
    # tail = worst banked scenes: lowest student-vs-ref in cert rows
    rows = sorted(cert["rows"], key=lambda r: r[4])
    tail_cids = [r[0] for r in rows[:6]]

    def student_depth(student, rgb):
        fmaps, ph, pw = student.forward_backbone(
            shared['preprocess'](rgb).to(device))
        with torch.no_grad():
            fused = neck(fmaps)
            return shared['head']([fused[3]], ph, pw).squeeze(0).cpu().numpy()

    gaps = {"e0_smooth": [], "dark": [], "tail": []}
    import student_grad as SG
    import subprocess as _sp
    import os as _os
    # Module-level RANK binds factor shapes at import: group
    # checkpoints by rank, one worker process per rank.
    groups = {}
    for _c, _r in CKPTS:
        groups.setdefault(_r, []).append(_c)
    only = sorted(groups[int(_os.environ.get('QUANT_RANK_ONLY', '0') or '0')]) \
        if _os.environ.get('QUANT_RANK_ONLY') else None
    todo = only or [c for _, cs in groups.items() for c in cs]
    ckpt_rank = {c: r for r, cs in groups.items() for c in cs}
    out_part = (ADAPT / 'runs' /
                f"quant_gaps_r{_os.environ.get('QUANT_RANK_ONLY', 'all')}.json")
    for ckpt_name in todo:
        student = SG.StudentBackbone(shared['backbone'])
        ckpt = torch.load(ADAPT / 'runs' / ckpt_name, map_location=device,
                          weights_only=False)
        with torch.no_grad():
            for (li, mm), tup in ckpt['factors'].items():
                A, B, b = student.factors[(li, mm)]
                if A.shape[1] != tup[0].shape[1]:
                    raise SystemExit(
                        f"RANK mismatch: need STUDENT_RANK={tup[0].shape[1]} "
                        f"for {ckpt_name}")
                A.copy_(tup[0].to(device))
                B.copy_(tup[1].to(device))
                b.copy_(tup[2].to(device))
        cfg = copy.deepcopy(dict(SEED_CONFIG))
        for cid in e0_cids:
            rgb, ref = by_cid[cid]
            t = run_pipeline(shared, cfg, [rgb])[0]
            s = student_depth(student, rgb)
            gaps["e0_smooth"].append(1.0 - _corr(s, t))
        rng = np.random.default_rng(1234)
        for cid in ['exploration-0', 'exploration-1', 'gate-1', 'audit-0']:
            rgb, ref = by_cid[cid]
            d = lowlight_augment(rgb, rng)
            t = run_pipeline(shared, cfg, [d])[0]
            s = student_depth(student, d)
            gaps["dark"].append(1.0 - _corr(s, t))
        for cid in tail_cids:
            rgb, ref = by_cid.get(cid, (None, None))
            if rgb is None:  # hold scenes live in fit_pool
                zf = np.load(ADAPT / 'fixtures' / 'fit_pool.npz',
                             allow_pickle=True)
                ids = [str(x) for x in zf['ids']]
                if cid.startswith('hold-'):
                    i = int(cid.split('-')[1])
                    rgb = zf['rgb'][i].astype(np.float32) / 255.0
                else:
                    continue
            t = run_pipeline(shared, cfg, [rgb])[0]
            s = student_depth(student, rgb)
            gaps["tail"].append(1.0 - _corr(s, t))
        print(f"{ckpt_name}: e0={np.mean(gaps['e0_smooth'][-6:]):.5f} "
              f"dark={np.mean(gaps['dark'][-4:]):.5f} "
              f"tail={np.mean(gaps['tail'][-6:]):.5f}", flush=True)

    rng = np.random.default_rng(0)
    (ADAPT / 'runs' / f"quant_gaps_tmp_{_os.getpid()}.json").write_text(
        json.dumps({"gaps": gaps, "ckpts": todo}, indent=1))
    print(f"wrote per-worker gaps ({len(todo)} ckpts)", flush=True)


if __name__ == '__main__':
    main()
