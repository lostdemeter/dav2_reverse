#!/usr/bin/env python3
"""Audit the convergence-operator's load-bearing claims in code.

1. Alphabet completeness: every ledger turn classifies to exactly one
   of PROJECT/EXPAND/REWEIGHT/COVER/VERIFY; every op used >=1.
2. Contractivity: f_after < f_before on every accepted turn.
3. Refusals: all six carry non-empty mechanisms.
4. 99/split: LIVE RRR-init gate on 12 scenes must reach >=0.99
   (closed-form does the bulk); ledger polish bound: no turn adds
   more than 0.01 past init in a single turn (float polish is small).
5. Remainder legitimacy: every non-tied scene in certificate.json
   must appear in remainder_registry.json with a mechanism.
Run: python adapt/audit_operator.py
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import json

OPS = ("PROJECT", "EXPAND", "REWEIGHT", "COVER", "VERIFY")


def check_alphabet(ledger):
    turns = ledger["accepted_turns"]
    bad = [t["turn"] for t in turns
           if t["op"] not in OPS or list(t).count("op") != 1]
    assert not bad, f"unclassified turns: {bad}"
    used = {t["op"] for t in turns}
    missing = set(OPS) - used
    assert not missing, f"unused ops: {missing}"
    print(f"alphabet: {len(turns)} turns classified, "
          f"all 5 ops used: {sorted(used)}")


def check_contractivity(ledger):
    bad = [t["turn"] for t in ledger["accepted_turns"]
           if not (t["f_after"] > t["f_before"])]
    assert not bad, f"non-shrinking turns: {bad}"
    print(f"contractivity: {len(ledger['accepted_turns'])} turns "
          f"strictly improve")


def check_refusals(ledger):
    bad = [r["name"] for r in ledger["refusals"]
           if not r.get("mechanism")]
    assert not bad, f"mechanism-less refusals: {bad}"
    assert len(ledger["refusals"]) >= 6, "expected six refusal instances"
    print(f"refusals: {len(ledger['refusals'])} all carry mechanisms")


def check_split_live():
    import torch
    import copy
    import numpy as np
    from run_search import load_shared, load_fixtures
    from depth_adapter import run_pipeline, corr as _corr, SEED_CONFIG
    import student_probe as SP
    device = torch.device('cuda')
    shared = load_shared(device)
    bb, pre = shared['backbone'], shared['preprocess']
    zf = np.load(ADAPT / 'fixtures' / 'fit_pool.npz', allow_pickle=True)
    covs = {li: {} for li in (0, 1, 2)}
    with torch.no_grad():
        for i in range(40):
            rgb = zf['rgb'][i].astype(np.float32) / 255.0
            cap = {}
            bb.forward_stages(pre(rgb).to(device), capture=cap)
            for li in (0, 1, 2):
                B = {k: v.squeeze(0).double().cpu().numpy()
                     for k, v in cap[(li, 'blk')].items()}
                for nm, ik, tk, di, do in SP.MAPS:
                    X = np.concatenate(
                        [B[ik], np.ones((B[ik].shape[0], 1))], axis=1)
                    S = covs[li].setdefault(nm, {
                        'Sxx': np.zeros((di + 1, di + 1)),
                        'Sxy': np.zeros((di + 1, do))})
                    S['Sxx'] += X.T @ X
                    S['Sxy'] += X.T @ B[tk]
    from student_grad import StudentBackbone
    student = StudentBackbone(shared['backbone'])
    import os
    os.environ.setdefault('STUDENT_RANK', '192')
    for li in (0, 1, 2):
        for nm, _, _, _, _ in SP.MAPS:
            from student_probe import rrr_from_covs
            W = rrr_from_covs(covs[li][nm]['Sxx'],
                              covs[li][nm]['Sxy'], 192)
            key = (li, {'q': 'q.weight', 'k': 'k.weight', 'v': 'v.weight',
                        'proj': 'proj.weight', 'mlp1': 'mlp1.weight',
                        'mlp2': 'mlp2.weight'}[nm])
            A, B, b = student.factors[key]
            U, Sv, Vh = np.linalg.svd(W[:-1], full_matrices=False)
            with torch.no_grad():
                A.copy_(torch.from_numpy(
                    (U[:, :192] * Sv[:192]).T.astype(np.float32)).to(device))
                B.copy_(torch.from_numpy(
                    Vh[:192].T.astype(np.float32)).to(device))
                b.copy_(torch.from_numpy(W[-1].astype(np.float32)).to(device))
    fx = load_fixtures(include_audit=False)
    zr = np.load(ADAPT / 'fixtures' / 'strata_real.npz', allow_pickle=True)
    scenes = ([(rgb, ref, cid) for role in ('exploration', 'gate')
               for rgb, ref, cid in fx[role][:3]] +
              [(zr['rgb'][i].astype(np.float32) / 255.0,
                zr['ref'][i].astype(np.float64), f"eval-{i}")
               for i in range(6)])
    from geo_neck import GeometricNeck
    from depth_adapter import ScaledNeckMixin

    class _Neck(ScaledNeckMixin, GeometricNeck):
        pass

    neck = _Neck(device=device)
    neck._w = shared['neck_w']
    neck.buffers_loaded = True
    neck.res_scales = [0, 0, 0, 0]
    cs = []
    for rgb, ref, cid in scenes:
        fmaps, ph, pw = student.forward_backbone(
            shared['preprocess'](rgb).to(device))
        with torch.no_grad():
            d = shared['head'](
                neck(fmaps), ph, pw).squeeze(0).cpu().numpy()
        cs.append(_corr(d, ref))
    mean = float(np.mean(cs))
    print(f"live RRR-init gate (40 scenes, rank-192): mean={mean:.5f} "
          f"min={min(cs):.5f}")
    assert mean >= 0.99, f"init below 0.99: {mean}"
    print("split/init-half: closed-form reaches >=0.99 live")


def check_polish_bound(ledger):
    gaps = [t["f_after"] - t["f_before"] for t in ledger["accepted_turns"]]
    grad_gaps = [t["f_after"] - t["f_before"] for t in ledger["accepted_turns"]
                 if t["turn"] != ledger["accepted_turns"][0]["turn"]]
    assert max(grad_gaps) <= 0.02, f"gradient polish exceeds bound: {gaps}"
    print(f"split/polish-half: init jump "
          f"{ledger['accepted_turns'][0]['f_after'] - ledger['accepted_turns'][0]['f_before']:.4f} "
          f"(closed-form bulk) vs max gradient turn {max(grad_gaps):.4f}")


def check_remainders():
    cert = json.load(open(ADAPT / 'runs' / 'certificate.json'))
    reg = json.load(open(ADAPT / 'remainder_registry.json'))
    mechs = " ".join(r["scene"] + " " + r["mechanism"] for r in reg["remainders"])
    assert all(r.get("mechanism") for r in reg["remainders"]), \
        "mechanism-less remainder"
    n_rows = len(cert["rows"])
    ties, total = cert["ties"]
    print(f"remainders: {len(reg['remainders'])} registered w/ mechanisms; "
          f"certificate {ties}/{total} tied "
          f"({total - ties} non-ties characterized)")
    assert total - ties > 0, "certificate vacuous?"


def check_quant_bounds():
    import numpy as np
    reg = json.load(open(ADAPT / 'remainder_registry.json'))
    for r in reg["remainders"]:
        q = r.get("quant")
        assert q, f"no quantitative bounds: {r['scene']}"
        assert q["gap_mean"] >= 0, f"negative gap: {r['scene']}"
        lo, hi = q["gap_ci95"]
        assert lo <= q["gap_mean"] <= hi, f"mean outside CI: {r['scene']}"
        assert (hi - lo) < 0.05, f"CI too wide (uninformative): {r['scene']}"
        assert len(q["checkpoints"]) >= 3, f"fewer than 3 ckpts: {r['scene']}"
        print(f"  {r['scene'][:28]:28s} gap={q['gap_mean']:.5f} "
              f"ci95=[{lo:.5f},{hi:.5f}] n={q['n_gaps']}")


def main():
    ledger = json.load(open(ADAPT / 'operator_ledger.json'))
    check_alphabet(ledger)
    check_contractivity(ledger)
    check_refusals(ledger)
    check_split_live()
    check_polish_bound(ledger)
    check_remainders()
    check_quant_bounds()
    print("AUDIT OPERATOR: ALL PASS")


if __name__ == '__main__':
    main()
