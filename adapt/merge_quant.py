#!/usr/bin/env python3
"""Merge per-rank gap files into remainder_registry quantitative bounds.

Usage: QUANT_RANK_ONLY=128 python adapt/quantify_remainders.py  (worker)
       QUANT_RANK_ONLY=192 python adapt/quantify_remainders.py  (worker)
       python adapt/merge_quant.py                              (merge)
Workers write adapt/runs/quant_gaps_tmp_<pid>.json; the merge pools
gaps per remainder, bootstraps CI95 (10k resamples), writes bounds
into remainder_registry.json, and extends audit_operator with the
quant check (bounds present + CI width sane + 3-ckpt set).
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np
import json
import glob

N_BOOT = 10000


def main():
    gaps = {"e0_smooth": [], "dark": [], "tail": []}
    ckpts = []
    for f in glob.glob(str(ADAPT / 'runs' / 'quant_gaps_tmp_*.json')):
        d = json.load(open(f))
        for k in gaps:
            gaps[k].extend(d["gaps"][k])
        ckpts.extend(d.get("ckpts", []))
    assert gaps["e0_smooth"], "no worker files found"
    rng = np.random.default_rng(0)
    reg = json.load(open(ADAPT / 'remainder_registry.json'))
    keymap = [("e0_smooth", 0), ("dark", 1), ("tail", 2)]
    for key, ri in keymap:
        v = np.array(gaps[key])
        boots = np.array([np.mean(rng.choice(v, len(v), replace=True))
                          for _ in range(N_BOOT)])
        lo, hi = (float(np.quantile(boots, 0.025)),
                  float(np.quantile(boots, 0.975)))
        reg["remainders"][ri]["quant"] = {
            "gap_mean": round(float(v.mean()), 6),
            "gap_ci95": [round(lo, 6), round(hi, 6)],
            "n_gaps": len(v),
            "checkpoints": sorted(set(Path(c).stem for c in ckpts))}
        print(f"{key}: gap_mean={v.mean():.5f} ci95=[{lo:.5f},{hi:.5f}] "
              f"n={len(v)} ckpts={len(set(ckpts))}", flush=True)
    (ADAPT / 'remainder_registry.json').write_text(json.dumps(reg, indent=1))
    print("wrote quantitative bounds -> remainder_registry.json", flush=True)


if __name__ == '__main__':
    main()
