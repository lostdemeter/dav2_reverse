#!/usr/bin/env python3
"""Scene-conditional tap study: WHEN does tap-2 (and other tap moves) hold?

All single-tap neighbor moves x all 79 scenes (synthetic fixtures +
strata_extra + strata_real), per-stratum hold/fail at 0.999 via the
seed pipeline. Pre-registered 2026-09-22: 3->2 holds on synthetics +
low-edge strata, fails on high-edge real strata; deeper taps fail
broadly. Run: python adapt/tap_study.py
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np
import copy
import json

MOVES = {
    "tap0_dn": [2, 6, 9, 12],
    "tap1_dn": [3, 5, 9, 12],
    "tap1_up": [3, 7, 9, 12],
    "tap2_dn": [3, 6, 8, 12],
    "tap2_up": [3, 6, 10, 12],
    "tap3_dn": [3, 6, 9, 11],
}


def load_all_scenes():
    from run_search import load_fixtures
    base = load_fixtures(include_audit=False)
    scenes = ([(rgb, ref, cid) for role in ('exploration', 'gate', 'retention')
               for rgb, ref, cid in base[role]])
    for path, prefix in ((ADAPT / 'fixtures' / 'strata_extra.npz', 'extra'),
                         (ADAPT / 'fixtures' / 'strata_real.npz', 'real')):
        if path.exists():
            z = np.load(path, allow_pickle=True)
            rids = ([str(x) for x in z['ids']] if 'ids' in z else None)
            for i in range(len(z['rgb'])):
                cid = rids[i] if rids else f"{prefix}-{i}"
                scenes.append((z['rgb'][i].astype(np.float32) / 255.0,
                               z['ref'][i].astype(np.float64), cid))
    return scenes


def main():
    import torch
    from run_search import load_shared
    from depth_adapter import run_pipeline, corr, SEED_CONFIG
    from strata import profile, assign

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared = load_shared(device)
    scenes = load_all_scenes()
    print(f"{len(scenes)} scenes", flush=True)
    medians = json.load(open(ADAPT / 'runs' / 'strata.json'))['medians']
    strata = [assign(profile(rgb), medians) for rgb, _, _ in scenes]

    cfgs = {"seed": list(SEED_CONFIG["taps"])}
    cfgs.update(MOVES)
    table = {}
    for name, taps in cfgs.items():
        cfg = copy.deepcopy(dict(SEED_CONFIG))
        cfg["taps"] = taps
        for (rgb, ref, cid), st in zip(scenes, strata):
            pred = run_pipeline(shared, cfg, [rgb])[0]
            c = corr(pred, ref)
            table.setdefault(st, {}).setdefault(name, []).append(c >= 0.999)
        print(f"{name} {taps}: done", flush=True)

    print("\nper-stratum hold fraction (seed vs moves):", flush=True)
    sts = sorted(table)
    print("stratum " + " ".join(f"{n:>8}" for n in ["seed"] + list(MOVES)),
          flush=True)
    for st in sts:
        row = []
        for name in ["seed"] + list(MOVES):
            vals = table[st].get(name, [])
            row.append(sum(vals) / len(vals) if vals else float('nan'))
        print(f"{st} " + " ".join(f"{v:8.2f}" for v in row), flush=True)


if __name__ == '__main__':
    main()
