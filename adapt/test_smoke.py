#!/usr/bin/env python3
"""Fast smoke test: DSL validation + neighbor enumeration. No torch/GPU/weights."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'third_party'))

from depth_adapter import validate_config, neighbor_configs, SEED_CONFIG
from experimenter import TrialSpec

fails = 0


def check(cond, msg):
    global fails
    print(('PASS ' if cond else 'FAIL ') + msg)
    if not cond:
        fails += 1


seed = validate_config(dict(SEED_CONFIG))
check(seed == SEED_CONFIG, "seed validates")
for bad in ({}, {"readout": 9, "res_scales": [0, 0, 0, 0], "head": "geo-conv"},
            {"readout": 3, "res_scales": [0, 0, 0], "head": "geo-conv"},
            {"readout": 3, "res_scales": [0, 0, 0, 9], "head": "geo-conv"},
            {"readout": 3, "res_scales": [0, 0, 0, 0], "head": "fluid"},
            {"readout": 3, "res_scales": [0, 0, 0, 0], "head": "geo-conv", "extra": 1}):
    try:
        validate_config(bad)
        check(False, f"rejects {bad}")
    except ValueError:
        check(True, f"rejects {bad}")

moves = neighbor_configs(seed)
check(len(moves) == 1 + 3 + 8, f"seed has 12 neighbors (got {len(moves)})")
specs = [TrialSpec(f"m{i}", dict(c), r) for i, (c, r, _p) in enumerate(moves)]
check(len({s.identifier for s in specs}) == len(specs), "neighbor identifiers unique")
check(all(validate_config(dict(s.config)) == dict(s.config) for s in specs),
      "neighbors re-validate")
print("SMOKE:", "PASS" if fails == 0 else "FAIL")
raise SystemExit(1 if fails else 0)
