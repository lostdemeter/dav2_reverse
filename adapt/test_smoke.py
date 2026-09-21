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
_good = dict(SEED_CONFIG)
for bad in ({}, dict(_good, readout=9),
            dict(_good, res_scales=[0, 0, 0]),
            dict(_good, res_scales=[0, 0, 0, 9]),
            dict(_good, head="fluid"),
            dict(_good, taps=[3, 6, 9]),
            dict(_good, taps=[3, 6, 9, 13]),
            dict(_good, taps=[3, 6, 6, 12]),
            dict(_good, taps=[4, 3, 9, 12]),
            dict(_good, tap_gains=[0, 0, 0, 2]),
            dict(_good, block_gains=[0] * 23),
            dict(_good, block_gains=[0] * 24 + [0]),
            dict(_good, block_gains=[0] * 23 + [3]),
            dict(_good, extra=1)):
    try:
        validate_config(dict(bad))
        check(False, f"rejects {bad}")
    except ValueError:
        check(True, f"rejects {list(bad)[-1] if bad else '{}'}")

moves = neighbor_configs(seed)
check(len(moves) == 2 + 3 + 8 + 7 + 8 + 48, f"seed has 76 neighbors (got {len(moves)})")
specs = [TrialSpec(f"m{i}", dict(c), r) for i, (c, r, _p) in enumerate(moves)]
check(len({s.identifier for s in specs}) == len(specs), "neighbor identifiers unique")
check(all(validate_config(dict(s.config)) == dict(s.config) for s in specs),
      "neighbors re-validate")

from efficiency import EfficiencyRule
from experimenter import Scorecard, Measurement, PromotionRule

LATEST = None


def _meas(exp_c, exp_t, gate_c, gate_t, ret_ok, ret_t, nbytes, mean=0.9995):
    def card(ds, c, t, ok_list=None):
        cases = []
        for i in range(t):
            good = i < c
            cases.append(__import__('experimenter').CaseResult(
                f"{ds}-{i}", True, good, ""))
        return Scorecard(ds, tuple(cases))
    exp = card("e", exp_c, exp_t)
    gate = card("g", gate_c, gate_t)
    ret = card("r", ret_ok, ret_t)
    diags = {r: {"mean_corr": mean} for r in ("exploration", "gate", "retention")}
    return Measurement(exp, gate, ret, nbytes, diags)


base = PromotionRule()
rule = EfficiencyRule()
inc = _meas(5, 6, 4, 4, 3, 3, 1000)
# pure tie + fewer bytes -> efficiency promotion
tie_small = _meas(5, 6, 4, 4, 3, 3, 900)
d = rule.compare(tie_small, inc)
check(d["action"] == "PROMOTE" and d["reasons"] == ["efficiency_gain_bytes"],
      "tie + fewer bytes promotes via efficiency")
# tie + same bytes -> still rejected
tie_same = _meas(5, 6, 4, 4, 3, 3, 1000)
check(rule.compare(tie_same, inc)["action"] == "REJECT",
      "tie + same bytes stays rejected")
# tie + MORE bytes -> rejected
tie_big = _meas(5, 6, 4, 4, 3, 3, 1100)
check(rule.compare(tie_big, inc)["action"] == "REJECT",
      "tie + more bytes stays rejected")
# gate regression + fewer bytes -> rejected (regressions stand)
bad = _meas(5, 6, 3, 4, 3, 3, 900)
check(rule.compare(bad, inc)["action"] == "REJECT",
      "gate regression + fewer bytes stays rejected")
# retention loss + fewer bytes -> rejected
ret = _meas(5, 6, 4, 4, 2, 3, 900)
check(rule.compare(ret, inc)["action"] == "REJECT",
      "retention loss + fewer bytes stays rejected")
# genuine accuracy win still promotes through base path
win = _meas(6, 6, 4, 4, 3, 3, 900)
d = rule.compare(win, inc)
check(d["action"] == "PROMOTE" and d["reasons"] != ["efficiency_gain_bytes"],
      "accuracy win promotes via base path")
# baseline with no incumbent
check(rule.compare(tie_small, None)["action"] == "BASELINE",
      "no-incumbent baseline passes through")
# margin-aware gate: tie + fewer bytes but mean_corr drops > margin -> rejected
thin = _meas(5, 6, 4, 4, 3, 3, 900, mean=0.9995 - 0.001)
d = rule.compare(thin, inc)
check(d["action"] == "REJECT" and d["reasons"] == ["margin_regressed:exploration"],
      "mean drop beyond margin blocks efficiency promotion")
# margin-aware gate: drop within margin still promotes
close = _meas(5, 6, 4, 4, 3, 3, 900, mean=0.9995 - 0.0001)
d = rule.compare(close, inc)
check(d["action"] == "PROMOTE" and d["reasons"] == ["efficiency_gain_bytes"],
      "mean drop within margin still promotes")
# bad margin value rejected at construction
try:
    EfficiencyRule(corr_margin=-1.0)
    check(False, "negative margin rejected")
except ValueError:
    check(True, "negative margin rejected")

from graduate import validate_widths, table_bytes, neighbor_widths, SEED as WSEED
w = validate_widths(dict(WSEED))
check(w == WSEED, "width seed validates")
check(table_bytes(w) == 1081356, f"seed tree tables = 1081356B (got {table_bytes(w)})")
for bad in ({"frac_cap": 999, "exp_span": 16, "dmax": 4096, "accum": "tree"},
            {"frac_cap": 8192, "exp_span": 7, "dmax": 4096, "accum": "tree"},
            {"frac_cap": 8192, "exp_span": 8, "dmax": 4096, "accum": "lut"}):
    try:
        validate_widths(bad)
        check(False, f"rejects widths {bad}")
    except ValueError:
        check(True, f"rejects widths {bad}")
wm = neighbor_widths(w)
check(len(wm) == 3 + 2 + 1 + 1, f"width seed has 7 moves (got {len(wm)})")
ids = [TrialSpec(f"w{i}", dict(c), r).identifier for i, (c, r) in enumerate(wm)]
check(len(set(ids)) == len(ids), "width neighbors unique")
tgt = {"frac_cap": 8192, "exp_span": 8, "dmax": 4096, "accum": "tree"}
check(table_bytes(tgt) == 557068, f"exp-halved tree tables (got {table_bytes(tgt)})")
# honesty: frac moves under tree change nothing (evaluator can't see them)
same = dict(tgt, frac_cap=2048)
check(table_bytes(same) == table_bytes(tgt),
      "frac width unread by tree costs zero bytes either way")
fixed = dict(WSEED, accum="fixed")
check(table_bytes(fixed) == 1200916, f"seed fixed tables (got {table_bytes(fixed)})")
print("SMOKE:", "PASS" if fails == 0 else "FAIL")
raise SystemExit(1 if fails else 0)
