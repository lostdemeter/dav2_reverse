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
            dict(_good, dropped=[0, 0]),
            dict(_good, dropped=[24]),
            dict(_good, dropped=[-1]),
            dict(_good, dropped="3"),
            dict(_good, extra=1)):
    try:
        validate_config(dict(bad))
        check(False, f"rejects {bad}")
    except ValueError:
        check(True, f"rejects {list(bad)[-1] if bad else '{}'}")

moves = neighbor_configs(seed)
check(len(moves) == 2 + 3 + 8 + 7 + 8 + 48 + 24, f"seed has 100 neighbors (got {len(moves)})")
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
same = dict(tgt, frac_cap=2048)
check(table_bytes(same) == table_bytes(tgt),
      "frac width unread by tree costs zero bytes either way")
fixed = dict(WSEED, accum="fixed")
check(table_bytes(fixed) == 1200916, f"seed fixed tables (got {table_bytes(fixed)})")

from depth_adapter import _model_bytes, ATTN_BLOCK_PARAMS, MLP_BLOCK_PARAMS
b0 = _model_bytes(dict(SEED_CONFIG), None)
b_attn = _model_bytes(dict(SEED_CONFIG, dropped=[0]), None)
b_mlp = _model_bytes(dict(SEED_CONFIG, dropped=[1]), None)
check(b0 - b_attn == ATTN_BLOCK_PARAMS * 4 == 592128 * 4,
      f"drop attn block saves exactly {ATTN_BLOCK_PARAMS * 4}B")
check(b0 - b_mlp == MLP_BLOCK_PARAMS * 4 == 1182336 * 4,
      f"drop mlp block saves exactly {MLP_BLOCK_PARAMS * 4}B")
check(_model_bytes(dict(SEED_CONFIG, dropped=[0, 1]), None)
      == b0 - (592128 + 1182336) * 4, "drops compose additively")

from baseline import BaselineFailed, check_baseline, fmt_counts, parity_gate
from experimenter import Measurement as _M


def _synth_meas(correct_map, means, nbytes=1000):
    def card(ds, n, c):
        return Scorecard(ds, tuple(
            __import__('experimenter').CaseResult(f"{ds}-{i}", True, i < c, "")
            for i in range(n)))
    exp = card("e", 7, correct_map[0])
    gate = card("g", 4, correct_map[1])
    ret = card("r", 3, correct_map[2])
    diags = {r: {"mean_corr": m} for r, m in
             zip(("exploration", "gate", "retention"), means)}
    return _M(exp, gate, ret, nbytes, diags)


class _FakeAdapter:
    def __init__(self, meas, flip=False):
        self._meas = meas
        self._flip = flip
        self._n = 0

    def seed(self):
        from experimenter import TrialSpec
        return TrialSpec("seed", {"k": 1}, "seed")

    def build(self, proposal):
        return {"k": 1}

    def evaluate(self, artifact):
        self._n += 1
        if self._flip and self._n > 1:
            other = _synth_meas((0, 0, 0), (0.5, 0.5, 0.5))
            return other
        return self._meas

    def fingerprint(self, artifact):
        import hashlib, json
        return hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()


full = _synth_meas((7, 4, 3), (0.9995, 0.9995, 0.9995)).as_dict()
check(fmt_counts(full) == "e7/7 g4/4 r3/3", "fmt_counts absolutes")
gate = parity_gate(0.99)
ok, reason = gate(full)
check(ok, f"passing baseline passes gate ({reason})")
thin = _synth_meas((2, 0, 1), (0.997, 0.995, 0.998)).as_dict()
ok, reason = gate(thin)
check(not ok, f"failing baseline fails gate ({reason})")
m = check_baseline(_FakeAdapter(_synth_meas((7, 4, 3), (0.9995, 0.9995, 0.9995))),
                   parity_gate(0.99), label="fake-pass")
check(m["exploration"]["counts"]["correct"] == 7, "pre-flight returns measurement")
try:
    check_baseline(_FakeAdapter(_synth_meas((2, 0, 1), (0.997, 0.995, 0.995))),
                   parity_gate(0.99), label="fake-fail")
    check(False, "pre-flight refuses failing baseline")
except BaselineFailed as e:
    check("refusing to search" in str(e), "pre-flight refuses failing baseline")
try:
    check_baseline(_FakeAdapter(_synth_meas((7, 4, 3), (0.9995, 0.9995, 0.9995)),
                                flip=True),
                   parity_gate(0.99), label="fake-flip")
    check(False, "pre-flight catches nondeterminism")
except BaselineFailed as e:
    check("nondeterministic" in str(e), "pre-flight catches nondeterminism")
print("SMOKE:", "PASS" if fails == 0 else "FAIL")
raise SystemExit(1 if fails else 0)
