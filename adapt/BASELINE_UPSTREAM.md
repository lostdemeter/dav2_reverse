# Proposal: baseline quality gate for the Foundry Experimenter

Status: PROPOSED (not yet filed upstream). Implemented domain-side here
as `adapt/baseline.py`; this doc specifies the core change so it can be
proposed to adaptation_foundry (and from there to Echion Revisited's
vendored copy at `echion/foundry/experimenter.py`).

## The incident (evidence)

On `experimental/adaptation-depth-styles`, the width search ran 10+
trials, promoted twice, sealed cleanly — with the seed scoring 2/14 in
its own harness. Every reason string was formally correct; every verdict
was noise. Root causes, two of them:
1. The bar (0.999, inherited from a float harness) exceeded what the
   integer harness delivers to ANY config. Domain fault, now fixed via
   seed-calibrated bars (`CORR_PASS_INT`, NOTES correction entry).
2. The controller never asked whether the BASELINE passes anything.
   `PromotionRule.compare` with `incumbent=None` returns BASELINE modulo
   the size cap — quality unchecked. Core gap. THIS proposal.

A secondary display fault compounded it: trial lines printed DELTAS
only (`exploration_correct: 0` reads the same for 14->14 and 2->2).
Fixed domain-side (`fmt_counts` absolutes on every trial line). The
measurement dicts always carried counts — the data was there, the
display hid it.

## Minimal core diff (against experimenter.py)

```python
class Experimenter:
    def __init__(self, adapter, max_trials=10, rule=None,
                 baseline_gate=None):
        ...
        # baseline_gate: None (legacy behavior, default) or a callable
        #   gate(measurement_as_dict) -> (ok: bool, reason: str)
        # supplied by the domain. Checked once, after the baseline trial
        # is decided. On failure: done/seal path with the ALREADY-EXISTING
        # stop reason "baseline_failed" (no new states, no new reasons).
        self.baseline_gate = baseline_gate
```

In `_finish_trial`, when the accepted decision is the first BASELINE:
evaluate `self.baseline_gate(deepcopy(self.measurement))`; on failure
set `self.error`, route to done with `stop_reason="baseline_failed"`.
Determinism re-check stays domain-side (it needs a second evaluation
the core shouldn't pay for on every run).

Why the core should own it (and the wrapper is only a stopgap):
- Every adapter otherwise reimplements pre-flight + determinism asserts.
- The stop reason already exists — the core anticipated a failing
  baseline for BUILD failures but not for QUALITY failures.
- Backward compatible: default None preserves current behavior exactly.

## Echion migration path

Echion vendors the core at `echion/foundry/experimenter.py` and drives
it from `echion/generation/ga_adapter.py` (GA candidates) and the
template search. Migration is one line per call site
(`baseline_gate=<their bar>`) plus stating each domain's bar — which
their PRINCIPLES doc already demands of gates generally. Their
zero-promotion runs would then distinguish "nothing beat a GOOD
baseline" from "the baseline itself fails," which is precisely the
ambiguity this incident exposed.

## License/provenance

Both repos are GPL-3.0-only; `adapt/baseline.py` + this doc are the
portable form of the proposal. File upstream against
lostdemeter/adaptation_foundry first; Echion consumes from there.
