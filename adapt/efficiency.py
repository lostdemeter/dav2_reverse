"""Efficiency objective for size-aware promotion (experimental).

The vendored PromotionRule cannot promote compression wins: ties on
accuracy reject with insufficient_new_correct_cases, so a candidate
with identical accuracy at half the bytes never promotes. EfficiencyRule
SUBCLASSES the base rule (the controller type-checks isinstance) and
adds exactly one documented promotion path: base rejects SOLELY for
no-correctness-gain (no regressions, no incomparability, size cap
satisfied) AND exploration didn't get worse AND the artifact is strictly
smaller AND no role's mean correlation drops by more than `corr_margin`
(the margin-aware gate: binary ties can hide real degradation, as the
first graduation run's overshoot showed). Core file untouched; the
deviation lives here, at the decision site, and every efficiency
promotion says so in its reasons.

Round-two note (margin + hard anchors): `corr_margin` compares the
`mean_corr` diagnostics the adapter already records per role. The base
`__post_init__` only allows int fields, so this class overrides it with
an explicit equivalent plus a float-margin check.
"""
from dataclasses import dataclass

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'third_party'))  # explicit vendored-core path
from experimenter import PromotionRule  # noqa: E402

# Reasons that together mean "pure tie, nothing regressed, size cap fine".
_TIE_REASONS = frozenset({"insufficient_new_correct_cases",
                          "exploration_utility_not_improved"})


@dataclass(frozen=True)
class EfficiencyRule(PromotionRule):
    corr_margin: float = 0.0002

    def __post_init__(self):
        for name in ("correct_reward", "wrong_penalty", "min_correct_gain",
                     "max_model_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("Promotion parameters must be positive integers")
        if (type(self.corr_margin) not in (int, float)
                or not (self.corr_margin >= 0)):
            raise ValueError("corr_margin must be a nonnegative number")

    def compare(self, candidate, incumbent=None):
        decision = super().compare(candidate, incumbent)
        if incumbent is None or decision["action"] != "REJECT":
            return decision
        reasons = set(decision["reasons"])
        deltas = decision.get("deltas", {})
        if not (reasons and reasons <= _TIE_REASONS
                and deltas.get("exploration_correct", -1) >= 0
                and deltas.get("model_bytes", 0) < 0):
            return decision
        # Margin-aware gate: no role's mean correlation may drop more
        # than corr_margin. Diagnostics are strict JSON floats already.
        cand = candidate.diagnostics if hasattr(candidate, "diagnostics") else {}
        inc = incumbent.diagnostics if hasattr(incumbent, "diagnostics") else {}
        for role in ("exploration", "gate", "retention"):
            c = (cand.get(role) or {}).get("mean_corr")
            i = (inc.get(role) or {}).get("mean_corr")
            if (isinstance(c, (int, float)) and isinstance(i, (int, float))
                    and c < i - self.corr_margin):
                return {"action": "REJECT",
                        "reasons": ["margin_regressed:" + role],
                        "deltas": deltas}
        return {"action": "PROMOTE",
                "reasons": ["efficiency_gain_bytes"],
                "deltas": deltas}
