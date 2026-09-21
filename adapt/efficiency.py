"""Efficiency objective for size-aware promotion (experimental).

The vendored PromotionRule cannot promote compression wins: ties on
accuracy reject with insufficient_new_correct_cases, so a candidate
with identical accuracy at half the bytes never promotes. EfficiencyRule
SUBCLASSES the base rule (the controller type-checks isinstance) and
adds exactly one documented promotion path: base rejects SOLELY for
no-correctness-gain (no regressions, no incomparability, size cap
satisfied) AND exploration didn't get worse AND the artifact is strictly
smaller. Core file untouched; the deviation lives here, at the decision
site, and every efficiency promotion says so in its reasons.
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
    def compare(self, candidate, incumbent=None):
        decision = super().compare(candidate, incumbent)
        if incumbent is None or decision["action"] != "REJECT":
            return decision
        reasons = set(decision["reasons"])
        deltas = decision.get("deltas", {})
        if (reasons and reasons <= _TIE_REASONS
                and deltas.get("exploration_correct", -1) >= 0
                and deltas.get("model_bytes", 0) < 0):
            return {"action": "PROMOTE",
                    "reasons": ["efficiency_gain_bytes"],
                    "deltas": deltas}
        return decision
