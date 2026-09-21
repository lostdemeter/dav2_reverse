"""Bounded, domain-neutral propose/build/evaluate/promote transactions.

The adapter owns candidate construction and the meaning of evaluation cases.
This controller owns isolation, comparable scorecards, declared promotion gates,
budgets, and an auditable incumbent history. Audit data is deliberately absent.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from typing import Protocol


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class TrialSpec:
    name: str
    config: dict
    rationale: str

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name or not isinstance(self.rationale, str)
                or not self.rationale or not isinstance(self.config, dict)):
            raise ValueError("A proposal requires a name, configuration, and rationale")
        object.__setattr__(self, "config", _json(self.config))

    @property
    def identifier(self):
        return hashlib.sha256(json.dumps(self.config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def as_dict(self):
        return {"id": self.identifier, **_json(asdict(self))}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    answered: bool
    correct: bool
    signature: str = ""

    def __post_init__(self):
        if (not isinstance(self.case_id, str) or not self.case_id
                or type(self.answered) is not bool or type(self.correct) is not bool
                or self.correct and not self.answered or not isinstance(self.signature, str)):
            raise ValueError("Case outcomes need an ID and consistent boolean outcomes")


@dataclass(frozen=True)
class Scorecard:
    dataset: str
    cases: tuple[CaseResult, ...]

    def __post_init__(self):
        object.__setattr__(self, "cases", tuple(self.cases))
        if (not isinstance(self.dataset, str) or not self.dataset or not self.cases
                or any(not isinstance(case, CaseResult) for case in self.cases)
                or len({case.case_id for case in self.cases}) != len(self.cases)):
            raise ValueError("A scorecard requires a dataset identity and unique, nonempty cases")

    def counts(self):
        answered = sum(case.answered for case in self.cases)
        correct = sum(case.correct for case in self.cases)
        return {"total": len(self.cases), "answered": answered, "correct": correct,
                "wrong": answered - correct, "abstained": len(self.cases) - answered}


@dataclass(frozen=True)
class Measurement:
    exploration: Scorecard
    gate: Scorecard
    retention: Scorecard
    model_bytes: int
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if (any(not isinstance(card, Scorecard) for card in (self.exploration, self.gate, self.retention))
                or type(self.model_bytes) is not int or self.model_bytes < 0
                or not isinstance(self.diagnostics, dict)):
            raise ValueError("Measurement requires three scorecards and a nonnegative artifact size")
        object.__setattr__(self, "diagnostics", _json(self.diagnostics))

    def as_dict(self):
        result = _json(asdict(self))
        for name in ("exploration", "gate", "retention"):
            result[name]["counts"] = getattr(self, name).counts()
        return result


@dataclass(frozen=True)
class PromotionRule:
    correct_reward: int = 3
    wrong_penalty: int = 5
    min_correct_gain: int = 1
    max_model_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("Promotion parameters must be positive integers")

    def utility(self, scorecard):
        counts = scorecard.counts()
        return counts["correct"] * self.correct_reward - counts["wrong"] * self.wrong_penalty

    def compare(self, candidate, incumbent=None):
        reasons, deltas = [], {}
        if candidate.model_bytes > self.max_model_bytes:
            reasons.append("artifact_size_limit")
        if incumbent is not None:
            for name in ("exploration", "gate", "retention"):
                before, after = getattr(incumbent, name), getattr(candidate, name)
                if before.dataset != after.dataset or {c.case_id for c in before.cases} != {c.case_id for c in after.cases}:
                    reasons.append("incomparable_" + name)
            if not any(reason.startswith("incomparable_") for reason in reasons):
                old_search, new_search = incumbent.exploration.counts(), candidate.exploration.counts()
                old_gate, new_gate = incumbent.gate.counts(), candidate.gate.counts()
                retained = {case.case_id: case for case in candidate.retention.cases}
                lost = sorted(case.case_id for case in incumbent.retention.cases
                              if case.correct and not retained[case.case_id].correct)
                deltas = {"exploration_correct": new_search["correct"] - old_search["correct"],
                          "exploration_utility": self.utility(candidate.exploration) - self.utility(incumbent.exploration),
                          "gate_correct": new_gate["correct"] - old_gate["correct"],
                          "gate_utility": self.utility(candidate.gate) - self.utility(incumbent.gate),
                          "lost_retention": lost,
                          "model_bytes": candidate.model_bytes - incumbent.model_bytes}
                if deltas["exploration_correct"] < self.min_correct_gain:
                    reasons.append("insufficient_new_correct_cases")
                if deltas["exploration_utility"] <= 0:
                    reasons.append("exploration_utility_not_improved")
                if deltas["gate_correct"] < 0:
                    reasons.append("gate_correctness_regressed")
                if deltas["gate_utility"] < 0:
                    reasons.append("gate_utility_regressed")
                if lost:
                    reasons.append("protected_cases_regressed")
        return {"action": "REJECT" if reasons else "BASELINE" if incumbent is None else "PROMOTE",
                "reasons": reasons or ["baseline_measured" if incumbent is None else "all_development_gates_passed"],
                "deltas": deltas}


class ExperimentAdapter(Protocol):
    def seed(self) -> TrialSpec: ...
    def propose(self, incumbent: TrialSpec, measurement: Measurement, history: list[dict], tried: tuple[str, ...]) -> TrialSpec | None: ...
    def build(self, proposal: TrialSpec) -> object: ...
    def evaluate(self, artifact: object) -> Measurement: ...
    def fingerprint(self, artifact: object) -> str: ...


class Experimenter:
    def __init__(self, adapter: ExperimentAdapter, max_trials=10, rule=None):
        if type(max_trials) is not int or max_trials < 1:
            raise ValueError("Trial budget must be positive")
        if any(not callable(getattr(adapter, name, None)) for name in ("seed", "propose", "build", "evaluate", "fingerprint")):
            raise TypeError("Adapter must implement the complete experiment interface")
        if rule is not None and not isinstance(rule, PromotionRule):
            raise TypeError("Expected a PromotionRule")
        self.adapter, self.max_trials = adapter, max_trials
        self.rule = PromotionRule() if rule is None else rule
        self.stage, self.tick = "propose", 0
        self.done = self.sealed = False
        self.stop_reason = None
        self.proposal = self.measurement = None
        self._candidate = self._incumbent = None
        self._incumbent_spec = self._incumbent_measurement = None
        self._candidate_digest = self._incumbent_digest = None
        self._accepted = []
        self.tried, self.history, self.rollbacks = [], [], []
        self.resources = {}
        self.error = None
        self._trial_spec = self._evaluated_measurement = None

    def model(self, baseline=False):
        if not self._accepted:
            return None
        return deepcopy(self._accepted[0][1] if baseline else self._incumbent)

    def _finish_trial(self, decision):
        self.proposal = deepcopy(self._trial_spec)
        before = self._incumbent_spec.identifier if self._incumbent_spec else None
        if decision["action"] in ("BASELINE", "PROMOTE"):
            self._incumbent = self._candidate
            self._incumbent_spec = deepcopy(self.proposal)
            self._incumbent_measurement = deepcopy(self.measurement)
            self._incumbent_digest = self._candidate_digest
            self._accepted.append((self._incumbent_spec, self._candidate, self._incumbent_measurement, self._candidate_digest))
        self.history.append({"trial": len(self.history) + 1, "proposal": self.proposal.as_dict(),
                             "measurement": self.measurement.as_dict() if self.measurement else None,
                             "decision": deepcopy(decision), "incumbent_before": before,
                             "incumbent_after": self._incumbent_spec.identifier if self._incumbent_spec else None,
                             "artifact_digest": self._candidate_digest,
                             "resources": deepcopy(self.resources)})
        self._candidate = None
        if self._incumbent is None:
            self.done, self.stage, self.stop_reason = True, "done", "baseline_failed"
        elif len(self.history) >= self.max_trials:
            self.stage, self.stop_reason = "seal", "trial_budget_exhausted"
        else:
            self.stage = "propose"

    def step(self):
        if self.done:
            return self.state()
        self.tick += 1
        if self.stage == "propose":
            try:
                proposal = (self.adapter.seed() if self._incumbent is None else self.adapter.propose(
                    deepcopy(self._incumbent_spec), deepcopy(self._incumbent_measurement), deepcopy(self.history), tuple(self.tried)))
                if proposal is not None and not isinstance(proposal, TrialSpec):
                    raise TypeError("Adapter must propose a TrialSpec or None")
            except Exception as error:
                self.error = type(error).__name__ + ": " + str(error)
                self.stop_reason = "proposal_failed"
                self.done = self._incumbent is None
                self.stage = "done" if self.done else "seal"
                return self.state()
            if proposal is None:
                self.stage, self.stop_reason = "seal", "proposal_space_exhausted"
            elif proposal.identifier in self.tried:
                self.stage, self.stop_reason = "seal", "duplicate_proposal"
            else:
                self.proposal = deepcopy(proposal)
                self._trial_spec = deepcopy(proposal)
                self.tried.append(proposal.identifier)
                self.measurement = self._candidate_digest = None
                self._evaluated_measurement = None
                self.resources, self.stage = {}, "build"
        elif self.stage in ("build", "evaluate"):
            current_stage = self.stage
            started = time.perf_counter()
            try:
                if current_stage == "build":
                    if self.proposal.as_dict() != self._trial_spec.as_dict():
                        raise ValueError("Proposal changed after it was recorded")
                    self._candidate = deepcopy(self.adapter.build(deepcopy(self._trial_spec)))
                    if self._candidate is None:
                        raise ValueError("None is not an experiment artifact")
                    self._candidate_digest = self.adapter.fingerprint(self._candidate)
                    if not isinstance(self._candidate_digest, str) or not self._candidate_digest:
                        raise ValueError("Artifact fingerprint must be a nonempty string")
                    self.stage = "evaluate"
                else:
                    working = deepcopy(self._candidate)
                    measurement = self.adapter.evaluate(working)
                    if not isinstance(measurement, Measurement):
                        raise TypeError("Evaluator must return a Measurement")
                    if self.adapter.fingerprint(working) != self._candidate_digest:
                        raise ValueError("Evaluation mutated the artifact")
                    self.measurement, self.stage = deepcopy(measurement), "decide"
                    self._evaluated_measurement = deepcopy(measurement)
            except Exception as error:
                self.resources[current_stage + "_seconds"] = time.perf_counter() - started
                self._finish_trial({"action": "REJECT", "reasons": [current_stage + "_failed"],
                                    "error": type(error).__name__ + ": " + str(error), "deltas": {}})
            else:
                self.resources[current_stage + "_seconds"] = time.perf_counter() - started
        elif self.stage == "decide":
            try:
                if (not isinstance(self.measurement, Measurement) or self.measurement != self._evaluated_measurement
                        or self.proposal.as_dict() != self._trial_spec.as_dict()
                        or self.adapter.fingerprint(self._candidate) != self._candidate_digest):
                    raise ValueError("Candidate, proposal, or captured measurement changed before promotion")
                decision = self.rule.compare(self._evaluated_measurement, self._incumbent_measurement)
            except Exception as error:
                decision = {"action": "REJECT", "reasons": ["transaction_integrity_failed"],
                            "error": type(error).__name__ + ": " + str(error), "deltas": {}}
                self.proposal = deepcopy(self._trial_spec)
                self.measurement = deepcopy(self._evaluated_measurement)
            self._finish_trial(decision)
        elif self.stage == "seal":
            try:
                intact = self._incumbent is not None and self.adapter.fingerprint(self._incumbent) == self._incumbent_digest
            except Exception as error:
                intact = False
                self.error = type(error).__name__ + ": " + str(error)
            if not intact:
                self.stop_reason = "incumbent_integrity_failed"
            else:
                self.sealed = True
            self.done, self.stage = True, "done"
        return self.state()

    def rollback(self):
        if self.sealed or self.done or self.stage != "propose" or len(self._accepted) < 2:
            raise ValueError("Rollback requires an unsealed promotion boundary and an earlier incumbent")
        previous = self._accepted[-2]
        if self.adapter.fingerprint(previous[1]) != previous[3]:
            raise ValueError("Prior incumbent integrity failed")
        discarded = self._accepted.pop()
        self._incumbent_spec, self._incumbent, self._incumbent_measurement, self._incumbent_digest = previous
        self.tick += 1
        self.rollbacks.append({"from": discarded[0].identifier, "to": previous[0].identifier,
                               "reason": "caller_requested_rollback", "tick": self.tick})
        return self.state()

    def state(self):
        return deepcopy({"tick": self.tick, "stage": self.stage, "done": self.done, "sealed": self.sealed,
                         "stop_reason": self.stop_reason, "max_trials": self.max_trials,
                         "proposal": self.proposal.as_dict() if self.proposal else None,
                         "measurement": self.measurement.as_dict() if self.measurement else None,
                         "incumbent": {"proposal": self._incumbent_spec.as_dict(),
                                       "measurement": self._incumbent_measurement.as_dict(),
                                       "artifact_digest": self._incumbent_digest} if self._incumbent_spec else None,
                         "history": self.history, "tried": self.tried, "rollbacks": self.rollbacks,
                         "rule": asdict(self.rule), "resources": self.resources, "error": self.error})
