"""Domain-neutral checked claims, constraints, and an evidence acquisition agenda.

Checked means a domain validator accepted the extraction and scope, not that the
real world or the source is infallible. Application policy decides what to do
with an unresolved assessment; this module never silently chooses a fallback.
"""

from copy import deepcopy
from collections import deque
from dataclasses import asdict, dataclass
from itertools import islice, product
import json
import math
from typing import Callable

from adaptive import AMBIGUOUS, SUPPORTED, UNSUPPORTED, Evidence, EvidenceRequest, Resolution


def _record(value):
    return json.loads(json.dumps(asdict(value), allow_nan=False))


def _evidence(items):
    if any(not isinstance(item, Evidence) or not item.source or not item.locator or not item.quote
           for item in items):
        raise ValueError("Claims require attributable evidence records")


@dataclass(frozen=True)
class Claim:
    identifier: str
    field: str
    value: object
    evidence: tuple[Evidence, ...] = ()
    checked: bool = False

    def __post_init__(self):
        if (not isinstance(self.identifier, str) or not self.identifier
                or not isinstance(self.field, str) or not self.field or type(self.checked) is not bool):
            raise ValueError("A claim needs an identifier, field, and explicit check flag")
        _evidence(self.evidence)
        if self.checked and (self.value is None or not self.evidence):
            raise ValueError("Checked claims need a value and evidence")
        json.dumps(self.value, allow_nan=False)

    def as_dict(self):
        return _record(self)


@dataclass(frozen=True)
class Supersession:
    new: str
    old: str
    evidence: tuple[Evidence, ...] = ()
    checked: bool = False

    def __post_init__(self):
        if (not isinstance(self.new, str) or not self.new or not isinstance(self.old, str)
                or not self.old or type(self.checked) is not bool):
            raise ValueError("Supersession needs two claim identifiers and an explicit check flag")
        _evidence(self.evidence)
        if self.checked and not self.evidence:
            raise ValueError("Checked supersession needs evidence")

    def as_dict(self):
        return _record(self)


@dataclass(frozen=True)
class Obligation:
    field: str
    kind: str
    detail: str
    candidates: tuple = ()
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self):
        if any(not isinstance(value, str) or not value for value in (self.field, self.kind, self.detail)):
            raise ValueError("An obligation needs a field, kind, and explanation")
        _evidence(self.evidence)

    def as_dict(self):
        return _record(self)


@dataclass(frozen=True)
class Constraint:
    name: str
    predicate: Callable[[dict], bool | None]
    detail: str

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name or not isinstance(self.detail, str)
                or not self.detail or not callable(self.predicate)):
            raise ValueError("A constraint needs a name, explanation, and callable")


@dataclass(frozen=True)
class Assessment:
    resolution: Resolution
    obligations: tuple[Obligation, ...]
    slots: tuple[dict, ...]
    claims: tuple[dict, ...]
    supersessions: tuple[dict, ...]
    constraints: tuple[dict, ...]
    assumptions: tuple[str, ...]
    candidate_count: int
    truncated: bool

    def as_dict(self):
        return _record(self)


def assess(required, claims, *, supersessions=(), obligations=(), constraints=(),
           assumptions=(), candidate_limit=64):
    """Combine explicitly checked statements without inventing source precedence.

    Constraints can reject candidate mappings. They cannot silently erase source
    conflicts, unknown semantics, or an unchecked supersession assertion.
    """
    required = tuple(required)
    if (not required or len(set(required)) != len(required)
            or any(not isinstance(field, str) or not field for field in required)
            or type(candidate_limit) is not int or candidate_limit < 1):
        raise ValueError("Supply unique required fields and a positive candidate limit")
    claims = tuple(sorted(claims, key=lambda claim: claim.identifier))
    supersessions = tuple(sorted(supersessions, key=lambda item: json.dumps(item.as_dict(), sort_keys=True)))
    assumptions = tuple(assumptions)
    by_id = {claim.identifier: claim for claim in claims}
    if len(by_id) != len(claims):
        raise ValueError("Claim identifiers must be unique")
    pending = list(obligations)
    if any(not isinstance(item, Obligation) for item in pending):
        raise TypeError("Expected structured obligations")
    explanatory_evidence = [item for obligation in pending for item in obligation.evidence]
    if any(not isinstance(item, str) or not item for item in assumptions):
        raise ValueError("Assumptions must be named explicitly")
    edges, relations, applicable = {}, [], []
    for relation in supersessions:
        newer, older = by_id.get(relation.new), by_id.get(relation.old)
        valid = (relation.checked and newer is not None and older is not None
                 and newer.checked and older.checked and newer.field == older.field)
        relations.append({**relation.as_dict(), "applicable": bool(valid)})
        if valid:
            edges.setdefault(relation.new, set()).add(relation.old)
            applicable.append(relation)
        else:
            pending.append(Obligation(newer.field if newer else "supersession", "unverified",
                                      "Supersession is unchecked, lacks an endpoint, or crosses fields",
                                      (relation.new, relation.old), relation.evidence))
    nodes = set(edges) | {target for targets in edges.values() for target in targets}
    degrees = dict.fromkeys(nodes, 0)
    for targets in edges.values():
        for target in targets:
            degrees[target] += 1
    queue = deque(sorted(identifier for identifier, degree in degrees.items() if degree == 0))
    visited = 0
    while queue:
        identifier = queue.popleft()
        visited += 1
        for target in sorted(edges.get(identifier, ())):
            degrees[target] -= 1
            if degrees[target] == 0:
                queue.append(target)
    cycle = visited != len(nodes)
    retired = set() if cycle else {target for targets in edges.values() for target in targets}
    if cycle:
        pending.append(Obligation("supersession", "conflict", "Supersession graph contains a cycle"))
        for relation in relations:
            relation["applicable"] = False
    options = {field: {} for field in required}
    proof = explanatory_evidence
    for claim in claims:
        if not claim.checked:
            pending.append(Obligation(claim.field, "unverified", "Claim semantics or scope have not been independently checked",
                                      (claim.value,), claim.evidence))
            proof.extend(claim.evidence)
        elif claim.identifier not in retired and claim.field in options:
            key = json.dumps(claim.value, sort_keys=True, allow_nan=False)
            options[claim.field].setdefault(key, {"value": claim.value, "claims": []})["claims"].append(claim.identifier)
            proof.extend(claim.evidence)
    if not cycle:
        for relation in applicable:
            proof.extend(relation.evidence)
            proof.extend(by_id[relation.new].evidence)
            proof.extend(by_id[relation.old].evidence)
    slots = []
    for field, values in options.items():
        entries = [values[key] for key in sorted(values)]
        slots.append({"field": field, "values": [entry["value"] for entry in entries],
                      "claim_ids": [identifier for entry in entries for identifier in entry["claims"]],
                      "status": "missing" if not entries else "conflict" if len(entries) > 1 else "unique"})
        if not entries:
            pending.append(Obligation(field, "missing", "No checked statement supplies this required field"))
        elif len(entries) > 1:
            pending.append(Obligation(field, "conflict", "Applicable checked sources disagree; no implicit winner",
                                      tuple(entry["value"] for entry in entries),
                                      tuple(item for entry in entries for identifier in entry["claims"] for item in by_id[identifier].evidence)))
    available = [slot for slot in slots if slot["values"]]
    count = math.prod(len(slot["values"]) for slot in available) if available else 0
    truncated = count > candidate_limit
    if truncated:
        pending.append(Obligation("candidates", "budget", "Candidate enumeration limit reached"))
    candidates = [dict(zip((slot["field"] for slot in available), values))
                  for values in islice(product(*(slot["values"] for slot in available)), candidate_limit)] if available else []
    checks = []
    complete = len(available) == len(required)
    for constraint in constraints:
        results, kept = [], []
        for candidate in candidates:
            if not complete:
                verdict = None
            else:
                try:
                    verdict = constraint.predicate(deepcopy(candidate))
                except Exception:
                    verdict = None
            results.append(verdict if type(verdict) is bool else None)
            if verdict is not False:
                kept.append(candidate)
            if complete and type(verdict) is not bool:
                pending.append(Obligation("constraint:" + constraint.name, "unverified", constraint.detail,
                                          (deepcopy(candidate),)))
        checks.append({"name": constraint.name, "detail": constraint.detail, "verdicts": results})
        candidates = kept
    if complete and count and not candidates and not truncated:
        pending.append(Obligation("constraints", "inconsistent", "No candidate satisfies the supplied constraints"))
    unique_pending = {json.dumps(item.as_dict(), sort_keys=True): item for item in pending}
    pending = list(unique_pending.values())
    unique_proof = {json.dumps(item.as_dict(), sort_keys=True): item for item in proof}
    proof = tuple(unique_proof[key] for key in sorted(unique_proof))
    if complete and len(candidates) == 1 and not pending and proof:
        result = Resolution(SUPPORTED, candidates[0], evidence=proof,
                            reason="One interpretation satisfies checked claims and constraints under the stated assumptions")
    else:
        status = AMBIGUOUS if any(item.kind == "conflict" for item in pending) else UNSUPPORTED
        result = Resolution(status, candidates=tuple(candidates), evidence=proof,
                            missing=tuple(dict.fromkeys(item.field for item in pending)),
                            reason="Unresolved obligations remain; no accepted interpretation")
    return deepcopy(Assessment(result, tuple(pending), tuple(slots),
                       tuple({**claim.as_dict(), "superseded": claim.identifier in retired} for claim in claims),
                      tuple(relations), tuple(checks), assumptions, count, truncated))


@dataclass(frozen=True)
class EvidenceProvider:
    kind: str
    supplies: tuple[str, ...]
    requires: tuple[str, ...] = ()

    def __post_init__(self):
        if (not isinstance(self.kind, str) or not self.kind
                or not isinstance(self.supplies, (tuple, list)) or not self.supplies
                or not isinstance(self.requires, (tuple, list))
                or any(not isinstance(field, str) or not field for field in (*self.supplies, *self.requires))):
            raise ValueError("A provider needs a name and explicit field sequences")


class EvidenceAgenda:
    """Ask each relevant specialist at most once per investigation, in supplied order."""

    def __init__(self, providers, max_requests=6):
        self.providers = tuple(providers)
        if (type(max_requests) is not int or max_requests < 1
                or len({p.kind for p in self.providers}) != len(self.providers)
                or any(not p.kind or not p.supplies for p in self.providers)):
            raise ValueError("Supply uniquely named providers and a positive request budget")
        self.max_requests = max_requests
        self.attempted = []

    def next_request(self, assessment, context):
        if assessment.resolution.status == SUPPORTED or len(self.attempted) >= self.max_requests:
            return None
        missing = tuple(dict.fromkeys(item.field for item in assessment.obligations))
        for provider in self.providers:
            wanted = tuple(field for field in missing if field in provider.supplies)
            if (wanted and provider.kind not in self.attempted
                    and all(key in context and context[key] is not None for key in provider.requires)):
                return EvidenceRequest(provider.kind, deepcopy(context), wanted,
                                       deepcopy(assessment.resolution.candidates))
        return None

    def record(self, request):
        if (request.kind not in {p.kind for p in self.providers} or request.kind in self.attempted
                or len(self.attempted) >= self.max_requests):
            raise ValueError("Request is unregistered, already attempted, or over budget")
        self.attempted.append(request.kind)

    def stop_reason(self, assessment, context):
        if assessment.resolution.status == SUPPORTED:
            return "requirements_satisfied"
        if self.next_request(assessment, context) is not None:
            return "provider_available"
        return "request_budget_exhausted" if len(self.attempted) >= self.max_requests else "no_relevant_untried_provider"
