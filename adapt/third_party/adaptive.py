"""Small, domain-neutral evidence and delegation interfaces.

This module coordinates specialists; it deliberately does not prescribe a learning
algorithm or turn a specialist's answer into a trusted training label.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from typing import Callable, Protocol


SUPPORTED = "SUPPORTED"
AMBIGUOUS = "AMBIGUOUS"
UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class Evidence:
    source: str
    locator: str
    quote: str
    facts: dict = field(default_factory=dict)

    def as_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))


@dataclass(frozen=True)
class EvidenceRequest:
    kind: str
    context: dict
    missing: tuple[str, ...] = ()
    candidates: tuple = ()

    def __post_init__(self):
        if (not isinstance(self.kind, str) or not self.kind or not isinstance(self.context, dict)
                or not isinstance(self.missing, (tuple, list))
                or any(not isinstance(item, str) or not item for item in self.missing)):
            raise ValueError("An evidence request needs a kind, context object, and named missing fields")

    def as_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))


@dataclass(frozen=True)
class Resolution:
    status: str
    value: object = None
    candidates: tuple = ()
    evidence: tuple[Evidence, ...] = ()
    missing: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self):
        if self.status not in (SUPPORTED, AMBIGUOUS, UNSUPPORTED):
            raise ValueError("Unknown resolution status")
        if self.status == SUPPORTED and (self.value is None or self.missing or not self.evidence):
            raise ValueError("A supported resolution needs a value, evidence, and no missing fields")
        if self.status != SUPPORTED and self.value is not None:
            raise ValueError("Unresolved candidates must not be exposed as an accepted value")
        if any(not isinstance(item, Evidence) or not item.source or not item.locator or not item.quote
               for item in self.evidence):
            raise ValueError("Evidence needs an attributable source, locator, and quotation")

    def as_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))


class EvidenceBroker:
    """Synchronous, bounded delegation with notifications and isolated records.

    Each top-level request has a fresh call budget. Handlers may delegate again,
    but cycles and excessive depth produce UNSUPPORTED rather than an infinite
    loop. Domain handlers remain responsible for validating their evidence.
    """

    def __init__(self, max_depth=3, max_calls=8):
        if type(max_depth) is not int or type(max_calls) is not int or min(max_depth, max_calls) < 1:
            raise ValueError("Delegation limits must be positive integers")
        self.max_depth, self.max_calls = max_depth, max_calls
        self.handlers: dict[str, Callable[[EvidenceRequest], Resolution]] = {}
        self.notifications: list[dict] = []
        self._subscribers: list[Callable[[dict], None]] = []
        self._active: list[str] = []
        self._calls = 0
        self._notifying = False

    def register(self, kind, handler):
        if not isinstance(kind, str) or not kind or not callable(handler):
            raise ValueError("Register a named callable specialist")
        if kind in self.handlers:
            raise ValueError("Specialist already registered")
        self.handlers[kind] = handler

    def subscribe(self, callback):
        if not callable(callback):
            raise ValueError("Notification subscriber must be callable")
        self._subscribers.append(callback)

    def _notify(self, event):
        self.notifications.append(deepcopy(event))
        del self.notifications[:-200]
        # Subscribers may dispatch another request. Record those events, but do
        # not recursively notify the same subscribers while they are executing.
        if not self._notifying:
            self._notifying = True
            try:
                for callback in tuple(self._subscribers):
                    try:
                        callback(deepcopy(event))
                    except Exception as error:
                        # Observability must not cancel a successful inference.
                        # Record the failure without recursively notifying it.
                        self.notifications.append({"event": "subscriber_error",
                                                   "request": deepcopy(event.get("request", {})),
                                                   "error": type(error).__name__})
                        del self.notifications[:-200]
            finally:
                self._notifying = False

    def resolve(self, request: EvidenceRequest) -> Resolution:
        if not isinstance(request, EvidenceRequest):
            raise TypeError("Expected an EvidenceRequest")
        if not self._active:
            self._calls = 0
        # The request identity deliberately ignores candidates: changing a guess
        # does not justify asking the same specialist for the same facts forever.
        identity = json.dumps([request.kind, request.context, request.missing],
                              sort_keys=True, allow_nan=False)
        if (identity in self._active or len(self._active) >= self.max_depth
                or self._calls >= self.max_calls):
            result = Resolution(UNSUPPORTED, missing=request.missing,
                                reason="Delegation cycle or request budget exhausted")
            self._notify({"event": "blocked", "request": request.as_dict(), "resolution": result.as_dict()})
            return result
        self._calls += 1
        self._active.append(identity)
        try:
            self._notify({"event": "request", "depth": len(self._active), "request": request.as_dict()})
            handler = self.handlers.get(request.kind)
            result = (handler(deepcopy(request)) if handler else
                      Resolution(UNSUPPORTED, missing=request.missing,
                                 reason="No specialist registered for " + request.kind))
            if not isinstance(result, Resolution):
                raise TypeError("Specialists must return a Resolution")
            self._notify({"event": "resolution", "depth": len(self._active),
                          "request": request.as_dict(), "resolution": result.as_dict()})
            return deepcopy(result)
        finally:
            self._active.pop()


class AdaptiveDomain(Protocol):
    def state(self) -> dict: ...
    def step(self) -> dict: ...
    def reset(self) -> dict: ...
    def evaluate(self) -> dict: ...
