#!/usr/bin/env python3
"""Baseline pre-flight gate (domain-side; vendored core stays pristine).

The failure mode it prevents: a bar above seed capability turns every
verdict into noise with correct-looking reasons (NOTES 2026-09-21
correction). Nobody noticed because trial lines printed DELTAS only
(0 looks the same for 14->14 and 2->2).

Two pieces:
  check_baseline(): build+evaluate the seed BEFORE any search loop.
    Refuses (raises BaselineFailed) unless the caller-supplied gate
    passes. Re-evaluates to assert determinism; nondeterminism is
    itself a finding. The later Experimenter re-runs the baseline as
    trial 1 (~seconds, accepted duplication that keeps core semantics
    identical, tried-set and audit intact).
  fmt_counts(): absolute correct/total per role for trial lines, so a
    human always sees the floor and the ceiling, not just the delta.

Upstream direction (see BASELINE_UPSTREAM.md): this belongs in the
foundry core as an optional Experimenter gate (the stop reason
`baseline_failed` already exists); here it lives domain-side so the
vendored copy is untouched.
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
sys.path.insert(0, str(ADAPT / 'third_party'))  # explicit vendored-core path


class BaselineFailed(Exception):
    """Seed does not pass its own gate: searching would measure noise."""


def fmt_counts(meas):
    """'e6/7 g4/4 r3/3' from a measurement as_dict() (has counts)."""
    parts = []
    for role, short in (("exploration", "e"), ("gate", "g"), ("retention", "r")):
        c = meas[role]["counts"]
        parts.append(f"{short}{c['correct']}/{c['total']}")
    return " ".join(parts)


def parity_gate(bar, roles=("exploration", "gate", "retention")):
    """Gate: every case correct in each role AND each role mean >= bar.

    Returns predicate over measurement as_dicts -> (ok, reason)."""
    def gate(meas):
        for role in roles:
            counts = meas[role]["counts"]
            if counts["correct"] != counts["total"]:
                return False, f"{role} {counts['correct']}/{counts['total']} below bar"
            diags = meas.get("diagnostics", {})
            mean = (diags.get(role) or {}).get("mean_corr")
            if isinstance(mean, (int, float)) and mean < bar:
                return False, f"{role} mean {mean:.5f} below {bar}"
        return True, f"all roles correct, means >= {bar}"
    return gate


def check_baseline(adapter, gate, label="seed"):
    """Pre-flight the seed through build+evaluate+fingerprint.

    Returns the baseline measurement as_dict. Raises BaselineFailed with
    the evidence attached. Verifies determinism by evaluating twice.
    """
    seed = adapter.seed()
    first = adapter.build(seed)
    m1 = adapter.evaluate(first).as_dict()
    fp1 = adapter.fingerprint(first)
    second = adapter.evaluate(adapter.build(adapter.seed()))
    m2 = second.as_dict()
    if m1 != m2 or adapter.fingerprint(adapter.build(adapter.seed())) != fp1:
        raise BaselineFailed(
            f"{label} nondeterministic across identical evaluations; "
            "fix determinism before searching")
    ok, reason = gate(m1)
    print(f"baseline pre-flight [{label}]: {fmt_counts(m1)} -> "
          f"{'PASS' if ok else 'FAIL'} ({reason})", flush=True)
    if not ok:
        raise BaselineFailed(
            f"{label} fails its own gate ({reason}); refusing to search. "
            f"counts: {fmt_counts(m1)}. Calibrate the bar to seed capability first.")
    return m1
