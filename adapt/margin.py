#!/usr/bin/env python3
"""Margin-aware verdicts, domain-side (core untouched).

Binary correct/incorrect aliases "equal" with "better in unmeasured
ways" AND "millipoints worse" with "garbage". The margin rule adds
resolution: a candidate ties a case iff c >= seed_c - m, where m is
calibrated from SEED capability (never tuned to flip a named
verdict):

    m = max(seed_corrs) - min(seed_corrs)   over the evaluated cases

Rationale: differences smaller than the seed's own variation across
cases are below the instrument's resolution. Same formula for every
candidate, computed blind before outcomes are compared. Widening m to
manufacture a win = moving the bar = ruled out (NOTES 2026-09-22).

Composes with EfficiencyRule: ties-with-fewer-bytes promote, where
"tie" is now margin-defined. Upstream follow-up to the baseline-gate
PR, not a local hack — this module is the portable proposal form.
"""
import json
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
sys.path.insert(0, str(ADAPT / 'third_party'))


def calibrate_margin(seed_corrs):
    """Margin from seed variation. seed_corrs: per-case floats."""
    cs = [float(c) for c in seed_corrs]
    return max(cs) - min(cs)


def margin_pass(cand_c, seed_c, m):
    """Tie iff within margin below seed (never above: c > seed_c passes)."""
    return float(cand_c) >= float(seed_c) - m


def apply_margin(records, m=None):
    """records: {cfg: [(cid, stratum, corr)]} with a 'seed' key.
    Returns (m, {cfg: {stratum: tie_fraction}})."""
    seed = {cid: c for cid, _, c in records["seed"]}
    if m is None:
        m = calibrate_margin(list(seed.values()))
    out = {}
    for cfg, rows in records.items():
        cell = {}
        for cid, st, c in rows:
            cell.setdefault(st, []).append(margin_pass(c, seed[cid], m))
        out[cfg] = {st: sum(v) / len(v) for st, v in sorted(cell.items())}
    return m, out


def report_margin_table(records, cfgs_order=None):
    """Print per-stratum tie fractions under calibrated margin."""
    m, table = apply_margin(records)
    print(f"calibrated margin m={m:.6f} "
          f"(seed range over {len(records['seed'])} cases)", flush=True)
    order = cfgs_order or ["seed"] + [k for k in records if k != "seed"]
    sts = sorted({st for cfg in table.values() for st in cfg})
    print("stratum " + " ".join(f"{n:>8}" for n in order), flush=True)
    for st in sts:
        print(f"{st} " + " ".join(f"{table[c].get(st, float('nan')):8.2f}"
                                  for c in order), flush=True)
    return m, table


if __name__ == '__main__':
    rec = json.load(open(ADAPT / 'runs' / 'tap_study.json'))["records"]
    report_margin_table(rec)
