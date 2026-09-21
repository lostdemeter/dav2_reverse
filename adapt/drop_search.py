#!/usr/bin/env python3
"""Block-drop search: find deletable backbone blocks (leaner model + proof).

Runs the architecture DSL under EfficiencyRule with honest param-byte
accounting (DepthAdapter size_mode='params'): a dropped block that holds
parity ties on accuracy at strictly fewer bytes -> PROMOTE with
efficiency_gain_bytes. Margin gate (2e-4) applies.

Run: python adapt/drop_search.py [--trials N]
"""
import json
import sys
import time
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))
sys.path.insert(0, str(ADAPT / 'third_party'))

from experimenter import Experimenter  # noqa: E402
from efficiency import EfficiencyRule  # noqa: E402
from run_search import load_shared, load_fixtures  # noqa: E402
from depth_adapter import DepthAdapter, run_pipeline, corr  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=60)
    ap.add_argument('--cap-mib', type=int, default=200,
                    help='EfficiencyRule size cap in MiB (model is ~100MB)')
    args = ap.parse_args()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    shared = load_shared(device)
    fixtures = load_fixtures(include_audit=False)
    adapter = DepthAdapter(shared, fixtures, size_mode='params')
    rule = EfficiencyRule(max_model_bytes=args.cap_mib * 1024 * 1024)
    lab = Experimenter(adapter, max_trials=args.trials, rule=rule)
    t0 = time.perf_counter()
    last = 0
    while not lab.done:
        st = lab.step()
        if len(st['history']) > last:
            last = len(st['history'])
            h = st['history'][-1]
            print(f"trial {h['trial']}: {h['proposal']['name']} -> "
                  f"{h['decision']['action']} {h['decision']['reasons']} "
                  f"{json.dumps(h['decision'].get('deltas', {}))}", flush=True)
    dt = time.perf_counter() - t0
    final = lab.model()
    print(f"\nstop: {lab.state()['stop_reason']} sealed: {lab.state()['sealed']}")
    print(f"incumbent: {json.dumps(final['config'])}")
    print(f"search time: {dt:.0f}s")

    audit = load_fixtures(include_audit=True)['audit']
    res = []
    fit = final.get('direct') if isinstance(final, dict) else None
    cfg = final['config'] if isinstance(final, dict) else final.config
    for rgb, ref, cid in audit:
        pred = run_pipeline(shared, cfg, [rgb], fit_explore=fit)[0]
        res.append((cid, corr(pred, ref)))
    print("audit (post-seal, report-only):")
    for cid, c in res:
        print(f"  {cid}: corr={c:.5f}")
    runs = ADAPT / 'runs'
    runs.mkdir(exist_ok=True)
    (runs / 'drop.json').write_text(json.dumps(
        {"incumbent": final, "audit": res, "seconds": dt}, indent=1, default=str))
    print("wrote adapt/runs/drop.json")


if __name__ == '__main__':
    main()
