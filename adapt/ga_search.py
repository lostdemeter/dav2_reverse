#!/usr/bin/env python3
"""GA calibration: rediscover the known efficiency win from random starts.

Population search over LUT-width genotypes (styles.py) using the SAME
evaluator the neighbor search used (WidthAdapter.evaluate: integer-head
stride-4 + softmax probe). The GA controller is domain-side code — the
vendored Experimenter is single-incumbent and the wrong shape here
(Echion's ga_adapter.py precedent: GA operators live in the adapter,
TrialSpec/Measurement/PromotionRule types carry the contracts).

PREDECLARED calibration bar: the final population must contain a
parity-holding config with exp_span==8 and fewer table bytes than seed
— i.e. the EXP-halving win the neighbor search found, rediscovered
without being handed it. frac is free under tree (proven), so the bar
does not pin frac.

Audit discipline: audit fixtures open ONCE, post-run, on the
best-bytes parity-holder only. Report-only.

Run: python adapt/ga_search.py [--pop 6] [--gens 8] [--seed 7]
"""
import json
import random
import sys
import time
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))
sys.path.insert(0, str(ADAPT / 'third_party'))

from experimenter import TrialSpec  # noqa: E402 contract types only
import styles as S  # noqa: E402
import graduate as W  # noqa: E402 evaluator + DSL + byte model


def score_of(meas):
    """(total_correct, bytes, per-case vector, mean corrs) from a Measurement."""
    correct, means, vec = 0, [], []
    for role in ("exploration", "gate", "retention"):
        card = getattr(meas, role)
        correct += sum(1 for c in card.cases if c.correct)
        vec.extend(1 if c.correct else 0 for c in card.cases)
        means.append(meas.diagnostics[role]["mean_corr"])
    return correct, meas.model_bytes, tuple(vec), tuple(means)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pop', type=int, default=6)
    ap.add_argument('--gens', type=int, default=8)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    shared, fixtures, probe = W.load_all(device)
    search_fx = {r: fixtures[r] for r in ('exploration', 'gate', 'retention')}
    adapter = W.WidthAdapter(shared, search_fx, probe)
    seed_bytes = W.table_bytes(dict(W.SEED))
    total_cases = sum(len(search_fx[r]) for r in search_fx) + 1  # +softmax probe

    rng = random.Random(args.seed)
    pop = [S.random_genotype(rng) for _ in range(args.pop)]
    cache = {}  # identifier -> (meas, score, fingerprints)

    def evaluate(cfg):
        spec = TrialSpec("ga", dict(cfg), "ga candidate")
        key = spec.identifier
        if key not in cache:
            art = adapter.build(spec)
            meas = adapter.evaluate(art)
            ncorr, nbytes, vec, means = score_of(meas)
            rec = S.behavior_record(vec, means, nbytes)
            cache[key] = (meas, (ncorr, nbytes),
                          S.sha_fingerprint(rec), S.zeta_fingerprint(rec), dict(cfg))
        return cache[key]

    def rank_key(item):
        """Lower is better: fewer Pareto dominators, then fewer bytes."""
        _, score, _, _, _ = item
        return score[1]

    t0 = time.perf_counter()
    for gen in range(args.gens):
        results = [evaluate(cfg) for cfg in pop]
        scores = [r[1] for r in results]
        front = S.pareto_front(scores)
        zds = []
        sigs = [r[3] for r in results]
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                zds.append(S.zeta_distance(sigs[i], sigs[j]))
        zmean = sum(zds) / len(zds) if zds else 0.0
        best = min(range(len(pop)), key=lambda i: (-scores[i][0], scores[i][1]))
        print(f"gen {gen}: evals={len(cache)} front={len(front)} "
              f"mean_zeta_dist={zmean:.0f} "
              f"best=correct {scores[best][0]}/{total_cases} "
              f"bytes={scores[best][1]} cfg={pop[best]}", flush=True)
        # elitist next generation: pareto front + tournament children
        # Feasibility-first ordering (correctness, then bytes) — mirrors the
        # EfficiencyRule philosophy. Pure Pareto rank let tiny-broken configs
        # dominate the front on bytes alone (round-one lesson, kept in NOTES).
        order = sorted(range(len(pop)),
                       key=lambda i: (-scores[i][0], scores[i][1]))
        elites = [pop[i] for i in order[:2]]
        children = []
        while len(elites) + len(children) < args.pop:
            a, b = rng.sample(range(len(pop)), 2)
            wa = order.index(a) <= order.index(b)
            pa, pb = (pop[a], pop[b]) if wa else (pop[b], pop[a])
            c, d = rng.sample(range(len(pop)), 2)
            wc = order.index(c) <= order.index(d)
            pc = pop[c] if wc else pop[d]
            child = S.crossover(rng, pa, pc)
            child = S.mutate(rng, child)  # always mutate: 48-config space
            children.append(child)        # needs churn, not convergence
        pop = elites + children

    dt = time.perf_counter() - t0
    results = [evaluate(cfg) for cfg in pop]
    scores = [r[1] for r in results]
    holders = [i for i in range(len(pop))
               if scores[i][0] == total_cases]
    winners = [i for i in holders
               if pop[i]["exp_span"] == 8 and scores[i][1] < seed_bytes]
    print(f"\n{len(holders)} parity-holders, {len(winners)} meet the bar "
          f"(exp8 + fewer bytes than seed {seed_bytes}) in {dt:.0f}s")
    ok = bool(winners)
    print("CALIBRATION:", "PASS" if ok else "FAIL")
    best = min(holders or range(len(pop)),
               key=lambda i: (scores[i][1], -scores[i][0]))

    # sealed audit, once, on the best-bytes parity-holder (or best effort)
    audit = fixtures['audit']
    from graduate import _integer_depth, _corr
    import geo_int as G
    from lut_pareto import build_frac_table, build_exp_table
    cfg = pop[best]
    saved = (G.FRAC_CAP, G._FRAC_LUT, G.DMAX, G._EXP_LUT)
    try:
        G.FRAC_CAP = cfg["frac_cap"]
        G._FRAC_LUT = build_frac_table(cfg["frac_cap"])
        G.DMAX = cfg["dmax"]
        G._EXP_LUT = build_exp_table(cfg["exp_span"])
        add = G.build_add_lut(dmax=cfg["dmax"])
        sub = G.build_sub_lut(dmax=cfg["dmax"])
        res = []
        for rgb, ref, cid in audit:
            pred = _integer_depth(shared, cfg, rgb, add, sub)
            res.append((cid, _corr(pred, ref)))
    finally:
        G.FRAC_CAP, G._FRAC_LUT, G.DMAX, G._EXP_LUT = saved
    print("audit (post-run, report-only):")
    for cid, c in res:
        print(f"  {cid}: corr={c:.5f}")
    runs = ADAPT / 'runs'
    runs.mkdir(exist_ok=True)
    (runs / 'ga_calibration.json').write_text(json.dumps(
        {"bar": "exp8 parity-holder under seed bytes", "pass": ok,
         "best": pop[best], "audit": res, "seconds": dt,
         "unique_evals": len(cache)}, indent=1, default=str))
    print("wrote adapt/runs/ga_calibration.json")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
