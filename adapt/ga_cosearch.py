#!/usr/bin/env python3
"""Joint architecture+width co-search (level-2 arena).

One genotype spans BOTH DSLs: {"arch": {...7 keys...}, "width": {...}}.
Each candidate runs the float architecture harness (13 scenes, 0.999
bar) AND the integer width harness (13 scenes + probe, 0.99 bar):
27 cases total. Joint bytes = param bytes (drops matter) + table bytes
(widths matter) — both levers move one counter, which is the point.

Predeclared race: full marks (27/27) + bytes below joint-seed bytes,
flat (--mode flat, uniform per-key over all 11 keys) vs style
(--mode style, cross-DSL group crossover). Metric: unique evals to
first target-hit, same seeds. If style groups capture real
co-adaptation, style should win; single-move neighbor search cannot
make these jumps at all.

Audit discipline: both harnesses' audit scenes open ONCE, post-run, on
the best full-marks candidate. Report-only.

Run: python adapt/ga_cosearch.py [--pop 6] [--gens 8] [--seed N] [--mode flat|style]
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
import graduate as W  # noqa: E402
from run_search import load_shared as load_arch_shared  # noqa: E402
from run_search import load_fixtures as load_arch_fx  # noqa: E402
from depth_adapter import (  # noqa: E402
    DepthAdapter, validate_config as validate_arch, SEED_CONFIG as ARCH_SEED,
    _model_bytes)


def joint_seed():
    return S.validate_joint({"arch": dict(ARCH_SEED), "width": dict(W.SEED)})


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pop', type=int, default=6)
    ap.add_argument('--gens', type=int, default=8)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--mutprob', type=float, default=0.3,
                    help='per-child mutation probability (low so crossover '
                         'does the work; the comparison under test)')
    ap.add_argument('--mode', choices=("flat", "style"), default="flat")
    args = ap.parse_args()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device} mode={args.mode}", flush=True)
    shared_w, fx_w, probe = W.load_all(device)
    shared_a = load_arch_shared(device)
    arch_fx = load_arch_fx(include_audit=False)
    search_w = {r: fx_w[r] for r in ('exploration', 'gate', 'retention')}
    arch_adapter = DepthAdapter(shared_a, arch_fx)
    width_adapter = W.WidthAdapter(shared_w, search_w, probe)

    seed = joint_seed()
    seed_bytes = (_model_bytes(seed["arch"], None)
                  + W.table_bytes(seed["width"]))
    total_cases = 13 + 14  # arch scenes + width scenes + probe

    cache = {}  # joint id -> (arch_meas, width_meas, score, fps, means, vec)

    def evaluate(geno):
        key = S.joint_identifier(geno)
        if key not in cache:
            g = S.validate_joint(geno)
            aspec = TrialSpec("co-arch", dict(g["arch"]), "co-search arch")
            aart = arch_adapter.build(aspec)
            am = arch_adapter.evaluate(aart)
            wm = width_adapter.evaluate({"config": dict(g["width"])})
            ac = sum(1 for r in ("exploration", "gate", "retention")
                     for c in getattr(am, r).cases if c.correct)
            wc = sum(1 for r in ("exploration", "gate", "retention")
                     for c in getattr(wm, r).cases if c.correct)
            nbytes = _model_bytes(g["arch"], aart["direct"]) + W.table_bytes(g["width"])
            vec, means = [], []
            for m in (am, wm):
                for r in ("exploration", "gate", "retention"):
                    card = getattr(m, r)
                    vec.extend(1 if c.correct else 0 for c in card.cases)
                    means.append(m.diagnostics[r]["mean_corr"])
            rec = S.behavior_record(vec, means, nbytes)
            cache[key] = (am, wm, (ac + wc, nbytes),
                          S.sha_fingerprint(rec), S.zeta_fingerprint(rec),
                          dict(g["arch"]), dict(g["width"]))
        return cache[key]

    # Pre-flight: joint seed must hold BOTH bars, or the race is noise.
    sm, _, (sc, sb), _, _, _, _ = evaluate(seed)
    swa = W.WidthAdapter(shared_w, search_w, probe).evaluate(
        {"config": dict(seed["width"])})
    sok = sc == total_cases
    print(f"pre-flight joint seed: {sc}/{total_cases} bytes={sb} -> "
          f"{'PASS' if sok else 'FAIL'}", flush=True)
    if not sok:
        raise SystemExit("seed fails joint bar; calibrate before racing")

    rng = random.Random(args.seed)
    # Seeded start: the joint seed (known-good) + randoms. Crossover must
    # ASSEMBLE the win from here; random starts in this space never hold
    # both bars at once (measured: 0 holders across 6 random-start runs).
    pop = [seed] + [S.random_joint(rng) for _ in range(args.pop - 1)]
    first_hit, prev_front_hash = None, None
    t0 = time.perf_counter()
    for gen in range(args.gens):
        results = [evaluate(g) for g in pop]
        scores = [r[2] for r in results]
        front = S.pareto_front(scores)
        fsha, _ = S.front_fingerprint(
            [(pop[i], scores[i][0], scores[i][1]) for i in front], seed_bytes)
        stable = "STABLE" if fsha == prev_front_hash else "moving"
        prev_front_hash = fsha
        sigs = [r[4] for r in results]
        zds = [S.zeta_distance(sigs[i], sigs[j])
               for i in range(len(sigs)) for j in range(i + 1, len(sigs))]
        zmean = sum(zds) / len(zds) if zds else 0.0
        best = min(range(len(pop)), key=lambda i: (-scores[i][0], scores[i][1]))
        hit = any(scores[i][0] == total_cases and scores[i][1] < seed_bytes
                  for i in range(len(pop)))
        if hit and first_hit is None:
            first_hit = (gen, len(cache))
        print(f"gen {gen}: evals={len(cache)} front={len(front)} "
              f"fronthash={fsha[:8]}:{stable} zeta={zmean:.0f} "
              f"best={scores[best][0]}/{total_cases} "
              f"bytes={scores[best][1]}", flush=True)
        order = sorted(range(len(pop)),
                       key=lambda i: (-scores[i][0], scores[i][1]))
        elites = [pop[i] for i in order[:2]]
        children = []
        cross = (S.style_crossover_joint if args.mode == "style"
                 else S.crossover_joint_flat)
        while len(elites) + len(children) < args.pop:
            a, b = rng.sample(range(len(pop)), 2)
            pa = pop[a] if order.index(a) <= order.index(b) else pop[b]
            c, d = rng.sample(range(len(pop)), 2)
            pc = pop[c] if order.index(c) <= order.index(d) else pop[d]
            child = cross(rng, pa, pc)
            if rng.random() < args.mutprob:
                child = S.mutate_joint(rng, child)
            children.append(child)
        pop = elites + children

    dt = time.perf_counter() - t0
    results = [evaluate(g) for g in pop]
    scores = [r[2] for r in results]
    holders = [i for i in range(len(pop)) if scores[i][0] == total_cases]
    winners = [i for i in holders if scores[i][1] < seed_bytes]
    print(f"\n{len(holders)} full-marks, {len(winners)} under seed bytes "
          f"({seed_bytes}) in {dt:.0f}s", flush=True)
    if first_hit is not None:
        print(f"first target-hit: gen {first_hit[0]}, "
              f"{first_hit[1]} unique evals (mode={args.mode})")
    else:
        print(f"target never hit (mode={args.mode}, {len(cache)} unique evals)")
    print("COSEARCH:", "PASS" if winners else "FAIL")

    best = min(holders or range(len(pop)),
               key=lambda i: (scores[i][1], -scores[i][0]))
    audit_cfg = {"arch": pop[best]["arch"], "width": pop[best]["width"]}
    from graduate import _integer_depth, _corr
    from depth_adapter import run_pipeline as arch_run, corr as arch_corr
    import geo_int as G
    from lut_pareto import build_frac_table, build_exp_table
    wcfg = audit_cfg["width"]
    saved = (G.FRAC_CAP, G._FRAC_LUT, G.DMAX, G._EXP_LUT)
    try:
        G.FRAC_CAP = wcfg["frac_cap"]
        G._FRAC_LUT = build_frac_table(wcfg["frac_cap"])
        G.DMAX = wcfg["dmax"]
        G._EXP_LUT = build_exp_table(wcfg["exp_span"])
        add = G.build_add_lut(dmax=wcfg["dmax"])
        sub = G.build_sub_lut(dmax=wcfg["dmax"])
        print("audit (post-run, report-only):")
        for rgb, ref, cid in fx_w['audit']:
            pred = _integer_depth(shared_w, wcfg, rgb, add, sub)
            print(f"  int {cid}: corr={_corr(pred, ref):.5f}")
    finally:
        G.FRAC_CAP, G._FRAC_LUT, G.DMAX, G._EXP_LUT = saved
    for rgb, ref, cid in load_arch_fx(include_audit=True)['audit']:
        pred = arch_run(shared_a, audit_cfg["arch"], [rgb])[0]
        print(f"  arch {cid}: corr={arch_corr(pred, ref):.5f}")
    runs = ADAPT / 'runs'
    runs.mkdir(exist_ok=True)
    (runs / f'cosearch_{args.mode}_{args.seed}.json').write_text(json.dumps(
        {"mode": args.mode, "winners": bool(winners),
         "first_hit": first_hit, "best": audit_cfg,
         "unique_evals": len(cache), "seconds": dt}, indent=1, default=str))
    print(f"wrote adapt/runs/cosearch_{args.mode}_{args.seed}.json")
    return 0 if winners else 1


if __name__ == '__main__':
    raise SystemExit(main())
