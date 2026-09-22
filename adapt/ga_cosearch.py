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
    ap.add_argument('--mate', choices=("random", "phase"), default="random",
                    help="second-parent choice: uniform random mating vs "
                         "phase-neighborhood mating (geometric proposer)")
    ap.add_argument('--groups', choices=("hand", "learned"), default="hand",
                    help="crossover unit decomposition: hand-drawn JOINT_GROUPS "
                         "vs machine-proposed LEARNED_GROUPS (style mode only)")
    ap.add_argument('--regroup', type=int, default=0, metavar="K",
                    help="self-dissolving structures: every K gens, re-derive "
                         "groups from measured co-success via learn_groups and "
                         "adopt them if they differ (0=off, fixed groups)")
    ap.add_argument('--regroup-min-n', type=int, default=20,
                    help="evidence guard: minimum top-half histories before "
                         "any dissolution is adopted")
    ap.add_argument('--regroup-mi-floor', type=float, default=0.05,
                    help="evidence guard: minimum best-pair MI for adoption")
    args = ap.parse_args()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device} mode={args.mode} mate={args.mate} "
          f"groups={args.groups}", flush=True)
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
    from bloom import ResonantBloom
    visited = ResonantBloom(n_hashes=5, n_bits=2**16)
    for _cfg in pop:
        visited.add(S.joint_identifier(_cfg))
    cur_groups = ([tuple(g) for g in S.LEARNED_GROUPS]
                  if args.groups == "learned"
                  else [tuple(g) for g in S.JOINT_GROUPS])
    regroup_events = []
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
        # Bloom observer reads BEFORE this gen's children are added below,
        # so est tracks evaluated space; fn must stay 0 (observer proof).
        fn = sum(1 for k in cache if not visited.check(k))
        print(f"gen {gen}: evals={len(cache)} front={len(front)} "
              f"fronthash={fsha[:8]}:{stable} zeta={zmean:.0f} "
              f"bloom_est={visited.estimated_count():.0f} fn={fn} "
              f"best={scores[best][0]}/{total_cases} "
              f"bytes={scores[best][1]}", flush=True)
        order = sorted(range(len(pop)),
                       key=lambda i: (-scores[i][0], scores[i][1]))
        elites = [pop[i] for i in order[:2]]
        children = []
        grp = cur_groups
        if args.mode == "style":
            def cross(rng_, x, y):
                return S.style_crossover_joint(rng_, x, y, grp)
        else:
            cross = S.crossover_joint_flat
        while len(elites) + len(children) < args.pop:
            a, b = rng.sample(range(len(pop)), 2)
            ai = a if order.index(a) <= order.index(b) else b
            pa = pop[ai]
            if args.mate == "phase":
                pc = pop[S.phase_mate_select(rng, sigs, ai)]
            else:
                c, d = rng.sample(range(len(pop)), 2)
                pc = pop[c] if order.index(c) <= order.index(d) else pop[d]
            child = cross(rng, pa, pc)
            if rng.random() < args.mutprob:
                child = S.mutate_joint(rng, child)
            children.append(child)
            visited.add(S.joint_identifier(child))
        pop = elites + children
        # Self-dissolving structures with evidence guard: re-derive groups
        # from measured co-success; adopt ONLY if should_regroup passes
        # (min histories + MI floor + real difference). Skips are logged —
        # a refused dissolution is evidence, not silence.
        if args.regroup and (gen + 1) % args.regroup == 0:
            hist = [({**v[5], **v[6]}, v[2][0], v[2][1]) for v in cache.values()]
            prop = S.learn_groups(hist)
            adopt, why = S.should_regroup(
                prop, cur_groups, min_n=args.regroup_min_n,
                mi_floor=args.regroup_mi_floor)
            if adopt:
                regroup_events.append(
                    {"gen": gen, "from": [list(g) for g in cur_groups],
                     "to": [list(g) for g in prop["groups"]],
                     "mi": prop["top_mi_pairs"], "n_top": prop["n_top"],
                     "why": why})
                cur_groups = [tuple(g) for g in prop["groups"]]
                print(f"gen {gen}: REGROUP "
                      f"{regroup_events[-1]['from']} -> "
                      f"{regroup_events[-1]['to']} ({why})", flush=True)
            else:
                regroup_events.append(
                    {"gen": gen, "from": [list(g) for g in cur_groups],
                     "to": None, "mi": prop["top_mi_pairs"],
                     "n_top": prop["n_top"], "why": "skip: " + why})
                print(f"gen {gen}: regroup skipped ({why})", flush=True)

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
    hist = [{"geno": {**v[5], **v[6]}, "correct": v[2][0], "bytes": v[2][1]}
            for v in cache.values()]
    (runs / f'cohistory_{args.mode}_{args.mate}_{args.seed}.json').write_text(
        json.dumps(hist, indent=0, default=str))
    (runs / f'cosearch_{args.mode}_{args.mate}_{args.groups}_{args.seed}.json'
     ).write_text(json.dumps(
        {"mode": args.mode, "mate": args.mate, "groups": args.groups,
         "regroup_every": args.regroup, "regroup_events": regroup_events,
         "winners": bool(winners),
         "first_hit": first_hit, "best": audit_cfg,
         "unique_evals": len(cache), "seconds": dt}, indent=1, default=str))
    print(f"wrote adapt/runs/cosearch_{args.mode}_{args.mate}_{args.groups}_{args.seed}.json")
    return 0 if winners else 1


if __name__ == '__main__':
    raise SystemExit(main())
