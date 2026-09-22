#!/usr/bin/env python3
"""Scene-stratified coverage: strata x rules matrix with obligations.

Thesis under test: verdicts transfer within scene strata, not across
arbitrary scene sets. Strata come from ORACLE-FREE input statistics
(edge/texture/luminance/verticality), so assignment needs no ground
truth. Each stratum gets a foundry Assessment (claims per candidate
with measurement evidence; obligations for empty/split cells).
Saturation = every non-empty stratum decided, read off the table —
never asserted.

Predeclared prediction test: held-out audit scenes get verdicts
predicted from their stratum majority (ties -> ambiguous -> scored
wrong, conservative). Bar: >= 15/20 correct (4 scenes x 5 candidates).
If strata don't predict, they're decoration — stated upfront.

Run: python adapt/strata.py [--fresh 12]
Writes adapt/runs/strata.json (matrix, obligations, prediction score).
Fresh scenes + oracle refs go to adapt/fixtures/strata_extra.npz
(gitignored, reproducible via --fresh + HF oracle).
"""
import json
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))
sys.path.insert(0, str(ADAPT / 'third_party'))

import numpy as np

STATS = ("edge", "texture", "lumspread", "vertical")
EXTRA_PATH = ADAPT / 'fixtures' / 'strata_extra.npz'
REAL_PATH = ADAPT / 'fixtures' / 'strata_real.npz'  # real RGB + HF-oracle refs (provenance inside)

CANDIDATES = {
    "seed": {},
    "tap2": {"taps": [2, 6, 9, 12]},
    "direct": {"head": "direct"},
    "analytic": {"head": "analytic"},
    "drop23": {"dropped": [23]},
}


def profile(rgb):
    """Oracle-free input statistics. rgb float HxWx3 in [0,1]."""
    from scipy.ndimage import sobel, gaussian_filter
    gray = np.asanyarray(rgb, dtype=np.float64)[..., :3].mean(axis=-1)
    gx, gy = sobel(gray, axis=1), sobel(gray, axis=0)
    edge = float(np.sqrt(gx ** 2 + gy ** 2).mean() / (gray.max() - gray.min() + 1e-9))
    smooth = gaussian_filter(gray, 3.0)
    hi = float(np.abs(gray - smooth).mean() / (gray.std() + 1e-9))
    lum = float(gray.std())
    h = gray.shape[0]
    vert = float(gray[:h // 2].mean() - gray[h // 2:].mean())
    return {"edge": edge, "texture": hi, "lumspread": lum, "vertical": vert}


def assign(prof, medians):
    """Stratum id like E1T0L1V0 from median splits (recorded, reproducible)."""
    bits = [prof[s] >= medians[s] for s in STATS]
    return "".join(f"{s[0].upper()}{int(b)}" for s, b in zip(STATS, bits))


def build_matrix(shared, scenes, refs, ids):
    """Run candidate panel over scenes. Returns (profiles, medians, table)."""
    import copy
    from depth_adapter import run_pipeline, corr, SEED_CONFIG, fit_direct_head
    from depth_adapter import DepthAdapter
    profs = [profile(rgb) for rgb, _, _ in scenes]
    medians = {s: float(np.median([p[s] for p in profs])) for s in STATS}
    strata = [assign(p, medians) for p in profs]
    # direct head needs one exploration fit (its own scenes, not audit)
    fits = {}
    if any(v.get("head") == "direct" for v in CANDIDATES.values()):
        dummy = DepthAdapter(shared, [])  # unused; fit needs shared only
        _ = dummy
        explore = [(rgb, ref, cid) for (rgb, ref, cid) in zip(
            [s[0] for s in scenes], [s[1] for s in scenes], ids)][:6]
        base = copy.deepcopy(dict(SEED_CONFIG))
        base["head"] = "direct"
        fits["direct"] = fit_direct_head(shared, base, explore)
    table = {}  # stratum -> {cand: [(cid, corr, verdict)]}
    for (rgb, ref, cid), st in zip(scenes, strata):
        for name, delta in CANDIDATES.items():
            cfg = copy.deepcopy(dict(SEED_CONFIG))
            cfg.update(copy.deepcopy(delta))
            fit = fits.get("direct") if cfg["head"] == "direct" else None
            pred = run_pipeline(shared, cfg, [rgb], fit_explore=fit)[0]
            c = corr(pred, ref)
            table.setdefault(st, {}).setdefault(name, []).append((cid, c, c >= 0.999))
    return profs, medians, strata, table


def assess_stratum(st, cell):
    """Foundry Assessment for one stratum: claims per candidate + obligations.

    Decided cells (unanimous) enter as CHECKED claims and slot uniquely;
    split cells enter UNCHECKED (value 'ambiguous') which the machinery
    itself converts to 'unverified' obligations. Resolution SUPPORTED
    therefore means exactly: stratum fully decided. The contracts do the
    classifying; this function only translates measurements.
    """
    from evidence_state import Claim, Obligation, Evidence, assess
    claims, obligations = [], []
    for cand, rows in sorted(cell.items()):
        vals = [bool(v) for _, _, v in rows]
        ev = (Evidence(source="strata-matrix", locator=f"{st}/{cand}",
                       quote=f"{sum(vals)}/{len(vals)} pass",
                       facts={"corrs": [round(float(c), 5) for _, c, _ in rows]}),)
        if all(vals):
            claims.append(Claim(identifier=f"{st}:{cand}", field=cand,
                                value="holds", evidence=ev, checked=True))
        elif not any(vals):
            claims.append(Claim(identifier=f"{st}:{cand}", field=cand,
                                value="fails", evidence=ev, checked=True))
        else:
            claims.append(Claim(identifier=f"{st}:{cand}", field=cand,
                                value="ambiguous", evidence=ev, checked=False))
    res = assess(sorted(cell), claims, obligations=obligations)
    return res.as_dict()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--fresh', type=int, default=12)
    args = ap.parse_args()

    import torch
    from run_search import load_shared, load_fixtures
    import build_fixtures as bf
    from harden_anchors import oracle_refs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}", flush=True)
    shared = load_shared(device)
    base = load_fixtures(include_audit=False)
    scenes = ([(rgb, ref, cid) for role in ('exploration', 'gate', 'retention')
               for rgb, ref, cid in base[role]])
    if EXTRA_PATH.exists():
        z = np.load(EXTRA_PATH, allow_pickle=False)
        for i in range(len(z['rgb'])):
            scenes.append((z['rgb'][i].astype(np.float32) / 255.0,
                           z['ref'][i].astype(np.float64), f"extra-{i}"))
        print(f"loaded {len(z['rgb'])} extra scenes", flush=True)
    else:
        fresh = [bf.make_scene(500 + i) for i in range(args.fresh)]
        refs = oracle_refs(fresh)
        np.savez_compressed(
            EXTRA_PATH, rgb=np.stack([(s * 255).astype(np.uint8) for s in fresh]),
            ref=np.stack(refs))
        for i, (rgb, ref) in enumerate(zip(fresh, refs)):
            scenes.append((rgb, ref.astype(np.float64), f"extra-{i}"))
        print(f"built {args.fresh} fresh scenes -> {EXTRA_PATH}", flush=True)
    if REAL_PATH.exists():
        zr = np.load(REAL_PATH, allow_pickle=True)
        rids = [str(x) for x in zr['ids']] if 'ids' in zr else None
        for i in range(len(zr['rgb'])):
            cid = rids[i] if rids else f"real-{i}"
            scenes.append((zr['rgb'][i].astype(np.float32) / 255.0,
                           zr['ref'][i].astype(np.float64), cid))
        print(f"loaded {len(zr['rgb'])} REAL scenes "
              f"({zr['source'][0] if 'source' in zr else 'unknown source'})", flush=True)

    rgb_list = [s[0] for s in scenes]
    ref_list = [s[1] for s in scenes]
    ids = [s[2] for s in scenes]
    profs, medians, strata, table = build_matrix(
        shared, list(zip(rgb_list, ref_list, ids)), ref_list, ids)
    print(f"medians: {json.dumps(medians, indent=0)}", flush=True)
    print(f"strata hit: {sorted(table)} ({len(scenes)} scenes)", flush=True)

    assessments, all_obs = {}, []
    for st, cell in sorted(table.items()):
        res = assess_stratum(st, cell)
        assessments[st] = {
            "verdicts": {
                c: f"{sum(1 for _, _, v in rows if v)}/{len(rows)}"
                for c, rows in sorted(cell.items())},
            "resolution": res["resolution"],
            "obligations": [o for o in res["obligations"]]}
        all_obs.extend((st, o) for o in res["obligations"])
    # empty-strata obligations: strata space is 2^4; list unhit ones
    import itertools
    from evidence_state import Obligation
    for bits in itertools.product("01", repeat=4):
        st = "".join(f"{s[0].upper()}{b}" for s, b in zip(STATS, bits))
        if st not in table:
            ob = Obligation(field=st, kind="missing",
                            detail="stratum unhit; acquire scenes")
            assessments[st] = {"verdicts": {}, "resolution": {"status": "UNSUPPORTED"},
                               "obligations": [ob.as_dict()]}
            all_obs.append((st, ob.as_dict()))
    n_split = sum(1 for st, o in all_obs if o["kind"] == "unverified")
    n_empty = sum(1 for st, o in all_obs if o["kind"] == "missing"
                  and st not in table)
    print(f"obligations: {n_empty} empty strata, {n_split} split cells "
          f"(+{sum(1 for st, o in all_obs if o['kind'] == 'missing' and st in table)} "
          f"missing-kind from machinery)", flush=True)
    saturated = (n_empty == 0 and n_split == 0)
    print("SATURATED" if saturated else "NOT saturated", flush=True)

    # Held-out prediction test on sealed audit scenes.
    audit = load_fixtures(include_audit=True)['audit']
    majority = {}
    for st, cell in table.items():
        for cand, rows in cell.items():
            vals = [bool(v) for _, _, v in rows]
            majority[(st, cand)] = (sum(vals) > len(vals) / 2) if vals else None
    from depth_adapter import run_pipeline as arch_run, corr as arch_corr
    import copy
    from depth_adapter import SEED_CONFIG
    n_ok, n_tot = 0, 0
    pred_detail = []
    for rgb, ref, cid in audit:
        p = profile(rgb)
        st = assign(p, medians)
        for name, delta in CANDIDATES.items():
            cfg = copy.deepcopy(dict(SEED_CONFIG))
            cfg.update(copy.deepcopy(delta))
            fit = None
            if cfg["head"] == "direct":
                from depth_adapter import fit_direct_head
                base0 = copy.deepcopy(dict(SEED_CONFIG))
                base0["head"] = "direct"
                fit = fit_direct_head(shared, base0, list(zip(
                    rgb_list[:6], ref_list[:6], ids[:6])))
            pred = arch_run(shared, cfg, [rgb], fit_explore=fit)[0]
            actual = arch_corr(pred, ref) >= 0.999
            guess = majority.get((st, name))
            scored = (guess is not None) and (guess == actual)
            n_tot += 1
            n_ok += scored
            pred_detail.append((cid, st, name, guess, actual, scored))
    print(f"held-out prediction: {n_ok}/{n_tot} "
          f"({'PASS' if n_ok >= 15 else 'FAIL'}, bar 15/20)", flush=True)
    runs = ADAPT / 'runs'
    runs.mkdir(exist_ok=True)
    (runs / 'strata.json').write_text(json.dumps(
        {"medians": medians, "assessments": assessments,
         "obligations": all_obs, "saturated": saturated,
         "prediction": {"score": [n_ok, n_tot], "detail": pred_detail}},
        indent=1, default=str))
    print("wrote adapt/runs/strata.json")
    return 0 if n_ok >= 15 else 1


if __name__ == '__main__':
    raise SystemExit(main())
