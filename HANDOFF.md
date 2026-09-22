# Handoff: fully-geometric Depth Anything V2 + learning machinery

Date: 2026-09-21. Branch: `experimental/adaptation-depth-styles`.
Authoritative detail lives in `adapt/NOTES.md` (running log) and the
repo README. This document is the map: what exists, where, what was
learned, what wasn't, and where the program goes next.

## Claim and scope (read first)

We replicate Depth Anything V2-Small **deterministically** (float
0.99999, integer paths bit-exact, C host tests green) and understand
its anatomy in measured detail. We do NOT possess its provenance:
DAV2's human training is unrecoverable, ~25M weights are baked
replicas, and nothing here exceeds the teacher. Claims below respect
that boundary; where we overreached (twice), NOTES records the
correction, not an edit.

Webcam float code (`phi_depth.py`, `geo_webcam.py` float stage) is a
DEMONSTRATION harness, not the model — floats there are a display
choice. The model itself is the phi-encoded weights + integer
datapath + C core.

## Branch structure

| Branch | Holds | Status |
|---|---|---|
| `main` | Geometric pipeline, integer datapath, C port, emitter v1, webcam demos | In sync with `origin/main` at `e7af766` |
| `experimental/adaptation-depth` | + adaptation_foundry scaffold: architecture/width DSLs, EfficiencyRule, graduation, emitter, C fixed/attention kernels, backbone stage/scale/gain/drop search | Pushed |
| `experimental/adaptation-depth-styles` (HERE) | + styles/GA/fingerprints, co-search, baseline gates, coverage matrix, stratification | Local + pushed per-session (see `git log`) |

Rule: `main` never receives experiment code; experiments merge outward
only by explicit decision. Vendored foundry core (`adapt/third_party/`)
stays pristine; all deviations live domain-side with provenance notes.

## The ledger

**Learned (search/fitting under gates):** LUT EXP halving (544 kB
tables; promoted, rediscovered blind, re-derived); 125-byte head
weights (least-squares fit); search-about-search (groups, margins,
guards, bars).
**Confirmed (explored, seed retained):** taps (+tap-2 tie), tap/block
gains (depth-graded sensitivity), all 24 drops (dispensability ranking,
non-monotonic), all 72 heads (near-miss ranking), readouts, fusion
scales, direct/analytic heads (tie / all-gate rejection).
**Untouched:** ~25M weights (baked); neck topology; all objectives,
bars, fitnesses (human-set); independent capability (zero — oracle is
the model); 13/16 strata empty; integer attention unwired in C;
fixed-point sensor path unwired.

## Machinery inventory (`adapt/`)

- DSLs: architecture (7 keys) + widths (4 keys) + joint (11 keys).
- Controllers: neighbor search (`run_search`, `drop_search`,
  `graduate`), GA (`ga_search`: flat/style, random/phase mating),
  co-search (`ga_cosearch`: +groups hand/learned, +regroup guard).
- Gates: `EfficiencyRule` (+margin), `baseline.py` pre-flight,
  absolute trial lines.
- Instruments: `styles.py` (fingerprints incl. zeta + front hashes,
  MI group-learner), `bloom.py` (resonant observer), `strata.py`
  (coverage matrix + obligations + held-out prediction),
  `stratify.py` (weight-direct roles), `emit_c.py` (artifact→C,
  bit-exact).
- Proposals for upstream: `BASELINE_UPSTREAM.md`.
- C core (`c_port/`): tree/fixed/fib solvers, linear/norm/gelu,
  batch head, late-bound tables; `make test` all green.

## Open threads (ordered)

1. **Active identification via coverage holes (NEXT — agreed).**
   Don't collect scenes; inject parametric stimuli per hole, fit
   minimal per-stratum rules, substitute machinery with rules under
   gates. Goal: routed ensemble (profiler → simplest verified rule
   per stratum, full model as fallback). Guards: interpolate
   synthetic↔natural (probe validity); fallback never dissolves.
2. Empty strata (13/16) + ambiguous cells + 4-scene audit: real scenes.
3. Finer ablations (half-MLP, single-head) if a bar justifies them.
4. Integer attention composition + fixed-point sensor in C.
5. Upstream: baseline-gate PR (foundry), vendor update path (Echion).
6. The substrate question (lattice ≅ contracts?): needs the experiment
   designed, not more discussion — same-harness exact-vs-mush verdict
   comparison.

## How to resume

```bash
git checkout experimental/adaptation-depth-styles
python adapt/test_smoke.py            # fast, no torch
cd c_port && make test               # C suite
python test_geometric_parity.py       # float replica
python geo_int.py                     # integer suite (slow)
python adapt/strata.py --fresh 0      # coverage (reuses scenes; see file)
python learned_webcam.py              # live learned model (needs camera)
```
`adapt/runs/*.json` are gitignored records; `adapt/fixtures/*.npz`
rebuild via `adapt/build_fixtures.py` (+HF oracle). Baked
`weights/geometric_*.npz` rebuild via `export_geometric_weights.py`.
