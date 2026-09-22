# adapt — experimental depth domain for the Adaptation Foundry

Can the foundry's propose/build/evaluate/promote machinery *learn* parts
of Depth Anything V2's process from a minimal primitive set? This
directory is that experiment. It lives on the `experimental/adaptation-depth`
branch and does not touch `main` or the upstream foundry repo.

## Boundary (mirrors the foundry's own)

| Layer | Lives here | Must not |
|---|---|---|
| Foundry core (evidence, experiment control) | `third_party/` (pristine vendor) | gain depth branches — propose upstream instead |
| Depth domain (primitives, learner, evaluator) | `depth_adapter.py`, fixtures | bypass the core's gates; see gate/audit data during search |
| Demo driver | `run_search.py` | leak audit fixtures into the adapter |

## The search space (TrialSpec DSL)

```json
{"readout": 3, "res_scales": [0, 0, 0, 0], "head": "geo-conv",
 "taps": [3, 6, 9, 12], "tap_gains": [0, 0, 0, 0]}
```

- `readout` 0..3 — which fused neck stage feeds the head (3 = exact replication).
- `res_scales` — four ints in [-2, 2]: phi-exponent scalings (`×φ^e`) of the
  four fusion residual branches (all 0 = exact replication).
- `head` — `geo-conv` (full geometric head), `direct` (64→1 linear fitted
  by least squares on exploration data, the `fit_weights.py` idea at 64ch),
  or `analytic` (zero-weight edge/texture/perspective head — the
  no-learning limit probe; correctly rejected on all gates).
- `taps` — which backbone ViT layers feed the neck's four reassemble
  stages (bands per position, strictly increasing; seed is HF's 3/6/9/12).
- `tap_gains` — four ints in [-1, 1]: `×φ^e` gain per tapped stage
  (free as an exponent add in the integer datapath).
- `block_gains` — 24 ints in [-2, 2] (attn0,mlp0,attn1,mlp1,…): learned
  backbone mixing ratios. Searched to exhaustion (77 trials): all moves
  rejected, with measured sensitivity decreasing with depth (L0-attn×φ
  → 0.47, L11-mlp×φ → 0.994). See NOTES.
- `dropped` — sorted unique block ids 0..23 to skip (residual-only).
  Searched under EfficiencyRule with honest param-byte accounting
  (`python adapt/drop_search.py`): all 24 singles + best pairs rejected
  at parity. Ranked dispensability is non-monotonic (L11-mlp 0.953 best,
  L5-mlp 0.264 worst). No deletable blocks — reported, not hidden.

Seed = exact replication. `propose()` enumerates single moves
(readout±1, one scale ±1, head flip) with rationale; the core's
`PromotionRule` (utility, no gate regression, no retention loss, 64MiB
cap) decides. The reference oracle is the HF model, hidden from the
learner exactly as the foundry roadmap asks for.

## Run

```bash
python adapt/build_fixtures.py   # one-time: HF oracle renders refs (needs baked weights + HF)
python adapt/test_smoke.py       # fast: DSL + neighbor enumeration, no torch
python adapt/run_search.py       # architecture search (GPU recommended)
python adapt/drop_search.py      # block-drop search under EfficiencyRule (leaner-model attempts)
python adapt/graduate.py         # LUT-width search under EfficiencyRule
python adapt/ga_search.py        # GA over widths: --mode flat|style (level-2 front fingerprints)
python adapt/emit_c.py           # emit C model from sealed incumbent (bit-exact test)
```

Level-2 (styles of styles): `adapt/styles.py` holds genotype ops,
sha/zeta fingerprints, and front fingerprints (member hashes +
hypervolume, measurement-grounded only). `--mode style` swaps uniform
per-key crossover for group crossover (G1={frac,dmax} /
G2={exp,accum} travel as units). Measured verdict on the 48-config
width space: no mode beats luck (all hits gen 0-1) — the discriminating
arena is the joint architecture+width co-search. See NOTES.

Co-search (`python adapt/ga_cosearch.py [--mode flat|style]`) — full
writeup in the level-2 section above; `--mate random|phase` selects
second-parent choice (uniform vs phase-neighborhood). Measured:
phase mating 0/3 vs random 2/3 at exploit 0.7 — over-exploitation,
kept as a parameter point.

Co-search (`python adapt/ga_cosearch.py [--mode flat|style]`): one
genotype spans both DSLs (7 arch + 4 width keys), one byte counter
(param bytes + table bytes), 27/27 joint bar, seeded starts. 3 seeds
per arm: style hits at 13/16 evals (2/3), flat once late (~20, 1/3).
Winners are cross-DSL assemblies (intact arch + exp8 tables) — the
move single-key mixing rarely builds. Suggestive (n=3), mechanism
visible. Joints audited on both harnesses post-run.

Geometric proposer (`adapt/styles.py`: `phase_mate_select`,
`learn_groups`): mating by phase position; regrouping from co-success
MI with converged keys reported separately. On 107 histories it
re-derived 3/4 hand groups and correctly refused to group constant
dmax. Learned-vs-hand group race: 2/4 tie each (loop closed
mechanically, no coronation).

Self-dissolving structures (`--regroup K`): groups re-derived from
measured co-success every K gens, adopted on difference with MI
evidence logged. `adapt/bloom.py` (resonant bloom, Echion addressing
math) observes visited space in constant memory: fn=0 throughout,
est tracks exact. Race 3 seeds: static 2/3 hits, regroup 1/3 —
dissolution on ~11-sample histories churns good structure (gen-2
regroups shatter into singletons). Lesson: dissolution needs a
minimum-evidence guard; strong pairs ([readout,taps],
[res_scales,tap_gains]) survive most events regardless.
`--regroup-min-n` / `--regroup-mi-floor` implement the guard
(`should_regroup`, smoke-tested): guarded 2/3 (13, 16), refusals
logged; min-n 8 fires a real adoption at gen 11 without breaking
anything. Thresholds on measured evidence, all the way down.

Fixtures (`adapt/fixtures/`, gitignored): 6 exploration + 4 gate scenes,
3 retention anchors (gradient/checker/disc), 4 sealed audit scenes, all
168px with HF reference depths. Audit fixtures are never passed to the
adapter; they open once, post-seal, for the report.

## Coverage matrix (`python adapt/strata.py`)

Scene-stratified rule coverage: 4 oracle-free stats median-split into
16 strata; candidate panel of 5; per-stratum foundry Assessments;
obligations for empty/split cells; saturation read off the table.
First run: 9/16 strata hit, 7 SUPPORTED, 18/20 held-out prediction
PASS — both misses localized to cliff stratum E1T1L0V1. See NOTES.

## Accumulation shootout (`python geo_int.py`, informational section)

Four traditions accumulate the same dots on our K=512 weights:

| Method | Single-add err | 32-wide head | Integer-only? |
|---|---|---|---|
| Tree LUT-adds (ours) | ~2e-4 | 0.9985–0.9998 | yes |
| Fixed-point bridge (ours) | — | 0.99924 | yes |
| Fibonacci exact (phi_lattice §11) | ~1e-11 | 0.999984 | yes (Python ints stand in for multi-limb) |
| Taylor series (phi_geist, 6 terms) | 2–9% | 0.85 random / −0.18 real | **no** (float correction at runtime) |

Fibonacci wins on accuracy (exact accumulation, single solve) at the
cost of unbounded ints (firmware needs the theory's multi-limb solver).
Taylor is table-free but its 2–9% single-add error compounds to garbage
over 32-deep chains — evidence against table-free for this workload.

## Graduation experiment (`python adapt/graduate.py`)

Known-answer test for the efficiency objective: seed full tables,
search LUT widths (+accumulator) with `EfficiencyRule`, expect the
Pareto 576 kB point. Run 2026-09-21: trials 3–4 PROMOTED
(`efficiency_gain_bytes`) — then OVERSHOT to {2048,8,4096} (552 kB),
because real features pass 0.999 where Pareto's random features read
0.977. Audit shows the price (0.9948–0.9979 vs seed's 0.9988–1.0).
Rejections all correct: exp→4 dies on the softmax probe, dmax→1024 and
accum→fixed both lose retention-0. Details + library lessons in
`adapt/NOTES.md`.

Round two (same day): anchor mining over 40 fresh scenes found all
seed-vs-narrow gaps exactly 0.00000 — proving frac_cap unobservable
under tree evaluation (no-op dimension). Fix: `table_bytes` counts only
tables the accum path reads. Re-run lands **{13312,8,4096} tree,
557,068 B**: EXP halving promotes (probe passes), frac correctly
unpromotable, dmax→1024 and fixed correctly rejected. Margin gate
(`corr_margin` 2e-4) armed but unexercised — reported, not hidden.

## LUT-width Pareto (`python adapt/lut_pareto.py`)

FRAC_CAP × EXP span × ADD/SUB DMAX vs head/conv/softmax corr + shippable
int32 LUT bytes. Knees: DMAX 1024 kills the head (0.999999→0.9816, the
target-mean tail matters); FRAC 4096 first cracks convs (1.0→0.99713);
FRAC 2048 collapses (0.79); EXP span 8 is free (halves tables, zero
loss), span 4 costs softmax (1.0→0.9972). Minimal all-green:
**FRAC 8192 + EXP 8 + DMAX 4096 ≈ 576 kB** (head 0.999999, conv 1.0,
softmax 1.0).
