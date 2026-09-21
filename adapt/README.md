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
{"readout": 3, "res_scales": [0, 0, 0, 0], "head": "geo-conv"}
```

- `readout` 0..3 — which fused neck stage feeds the head (3 = exact replication).
- `res_scales` — four ints in [-2, 2]: phi-exponent scalings (`×φ^e`) of the
  four fusion residual branches (all 0 = exact replication).
- `head` — `geo-conv` (full geometric head) or `direct` (64→1 linear fitted
  by least squares on exploration data, the `fit_weights.py` idea at 64ch).

Seed = exact replication. `propose()` enumerates single moves
(readout±1, one scale ±1, head flip) with rationale; the core's
`PromotionRule` (utility, no gate regression, no retention loss, 64MiB
cap) decides. The reference oracle is the HF model, hidden from the
learner exactly as the foundry roadmap asks for.

## Run

```bash
python adapt/build_fixtures.py   # one-time: HF oracle renders refs (needs baked weights + HF)
python adapt/test_smoke.py       # fast: DSL + neighbor enumeration, no torch
python adapt/run_search.py       # full search (GPU recommended)
```

Fixtures (`adapt/fixtures/`, gitignored): 6 exploration + 4 gate scenes,
3 retention anchors (gradient/checker/disc), 4 sealed audit scenes, all
168px with HF reference depths. Audit fixtures are never passed to the
adapter; they open once, post-seal, for the report.

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

## LUT-width Pareto (`python adapt/lut_pareto.py`)

FRAC_CAP × EXP span × ADD/SUB DMAX vs head/conv/softmax corr + shippable
int32 LUT bytes. Knees: DMAX 1024 kills the head (0.999999→0.9816, the
target-mean tail matters); FRAC 4096 first cracks convs (1.0→0.99713);
FRAC 2048 collapses (0.79); EXP span 8 is free (halves tables, zero
loss), span 4 costs softmax (1.0→0.9972). Minimal all-green:
**FRAC 8192 + EXP 8 + DMAX 4096 ≈ 576 kB** (head 0.999999, conv 1.0,
softmax 1.0).
