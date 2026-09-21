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
