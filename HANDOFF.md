# Handoff: fully-geometric DAV2 + learned depth-graded student

Date: 2026-09-25. Branch: `experimental/adaptation-depth-styles`
(pushed). Authoritative detail: `adapt/NOTES.md` (running log, ~2400
lines), `adapt/CONVERGENCE_OPERATOR.md` (theory note v2), repo README.
This document is the map: what exists, where, what was learned, what
wasn't, and where the program goes next.

## Claim and scope (read first)

We replicate Depth Anything V2-Small **deterministically** (float
0.99999, integer paths bit-exact, C host tests green) AND we
distilled a depth-graded narrow-early student to **0.99788**
(Phase-E best, banked as drop-in phi npz + live webcam runner).
We do NOT possess DAV2's provenance: human training is
unrecoverable, ~25M teacher weights are baked replicas, and nothing
here exceeds the teacher. Where we overreached (unreachable bar,
dormant k-bug tables, resampling artifact, joint-codebook
ambiguity, phase-2-overclaim retracted as noise, clobbered
checkpoints, actmax insufficiency), NOTES records each correction,
not an edit.

Webcam float code is a DEMONSTRATION harness, not the model. The
model is the phi-encoded weights + integer datapath + C core +
banked student artifact.

## Branch structure

| Branch | Holds | Status |
|---|---|---|
| `main` | Geometric pipeline, integer datapath, C port, emitter v1, webcam demos | In sync with `origin/main` at `e7af766` |
| `experimental/adaptation-depth` | + adaptation_foundry scaffold (pushed) | Pushed |
| `experimental/adaptation-depth-styles` (HERE) | + everything below | Pushed per-session |
| Upstream `feature/baseline-gate` (adaptation_foundry) | baseline_gate + tests + margin proposal | Pushed, PR unopened (no gh CLI here) |

Rule: `main` never receives experiment code. Vendored
`adapt/third_party/` stays pristine. Commit locally often; NEVER
push/merge without explicit user approval (granted per-push this
program — reconfirm if resuming later).

## The ledger, part 1: replica anatomy (closed)

**Learned:** LUT EXP halving (544 kB); 125-byte head; search-about-
search (groups, margins, guards, bars).
**Confirmed (seed retained):** taps (tap-2 now scene-conditional,
not a tie — splits on 6 real strata), gains (depth-graded), 24/24
drops fail, 72/72 heads fail, readouts, fusion scales, direct/
analytic rejections, neck v4 (12/12 fail; margin gate caught 3
would-be false wins).
**Weight-pathway irreducibility:** codebook ~0.9988 near-miss
(saturates below bar), low-rank dead (full-rank spectra),
sparsity dead at 5%, sharing dead (MLP offline, attention probe
0.36).
**Closed threads:** integer attention in C (bit-exact 0/3072);
fixed-point sensor path (tokens 0.999986); real scenes (40 COCO +
oracle, 9→11 strata, then 79 scenes 12/16 with corners
documented); substrate experiment (verdicts substrate-invariant
at recalibrated bars); bigger co-search arena (hand 4/5 vs random
5/5 — style-content thesis falls; granularity wins, content ties);
upstream PR prepared + branch pushed.

## The ledger, part 2: distillation (closed except noted)

**Result:** depth-graded student (L0–2 @128–192, rest frozen),
0.9928 → 0.99788 across phases A→E; banked
(`weights/student_backbone_r192-phaseE0.99788.npz`, phi-parity
0.99846); live webcam 0.975–0.995 by scene; Phase-D certificate
7/31 ties under margin 0.00027.
**Division of labor (measured):** closed-form RRR init does ~0.99
(~40 scenes identify maps); gradients polish +0.004; random-init
stalls 0.77 (basin real). Few-shot designs, gradients fit.
**Three inversions:** robustness ≠ linearity ≠ correctability
(gain-robust L11 is most nonlinear; absorbable ≠ fixable).
**Failed with mechanisms:** codebook saturation, rank, sparsity,
sharing, unfreeze-late (0.99502), dark-aug v1/v2 (label flaw then
oscillation; dark accepted as limitation), full-width (0.99639 —
narrowing exonerated), surgical edits F1/F2/G + joint-bias +
mapsel (no post-hocוש scoped-retraining moves work; stationarity).
**Coreset finding:** random ties greedy; core is statistical
(~5–25 diverse scenes); poison autopsy corrected (no poison
scenes — conditioning knife-edge). Synth pools trail reals
early; heightfields sit middle.
**Inversion tool:** `adapt/invert.py` (live GUI); ceilings ~0.76
(norm ambiguity + attention); teacher/student diff 0.10, no
catastrophe. Actmax proper failed twice (method insufficient).

## Open threads (ordered, with verdicts where closed)

1. Dark limitation (ACCEPTED with mechanism: regime competition,
   overlap 0.92 same span; webcam dims track illumination).
2. Upstream PR click (branch pushed; PR body staged
   `/tmp/opencode/` — machine-local, recheck if gone).
3. C-assembly of banked student from wired kernels (follow-up,
   untracked here).
4. Holographic transfer: STOPPED (H3 null; dead channels 3%
   vs theory's 42%).
5. Convergence-operator note v2 (`adapt/CONVERGENCE_OPERATOR.md`):
   move alphabet, six refusals, trace null (reshape atomic),
   F* resting semantics. Audited green by
   `adapt/audit_operator.py` (ledger + live RRR-init gate +
   remainder registry with bootstrap CI95 bounds).
6. THE OPEN QUESTION (user-framed): what the teacher knows that
   the student can't learn — candidates ranked: (a) tail
   coverage [largest, testable], (b) pretraining trajectory
   [likely irreducible], (c) capacity-at-scale [open quadrant].
   Per-remainder quantitative bounds live in
   `adapt/remainder_registry.json`.

## Key files added this era

`adapt/student_grad.py` (env-flag training rig), `bank_student.py`,
`margin.py`, `neck_search.py`, `tap_study.py`, `layerfit_probe.py`,
`blockfit_probe.py`, `student_probe.py`, `substrate_probe.py`,
`fetch_real/holes/fitpool.py`, `gen_synthpool/heightfields.py`,
`coreset/verify_sets/phase2_gate/curves.py`, `autopsy.py`,
`surgical_edit.py`, `singular_gain.py`, `opcodes.py`,
`holographic_ablate.py`, `invert.py`, `actmax.py`,
`covconverge.py`, `synth_resid.py`, `certify.py`, `failmap.py`,
`audit_operator.py`, `quantify_remainders.py`, `merge_quant.py`,
`trace_analysis.py`, `student_webcam.py`, `CONVERGENCE_OPERATOR.md`,
`operator_ledger.json`, `remainder_registry.json`.
Checkpoints/weights/fixtures/runs: gitignored, local-only; full
copies in `~/dav2_weight_backup_20260925/` (sha256 manifest).

## How to resume

```bash
git checkout experimental/adaptation-depth-styles
python adapt/test_smoke.py            # fast, no torch
python adapt/audit_operator.py        # theory claims green (needs GPU ~10min)
cd c_port && make test               # C suite
python test_geometric_parity.py       # float replica
python student_webcam.py --frames 5 --compare-hf   # banked student live
# any checkpoint: student_webcam.py --ckpt adapt/runs/<name>.pt
# inversion microscope: python adapt/invert.py  (X quits)
```
First action in any session: `git status`, `git log --oneline -5`,
re-run smoke before believing anything. Machine-local paths
(`/tmp/opencode/*` logs, PR body) do not survive reboot — regenerate
as needed. GPU: single RTX 3090 Ti 24GB; serialize heavy jobs.
