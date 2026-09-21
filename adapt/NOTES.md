# Adaptation-depth: running notes

Working log for the experimental branch. Kept because several findings
generalize to adaptation_foundry as a library (flagged LIBRARY below).

## 2026-09-21 — scaffold + first search

- The vendored core worked untouched: seed/baseline, neighbor proposals,
  build/evaluate/decide, seal, post-seal audit. No core changes needed
  to run a vision/integer domain. Good sign for generality.
- LIBRARY: strict JSON (`allow_nan=False`) turned a degenerate model
  output (NaN corr from constant predictions) into `evaluate_failed`.
  Fixed domain-side (degenerate scores as clean failures, corr=-1.0).
  Consider documenting the pattern: adapters must sanitize floats
  BEFORE constructing Measurements; the core is right not to accept NaN.
- First search: 13 trials to `proposal_space_exhausted`, zero promotions.
  Wrong readouts rejected with retention loss; scale moves tied; direct
  64→1 head tied everywhere (+552 bytes). Honest zero — matches the
  foundry's own recorded run character.

## 2026-09-21 — gates are blind to compression

- A candidate with identical accuracy at half the bytes gets REJECTED
  (`insufficient_new_correct_cases`): PromotionRule measures correctness,
  retention, and a size CAP — never size improvement. Ties can't promote.
- This is not an adapter bug; it's a missing objective dimension. The
  foundry roadmap anticipates it (Milestone 5: objective profiles, Pareto
  views of rejected-but-useful alternatives).
- LIBRARY: efficiency-shaped problems need an explicit objective profile.
  Response: `adapt/efficiency.py` — EfficiencyRule wraps PromotionRule
  untouched and promotes ONLY on (base rejects solely for no-correctness-
  gain) + (no regressions anywhere) + (strictly fewer bytes). All other
  rejections stand. The deviation is documented at the decision site.

## 2026-09-21 — graduation experiment design

- Known-answer test: seed full tables {13312,16,4096}, search width moves
  + accum flip, expect discovery of {8192,8,4096} (576 kB, from Pareto).
- No-op dimensions waste trials: accumulation choice was DELIBERATELY
  kept out of the architecture DSL because the float-pipeline evaluator
  can't see it. Lesson: every DSL dimension must be observable by the
  evaluator, or search burns budget on phantoms. The width DSL gets its
  own integer-path evaluator for exactly this reason.
- Evaluation policy (explicit, like Futhark's beam/radius): integer head
  on a stride-4 subsample of 518px features (130² grid), refs resized
  168→130 once in the harness (alignment interp is measurement, not
  artifact). Same grid for tree/fixed configs so case verdicts stay
  comparable across trials.
- Harness patching pattern: LUT widths live in geo_int module globals;
  evaluate() patches with save/restore (try/finally), single-threaded.
  The core's fingerprint/integrity checks cover the ARTIFACT (untouched);
  harness patching is documented, not hidden.

## 2026-09-21 — graduation experiment: first efficiency promotions

- Ran `adapt/graduate.py` (14-trial budget, sealed at 10 on exhaustion):
  trials 3 AND 4 PROMOTED via `efficiency_gain_bytes` — the first
  promotions on efficiency grounds in this lineage. The machinery
  terminates, seals, audits cleanly.
- Gate behavior verified by the rejections around them: exp_span→4
  fails on the softmax probe (exploration -1); dmax→1024 and
  accum→fixed both lose retention-0 (protected_cases_regressed) despite
  byte savings. The probe and the anchors do exactly their jobs.
- OVERSHOOT, reported honestly: incumbent landed at
  {2048,8,4096} (552 kB), past the Pareto-predicted {8192,8,4096}.
  Pareto used RANDOM features (frac2048 head: 0.977); real backbone
  features concentrate on lattice levels, so the same widths pass 0.999
  on fixtures. Audit shows the cost: 0.9948–0.9979 vs the seed's
  0.9988–1.0. The objective did what it says (ties + fewer bytes), but
  the 0.999 threshold is coarse enough to hide real degradation.
- LIBRARY: binary correct/incorrect verdicts alias "equal" with "better
  in unmeasured ways". Follow-ups: (a) harder retention anchors near
  the threshold cliff, (b) margin-aware gates (require corr >= max over
  incumbent per case, not a fixed bar), (c) Pareto-frontier reporting
  instead of single-incumbent when accuracy and bytes trade off.
  None of these change the core — all live in adapter/rule code.

## 2026-09-21 — learned head runs live on webcam

- `geo_webcam.py --int-head` loads `adapt/runs/graduation.json`, patches
  DMAX + head LUTs, runs float backbone/neck/convs for 32ch features,
  then IntegerPhiHead (JIT) per frame. Headless 3-frame test:
  int-vs-float 0.999929, HF parity 0.999456, ~4 FPS (vs ~100 float).
  Bottleneck is the per-pixel Python path around the JIT kernel
  (numpy log-encode + decode list-comp over 196k px), not the kernel
  itself (0.1 us/px). Vectorizing the boundary is the obvious speedup.
- THE K-BUG, caught by this test: `build_add_lut(dmax)` passes dmax
  positionally into the `k` slot (`(k=K, dmax=DMAX)`), silently building
  tables on the wrong grid — first run gave corr -0.08 (garbage).
  Dormancy analysis, verified: the misbuilt arrays were consumed ONLY
  by the webcam path (which assigns them onto the head instance).
  `graduate.py`/`lut_pareto.py` built the same misbuilt arrays but never
  read them — tree/fixed evaluation uses the head instance's own
  default-built LUTs, and dmax reaches evaluation through the `G.DMAX`
  global (clipping threshold), which was always patched correctly.
  Proof: graduation re-run after the fix is trial-for-trial identical
  (same promotions, same overshoot, same audit). Pareto's FRAC/EXP
  findings never touched add/sub at all. Fix applied at all 4 call
  sites (keyword `dmax=`); no published numbers change.
- LIBRARY: two lessons. (1) Dead parameters are silent: `_integer_depth`
  accepts `add, sub` and ignores them — the real dmax path is a module
  global. Plumb effectful parameters explicitly or drop them; an unused
  argument is a latent wrong-result bug, as demonstrated. (2) A live
  end-to-end test caught what the parity suites couldn't, because the
  suites never exercised the assignment path. New-behavior demos should
  run before results are written up, not after.

## 2026-09-21 — round two: honest bytes + margin gates + hard anchors

- Anchor mining (`adapt/harden_anchors.py`, 40 fresh scenes, seed vs
  narrowed integer-tree scores): ALL gaps exactly +0.00000, and no scene
  cleared the 0.9995 seed bar. This is not a failed mining run — it is
  the no-op diagnosis confirmed empirically: tree evaluation never
  reads FRAC tables, so frac_cap is unobservable by construction. Kept
  zero anchors; kept the script as the method record.
- Honesty fix (`table_bytes` now counts only tables the accum path
  reads; EXP counted in both as shared attention cost): frac moves under
  tree change 0 bytes and correctly can't promote. This is the ROADMAP
  principle "measure complete costs" applied to the artifact itself.
- Margin gate (`corr_margin` 2e-4 on per-role mean_corr diagnostics,
  base `__post_init__` overridden explicitly since it only allows ints):
  armed, 3 new smoke tests green — and NOT exercised this run (no
  `margin_regressed` verdicts), which is itself reported, not hidden.
- Re-run outcome: incumbent **{13312,8,4096} tree, 557,068 B** — the
  genuine win is EXP halving (probe still passes), frac correctly
  unpromotable, dmax→1024 rejected on retention-0, fixed rejected
  (+119,560 B, no gain). Audit 0.9948–0.9979 (same integer path as
  before, deterministic). No overshoot: the objective now sees what the
  evaluator sees.
## 2026-09-21 — C firmware on live webcam + leanness payoff verdict

- `phi_head_predict_batch` (one ctypes call/frame) + `geo_webcam.py
  --c-head`: live C integer head, bit-IDENTICAL to the Python int path
  on camera frames (corr 1.000000, maxabs 0.0 — not close, exact).
- Measured per-frame budget (196k px, CUDA): float stage 7ms steady;
  encode boundary (numpy log) 92ms; C batch 38ms; vectorized decode
  ~10ms. Python int-head path was ~210-455ms total. So the C kernel is
  ~12x faster than the Python/JIT head — and the remaining cost is the
  FLOAT encode boundary, not integer math. True embedded has no float
  encode either (fixed-point sensor path, still future work).
- LEANNESS VERDICT, honest: on host FPS, learned narrowness pays
  ~nothing for the tree head (ADD/SUB sizes unchanged at dmax 4096;
  EXP halving is untouched by this path). The payoff is footprint —
  544 kB vs 1.1 MB tables on flash — not speed. The speed levers were
  dropped blocks (none held parity) and the C kernel itself (12x, won
  above). Footprint is the right claim; FPS is not. Saying so plainly.

## 2026-09-21 — finer granularity: heads nearly hold, unification ships

- Head-ablation sweep (all 72: single-head zeroing, exploration corrs):
  best L1h0 0.99882 / L0h4 0.99873 — but mins 0.9976/0.9980, below the
  0.999 bar on every head. Verdict: finer granularity does NOT hold
  parity. Chasing it would mean moving the bar, which is ruled out.
  The near-miss ranking is kept as evidence (early-layer heads most
  redundant — same direction as the tap-2 tie, interestingly).
- So: emitter + C unification, per the agreement. `phi_int.h` now uses
  unsized externs + `#ifndef`-guarded caps (DMAX/FRAC_CAP/EXP_MAX);
  `make test` unchanged (defaults identical, ALL PASS). The emitter
  compiles per-model tables + caps against the unified core — the
  `model_head.c` duplication is DELETED, tree and fixed both 0/64
  bit-exact. Fixed expectations are computed under patched narrowed
  tables (graduate.evaluate's save/restore pattern), so they hold at
  any widths. Firmware-style `--gc-sections` linking throughout.
- LIBRARY: the unification pattern — core binds sizes late (unsized
  externs + guarded caps), emitter binds them per artifact, bit-exact
  tests per binding. Any adapter can copy this shape.
- Shippable statement, earned: `python adapt/emit_c.py` turns the
  sealed {13312,8,4096}-tree incumbent into 544 kB of tables + weights
  + a compiled, bit-exact-tested C model. Search → ship, no human in
  the middle.

## 2026-09-21 — both tracks: emitter scaffold + analytic candidate

- EMITTER (`adapt/emit_c.py`): sealed width artifact → narrowed
  ADD/SUB/EXP tables + head (sign,exp) vectors + size-aware head loop +
  manifest, compiled against the 64px vectors: **0/64 bit-exact, EMIT:
  ALL PASS**, 544 kB emitted for the {13312,8,4096} incumbent. The
  size-aware `phi_add` copy is labeled as honest duplication with the
  unification path named (parameterize `phi_int.c` by table size).
  Emission refuses non-width configs and non-tree accums loudly —
  scaffolds should fail closed.
- ANALYTIC CANDIDATE (architecture DSL +1 move: zero-weight
  edge/texture/perspective head, 0 fitted bytes): tried trial 3,
  rejected on EVERY gate (exploration -6, gate -4, all retention lost).
  The gates treat a qualitatively different candidate correctly — big
  savings can't buy failed accuracy. Seed space now 13 neighbors.
- LIBRARY (for the eventual upstream note): this pair is the
  "search → ship" loop in miniature — the same JSON the gates judged
  is the JSON the emitter compiles. The foundry never sees C; the
  domain owns the backend. That separation is what makes the pattern
  reusable: any adapter with a serializable artifact + a kernel
  registry gets deployment for free.

## 2026-09-21 — fixed-bridge C + backbone kernels (all bit-exact)

- New `fixed_nn.{h,c}`: fixed_dot, rescale, head_fixed, linear,
  layernorm (+isqrt), GELU-vec. `make test` → **C FIXED-NN: ALL PASS**
  with 0 mismatches everywhere incl. 1536 layernorm outputs at real
  384 width; head_fixed corr 0.999851 vs float refs.
- The one real trap, handled explicitly: Python `//`/`>>` floor toward
  −inf, C truncates. Every negative-capable site uses floor_div /
  floor_div128 / Newton isqrt; sums wrap mod 2^64 via uint64_t
  (identical to numpy int64); head dot uses __int128 (matches Python's
  unbounded ints at these sizes). Bounds documented in the header.
- Debugging note: the first fixed_dot run failed 7/8 — and the C code
  was RIGHT. The generator had flattened all cases into one max-scale
  group while the kernel (correctly) uses per-dot max. Generators must
  call the exact function under test (`fixed_dot_terms` on (T,1)), not
  a hand-rolled equivalent. Same lesson as the k-bug: test the path,
  not a lookalike.
- Emitter now covers fixed too (`EMIT-FIXED: ALL PASS`, 0/64): full
  FRAC/COARSE/FINE + ADD/SUB/EXP as PHI_* definitions linking in place
  of luts.c, tested via fixed_nn.c's phi_head_fixed. Narrowed fixed
  linking stays open (kernels bind full-size externs) — manifest
  records narrowed targets vs actuals explicitly.
- On "fully learned (head, neck, backbone)": current learned surface
  is widths + neck/head architecture; backbone weights remain baked
  replicas. The C kernels above are the prerequisite either way
  (a learned backbone still executes through linear/norm/gelu). Next
  learning step proposed: backbone stage/scale search in the DSL —
  not started; saying so plainly rather than implying it.

## 2026-09-21 — backbone stage/scale search lands (taps + tap_gains)

- DSL v2 adds `taps` (which ViT layers feed the neck, bands per
  position + strict increase so pyramid roles stay sane) and
  `tap_gains` (per-stage `×φ^e`, free as an exponent add in
  integer-land). `forward_stages` takes optional taps; default path
  verified bit-identical. Seed space grows 13 → 28 neighbors.
- 29-trial run to `proposal_space_exhausted`, incumbent retained
  (exact replication including taps 3/6/9/12). The backbone interface
  is genuinely sensitive: tap 12→11 collapses everything (retention
  lost), 9→8 regresses the gate, 3→4 loses exploration — but tap
  3→2 TIES everywhere (no gate/retention damage). A real lead for
  follow-up: is layer 2 as good as 3 across harder scenes, or a
  fixture-resolution artifact? Not claimed either way here.
- Gains: mostly ties or small losses; stage-3 gain+1 regresses the
  gate. Zero promotions overall — correct, since ties with unchanged
  bytes can't promote under the base rule.
- LIBRARY (measurement hygiene): `model_bytes` deltas of ±3 in this
  run are pickle-encoding noise from small-int values, not real size
  changes. Byte deltas near zero should be treated as zero before any
  efficiency reasoning touches them — a tolerance the EfficiencyRule
  doesn't have yet. Logged as the next rule refinement.
- RESOLVED (drop search): `_model_bytes` now excludes pickle framing
  entirely (params + fit bytes only), so the tolerance issue is moot
  for this driver — drops move megabytes, noise was bytes.

## 2026-09-21 — block-drop search: the early win did NOT land

- Design: DSL `dropped` (unique block ids 0..23), backbone skips blocks
  (residual passthrough), drop-add moves at priority 9 so all 24 singles
  run first (trials 2–25, verified), `_model_bytes` = remaining backbone
  float32 + neck + head + fit bytes (constants measured, not assumed:
  backbone 22056192, attn block 592128, mlp 1182336 — smoke asserts the
  exact deltas), driver under EfficiencyRule with 200MiB cap (model is
  ~100MB; the default 64MiB cap would reject everything).
- Outcome, plainly: NO deletable blocks at parity. All 24 singles
  rejected with full exploration collapse; best pairs (L11mlp+L9attn
  0.847, L11mlp+L10mlp 0.873, triple 0.682) worse than singles.
- The consolation is structural, and it surprised us: dispensability is
  NOT monotonic with depth. Ranked single-drop means: L11-mlp 0.953 >
  L9-attn 0.933 > L10-mlp 0.859 > … > L5-mlp 0.264. The single-scene
  0.97 that motivated this search was L11-mlp on scene 0 (min 0.896
  across scenes — correctly held by the gates). Late blocks lead, but
  mid-network blocks (L5-mlp, L3-attn) are the most load-bearing, not
  early ones. The naive "early fragile, late dispensable" story from
  the gain search needs this correction: gains and drops probe
  different things (scaling vs removal).
- What this means for the leanest-model thesis: the backbone is
  load-bearing at whole-block granularity. Remaining leanness plays:
  finer granularity (half-width MLP? single-head ablation?),
  the already-won width tables, and the emitter. NOT on the table:
  relaxing the bar to manufacture a win.

## 2026-09-21 — backbone weights learned (24 block gains), exhaustion

- Scope, stated first: NOT 22M-param fine-tuning (out of framework,
  out of data, out of thesis). The learnable surface is 24 per-block
  output gains (12 attn + 12 mlp) as phi exponents in [-2,2] — the
  residual-stream mixing ratios. Uniform weight scaling would be
  absorbed by LayerNorm; output gains survive it, and ship free as
  exponent adds. `forward_stages` takes optional `bgains`; default path
  verified bit-identical. DSL v3, 76 seed neighbors.
- 77-trial run to `proposal_space_exhausted` (22s), incumbent retained
  (all-zeros). Every single gain move (±1, all 24 blocks) fails the
  0.999 bar — but the follow-up measurement matters more than the
  verdicts: L0-attn×φ collapses to 0.47 mean, L6-attn×φ to 0.78,
  L11-mlp×φ degrades gracefully to 0.994. Sensitivity DECREASES with
  depth: early layers set coordinates everything downstream uses, late
  MLP tweaks are absorbable. That is evidence about DINOv2's residual
  structure, not just gate output.
- The seed itself sits at min 0.99901 / mean 0.99956 on exploration —
  the bar is at the seed's floor, so all these verdicts are razor
  thin. Another vote for margin-aware comparison in the base rule
  (built for efficiency, still absent here). Binary verdicts keep
  aliasing "slightly worse" with "garbage"; the post-hoc corr
  measurements above are doing the work the scorecards can't.
