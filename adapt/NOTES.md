# Adaptation-depth: running notes

Working log for the experimental branch. Kept because several findings
generalize to adaptation_foundry as a library (flagged LIBRARY below).

## 2026-09-21 — guard tested: refusal restores parity, adoption fires clean

- `should_regroup` (min histories + MI floor + real-difference) unit-tested
  (4 smoke checks) and wired in (`--regroup-min-n`, `--regroup-mi-floor`).
- Scoreboard, 3 seeds each: STATIC 2/3 (13, 16) · UNGUARDED-REGROUP 1/3
  (17) · GUARDED-REGROUP 2/3 (13, 16). The guard hypothesis is confirmed
  in the precise predicted form: refusals at gen 2/5 (n_top 4-7 < 20),
  performance restored to static level. No better, no worse — the guard
  prevents harm; it doesn't create wins.
- Min-n 8 variant (seed 2, 14 gens): adoption FIRED at gen 11
  (n_top=8, MI 0.142) — and again the strong pairs ([readout,taps],
  [res_scales,tap_gains]) survived dissolution intact. The hit (gen 7,
  13 evals) preceded adoption; adoption neither caused nor broke it
  (5 holders, 4 under bytes at close). Third replication of the same
  pattern: the grouping signal is real, the adoption policy is what
  needs the threshold.
- On the fundamental-truth question: three levels now show the same
  shape — verdicts needed calibrated bars, dissolution needed evidence
  floors, and in both cases the unguarded variant failed in exactly the
  predicted way while the guarded variant restored parity. Coincidence
  twice is interesting; three times with a mechanism (thresholds on
  measured evidence) is a pattern with a name: legibility all the way
  down. Still not proof — but no longer a single anecdote either.

## 2026-09-21 — self-dissolving groups: mechanism works, verdict negative

- Built: `--regroup K` re-derives groups from measured co-success via
  `learn_groups` every K gens and adopts on difference (events logged
  with MI evidence); bloom observer alongside (fn=0 throughout, est
  tracks exact — instrument proven, selection untouched).
- Race, static vs regroup3, 3 seeds: STATIC 2/3 hits (13, 16 evals),
  REGROUP 1/3 (17 evals). Dissolution hurt or did nothing.
- Why, read off the events: gen-2 regroups fire on ~11 histories and
  shatter hand groups into singletons + one blob (MI over a dozen
  samples is noise-following). Strong pairs ([readout,taps],
  [res_scales,tap_gains]) DO survive most dissolutions — the mechanism
  partially works — but adopting full regroups on thin evidence churns
  good structure away. Seed 2 is the exhibit: static hits at 13,
  regrouped misses entirely.
- Design lesson (the actual finding): self-dissolving structures need
  a MINIMUM-EVIDENCE guard — min histories, or an MI significance
  margin over the incumbent grouping, before adopting. Dissolution
  without an evidence threshold is just noise with a log line.
  Follow-up: guard + rerun; the events format already carries what's
  needed to gate on.
- LIBRARY: same lesson as the bar incident, one level up. Every
  self-modifying mechanism needs a "when am I allowed to believe my
  own statistics" threshold, or autonomy degrades into churn. Put the
  threshold in the mechanism, not in a human watching the logs.

## 2026-09-21 — stability round: everything stands (research paused)

- Runs/*.json backed up before touching anything; restored after.
- Battery, all green with recorded numbers: smoke PASS; C suite ALL
  PASS (all zero-mismatch lines identical); float parity 0.999995 /
  0.999998; integer suite OVERALL PASS incl. layer0 0.999996 and the
  full shootout table unchanged; emitter tree 0/64 EMIT ALL PASS;
  seed width-eval determinism IDENTICAL across runs.
- Learned artifacts: webcam int-vs-float 0.99956 today (vs 0.99982
  last session — different frames, scene variation, expected, far
  above every bar); graduation machinery spot-check passes pre-flight
  e7/7 g4/4 r3/3 with absolutes visible; incumbent config intact.
- Standing position, user's call recorded: DAV2's human training is
  unrecoverable in full and we claim nothing about swaths we can't
  see. What we claim: understanding + deterministic repeatability,
  both re-verified above. No new research builds this turn.

## 2026-09-21 — group race: learned ties hand (loop closed, no coronation)

- Raced LEARNED_GROUPS vs JOINT_GROUPS as crossover units, 4 seeds each,
  style crossover + random mating fixed: HAND hits at 13/16 evals (2/4),
  LEARNED hits at 14/16 (2/4). Dead heat — seed 4 beats both.
- Read exactly: the loop closed MECHANICALLY (machine proposal encoded,
  raced fairly, measured honestly) but proved nothing about superiority.
  Expected in hindsight: the decompositions differ only in dmax
  placement, and dmax never varies among survivors — so the two group
  sets generate nearly the same children. The race was fair; the
  contrast was thin.
- What stands: (1) self-modifying search runs end-to-end on this
  branch — propose-from-data is infrastructure now, not a sketch;
  (2) the discriminating test needs a space where group structure
  bites (deeper co-search: analytic kernels, emitter flags, backbone
  stage selection); (3) the converged-set mechanism ([dmax] flagged
  constant) is arguably the more valuable output — it tells you which
  dimensions NOT to spend trials on.
- LIBRARY: report ties as ties. A 2/4-vs-2/4 with a good story attached
  is how Kadmon-type overclaiming starts. The receipt here is the loop,
  not a leaderboard.

## 2026-09-21 — the proposer closes one loop (learned groups ≈ hand groups)

- `learn_groups` (MI + agglomerative over top-half histories, converged
  keys reported separately) ran on 107 evaluated genotypes from 6
  co-search runs. Proposed: [accum,block_gains,dropped,head] +
  [exp_span,frac_cap] + [readout,taps] + [res_scales,tap_gains], with
  dmax flagged CONVERGED (never varies among winners — true: dmax1024
  always fails). Hand design: identical on 3 of 4 groups; the only
  difference is the learner refusing to group a constant. Top MI pairs
  (block_gains×head 0.374, res_scales×tap_gains 0.364) are the
  co-adaptation pairs a human would draw.
- Stated carefully: n=107 histories, mostly failures, coarse statistic.
  But the direction is right — the system re-derived its own DSL
  decomposition from gate verdicts alone. That is one full turn of the
  loop the grand-strategy discussion asked for: search producing the
  guidance for the next search. Next: race learned groups vs hand
  groups as crossover units (the receipt that decides).
- Phase mating, honestly negative: 0/3 hits vs random mating's 2/3
  (same seeds, style crossover fixed). exploit_p=0.7 nearest-neighbor
  mating inbreeds — small steps, many unique evals (22-24 vs 11-16),
  no hits. Over-exploitation, and k=3 neighborhoods at pop 6 are too
  weak an instrument to conclude anything general about geometric
  mating. Kept as a parameter point, not a verdict on the idea.

## 2026-09-21 — co-search: style crossover beats flat mixing (n=3/arm, suggestive)

- Joint genotype (7 arch keys + 4 width keys), joint bytes (param bytes
  + table bytes — both levers move one counter), joint bar 27/27 at
  respective harness bars, seeded starts (random starts provably
  hopeless: 0 holders in 6 runs — the space needs good material first,
  another honest negative).
- Race, pop 6 / gens 8 / mutprob 0.3, same seeds: STYLE hits at 13 and
  16 unique evals (2/3 seeds); FLAT hits once post-loop (~20 evals),
  misses twice. Winners are exactly cross-DSL assemblies both times:
  intact arch (seed taps/gains/head) + exp8 tables ({4096,8,4096} and
  {13312,8,4096}) — the combination single-key mixing rarely builds
  but group moves (G4 tables as a unit) construct directly.
- Stated carefully: n=3 per arm is SUGGESTIVE, not conclusive. What IS
  conclusive: the machinery runs clean (pre-flight gates, sealed
  audits both harnesses, front hashes), and the winners' form matches
  the mechanism (cross-DSL bundles, not scattered keys).
- LIBRARY: co-search is the first workload here that single-move walk
  structurally cannot do well (it never holds two DSLs at once while
  changing both). If the foundry wants a flagship demo beyond Futhark,
  this shape — joint genotype, unified byte objective, style-group
  crossover, sealed dual audit — is the template.

## 2026-09-21 — CORRECTION: the integer-harness bar was unreachable

- The GA calibration failed twice (best 2/14), which smelled wrong and
  was: the SEED ITSELF scores 2/14 in the integer stride-4 harness
  (0.995–0.998; full-res integer head tops ~0.9990 vs HF). The 0.999
  float-pipeline bar exceeds what this harness delivers to ANY config.
  Float pipeline intact (0.999995) — harness problem, not model problem.
- Consequence, stated plainly: the round-one/round-two graduation
  "promotions" were ties-among-failures + real byte savings. The byte
  finding (EXP halving, probe at 1.0) stands; the "parity held" language
  in earlier entries was wrong. This entry corrects them; history kept.
- Fix: `CORR_PASS_INT = 0.99`, calibrated from measured seed capability
  (separates working >=0.994 from broken <=0.90); finer distinctions
  stay in mean_corr + margin rule. Float harness keeps 0.999.
- Re-run under the corrected bar: trial 3 (exp8) PROMOTES on a genuine
  14/14 tie; trial 4 (dmax1024) is correctly rejected WITH retention
  loss (so the dmax finding survives honestly); frac moves correctly
  unpromotable. Same incumbent {13312,8,4096}, now earned.
- LIBRARY: bars must be calibrated to what the harness can deliver to
  the SEED, measured before any search runs. A bar above seed
  capability turns every verdict into noise with correct-looking
  reasons — the most dangerous failure mode in this file, because the
  machinery looked like it was working. Add "measure seed first" to
  any experiment checklist.
- GA CALIBRATION: PASS under the corrected bar — 7 parity-holders,
  5 meet the bar (exp8 + fewer bytes), including {4096,8,4096}.
  Plus a controller lesson from round one: pure-Pareto selection let
  tiny-broken configs own the front on bytes alone; feasibility-first
  ordering (correctness, then bytes) + always-mutate fixed it.
  Diversity stayed healthy throughout (mean zeta distance 500-900,
  never collapsed to 0) — the fingerprint instrumentation earned its
  keep as an observer even before niching uses it.

## 2026-09-21 — level-2: front fingerprints + style crossover (measured non-result)

- Built: `front_record`/`front_fingerprint` (member hashes + hypervolume,
  measurement-grounded only), `style_crossover` (G1={frac,dmax} /
  G2={exp,accum} move as units), `--mode flat|style` in ga_search with
  first-hit tracking (gen + unique evals) and per-gen front hashes.
- 5 seeds × both modes, identical starts: EVERY run hits at gen 0-1
  (7-13 evals), flat and style indistinguishable. The 48-config width
  space saturates under random sampling — no mode can demonstrate an
  advantage where luck already wins.
- Verdict, stated exactly: the sublinear-sampling claim is UNTESTED,
  not refuted. The machinery works (front hashes track, diversity
  observed 300-900, both modes pass honestly), but the arena is too
  small to discriminate. The required next arena is the JOINT
  architecture+width co-search (1000s of configs, cross-DSL style
  groups) — which is now motivated by measurement rather than assumed.
  Neighbor search walks single moves; style crossover should win exactly
   there, or the thesis takes the hit it deserves.

## 2026-09-21 — fixing the failure mode itself (3 layers)

- Trial lines now print ABSOLUTES (`cand[e6/7 g4/4 r3/3] inc[e7/7...]`)
  in all four drivers — the display fault that hid 2/14 ties is gone.
  Verified live: the exp4 rejection now reads as what it is.
- `adapt/baseline.py`: pre-flight gate (build+evaluate seed, refuse on
  failure, assert determinism by double evaluation) wired into all four
  drivers; 6 new smoke tests green. Overhead ~2 extra evaluations,
  documented as accepted duplication that keeps core semantics intact.
- `adapt/BASELINE_UPSTREAM.md`: the core proposal — optional
  `baseline_gate` on Experimenter reusing the existing
  `baseline_failed` stop reason, why wrapper-only is a stopgap, and the
  one-line-per-callsite Echion migration. Filed here first as portable
  form; upstream PR + Echion vendor update are the next steps, not this
  branch's job.

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

## 2026-09-21 — stratification: weight-direct roles (pre-registered predictions)

- `adapt/stratify.py` characterizes all 12 layers from baked weights
  alone (phi-level stats, isotropy, rank ratios, layer-scales; position
  excluded; attention span omitted as stated — needs activations).
- k=3 agglomerative roles: {L0}, {L1..L10}, {L11} — contiguous,
  degenerate-looking but real: L0 extreme scaling (attn_iso 6.25),
  L11 extreme drive (ls1 0.335, ls2 -0.579 vs ~±0.09 elsewhere). No
  rotation-like layers anywhere (all attn_iso 2.6+).
- TWO rival predictions, both fixed BEFORE gate probes run:
  T1 (weight-character): fragile-first
  [11,10,9,8,6,7,5,4,2,3,1,0] — scaling+drive = fragile.
  T2 (positional fan-out, suggested BY exploration gain data — stated):
  fragile-first [0..11] — early layers amplify downstream.
  NOTE: T1 already conflicts with exploration patterns (L0 collapse);
  the gate probe arbitrates. If T1 loses, the scaling_score theory is
  falsified as written and must be revised, not tuned.

## 2026-09-21 — stratification probe: theories weak, scene interaction dominates

- Gate-scene probes (roles fixed beforehand): gain T1 rho=+0.449
  (p=0.028, marginal), T2 rho=-0.447 (mirror); drop T1/T2 both ns.
  Verdict: WEAK at best. A p=0.028 on n=24 with rank ties is noise
  wearing a lab coat — reported, not celebrated.
- The finding that matters is not the rhos but the RAW comparison:
  L0-attn gain scores 0.47 on exploration scenes vs 0.89 on gate
  scenes. Same block, same intervention, 0.4 corr apart. Gain
  fragility is block×scene INTERACTION, so any block-only ranking
  (T1 or T2) has a low ceiling by construction. The heterogeneity
  thesis survives — strengthened, even — but at a deeper level than
  "different layers do different jobs": layer behavior differs PER
  INPUT. Roles from weights alone cannot capture that; they describe
  capacity, not conduct.
- Standing: role map {L0},{L1..L10},{L11} kept as weight-direct
  description (L0 scaling-extreme, L11 drive-extreme, both contiguous
  and real); scaling_score retired as a fragility theory (falsified as
  written — L11 predicted most fragile, measured among most robust).
  Next probe design must be scene-stratified, or it measures scenes
  and calls them layers.

## 2026-09-21 — coverage matrix: strata predict, misses localized

- `adapt/strata.py`: 4 oracle-free stats (edge/texture/lumspread/
  vertical) median-split into 16 strata; 25 scenes hit 9; candidate
  panel of 5 (seed, tap2, direct, analytic, drop23); per-stratum
  foundry Assessments (decided cells CHECKED, splits UNCHECKED so the
  machinery itself emits 'unverified' obligations); 13 empty strata
  get 'missing' obligations. NOT saturated — stated by the table.
- 7 strata resolve SUPPORTED, 2 UNSUPPORTED with the obligations to
  prove it. Analytic fails 0/N and drop23 fails 0/N in EVERY stratum
  (search rejections, independently reproduced); tap2 == seed in all
  9 strata (the tie, reproduced a third way); direct holds everywhere.
- Held-out prediction 18/20 PASS (bar 15/20). Both misses are audit-2
  in E1T1L0V1 for seed/tap2 — predicted hold from a 6/7 majority,
  actual fail. The misses have an address: E1T1L0V1 is the cliff
  stratum where strong candidates crack. Next evidence goes there by
  name, not by vibes.
- LIBRARY: saturation is now a readable property (SUPPORTED everywhere
  + zero obligations), not a claim. And the matrix is the portfolio
  key the earlier discussion asked for: entries addressable by the
  strata they cover, holes drawn as obligations.

## 2026-09-21 — CORRECTION: the cliff was a resampling artifact (major)

- `depth_adapter.corr()` UPSAMPLED the 168px reference to 518px before
  correlating. Bilinear-upsample blur mismatches sharp predictions
  exactly at edges — more edges, more mismatch. This fabricated:
  (a) the entire B-family dose-response (0.99997->0.99850 becomes ALL
  >=0.99994 under the fixed protocol), and (b) the E1T1L0V1 cliff
  (seed 6/7 + audit miss becomes seed 7/7 + 20/20 prediction).
  Both retracted. The active-identification experiment still earned
  its keep: the injected-signal dose-response is what EXPOSED the
  metrology bug (a real mechanism would not vanish under correct
  resampling). Fixed rule: downsample predictions to reference
  resolution; NEVER upsample ground truth.
- Verdict audit (buggy -> fixed protocol), re-ran everything cheap:
  STANDS: drops catastrophic (e0/6 both), analytic/wrong-readout
  catastrophes, block-gain collapses + depth gradient (L0 0.67,
  L11 0.994), drop ranking shape (L11 best 0.941, mid worst),
  head near-miss ranking (L1h0 now 0.99918, still 0 full passes),
  dmax1024 retention loss, exp8 genuine 14/14 promotion, co-search
  race outcome (style 2/3, re-ran), ALL width/integer results
  (separate clean protocol, verified identical code path).
  STRENGTHENED: tap-2 tie is now a genuine 6/6 tie.
  CORRECTED: direct genuinely fails 2 scenes (was masked when the
  buggy protocol also failed seed there).
  FIXED (code): flat-mode lambda-precedence crash (crashed loudly, no
  silent corruption; prior flat runs predate the bug — verified via
  run JSONs).
- E1T1L0V1 reframed, not deleted: not a cliff (seed holds 7/7) but the
  most DISCRIMINATING stratum (direct splits 2/7 there and only
  there). Strata prediction now 20/20.
- LIBRARY, third verse: metrology bugs wear the best disguises —
  this one produced a smooth dose-response curve, the shape most
  likely to be trusted. Defenses, all now in place: never upsample
  ground truth (rule, not advice); distrust any finding whose effect
  size tracks a preprocessing parameter; injected-signal sweeps as
  standard audit for dose-response claims (the sweep that caught this
  is now a reusable pattern: if the curve survives correct
  resampling, it's mechanism; if not, it's measurement).

## 2026-09-21 — sweep re-run: flat everywhere, hunt over

- Full 29-stimulus re-run under the fixed protocol: B-family
  0.99993–0.99998 (no trend), A-family ≥0.99994, C-family ≥0.99995.
  Two isolated dips (A-f12-a0.15 0.99930, A-f16-a0.15 0.99948),
  non-monotonic with neighbors — grating-frequency curiosities, noted
  as such, claimed as nothing. No milder edge effect hiding under the
  artifact; the model handles edges and texture to ≥0.9993 across all
  tested ranges.
- The instrument is now calibrated: flat-on-healthy is the expected
  signature, and future sweeps compare against this baseline instead
  of against zero. E1T1L0V1 keeps exactly one honest distinction:
  most discriminating stratum (direct splits 2/7 there, nowhere
  else) — not a cliff, a lens.

## 2026-09-22 — PRE-REGISTERED: real-scene strata fill + weight-pathway program

- Program decision (user): pursue ALL FOUR weight-pathway reductions
  (codebook, low-rank, structured sparsity, cross-layer sharing),
  codebook first (cheapest). Thread #1 (real scenes) runs first
  because every reduction gate needs non-synthetic ground to stand on.
- Method: ~40 COCO-train RGB (HF `detection-datasets/coco`, strided
  sampling for diversity) + HF-oracle refs at 168px via
  `harden_anchors.oracle_refs` (same protocol as build_fixtures).
  Saved to `adapt/fixtures/strata_real.npz` (gitignored) with source
  ids; `strata.py` loads it alongside `strata_extra.npz` (provenance
  noted, vendored core untouched). Webcam `captures/*.png` NOT used
  (combined figures, not RGB scenes — stated so nobody re-tries).
- Predictions (fixed BEFORE fetching):
  P1: pooled medians shift toward real stats (edge/texture up,
    lumspread down, vertical toward positive) and coverage spreads
    from 9 hit to >=12 hit strata.
  P2: seed holds >=0.999 on the majority of real scenes but records
    1-3 misses (reals harder than synthetics); any candidate splits
    localize in newly filled strata.
  P3: held-out audit prediction stays >=15/20 after recompute.
  P4 (codebook, later probe): per-layer codebook to 2048 levels
    holds parity with a byte win; flat global 512 fails retention.
  If P1 fails (reals still collapse into 2 strata after recompute),
  the median-split stat set is falsified as a stratifier and must be
  revised, not patched with hand-picked bins.

## 2026-09-22 — OUTCOME: real scenes land (P1 near-miss, P2 good-miss, P3 holds)

- Fetched 40 COCO-train RGB (`adapt/fetch_real.py`, strided streaming)
  + HF-oracle refs at 168px; `strata.py` now also loads
  `strata_real.npz` (provenance inside; synthetic path untouched).
- P1 (coverage >=12): MISS on the number, mechanism confirmed.
  9 -> 11 hit strata; medians shifted exactly as predicted (edge
  0.062->0.247, texture 0.024->0.215, vertical -0.44->-0.08).
  Remaining empty 5: E0T0L1V1, E0T1L0V0, E0T1L1V0, E0T1L1V1,
  E1T0L1V0. The stat set stratifies (reals spread, not collapse),
  so no revision triggered — but the bar was missed, stated plainly.
- P2 (1-3 seed misses on reals): MISS in the good direction. Seed
  holds 65/65 across all 11 hit strata — the geometric replica
  matches HF on real photos, not just synthetics. The news is in
  tap2: synthetic-genuine tie now SPLITS on reals (E0T0L0V1 4/5,
  E1T1L0V1 10/13, E1T1L1V0 1/2, E1T1L1V1 6/8 where seed holds).
  The tap-2 watch-list item just got its margin evidence: tie on
  synthetics, qualified on reals. Direct holds only in E0T0L1V0
  (15/21, synthetic-dominated) and ~nowhere on reals.
- P3 (held-out >=15/20): HOLDS, 19/20.
- COUNTING BUG (conservative, no verdict change): the obligations
  print counts kind=='missing' as "missing strata" (13) and
  kind=='ambiguous' as splits (always 0 — the machinery emits
  'unverified', never 'ambiguous'). Truly-empty strata are 5, split
  cells 8. Saturation logic (needs n_empty==0) stays conservative;
  fixing the labels + split count next.

## 2026-09-22 — CORRECTION + codebook verdict (P4 fails honestly, locality wins big)

- CORRECTION (own bug, caught by measurement): the first codebook
  folded sign into the level (lvl=sign*(e-BIAS)/K). lvl=+35.9 is
  BOTH tiny-negative and huge-positive — ambiguous space, decoded
  garbage (max rel err 3e15, corr ~0.21 flat across C). Retracted;
  magnitude-only codebook (mu=(e-BIAS)/K, signs exact 1-bit sidecar).
  Joint-codebook P4-as-first-written is falsified as formulated.
- Global magnitude codebook dose-response (6 scenes, 3 synth + 3
  real): C=4096 0.852 / 2048 0.30 / 512 0.11 / 128 negative. Even
  4096 (mean rel err 0.2%/weight) fails — 22M tiny roundings
  accumulate through 12 softmax attention layers (the anti-averager).
- Per-layer codebooks (P4's letter): 2048 0.99829 (1/6), 4096
  0.99810 (3/6), 8192 0.99876 (3/6). Locality buys +0.15 over
  global at comparable levels (distributions are layer-local) — the
  pathway-relevant win. But the curve SATURATES ~0.9988 below the
  0.999 bar: more levels don't close it. P4 FAILS as stated; no
  promotion (EfficiencyRule correctly refuses — these are
  regressions, not ties).
- Guard probe (outliers |mu|>15 kept exact, 1.07M residuals, +4MB):
  zero gain (0.99801 vs 0.99829). Damage is in the dense core, not
  the tails — evidence against few-load-bearing-outliers for this
  axis. Guard variant retired.
- Standing: codebook alone = genuine near-miss (34-42MB vs 66MB
  phi, ~0.9988). Kept as evidence + stacking candidate. Next axis:
  low-rank (float factors first = thesis upper bound, then
  phi factors = deployable). Pre-registered below.

## 2026-09-22 — PRE-REGISTERED: low-rank pathway probe

- Per-matrix truncated SVD at rank fractions {1/2, 1/4, 1/8},
  float factors patched into buffers (upper bound — no phi rounding
  confound), same 6 probe scenes, seed pipeline, absolutes printed.
- Predictions: rank-1/2 holds >=0.999 on >=4/6 (directional
  redundancy is real); rank-1/8 collapses (<0.95 mean); attention
  proj/qkv more sensitive per-rank than MLP (softmax amplification).
  If rank-1/2 fails everywhere, the rank thesis is dead and the
  program moves to sparsity/sharing without mourning.

## 2026-09-22 — OUTCOME: rank thesis dead, sharing half-retired (measure-first pays)

- Low-rank (float factors, upper bound): 0/6 in ALL 9 configs —
  rank-1/2 all 0.08, attn-only 0.26, mlp-only 0.25; rank-1/8 worse.
  All three pre-registered predictions fail. Mechanism measured, not
  mourned: spectra are full-rank (L5-mlp1 needs 362/384 dims for 99%
  energy; q needs 259/384). Directional redundancy is zero — the
  weights use every direction they have. Rank axis RETIRED.
- Cross-layer sharing, decided OFFLINE (no pipeline runs spent):
  top-64 subspace overlap L0-L1/L0-L11/L5-L6: q 0.33-0.38 (vs ~0.13
  random — weak-moderate shared structure in attention), mlp1 0.18
  (≈ noise floor). MLP-sharing probe RETIRED by measurement before
  spending a single forward pass. Attention-sharing stays a
  qualified-maybe (needs a probe to decide; 0.35 overlap won't carry
  parity, stated upfront).
- Standing pattern across axes: precision (codebook 0.9988 saturates
  below bar), direction (full-rank), blocks/heads/gains (all
  load-bearing from prior searches). The model is dense and
  irreducible at every granularity tried. Remaining: structured
  sparsity (last unmeasured axis).

## 2026-09-22 — PRE-REGISTERED: structured sparsity probe

- Zero whole output-rows (neurons) by row-L2 at row fractions
  {5%, 10%, 25%} on 2D backbone weights (norms/scales untouched),
  same 6 probe scenes, absolutes printed.
- Predictions (weak, stated): 5% degrades to ~0.99 (marginal, not
  parity); 25% collapses (<0.9). Dense-core sensitivity from the
  codebook axis says small weights matter. If 5% HOLDS parity it is
  the first structural win and goes to the gate immediately.

## 2026-09-22 — OUTCOME: sparsity dead at 5%; all four axes closed

- Structured sparsity: 5% rows zeroed -> mean 0.28 (0/6); 10% 0.06;
  25% 0.12. The "5% ~0.99" prediction fails hard — even the weakest
  neurons are load-bearing. No structural redundancy at row grain.
- Weight-pathway program, final ledger:
  codebook ~0.9988 near-miss (stacking candidate only) · low-rank
  DEAD (full-rank spectra) · sparsity DEAD at 5% · sharing DEAD for
  MLP offline (0.18), weak-moderate for attention (0.35, won't carry
  parity — no probe spent, stated).
- Thesis verdict: the backbone is IRREDUCIBLE at every granularity
  tried (blocks, heads, gains, precision, direction, rows,
  cross-layer). Matches all prior coarse searches (24/24 drops fail,
  72/72 heads fail). The "leanest backbone" line is closed; won
  leanness stays where earned (EXP-halving 544kB, C kernel, 125B
  head, per-layer codebook as evidence). This is a result, not a
  defeat: first measured irreducibility map of a DAV2 replica.

## 2026-09-22 — PRE-REGISTERED: integer attention composition in C

- Open row in c_port/README (kernels exist, composition unwired):
  add nn_attn_scores + nn_softmax_rows + nn_attn_av +
  nn_attention_fixed to fixed_nn.{h,c} mirroring geo_int's
  int_attention_fixed body op-for-op (floor_div128 at the two
  negative-capable sites: (dot+2^13)>>14 and //8; den==0 guard).
- Vectors: softmax-rows vs G.int_softmax_fixedvals EXACT (random +
  wide-spread clipping + constant rows); attention end-to-end vs
  G.int_attention_fixed EXACT (real baked layer0 W, realistic random
  X N=8). Scores/av covered by end-to-end bit-exactness (stated, not
  hidden — no exact standalone fns exist for them).
- Predictions: 0 mismatches on softmax first run (pure LUT path);
  1-2 floor-trap iterations on scores/av composition before green
  (the //8 and (acc+HALF)>>F sites on negative dots). Zero-float
  grep stays clean; make test green before any claim.

## 2026-09-22 — OUTCOME: integer attention wired in C, bit-exact

- `nn_attn_scores` + `nn_softmax_rows` + `nn_attn_av` +
  `nn_attention_fixed` in fixed_nn.{h,c}; `make test` green
  (softmax 0/150, attention 0/3072 on real baked layer0 W, N=8;
  full suite ALL PASS incl. prior kernels). Zero-float grep clean.
- Two bugs, both harness-side on first run, stated: (1) test used
  6x24 rows against a square-assumed kernel — generalized kernel to
  (rows,cols); (2) per-row av call passed rows=1 as the reduction
  length (computed 1 of 8 terms) — changed to per-row DEN array,
  one call. Q/scores intermediates verified equal before the fix
  localized it past softmax. Prediction score: softmax-first-try
  wrong (shape bug, not floor trap), floor sites correct as written.
- Thread 1 of the closing program DONE. Remaining: fixed-point
  sensor path, upstream PR, bigger co-search arena, substrate
  experiment, 5 empty strata.

## 2026-09-22 — PRE-REGISTERED: fixed-point sensor path (v1, 518-native)

- Gap (README honest boundary): uint8->normalized + patch-embed +
  CLS/pos still float. v1: `sensor_encode_fixed` (per-channel integer
  affine, K=8 extra bits) + `sensor_patch_tokens` (im2col +
  int_linear_fixed + fixed CLS/pos) in geo_int; `nn_sensor_encode` +
  `nn_patch_tokens` in C, bit-exact vs the exact Python fns.
- Scope, stated: 518-native only (pos_embed is 1370x384 baked; resize
  interp stays host-side like decode-for-display). Accuracy gate is
  corr-based (integer sensor ~= float preprocess within 1 ULP; no
  bit-exact-vs-float claim possible): fixed tokens vs float backbone
  tokens >=0.9999, and fixed-sensor full layer0 vs HF >=0.999.
- Predictions: C bit-exact first try (affine + gather, no floor
  subtlety beyond floor_div); corr gates pass with margin (>=0.9999
  tokens, layer0 stays >=0.999).

## 2026-09-22 — OUTCOME: sensor path wired (Python + C, gates pass)

- geo_int: `sensor_encode_fixed` + `bake_sensor_fixed` +
  `sensor_patch_tokens` + `sensor_parity`. Tokens vs HF embeddings
  0.999986 (maxabs 0.003 ~ bicubic-resample residue, stated);
  sensor-fed layer0 vs HF 0.999946. Both bars pass with margin.
- C: `nn_sensor_encode` + `nn_patch_tokens`, bit-exact 0/2352 +
  0/1920 first try; `make test` ALL PASS x3 suites; zero-float grep
  clean. One macro collision (SN_T count vs array) fixed at build.
- Thread 2 of the closing program DONE. Honest remainder: resize
  interp + decode-for-display stay host-side (documented v1 scope).
  Remaining: upstream PR, bigger co-search arena, substrate
  experiment, 5 empty strata.

## 2026-09-22 — Upstream PR prepared (NOT filed — needs explicit push approval)

- Spec BASELINE_UPSTREAM.md verified against the live core: the
  `baseline_failed` reason exists but fires only on missing
  incumbent, never on quality — the gap is real, the diff applies.
- Patch implemented in scratch checkout (/tmp/opencode/
  adaptation_foundry, NOT our repo): optional `baseline_gate`
  callable on Experimenter, checked once after baseline accept,
  fail-closed to done/`baseline_failed`, reason on self.error.
  Convention identical to adapt/baseline.py gates (pass-through).
- 5 new BaselineGateTests + full upstream suite 73/73 green.
  Diff + PR body staged at /tmp/opencode/baseline_gate_pr.diff +
  baseline_gate_pr_body.md. Filing = pushing upstream = needs the
  user's explicit say-so (house rule 6). Nothing pushed.
- Echion recon: vendor copy IN SYNC with upstream; NO direct
  Experimenter( call sites exist there yet (contracts only) — so the
  "one-line migration" is vacuous today, just a vendor sync post-merge.
  Spec's migration paragraph corrected accordingly in the PR body.

## 2026-09-22 — PRE-REGISTERED: bigger co-search arena (thread 4)

- Diagnosis of the 2/4 tie: the old arena (synthetic-only, pop6/gens8,
  4 seeds) couldn't discriminate — synthetics agree with each other,
  and hand/learned differed only in dmax placement. Bigger arena =
  harder cases + more power + fresh contrast, in that order:
  A. Real-scene panel in the co-search arch evaluation (--real N,
     default 8 COCO scenes w/ oracle refs; total_cases grows; seed
     pre-flight recalibrated, or the race is noise).
  B. One big hand-style run (pop12/gens16) -> history pool ->
     learn_groups -> LEARNED2 (same-denominator histories only;
     old synthetic-only histories NOT pooled — different
     denominators would bias the ranking, stated).
  C. Race hand vs LEARNED2 (5 seeds/arm, pop10/gens12), metric unique
     evals to first target-hit.
- Predictions: target-hit rate drops overall (reals harder — tap2
  splits prove the bar bites); style-vs-flat gap widens on the harder
  landscape; hand vs LEARNED2 discriminates (fresh groups differ on
  more than dmax). If LEARNED2 == hand structurally, the contrast is
  thin again — report it, race anyway, ties are ties.

## 2026-09-22 — AMENDMENT: proposer blobs at n=59; permutation race instead

- Big hand-style run (pop12/gens16, 35-case bar): PASS, first hit
  gen 11/49 evals; winner = intact arch + exp8 (same cross-DSL form
  as the old arena, now holding 8 reals). 9 holders under seed bytes.
- learn_groups on the 59-history pool: one 8-key BLOB + 3 singletons
  (MI ~0.2 chains everything). The evidence guard PASSES it
  (n_top=29>=20, MI>=0.05) — the guard checks quantity, not shape.
  LIBRARY: self-modifying mechanisms need a shape check too
  (no-blob: refuse if largest group >60% of keys). Racing blob-vs-hand
  is meaningless (blob crossover ~= flat mixing).
- New contrast (cleaner thesis test): hand JOINT_GROUPS vs RANDOM
  4-partitions (permutation test). If hand beats random partitions,
  group structure bites — the thesis holds even though the proposer
  can't resolve it at n~60. Prediction: hand wins on first-hit evals
  (its bundles — tables-together, structure-together — are real
  co-adaptation); random partitions scramble them. 5 seeds/arm,
  pop10/gens12, --real 8. If hand ties random, the style thesis
  takes the hit it deserves.

## 2026-09-22 — OUTCOME: random >= hand; style thesis takes the hit

- Permutation race (--real 8, pop10/gens12, 5 seeds/arm): hand 4/5
  (25, 27, 36, 37 evals + one miss) vs random-partition 5/5 (47, 28,
  35, 24, 28; median 28 vs hand 31.5). Prediction falsified.
- Read exactly: GROUP MOVES still beat flat mixing (prior 2/3 race
  stands — the move class matters), but hand grouping content adds
  nothing over arbitrary chunks (this race — the content doesn't).
  Neither machine (blob at n=59) nor human resolves better-than-
  random decompositions at this arena scale. Winners' form unchanged
  (intact arch + exp8 assemblies — reachable by many decompositions,
  which is WHY content doesn't discriminate).
- Thread 4 DONE. The co-search machinery stands (harder arena runs
  clean, 35-case bar, audits sealed); the style-content thesis does
  not. LIBRARY: permutation-test your inductive bias — a mechanism
  can win (groups move) while its content loses (which groups).

## 2026-09-22 — PRE-REGISTERED: substrate experiment (thread 5)

- Question (lattice ≅ contracts?): do gate verdicts float free of
  the phi substrate, or measure the lattice+model joint?
- Design (same-harness exact-vs-mush, learned from the bar
  correction): mush = per-layer C=2048 codebook backbone (measured
  0.9983 — a real, principled coarsening, not arbitrary noise).
  Width DSL + integer harness UNCHANGED (backbone is the substrate,
  widths the variable). Recalibrate the bar to MUSH-seed capability
  (same rule as CORR_PASS_INT — absolute bars would make everything
  trivially fail, which proves only tightness). Rerun the decisive
  graduation trials: exp8 (promoted 14/14) + dmax1024 (rejected,
  retention loss) + a frac move (correctly unpromotable).
- Predictions: mush-seed capability lands 0.997-0.9985; verdict
  PATTERN reproduces (exp8 promotes, dmax1024 rejects, frac stays
  unpromotable) → contracts substrate-invariant at recalibrated
  bars. If the pattern scrambles (e.g. dmax1024 promotes under mush),
  verdicts are substrate-bound — stated, not mourned.
- Compute needs GPU (width-harness evals); runs after the race frees it.

## 2026-09-22 — OUTCOME: verdicts substrate-invariant (thread 5 DONE)

- Mush-seed means 0.9951/0.9906/0.9959 (pre-registered 0.997-0.9985:
  slightly high, stated). Recalibrated mush bar 0.985 (same rule as
  CORR_PASS_INT: below mush-seed floor, above broken <=0.90).
- Pattern at respective bars: exp8 IDENTICAL to seed to 5 decimals
  under both substrates (0.99735 exact / 0.99509 mush — the tie is
  numerical, not just verdict-wise; tree never reads EXP, consistent
  with the honesty fix); dmax1024 catastrophic under both (worse
  under mush: 0.59/0.36/0.76 vs 0.70/0.56/0.85, same verdict);
  frac2048 ≡ seed under both. One gate scene the mush seed misses
  at 0.99 is a capability limit (all mush configs miss it), not a
  verdict — exactly why bars recalibrate.
- Answer: CONTRACTS CARRY IT. Verdict structure reproduces across
  substrates at calibrated bars. Thread 5 DONE. Remaining: 5 empty
  strata (14 hole-fillers fetched, strata re-running).

## 2026-09-22 — OUTCOME: 12/16 strata, unreachable corners documented (thread 6 DONE)

- 14 hole-fillers appended (54 reals); recompute: 79 scenes, 12/16
  hit, 19/20 held-out PASS. Seed 79/79 everywhere; tap2 splits in 6
  strata; analytic/drop23 fail 0/N in all 12 (fourth independent
  reproduction of every search rejection).
- Honest remainder, two kinds: (a) E0T1L1V0/V1 — 0 hits in 6452 COCO
  images (train1500 + val4952): low-edge + high-texture +
  high-lumspread is ≈absent in natural photography. Stratification
  finding, not a data failure: the median-split stat set has
  degenerate corners. (b) E1T0L0V0/V1 — median-shift casualties
  (hit before, empty after refilling). Holes are moving targets
  under recomputed medians — the price of honest stratification.
- Thread 6 closed at 12/16 with the corners documented. Chasing two
  never-observed strata with exotic sources is negative-value work;
  the matrix already discriminates all live questions.

## 2026-09-22 — PRE-REGISTERED: pooled group learning (269 histories)

- The blob verdict was about n=59. Pool now: big hand run (59) +
  random-arm seeds 11-15 (210), all 35-case denominator (--real 8).
  Excluded: older 27-case files (different denominators bias the
  ranking) and the overwritten hand arm (lost to filename collision
  — runs now write arm-stamped files; stated). Random-arm bias noted:
  broader key-combo coverage, arguably good for MI.
- Predictions: resolves non-blob structure (largest group <=6 keys);
  top-MI pairs reproduce block_gains×taps / readout×taps (>=2 of top-3
  match the n=59 run's pairs). If blob again at n=269, the proposer
  is retired at scale too — the failure is algorithmic (chaining),
  not sample size.

## 2026-09-22 — OUTCOME: blob at 269, proposer retired (content moot anyway)

- Same 8-key blob + accum/dmax/exp_span singletons. MI sharpened
  (0.33 vs 0.23) but clustering still chains. Pair reproduction 1/3
  (only block_gains×head survives). Both predictions fail.
- Reading, stated carefully: chaining-vs-genuine-nonfactorizability
  is undecidable from inside — BUT the permutation race makes it
  moot (hand ≈ random means content doesn't matter either way). The
  coherent story: nothing factorizes (matches irreducibility
  everywhere) → all groupings equivalent → only move GRANULARITY
  matters (multi-key vs scattered, the one standing style result).
  The singletons are the real output: accum/dmax/exp_span vary
  independently — exactly the levers that ever won anything.
- Proposer RETIRED (usefulness, not correctness). Fix applied:
  race files now arm-stamped (no more hand/random overwrites).

## 2026-09-22 — PRE-REGISTERED: scene-conditional tap study

- Tap-2 went tie -> splits-on-6-strata. New question: WHEN does it
  hold? All single-tap moves (3->2, 6->5/7, 9->8/10, 11/12 neighbors)
  x all 79 scenes, per-stratum hold/fail at 0.999, seed pipeline.
- Predictions: tap 3->2 holds on synthetics + low-edge strata,
  fails on high-edge real strata (E1T1L0V1 et al) — i.e. the split
  tracks edge/texture load, not random; deeper taps (9/12 neighbors)
  fail broadly (matches prior tap search). If 3->2 holds everywhere
  including E1T1L0V1, the strata splits came from elsewhere (direct
  fit? noise?) — stated alternative.
