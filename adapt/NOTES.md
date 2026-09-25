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

## 2026-09-22 — OUTCOME: tap sensitivity is depth-graded AND scene-graded

- 6 tap moves x 79 scenes: tap3 (12->11) collapses 0.00 in ALL 12
  strata (irreplaceable everywhere — matches prior search);
  tap0 (3->2) holds 1.00 on 6 strata, degrades to 0.67-0.86 on
  high-edge/texture strata (E1T1L0V1 0.76, E1T1L1V0 0.67) — partial,
  not collapse. Mid taps intermediate (tap1_up hits 0.40 in
  E1T0L1V0). Seed 1.00 everywhere.
- Opinion formed: the tap question is closed AS A MAP (which layer
  tolerates substitution under what scene load), not as a verdict.
  Layer 12 = load-bearing everywhere; layer 3 = substitutable except
  under edge load; the split tracks input statistics. No bar moved,
  no win claimed — understanding claimed, with numbers.

## 2026-09-22 — PRE-REGISTERED: attention-share probe (cheap close-out)

- All 12 layers share layer5's q.weight (float buffers, same seed
  pipeline, 6 probe scenes). Thesis: 0.33-0.38 subspace overlap
  carries the Q function. Prediction: FAILS (<0.95 mean) — overlap
  is real but far from substitutability; biases/norms stay
  layer-local so the test is pure. If it HOLDS (>=0.999), it is the
  biggest structural win in the program and goes to the gate.

## 2026-09-22 — OUTCOME: shared-Q collapses (0.36); opinions on the rest

- Shared layer5-Q across 12 layers: mean 0.36 (0/6, one negative).
  Overlap ≠ substitutability — closed as predicted. Sharing axis
  fully retired (MLP offline 0.18, attention probe 0.36).
- Stacking: NO MATERIAL — opinion formed without running. Only
  codebook ever near-missed (0.9988); every other axis is dead, not
  marginal. There is nothing to stack WITH. Stacking two failures
  is not an experiment.
- Margin gates: NEEDED, evidenced (0.9988 and L1h0 both sit in the
  binary blind spot). Implementation is an upstream follow-up to the
  filed baseline-gate PR (same story: gates must resolve ties), not
  a local hack. Opinion recorded; code waits for the PR thread.
## 2026-09-22 — PRE-REGISTERED: linear-layer pilot (distillation crux)

- Question: does a closed-form LINEAR reparameterization of a teacher
  layer exist? Ridge 384->384 (same-width: capacity is not the
  variable, linearity is) fit on teacher (input, output) token pairs
  from real images, tested on held-out images. One image = 1370
  pairs; fit on 10 reals, test on 5 reals + 3 synthetics. Layers
  {0, 5, 11} (early/mid/late). CPU ridge after one GPU activation
  pass. No gradient descent anywhere — that is the point.
- Predictions: FAILS everywhere (<0.95 test corr) — GELU/softmax/
  LayerNorm all nonlinear and spectra show full-rank use. Gradient
  expected: L0 worst (sets coordinates), L11 least-bad (absorbable
  tweaks, matches gain-search depth gradient). corr here is
  activation-corr vs teacher, NOT depth corr — different meter,
  stated.
- Reading rules (fixed before running): >=0.999 anywhere -> rethink
  everything (layers linear on-manifold, phi-from-birth trivial);
  <0.95 everywhere -> greedy-linear is dead, the student must carry
  real nonlinear capacity and "mathematical training" needs a
  nonlinear solver (named open problem, not a refutation of
  phi-from-birth — reparameterization can still exist, it just
  isn't ridge-reachable).

## 2026-09-22 — OUTCOME: greedy-linear dead; gradient inverted (interesting)

- Ridge 384->384 (unregularized lstsq, generous to linearity):
  L0 0.938 / L5 0.881 / L11 0.824 held-out real (synth worse:
  0.93/0.80/0.57). FAIL everywhere as predicted — no layer is
  linear on-manifold.
- Gradient prediction INVERTED (stated): expected L0 worst, got L11
  worst. Robustness-to-scaling (L11 most robust per gain search) is
  NOT linearity — late layers are the most nonlinear (or most
  input-dependent). Two different axes of "sensitivity," now
  measured both ways. L11-synth 0.57 is the worst transfer in the
  program's history — scene interaction strikes again.
- Consequence, exactly as pre-registered: the student must carry
  real nonlinear capacity; "mathematical training" needs a nonlinear
  solver. Candidates: (a) conventional gradient descent on a small
  student (paper's recipe, abandons few-shot); (b) greedy NONLINEAR
  fit — same-width student layer with GELU/softmax trained by...?
  (c) layer-wise LEAST-SQUARES on linear sub-blocks composed with
  frozen teacher nonlinearities (hybrid: learn projections, keep
  activations). (c) is untested and cheap — opinion: try it next.

## 2026-09-22 — PRE-REGISTERED: blockfit pilot (hybrid RRR, activation rank)

- Refinement forced by honesty: teacher LINEAR blocks are exactly
  fittable at full width (they ARE linear) — the only question is
  reduced width, and weight-space rank is dead. So (c) must measure
  ACTIVATION-space compressibility (never measured): reduced-rank
  regression per linear map (q/k/v/proj/mlp1/mlp2) at ranks
  {192,96,48}, all narrowed maps composed SIMULTANEOUSLY with the
  exact nonlinearity FORMS (softmax/GELU/LN recomputed, error
  compounding included), block-output corr on held-out reals+synth.
  Streamed covariances (no token storage); ridge 1e-6 for stability.
- Context diagnostic first: activation effective rank (99%/99.9%
  energy) per block I/O on real tokens — if full-rank, RRR is
  expected to fail and the failure is located precisely.
- Predictions: activation rank high (>=300/384 @99.9%); RRR-192
  partial (~0.95); RRR-96 fail (<0.90); attention worse than MLP
  (softmax compounding of Q/K errors). A hold (>=0.999) at 96
  anywhere = first compression win in program history → gate.
- Reading rules: fail everywhere -> hybrid dead; "mathematical
  training" then requires either end-to-end discrete search over a
  small student DSL (foundry-style, few-shot preserved) or gradient
  descent (few-shot abandoned). Both named, neither started.

## 2026-09-22 — OUTCOME: hybrid partial — early layers compressible (first!)

- OLS ceiling 1.0000 everywhere (method valid; gaps are pure
  capacity). Activation effrank depth-graded: L0 q needs 27/384
  @99%, L5 144, L11 153 (mlp2: 127/313/177). Weight-space full-rank
  but ACTIVATION-space compressible early — never measured before.
- Composed (all maps narrowed + exact forms, compounding incl.):
  L0 r192 layer 0.99759 (attn block 0.99995!), r96 0.98826;
  L5 r192 0.92235; L11 r192 0.85616. Predictions partially hold
  (L5/L11 fail as said; L0 far better than ~0.95; "attention worse"
  true only at L11). One own-bug caught: 'ols' row first printed
  rank-48 values (min vs max key) — rerun, ceiling confirmed.
- COHERENT PICTURE (three axes triangulate): early = linear-ish,
  compressible, gain-fragile (sets coordinates); late = nonlinear,
  incompressible, gain-robust (absorbable tweaks). Weight-space is
  full-rank everywhere; removal impossible anywhere; but
  activation-space early layers have slack.
- For phi-from-birth: FIRST compression-compatible finding — a
  depth-graded student prior (narrow early, full late) with
  closed-form RRR block fits (few-shot preserved). Not a win
  (0.988 < bar, depth-parity unmeasured) — a direction with
  numbers. Proposed next: depth-graded student pilot, depth-gated.

## 2026-09-22 — PRE-REGISTERED: depth-graded student pilot (real meter)

- Assemble RUNNING student backbones: RRR-fit rank-r maps (dense
  reconstruct — tests function, not speed) for early layers,
  teacher-exact late. Configs: A=L0-2@96, B=L0-3@96, C=L0-5@192,
  D=L0-2@96+L3-5@192. Full pipeline (student backbone + neck +
  head) vs HF-oracle refs, 12 scenes (6 fixture + 6 real), bar
  0.999 + margin reading. Float maps first (upper bound);
  phi-encode after only if something holds. Bytes: factor pairs
  reported, not claimed.
- Predictions: A closest (depth >=0.995, below bar — block 0.988
  compounds); D degrades further; NONE >=0.999 (near-miss, not
  win). A hold (>=0.999) on any config = first genuine compression
  lead -> gate + emitter path + phi-encode.

## 2026-09-22 — OUTCOME: single-layer holds, compounding kills

- F (L0-only @192): mean 0.99914, 11/12 pass (only miss:
  exploration-2 synthetic 0.9930; all 6 reals 0.9998-1.0). E
  (L0-only @96): 0.9926, 3/12. Multi-layer A-D: 0.89-0.97.
  Prediction direction held (no full win) but A-number missed
  (0.962 vs hoped >=0.995).
- Located precisely: single-layer capacity SUFFICES (F ~holds);
  independent greedy fits DON'T COMPOSE (A << F). The killer is
  compounding, and it prescribes its own fix: SEQUENTIAL RRR —
  fit each layer against teacher targets GIVEN STUDENT (perturbed)
  inputs from already-fitted prefix layers. Still closed-form,
  still few-shot.

## 2026-09-22 — PRE-REGISTERED: sequential RRR (config A chain)

- Fit L0 maps (teacher inputs) -> student L0 outs (numpy exact
  forms) -> L1 covariances on (student-outs -> teacher L1 targets)
  -> fit L1 -> student L1 outs -> L2 likewise. Depth-gate A-seq vs
  A-indep 0.962 on same 12 scenes.
- Predictions: A-seq >=0.99 (compounding substantially absorbed;
  residual gap from rank capacity itself); per-layer student-input
  drift shrinks vs indep (measured: mean |student-in - teacher-in|
  per layer). If A-seq <= A-indep, sequential adaptation is void
  and compounding is irreducible-by-fitting -> end-to-end search
  or gradients, stated.

## 2026-09-22 — OUTCOME: sequential WORSE (0.895 < 0.962); overfit suspects

- A-seq 0.89462 vs A-indep 0.96230. Drift grows per layer
  (0.0048/0.0086/0.0110) yet depth gets worse — adapting to
  drifted inputs backfires. Prime suspect: OVERFIT — 13.7k
  correlated tokens vs ~37k map params at rank-96 (+near-zero
  ridge): the fits memorize train perturbations, break on test.
  Same ratio afflicts indep fits, but teacher-inputs are cleaner.
- Decisive follow-up running: N_FIT=30 diverse reals. A-seq-30 >>
  A-seq-10 -> data-starved, keep pushing closed-form. A-seq-30 ~=
  A-seq-10 -> structural: compounding irreducible-by-fitting ->
  the fork (end-to-end discrete search vs gradients). No conclusion
  until the 30-image run lands.

## 2026-09-22 — OUTCOME: data-starved, not structural (A-seq 0.895->0.976)

- 30 diverse reals: A-seq 0.97610 (from 0.89462, +0.08), now ABOVE
  A-indep (0.955-0.962). Drift unchanged (capacity-structural) but
  better maps handle it. Single-layer F stable ~0.999 (9-11/12);
  E (L0@96) flat 0.990-0.993 (capacity-bound, not data-bound).
- Closed-form sequential fitting WORKS — it was starved, not void.
  Trajectory favors more scenes. Running N_FIT=100 (still few-shot
  vs millions). Pre-registered read: A-seq-100 >=0.99 (gains
  continue, below bar) keeps the program open; >=0.999 goes to the
  gate; saturation <0.99 closes closed-form (diminishing) -> the
  fork.

## 2026-09-22 — OUTCOME at data-cap (A-seq 0.982, still rising — not closing yet)

- 48 reals (all we own): A-seq 0.98179 (10->0.895, 30->0.976,
  48->0.982). Still climbing but flattening. F single-layer stable
  ~0.999 (8/12); E flat ~0.99.
- Fairness check on the pre-registered read: the cap is OUR DATA
  (54 reals), not the method's curve — closing closed-form while
  capped-and-rising would be verdict-by-data-limit. Fetching 150
  more fit scenes (SEPARATE file fit_pool.npz — NOT strata_real,
  which would shift medians; stated) then one A-seq rerun decides:
  >=0.99 open, >=0.999 gate, flat fork.

## 2026-09-22 — OUTCOME at 198 scenes: saturated 0.976 -> the fork

- A-seq: 0.895 (10) -> 0.976 (30) -> 0.982 (48) -> 0.976 (198).
  Saturated below 0.99 with 4x data past the knee. Per the
  pre-registered read: closed-form layer-wise fitting is CLOSED
  (compounding irreducible-by-fitting). F single-layer stable
  ~0.999 (8-11/12); one narrowed layer absorbable, three are not.
- Fork decided by reasoning (stated, no compute spent): end-to-end
  DISCRETE search can't fix map values (RRR already optimal per
  layer; search would rediscover F) — retired. GRADIENT descent on
  a depth-graded narrow student, RRR-INITIALIZED: closed-form gives
  the start, gradients fix the compounding. Few-shot designs,
  gradients fit — each does what it can do.
- Proposed pilot (needs go-ahead, bigger build): student backbone,
  12 layers, early linears bottlenecked (L0-2 @96-192, rest full),
  RRR init, Adam on teacher pseudo-labels (fit_pool 204 scenes,
  SSI + gradient-matching per paper), gated same 12 scenes +
  margins. If gradients can't close 0.976->0.999 with RRR start,
  narrow-early is dead too and the student keeps full early width.

## 2026-09-22 — PRE-REGISTERED: gradient pilot (RRR-init, bottleneck L0-2)

- Student = teacher with L0-2 linears replaced by trainable
  bottlenecks (384->r->dout, r=128; norms/scales/embeddings/pos
  FROZEN teacher-exact — isolates linear learning). RRR init from
  198-scene covariances (0.976 start, free). Late layers frozen
  exact. Pilot isolates compounding-fix, not full student.
- Labels: teacher 518px depths, cached (deterministic). Loss: SSI
  (per-image least-squares align + MAE) + gradient-matching x2
  (single-scale Sobel L1); top-10% residual mask; NO feature
  alignment (semantics preserved by RRR init + frozen norms —
  stated reason, not oversight).
- Eval hygiene: held-out MUST exclude fit ids (fit_pool stride-7
  overlaps strata_real stride-25 stream) — fixtures + audit +
  non-overlap reals only, verified by id check before training.
- Predictions: closes 0.976 -> >=0.998 (near; compounding
  substantially fixed); HOLD (>=0.999) possible, not promised.
  Train->1.0 while held-out stalls <0.99 = data verdict (204
  scenes insufficient for gradients) -> stream more. Flat
  (held-out ~= 0.976 after 100 epochs) = narrow-early dead even
  for gradients -> student keeps full early width.

## 2026-09-22 — OUTCOME: gradients close half the gap (0.976->0.9928)

- 100 epochs, best ep29 0.99276 (4-5/23 pass), oscillating plateau
  after — no late breakthrough. Prediction missed (wanted >=0.998).
  Train/held-out gap NOT the story (loss fell steadily; held-out
  tracked it, oscillating) — capacity/schedule is.
- Diagnosis order (one variable at a time): rank first (E@96 flat
  0.99 vs F@192 0.999 single-layer says rank matters enormously at
  depth). Rerun RANK=192 same protocol: >=0.998 -> capacity story,
  continue up; ~0.993 -> not capacity -> schedule/data/unfreeze
  next. Narrow-early NOT retired (untried capacity gradient).

## 2026-09-22 — OUTCOME: capacity refuted (192 < 128 held-out); overfit diagnosed

- Rank-192: best 0.98979 (ep84) vs rank-128's 0.99276 — MORE
  capacity, LOWER train loss, WORSE held-out. Classic overfit
  (2.6M+ params / 191 scenes; 128-run train fell while held-out
  peaked ep29). Capacity gradient inverted under gradients, same
  pattern as the L-gradient inversion. Not capacity.
- Next variable: DATA (diagnosed cause, not a guess). Extend
  fit_pool to ~500 scenes (cheap streaming + oracle), retrain
  rank-128 identical protocol. Pre-registered: >=0.998 ->
  data story confirmed, continue scaling; ~0.993 again -> not
  data -> schedule (cosine) then rank-64 then unfreeze, in order.

## 2026-09-22 — OUTCOME at ~590 scenes: 0.99386, data helped but saturated

- Init 0.98314 (191) -> 0.99037 (590): data moved the CLOSED FORM
  +0.007. Gradients added +0.0035 more (best 0.99386, 8/56 pass).
  Pre-registered >=0.998 MISSED. Data was real but insufficient;
  the "not data" branch is now live.
- Against hasty ladder-climbing: the plateau (steady climb then
  flat, not oscillatory) + init-already-0.990 says optimization is
  NOT the limiter. Next diagnostic (cheap, no training): per-scene
  failure map of the best checkpoint by stratum — if failures
  concentrate, targeted data/strata beats blind scaling; if
  diffuse, unfreeze-late (absorbability: OUR finding says late
  layers absorb tweaks) outranks schedule.

## 2026-09-22 — Failure map: SYNTHETICS fail, reals hold (coverage, not capacity)

- Best r128 by stratum: E1T1L0V1 0.9974, E1T1L1V1 0.9966 (reals
  hold); E0T0L1V0 0.9847 uniform-mild; E0T0L0V0 0.9719 with
  exploration-2 0.9489, exploration-4 0.9526, audit-1 0.9436.
  Failures = SYNTHETIC scenes the fit never saw (fit is 100%
  real COCO). Same scene-interaction signature as the linear
  pilot's L11-synth 0.57 — teacher acts differ on synth, fitted
  maps never learned them.
- Fix (targeted, pre-registered): mix 9 synthetic scenes
  (exploration-6 + retention-3) into fit; eval keeps gate-4 +
  audit-4 + holdout-reals (never fit). Predictions: E0T0L0V0
  rises >=0.99, overall >=0.996; audit-synth (never fit either
  side) is the honest judge — if audit holds too, coverage was
  the whole story.

## 2026-09-22 — OUTCOME mixed run: 0.99594, coverage helped, not whole story

- 567 train (+9 synth), 54-scene eval, 100 epochs: best ep89
  0.99594 (10/54 pass) vs 0.99386 before. E0T0L0V0 0.9719->0.9870
  (exploration-2 0.949->0.971, exploration-4 0.953->0.994);
  E0T0L1V0 0.9847->0.9908; reals 0.9975+. Direction as predicted,
  both bars MISSED (E0T0L0V0 <0.99; overall <0.996 by 6e-5).
- Honest judge speaks: audit-1 (never fit either side) still
  0.9682 — worst scene. Fitted synthetics learned, unfitted
  synthetics lag. Coverage was real but NOT the whole story;
  residual gap is capacity/absorption on low-edge gradient scenes.
- Next (last untried lever before retiring narrow-early):
  UNFREEZE-LATE — late layers at small LR absorb early-narrowing
  errors (absorbability is OUR finding: late robust, early
  fragile). Pre-registered: unfreeze L9-11 @1e-6 alongside
  bottleneck training; >=0.998 continues, >=0.999 gates, flat
  retires narrow-early (student keeps full early width).

## 2026-09-22 — OUTCOME unfreeze: 0.99502, absorbability-fix fails

- L9-11 @1e-6 (+5.3M params), same protocol: best ep69 0.99502
  (7/54), BELOW frozen 0.99594, more late oscillation. Late layers
  do NOT soak up early-narrowing errors at this LR — absorbability
  (robustness to scaling) ≠ correctability (fixing others'
  errors). Third inversion in the program: robustness, linearity,
  and correctability are three different axes.
- Narrow-early @128 ceiling stands ~0.996 (four interventions:
  data +0.002, synth-mix +0.002, rank192 -0.003, unfreeze -0.001).
  Untried: cosine schedule, rank-64. Both cheap; neither
  conceptually favored (no finding points at them — stated).
  Decision point: spend two more runs, or accept 0.996 and move
  the program to full-early-width / phi-encode-best.

## 2026-09-23 — OUTCOME ladder (both cheap runs done, for completeness)

- Cosine r128: best ep89 0.99624 (16/54), train loss 0.041 (vs
  0.086 const) — fits train much harder, lifts held-out +0.0003.
  Crosses the 0.996 line by 2.4e-4. Best narrow-early number in
  program history; still 0.003 from hold.
- Rank-64 const: best ep79 0.99256, late degradation (min 0.789
  at ep74) — capacity floor confirmed from below (single-layer
  E@96 sat ~0.99, but 3-layer @64 compounds worse than @128).
- Ladder CLOSED. Narrow-early ceiling: 0.9962 (cosine). Seven
  interventions total; none reach 0.999. Standing decision:
  accept ~0.996 as the narrow-early result (bank depth-graded
  prior + RRR-init hybrid), or full-early-width student next.

## 2026-09-23 — PRE-REGISTERED: basin-breaker trio (user-approved)

- Q: are we trapped in the RRR basin (init 0.992, all gains
  ±0.004, cosine halves train loss for +0.0003 held-out)?
- (b) Synth-residual probe: RRR-128 bases from real-dominated
  covs; relative reconstruction residual on held-out REAL vs
  SYNTHETIC activations per map (L0-2). Predicts synth residual
  >> real residual (mechanism for E0T0L0V0 failures). If equal,
  coverage-in-activation-space is dead as a mechanism.

## 2026-09-23 — OUTCOME (b): bases span synth fine; mechanism dead

- L0 real/synth residuals near-identical (q 0.0031/0.0030, v
  0.033/0.035, mlp2 0.095/0.114); L1/L2 mixed, mostly <1.5x,
  inconsistent direction (L1-mlp2 BETTER on synth 0.155 vs
  0.205). Predicted strong synth>>real effect ABSENT.
- E0T0L0V0 failures are NOT poor fits on synth tokens — bases
  cover them. Residual suspects: error AMPLIFICATION through
  frozen late layers on smooth scenes, or corr-denominator
  sensitivity on low-variance refs. Both live downstream of the
  fit, consistent with deep supervision (c) still being the live
  basin-breaker.
- (a) Random-init run: proper small-random factors (NOT the
  dead A=0 default), same protocol. Lands >=0.994 -> no trap,
  gradients do the work, RRR-init unremarkable. Lands <=0.97 ->
  basin real, RRR carries the result, gradients only polish.

## 2026-09-23 — OUTCOME (a): basin real (random-init 0.77, 0/54)

- 0.147 init -> 0.77 best (ep~94), flat from ep30. Gradients
  alone on 576 scenes buy 0.62 of correlation; RRR-init buys
  0.99 before epoch 0. Few-shot training IS the closed-form
  init — gradients polish (+0.004) but cannot discover.
  The program's division of labor stands: closed-form designs,
  gradients fit.
- (c) Deep supervision: student L0-2 layer-outs vs teacher
  layer-outs (online teacher forward, no_grad) as extra loss
  term (lambda 0.1, variance-normalized) + depth loss. Directly
  penalizes compounding drift; changes landscape, not position.
  Predicts >=0.997 (beats cosine best) — the basin-breaker
  candidate. Flat (~0.996) -> drift-penalty insufficient.

## 2026-09-23 — OUTCOME (c): 0.99619, faster + tighter, same ceiling

- Best ep84 0.99619 (11/54), min 0.98269 — best worst-case in
  program history. Matches cosine 0.99624 on mean, tighter
  floor. But pre-registered >=0.997 MISSED: drift-penalty
  insufficient as a basin-breaker. Learns faster (0.99595 by
  ep49 vs ep89 depth-only), same ceiling.
- TRIO CLOSED. (b) coverage-mechanism dead; (a) basin real
  (0.77); (c) landscape-change insufficient. Nine interventions,
  ceiling 0.9962. The residual lives in synthetic-gradient
  scenes and survives: init, data x4, synth-mix, schedule,
  rank±, unfreeze, deep supervision. Next candidates worthy of
  the name: full-early-width student (concede narrowing), or
  accept ~0.996.

## 2026-09-23 — PRE-REGISTERED: full-width student (causal isolation)

- Design: L0-2 FULL direct maps as Parameters (teacher-exact
  init — no bottleneck, no RRR), rest frozen; same labels,
  loss, protocol, gates. Asks the causal question the whole
  program has circled: is the 0.996 gap narrowing capacity or
  the training itself?
- Predictions: HOLDS >=0.999 (pipeline sound; gap = narrowing
  capacity, isolated cleanly — the geometric account closes:
  rank-deficiency in early maps is THE mechanism). DROPS to
  ~0.996 too -> training-limited (loss/data/pipeline), rethink
  everything — narrowing exonerated, bigger finding.
- Either outcome is decisive; that is the point of the run.

## 2026-09-23 — OUTCOME full-width: 0.99639, narrowing EXONERATED

- Teacher-exact init 0.99998 (54/54) -> training DEGRADES it
  (0.983 by ep9), oscillates 0.989-0.996, best ep14 0.99639.
  Full capacity (5.3M free params) lands within 0.0002 of the
  bottlenecked cosine best (0.99624). The bottleneck was NEVER
  the binding constraint under gradient training.
- Causal chain closed: not capacity (this run), not optimization
  effort (cosine halves train loss, +0.0003 held-out), not
  activation coverage (bases span synth fine), not drift penalty
  (deep-sup same ceiling). Remaining: the SUPERVISION itself —
  SSI aligns scale+shift per image (absolute geometry
  unsupervised), 576 scenes underdetermine millions of params,
  and the walk off the teacher manifold happens in
  loss-invisible directions. Low-variance synth refs amplify
  tiny absolute errors into corr failures.
- The geometric account of why distillation works AT ALL here:
  teacher early-layer activations are low-rank ON-MANIFOLD
  (L0-Q effrank 27/384), so closed-form RRR projection loses
  almost nothing (0.992 before epoch 0); late layers are
  full-rank but gain-robust. Gradients polish (+0.004),
  random-init stalls (0.77). Small nets aren't generally
  capable — THIS teacher's early computation is approximately
  low-rank on real data, and that is the whole ballgame.
- Per prior agreement: bank-and-encode next (cosine r128
  0.99624 best-mean; deep-sup 0.99619 best-floor — bank the
  cosine artifact, note the floor).

## 2026-09-23 — PRE-REGISTERED: covariance convergence (teacher's data needs)

- Q (user): minimum set of data to train THIS teacher's early
  computation? Direct measurement: stream RRR covariances over
  fit_pool in fixed order; snapshot W_N at N={5,10,20,40,80,160,
  320}; per-map relative drift ||W_N-W_final||/||W_final|| +
  held-out token-output corr of W_N (2 reals + 1 synth).
- Predictions: elbow at 20-40 scenes (drift <1%); mlp2 slowest
  (largest map); token-corr saturates with drift. No elbow by
  320 -> minimal-set claim weakens (covariances keep moving).
  This quantifies the closed-form 90% of our result, not the
  gradient polish.

## 2026-09-23 — OUTCOME: maps identified by ~40 scenes; drift is null

- Drift: q/k tiny by N=5 (L0-Q 1.3% — 5 scenes identify it,
  effrank-27 consistent); v/proj/mlp2 still 9-15% at N=160,
  decaying ~1/sqrt(N), no sharp elbow. mlp2 slowest ✓.
- Tokcorr: functionally FLAT from N=20-40 on (L1-v 0.9777 at
  20 vs 0.9810 final; L2-mlp2 0.9675 vs 0.9704). Drift keeps
  falling past 40 but tokcorr doesn't move — late drift lives
  in functionally-NULL directions.
- Teacher's data needs, quantified: ~40 scenes identify the
  early maps functionally; ~5 scenes identify L0-Q/K. The
  A-seq gains past 40 came from the gradient phase, not the
  maps. Minimum set for the closed-form 90%: dozens, not
  thousands. Prediction half-holds (elbow real in function
  space, absent in parameter space — the distinction IS the
  finding).

## 2026-09-22 — Bars by hand: what they did and didn't decide

- Asked directly (user): could hand-set bars have affected results?
  Answer, split honestly. ROBUST (any sane bar decides the same):
  drops (0.2-0.9), shared-Q (0.36), sparsity (0.28), rank (0.08),
  dmax1024 (0.6), analytic/direct rejections. FRAGILE (bar position
  is the verdict): codebook 0.9988 vs 0.999, L1h0 0.99918, tap2
  split counts, per-layer-4096 3/6. The bar was ALSO set at the
  seed's floor (0.99901), so all fragile verdicts are razor-thin by
  construction — conservative (nothing undeserved promotes) but
  brittle (millipoint noise decides ties).
- Consequence: margin gates (direction 5) don't just add resolution —
  they re-examine every fragile verdict. Guardrail, pre-stated:
  margins calibrate from SEED variance, never tuned to flip a named
  verdict; margins pre-register before outcomes are seen. Widening a
  margin to manufacture a win = moving the bar = ruled out.
- Order of work: margins first (they condition how neck verdicts are
  read), neck second. Margin re-analysis needs per-case corrs —
  tap_study printed fractions only, so it re-runs once WITH a JSON
  record; every future probe records per-case numbers by default.

## 2026-09-22 — PRE-REGISTERED: margin re-analysis + neck DSL v4

- M1: tap_study rerun recording per-case corrs (79 scenes x 7 cfgs);
  margin rule (per-case pass iff c >= seed_c - m, m from seed
  variance: max seed shortfall across cases + epsilon) applied to
  tap moves, L1h0-class near-misses, codebook probe means.
  Predictions: tap0 holds-as-tie in MORE strata under margins (its
  0.67-0.86 fractions sit millipoints under, not collapsed);
  codebook 0.9983 stays a REGRESSION (gap 0.0017 >> any
  seed-calibrated margin — margins add resolution, not wins);
  tap3 collapse unchanged.

## 2026-09-22 — OUTCOME M1: margins resolve tap0 as regression (both ways work)

- Calibrated m=0.000274 (seed max-min range over 79 cases; NOTES
  said "max shortfall + epsilon" ambiguously — the range formula in
  margin.py is authoritative, same for all candidates, computed
  blind. Clarification recorded, not edited.)
- tap0 under margins: full ties in only 3 strata (E0T0L0V0/L1V0,
  E0T1L0V0); 0.00-0.50 across E0T1L0V1, E1T0L1V0, E1T1Lx. The
  "holds 6/12" binary read was bar-artifact: tap0 corrs sit above
  0.999 but systematically >m short of seed. Prediction inverted
  honestly: margins made a tie into a MEASURED REGRESSION.
  Resolution cuts both ways — that is the point of the instrument.
- codebook 0.9983: gap 0.00168 >> m, stays regression ✓. tap3: 0.00
  everywhere ✓. L1h0-class: same re-read available (recorded
  0.99918 vs seed floor — within a seed-calibrated margin? seed
  range here is 0.000274 but that sweep ran on different scenes;
  margins calibrate per-evaluation, never imported — stated rule).
- Bar-effect question CLOSED quantitatively: hand bars decided the
  fragile verdicts (tap0 "tie" was bar-artifact); robust verdicts
  untouched. MarginRule module (adapt/margin.py) ships as the
  portable form; upstream follow-up to the baseline-gate PR.
- M2: MarginRule domain-side (adapt/margin.py; EfficiencyRule
  composes, core untouched) for the neck search.
- N1: neck DSL v4 (per-stage channel halving 48/96/192/384 +
  single fusion-stage drop; honest NECK_PARAMS bytes) under
  EfficiencyRule+MarginRule. Predictions stand (stage drops fail;
  48-stage halves tie, 384-stage fails).

## 2026-09-22 — PRE-REGISTERED: neck v4 search (drops + stream zeros)

- Scope correction (honest): channel HALVING needs fitted narrower
  weights (no fitting harness exists) — v4 probes what deletion can:
  fusion-stage drops (1 of 4, size chain bridged by harness interp,
  documented) + reassemble-stream zeros (1 of 4 pyramid scales) +
  conv-channel zero-halves (structured, no fitting). Byte model:
  per-stage neck params measured EXACT from npz shapes (same
  discipline as ATTN/MLP block bytes).
- Driver adapt/neck_search.py (own DSL {stage_drop, stream_zero,
  chan_half}, EfficiencyRule, same fixtures/corr/margin reading).
  ~20 trials to exhaustion, minutes not hours.
- Predictions: stage drops fail (pyramid roles load-bearing, like
  taps); stream-zero small-scale (i=3) nearest to tie; channel
  halves fail (backbone sparsity died at 5%). Any hold+bytes-win
  goes to the emitter immediately.

## 2026-09-22 — OUTCOME N1: neck holds everywhere; margin gate catches 3

- 12/12 rejected, incumbent retained. Stage drops all collapse
  (means 0.13 to -1.0). Stream-zero i=3 collapses HARDEST (-1.0) —
  prediction inverted on which stream (deep stream matters most,
  consistent with tap-12 irreplaceability); nearest-to-tie is
  zero1/zero0, not i=3. Stated.
- The instrument paid for itself: zero1, half0, half1 are binary
  FULL TIES (e6/6 g4/4 r3/3) with millipoint mean drops — rejected
  SOLELY by `margin_regressed:exploration`. Without the margin gate
  all three PROMOTE on bytes (false wins). Real panel agrees
  (3/8, 6/8, 5/8 — rejection direction). This is the second time a
  guard earned its keep by refusing (dissolution guard was first).
- Neck CLOSED (drops fail, halves fail-or-margin, zeros fail).
  Direction 6 DONE. Remaining open: distillation design (discussion).

## 2026-09-22 — PRE-REGISTERED: neck topology search

- DSL v4: fusion depth (2 vs 3 residual blocks per stage? — NO:
  start with what exists) — honest scope: reassemble channel widths
  (48/96/192/384 -> narrowed variants), fusion stage skips (drop 1 of
  4 fusion stages), readout is covered. Moves: per-stage channel
  halving + single-stage drop, EfficiencyRule with honest neck bytes
  (2.7M params, measured NECK_PARAMS).
- Predictions: stage drops fail (pyramid roles load-bearing, like
  taps); channel halving on the 384-stage fails, on 48-stage ties
  (capacity-graded, mirroring tap depth-grading). If the 48-stage
  halves AND holds, it is a real neck win and goes to the emitter.

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

## 2026-09-23 — PRE-REGISTERED: activation-maximization pilot (core inputs)

- Q (user): the irreducible CORE inputs — what stimuli do the
  neurons themselves ask for? Convergence used fit_pool COCO reals
  only (no synthetics in the identification set). Gradient ascent
  on 518px pixels (mean-init) maximizing L0 post-GELU channels
  (high-variance picks) + one attention-output channel; jitter
  (random roll) + TV + L2 priors, ~200 Adam steps. Teacher path
  untouched (forward_stages takes pixel_values directly).
- Predictions: stimuli show edge/texture structure, not objects
  (early ViT); target selectivity >5x vs other sampled channels;
  synth stimuli drive their channel harder than any of 10 real
  scenes do (else the prior failed, not the neuron).
- Reading: structured + selective -> stimuli span the manifold
  directions; feed them to RRR next (close the loop: failures
  -> target neurons -> synthesized stimuli -> re-fit -> gate).
  Mush/unselective -> pilot method insufficient, core-inputs
  question stays open.

## 2026-09-23 — OUTCOME actmax: method insufficient (both configs fail)

- Mean-init (LR 0.05): all stimuli converge to near-gray (±0.04),
  drive targets at 0.2-0.3x of best-real. Under-movement (per-pixel
  grads of a token-mean objective dilute ~1/1369; Adam can't
  compensate from a flat start).
- Real-init (LR 0.5, 300 steps): worse — 3/4 targets end BELOW
  their starting image's activation (mlp378 final -0.08);
  selectivity rows identical across stimuli (all runs hug the
  shared init). Stochastic jitter + joint token competition
  defeat naive ascent.
- Per pre-registration: pilot method insufficient. Pixel-space
  ascent on ViT without serious priors (Fourier/decorrelated
  parameterization) is real engineering, not a pilot. Pivot
  proposed: CORESET SELECTION instead — greedy forward selection
  of the minimal real-scene subset identifying the maps
  (covariances are additive, evaluation is tokcorr — answers
  "core inputs" empirically with no synthesis).

## 2026-09-23 — PRE-REGISTERED: greedy coreset (core inputs, empirical)

- Pool 150 fit_pool scenes (fixed order); per-scene covariances
  cached (L0-2, all maps). Probe: 2 held-out reals + 1 synth
  (tokens cached). Greedy to K=25 on mean held-out tokcorr
  (OLS-scored, held-out = overfit guard); RRR-128 verify final;
  random-25 x3 control.
- Predictions: greedy beats random-25 clearly (>=0.002 mean
  tokcorr); elbow <=15 scenes (gains concentrate early);
  selected scenes span >=4 strata by scene ~10 (diversity
  emerges unenforced). Random ties greedy -> scenes
  interchangeable, core = "any ~25 reals" (weaker but still an
  answer).

## 2026-09-23 — OUTCOME coreset: random ties greedy; core is statistical

- Greedy-25 RRR: mean 0.99192 / worst 0.95444. Random-25 x3:
  means 0.99195/0.99222/0.99180, worsts 0.953-0.954. TIE
  (within noise on both metrics). "Greedy wins >=0.002" FAILED.
- Elbow real: worst 0.9935 -> 0.9999 by step 2, 1.0000 by step
  5 (OLS); gains zero after. Diversity emerges unenforced
  (5 strata by scene 10, 9 by scene 25) ✓.
- Reading: the irreducible core is STATISTICAL, not specific.
  No special scenes exist — any ~5-25 diverse reals identify
  the maps. This explains covconverge (~40 random scenes work)
  and reframes the synthetic question: synthetic inputs need
  only match the activation DISTRIBUTION (strata edge/texture
  stats), not specific content. Caveat, stated: greedy
  optimized OLS-worst, verified RRR — truncation may wash out
  selection differences; the tie is reported, not oversold.

## 2026-09-23 — PRE-REGISTERED: synthesis loop (user-directed)

- Q: can we GENERATE the core instead of selecting it? Parametric
  generator (gradients + occluder discs + checker/noise texture,
  wide ranges) -> 200-scene pool -> greedy-25 by worst-tokcorr
  (same machinery) vs greedy-real-25 (mean 0.99192/worst 0.95444)
  + random controls. Stimuli saved to captures/synth_core/ with
  a contact sheet for human viewing.
- Predictions: synth-25 within 0.002 of real-25 on mean but lags
  on worst-case (natural phase structure matters); refinement
  (mutate winners, re-greedy) closes half the gap. Synth matches
  reals -> training data fully synthesizable (huge). Synth <0.98
  -> natural statistics essential beyond 4 stats (also an
  answer).
- Phase 2 (only if phase 1 close): RRR-fit student on synth set,
  depth-gate. Refinement loop proper (v2) follows phase-1 numbers.

## 2026-09-23 — PRE-REGISTERED: heightfields + inversion (user-directed)

- Heightfields (analytic 3D): RGB + TRUE depth, zero oracle.
  Height = planes + Gaussian bumps (known objects/placements)
  + steps; Lambertian shading (random light) + albedo texture.
  Knobs: range/lighting/texture exactly as user specified.
  Pool 200 -> greedy-25 vs real-25/synth-25. Predicts BETWEEN
  (beats parametric 2D, trails photos: shading carries shape
  cues parametrics lack, misses natural clutter).
- Inversion (not maximization): reproduce a measured L0-block
  activation pattern from noise init (TV + jitter). Success =
  activation match + structured image. Predicts: works for
  single-pattern match (overdetermined: 268k pixels -> 1370x384
  targets... actually UNDERdetermined, should fit); the SET
  question (spanning inversions as training data) stays open
  pending single success.

## 2026-09-24 — PRE-REGISTERED: post-hoc verify + phase 2 (killed runs)

- Killed both greedy runs (steps 7-9, gains ~0, ~1 day left).
  Post-hoc: RRR-128 verify greedy-7 per pool + random-7 x3 on
  the fixed 7-scene probe. Predicts real-7 >= synth/hf-7 on
  worst-case (gap from early steps persists); hf-7 >= synth-7.
- Phase 2: RRR-128 students fit on greedy-7 sets, depth-gated
  on 12 scenes. Predicts real-7 best, hf middle, synth last;
  all below bar (7 scenes << 40-scene identification knee).
  The comparison across sets (not the absolute) is the result.

## 2026-09-24 — OUTCOME phase 2: synth/hf BEAT real-7 (inverted, informative)

- Depth-gated RRR-128 on greedy-7 sets: synth-7 0.97911 (min
  0.93, 1/12) > hf-7 0.97776 (min 0.93) > real-7 0.97139 (min
  0.78). Predicted order REVERSED.
- Reading (two rivals, stated): (i) small-N noise dominates —
  all three starved (7 << 40-scene knee), tokcorr ranking need
  not transfer to depth ranking; min gaps (0.78 vs 0.93) argue
  against pure noise. (ii) STRUCTURED hypothesis: simple scenes
  give stable covariances (well-conditioned fits), diverse
  reals give high-variance covariances (overfit) at tiny N —
  synthetics better PER-SCENE few-shot, reals win with volume
  (the A-seq curve). Discriminating test: learning curves per
  pool (depth vs N at 7/15/30) — crossover predicted.
- Either way the coreset->training link is qualified:
  identification-quality != training-quality at small N.

## 2026-09-24 — PRE-REGISTERED: learning curves per pool (crossover test)

- RRR-128 L0-2 students on random-N sets (2 draws/cell),
  N={7,15,30,60} x {real,synth,hf}, depth-gated same 12 scenes.
  Random (not greedy): random≈greedy established, keeps cells
  comparable. Predicts CROSSOVER: synth/hf above real at 7
  (phase-2 result), gap narrows by 15, real ahead by 30-60
  (volume wins). All below bar (absolute secondary to order).
- No crossover either way also decisive: synth leads
  throughout -> structured hypothesis strengthened (clean
  statistics dominate); real leads throughout -> phase-2 was
  noise, retract the per-scene claim.

## 2026-09-24 — OUTCOME curves: no scaling, no separation, one poison

- Means sit 0.96-0.99 in ALL 24 cells (pool x N x draw) save
  one: synth-N60d0 collapses (0.717, min 0.19), d1 fine
  (0.987). No crossover, no N-scaling (random-60 ~= random-7),
  no pool separation. Phase-2 "synth>real" RETRACTED as draw
  noise (0.008 gap < ±0.01 draw variance measured here).
- The poison cell is the finding: synth pool contains scenes
  whose inclusion wrecks fits (hf random-verify worst -0.48
  was the same phenomenon). Photos show no poison in any
  draw. Selection matters for synthetics as POISON AVOIDANCE,
  not magic-scene discovery. Independent RRR plateaus by
  N≈7-15 everywhere; A-seq/gradient gains came from fitting
  machinery, never data volume.
- Next: poison autopsy (which scenes? what stats?) to constrain
  generator ranges — the refinement loop's actual job spec.

## 2026-09-24 — PRE-REGISTERED: poison autopsy

- Reproduce synth-N60d0 indices (rng(0) consumption order);
  stats screen (profile + activation norms vs d1/pool);
  drop-one-out tokcorr screening (CPU) to rank suspects;
  confirm by refit-minus-suspects + depth-gate (1 GPU run).
- Predicts: 1-3 scenes carry the damage (removal restores
  >=0.97); suspects are stat outliers (extreme edge/texture
  or flat-black — generator range excess). Diffuse damage
  (no single removal helps) -> conditioning story, not
  poison scenes -> constrain overall ranges instead.

## 2026-09-24 — OUTCOME autopsy: NO poison scenes (self-correction)

- Confirm: d0-with-poison 0.71687 reproduces; d0-minus-top3
  0.98570 recovers to d1-clean 0.98706. Suspects (synth-27/
  138/49) are input-stat ORDINARY (all E0T0L0V1, mid-range
  edge/texture — kills the range-excess prediction).
- CORRECTION (control run): d0-minus-RANDOM3 recovers
  IDENTICALLY (0.98548, min 0.92). The top-3 are NOT special
  (previous message crowned them prematurely — corrected).
  ANY 3 removals restore parity; d1 (N=60, 19 scenes shared
  with d0) was always fine. The collapse is a knife-edge in
  the specific 60-combination's conditioning, not a scene
  property at all.
- Consequences (refinement-loop spec rewritten): blacklist
  scenes VOID; input-stat range constraints VOID (suspects
  ordinary); "poison" framing RETIRED — it is a fitting
  instability. What works: cap set size at the elbow (~10,
  gains nil past it anyway) + ridge scaling with N. The loop's
  job is conditioning control, not scene selection.

## 2026-09-24 — OUTCOME webcam sanity: works live, dark-room gap diagnosed

- Student runs live (25ms/frame post-warmup, plausible depths)
  via student_webcam.py (teacher + student + optional HF side
  by side). Sanity PASSES as a pipeline.
- Fidelity on dark-room scene: student-vs-teacher 0.71-0.81
  (mean 0.77) vs 0.996 held-out; teacher-vs-HF holds 0.96 on
  the same frames. Gap is student-specific, scene-specific.
- Ruled out: uniform darkness (0.3x costs 0.01, teacher holds
  0.993); global photometrics (histogram EQ: helps 2/5, hurts
  3/5). Remaining: dark textured low-light content itself —
  crushed blacks + sensor noise + casts, a regime with ZERO
  representation in fit (bright COCO + bright synth). Effect
  direction matches all prior scene-interaction findings.
- Fix path (not yet run): low-light augmentation (darken +
  clip + noise + tint) in fit/training, re-gate. Targeted,
  cheap, pre-register on launch.

## 2026-09-24 — PRE-REGISTERED: consolidation sequence (user-approved)

- Order: cosine+deepsup combined (best mean + best floor, one
  run) -> low-light augmentation retrain (diagnosed dark gap)
  -> unfreeze-late at real LR (1e-6 tested nothing) ->
  factored inference (bit-exact vs dense) -> encode winner
  (phi + C emit) -> inversion tool.
- This run: STUDENT_DEEPSUP=0.1 + --schedule cosine, same
  protocol/labels/eval. Predicts >=0.99624 (matches cosine
  best) with min >=0.97 (keeps deepsup floor); a HOLD would
  be the program's first genuine compression lead.

## 2026-09-24 — OUTCOME combined: 0.99688, new best (both parents)

- Best ep59 0.99688 (13/54, min 0.98725): beats cosine-only
  0.99624 on mean AND keeps deepsup's floor (min 0.987 vs
  0.971). Combination strictly dominates both parents.
  Pre-registered bars pass; HOLD (0.999) still distant, stated.
- Next per sequence: low-light augmentation retrain.

## 2026-09-24 — PRE-REGISTERED: low-light augmentation retrain

- Dark-room gap (webcam 0.77): fit has zero dark scenes.
  Online aug in gradient phase only (RRR init unchanged):
  per-batch darken xU(0.25,0.6) + shadow clip + Gaussian
  noise + tint jitter; geometry preserved so clean teacher
  labels stay valid targets. Eval: same 54-gate + dark-eval
  (0.3x versions) + webcam re-test on saved dark frames.
- Predicts: dark-eval/wet-cam fidelity rises (>=0.90 on saved
  dark frames); clean held-out holds >=0.996 (no aug
  regression). Clean regresses -> aug too strong, dial back.

## 2026-09-24 — OUTCOME aug-v1: destructive interference, killed at ep9

- Dark-eval 0.933 (init) -> 0.758 (ep4) -> 0.705 (ep9) while
  clean held 0.995. Gradient-phase-only dark batches vs a
  clean RRR basin: off-manifold gradients destroy rather
  than teach. Killed (primary metric decisively worsening).
- Redesign (v2): dark enters the BASIN (augmented RRR
  covariances) + gentler photometrics. Pre-registered: init
  dark-eval starts higher; end dark >=0.90, clean >=0.996.

## 2026-09-24 — Label flaw found + killed v2, v3 pre-registered

- Teacher(dark)-vs-teacher(clean): 0.78 gentle / 0.75 strong
  (min NEGATIVE). Darkening changes teacher predictions, so
  v1/v2's dark-RGB->clean-label pairing mis-supervised dark
  inputs — the student learned a mapping the teacher doesn't
  implement, and dark-eval (vs teacher-on-dark) correctly
  punished it. Killed v2 at ep14 (dark 0.76, clean 0.996).
- v3 (correct-labels): STUDENT_DARKPOOL=N appends fixed dark
  variants as first-class pairs; labels computed ON dark
  images; RRR + gradients flow automatically. No online aug.
  Pre-registered: dark-eval climbs from init (no crash);
  clean holds >=0.996; saved webcam dark frames >=0.85.

## 2026-09-24 — CHECKPOINT HYGIENE FAILURE (own fault, recorded)

- v3's first launch reused BEST filename _ds0.1_cos (no
  darkpool suffix) and overwrote the 0.99688 champion at ep4
  with a 0.98267 intermediate. runs/ is gitignored: weights
  unrecoverable except by re-running combined (~3h GPU).
- Fixed: BEST_STEM now stamps every lever (_darkpoolN);
  dark-eval runs when DARKPOOL_N>0 too. Rule going forward:
  checkpoint filename must be injective in config — a
  collision is a data-loss bug, treated as such.
- Cost: champion must be re-earned if needed for encode.
  Mitigation accepted; combined config re-runnable exactly.

## 2026-09-24 — OUTCOME v3: dark oscillates, never stabilizes (killed ep34)

- Dark-eval: 0.939 init -> dip 0.868 -> climb 0.920 -> crash
  0.825 -> 0.877/0.896, min stuck ~0.5. Clean held 0.994-0.996
  throughout. Correct labels fixed the DIRECTION (no more
  monotonic collapse) but not the VOLATILITY: shared rank-128
  bottleneck alternates between regimes (capacity competition).
- Next: effrank-on-dark diagnostic (does dark need different/
  more subspace directions?) BEFORE any rank-192 run — quantify,
  then decide. Killed v3 (pattern established over 30 epochs).

## 2026-09-24 — Subspace overlap kills capacity story; interference test

- Effrank clean-vs-dark: identical rank (L0 101/152 vs
  106/170; L5/L11 identical). Top-128 subspace overlap
  0.91-0.93 (vs ~0.06 random): SAME span, different
  loss-landscapes. Bottleneck capacity exonerated twice;
  remaining mechanism is optimization interference across
  regimes sharing one span.
- Test (pre-registered): dark fraction 0.26 -> 0.15 (same
  v3 design otherwise). Interference scales with fraction:
  dark stabilizes >=0.90 sustained + clean holds >=0.996.
  Still oscillates -> accept dark limitation, bank combined,
  move to factored inference + encode.

## 2026-09-24 — OUTCOME fraction test: dark limitation accepted

- 0.15 fraction: dark oscillates 0.86-0.91 (min ~0.3-0.45),
  never >=0.90 sustained; clean holds 0.993-0.996. Same
  verdict as 0.26-fraction: interference damps with fraction
  but never stabilizes. Killed ep30 (pattern replicated).
- Representation (overlap 0.92) vs optimization capacity
  stands as the fourth axis distinction: one span serves
  both regimes for FITS, joint GRADIENT optimization cannot
  hold both. Dark limitation ACCEPTED: document (webcam dim
  rooms degrade), don't chase. Clean line unaffected.
- Sequence update: unfreeze-late@real-LR (still open) drops
  below factored-verify + encode in priority — dark work is
  closed, ship the clean winner.

## 2026-09-24 — Clobber confirmed; encode plan (bank cosine-0.99624)

- Combined 0.99688 UNRECOVERABLE (file holds first-v3 ep4
  0.98267; runs/ gitignored). 6e-4 above surviving best —
  NOT worth 3h to re-earn; noted as optional rerun.
- Encode candidate: cosine-only r128 (0.99624, best surviving
  bottleneck mean; deepsup 0.99619 best floor noted).
  Fullwidth file (0.99639) excluded: full-size maps, nothing
  to compress — retrained teacher, not a student.
- Encode = phi-encoded student backbone npz (drop-in for the
  geometric loader: L0-2 maps phi-quantized, rest teacher
  baked) + integer-path parity + (follow-up, stated) C
  assembly from existing kernels. Factored mirror verifies
  at maxabs 1.1e-4 (rounding order, expected).

## 2026-09-24 — OUTCOME bank: phi cost 1e-5, artifact exact

- Banked npz (41.7MB, 5.3M student params phi-encoded, rest
  teacher): 12-scene subset mean 0.99498/min 0.97225 —
  IDENTICAL to float factors (0.99499/0.97230) to 4dp. Gate
  weakness lives in the checkpoint, not the encoding. Integer
  bake path accepts the npz (bake_layer_fixed ran clean).
- Factored mirror: maxabs 1.1e-4 vs composed (rounding order);
  measured 1.06x (8.4->8.0ms) — attention/norms dominate, not
  linears. torch.compile fails (env inductor bug, not our
  code). Speed win banked as-is (12% smaller, 6% faster).
- Remaining encode: C-assembly from existing kernels
  (follow-up, stated — kernels all wired, firmware assembly
  untracked here). Receipt: adapt/runs/student_banked.json.

## 2026-09-24 — OUTCOME banked webcam re-test: 0.975 (light-dependent)

- Banked cosine student live: student-vs-teacher 0.959-0.988
  (mean 0.975), student-vs-HF 0.984-0.992. Different view,
  dim-but-structured scene (cables/fabric, still darkish).
- vs prior 0.767 run: same room, different light/composition.
  Corroborates the diagnosis directionally (fidelity tracks
  illumination regime, not pipeline): bright/structure good,
  near-black fails. The accepted dark limitation stands with
  a live number on each side.

## 2026-09-24 — Inversion bottleneck diagnosed (norm ambiguity + attention)

- 2000-step teacher inversion stalls at 0.762 (not 0.99).
  Per-tensor: K 0.948 / Oattn 0.838 / Q 0.751 / G 0.727 /
  Y 0.659 / V 0.488 / C 0.364 / H 0.269. Spread, not uniform.
- Mechanism: LayerNorm scale-null-space (infinite pixel inputs
  map to same H — H structurally unrecoverable at 0.27) +
  softmax winner-take-all attention patterns (C 0.36; tiny
  pixel changes flip patterns -> gradient chaos). K easy
  (direct linear, smooth). Ceiling is architectural, not
  optimizer weakness — a characterization result in itself.
- Tool proceeds as comparison (teacher-vs-student under
  identical protocol): same ceiling structure both sides,
  differences still informative. Student 2000-step run next.

## 2026-09-24 — Inversion comparison ready; output-hygiene fix

- Student inversion (identical protocol): match 0.786, H 0.63
  vs teacher H 0.27 — narrowed maps invert EASIER (smoother,
  lower-rank landscape). Characterization, not malfunction.
- Output collision (same class as ckpt clobber): student strip
  overwrote teacher's (shared captures/invert/). --tag flag
  added (captures/invert_TAG). Re-running teacher tagged for
  the difference analysis. Rule restated: every run artifact
  path must be unique in config.

## 2026-09-24 — OUTCOME inversion tool: works, no hidden catastrophe

- Teacher (0.762) vs student (0.786) inversions of the dark
  frame: both recover coarse layout + texture at 14px patch
  granularity; mean|diff| = 0.102 pixel units, p99 0.341
  (measured — an earlier draft of this entry guessed 0.03,
  corrected). Same coarse structure, moderate pixel-level
  differences; student blurrier, consistent with lower-rank
  maps. No catastrophic blindness, but the gap is visible,
  not millipoint-only at this level.
- Tool stands (adapt/invert.py: live GUI + headless + --tag +
  --student + per-tensor diagnostics). Lesson: L0-inversion
  ceiling (~0.76, norm ambiguity + attention competition)
  dominates over teacher/student differences at this level.
  Deeper-layer inversion (where L11-nonlinearity lives) is
  the follow-up if differences are wanted sharper.

## 2026-09-24 — Latest-model webcam: 0.991 live (sanity PASSES)

- frac15-darkpool mid-training checkpoint live: 0.990-0.994
  vs teacher (mean 0.9906), 0.982-0.995 vs HF, 24ms/frame.
  Scene has both bright curtain and near-black corners; the
  student tracks the teacher across the range including dim
  regions (brighter room than the 0.767 run, dimmer than
  lab scenes). Live processing confirmed on the latest
  weights, not just the banked ones.

## 2026-09-24 — PRE-REGISTERED Phase A1 (multi-scale GM)

- Winner config (deepsup+cosine, clean protocol) + GM scales
  1/0.5/0.25. Absolute-anchor variant DROPPED by reasoning
  (meter is affine-invariant; absolute errors gate-irrelevant
  — stated, no compute spent). Predicts overall >=0.99688
  (beats combined best) with E0T0L0V0 min rising. Flat ->
  supervision-specificity dead, move to Phase B.

## 2026-09-24 — OUTCOME Phase A1: 0.99702 overall, E0 NOT fixed

- Best ep54 0.99702 (14/54, min 0.98304): beats combined
  best; first 0.997+ in program history. Overall bar PASSES.
- E0 bar FAILS: E0T0L0V0 0.98178 (was 0.98695); the gradient
  scene exploration-2 WORSENED 0.971->0.919 while audit-1
  improved 0.968->0.994. Multi-scale GM helps edge/texture
  scenes (E1T1L0V1 holds 1.0000) and HURTS smooth ones —
  gradient loss has nothing to match on smooth scenes, so it
  pulls maps away from them. Mechanism understood, stated.
- Phase B mandated with teeth: E0 needs COVERAGE (scenes in
  fit), not sharper loss. exploration-2 (0.919) is now the
  single worst scene and the cleanest target in program
  history.

## 2026-09-24 — PRE-REGISTERED Phase B (synth-gradient coverage)

- Winner config (deepsup+gms+cosine, clean) + 100 parametric
  synth scenes (E0-heavy) as first-class pairs, teacher labels
  via staleness rebuild. Predicts: exploration-2 rises >=0.97,
  E0T0L0V0 mean >=0.99, overall holds >=0.997 (no coverage-shift
  regression). E0 flat -> coverage insufficient too; rethink
  smooth-scene representation (Phase C capacity).

## 2026-09-24 — OUTCOME Phase B: E0 +0.003, overall -0.0008 (trade)

- Best 0.99620 (ep19). E0T0L0V0 0.98443 (was 0.98178);
  exploration-2 0.9360 (was 0.9191); audit-1 0.9912.
  Coverage helps E0 marginally but dilutes everything else:
  overall 0.99702 -> 0.99620. No bar passes.
- Interaction hypothesis (mechanism-grade): fine-scale GM
  (0.25) punishes smooth scenes; coverage supplies them; the
  two fight. Test: synthpool100 + GM scales 1/0.5 only (drop
  0.25). Predicts E0 >=0.99 AND overall >=0.997 (both bars
  together). E0 flat again -> smooth representation is the
  wall -> Phase C / accept.

## 2026-09-24 — OUTCOME AxB: E0 best-ever, overall drops (tension located)

- Best 0.99578 (ep44). E0T0L0V0 0.98823 (best ever);
  exploration-2 0.9756 (best ever, was 0.919-0.936).
  Overall 0.99578 < A1 0.99702. Interaction half-holds:
  dropping 0.25-scale GM fixes E0 (fine GM was hurting
  smooth scenes) but weakens edge-scene supervision.
- Tension located precisely: edge scenes want fine-scale GM,
  smooth scenes want it gone; one global loss can't serve
  both. Next (Phase A2, mechanism-derived): edge-weighted GM
  — per-image GM weight from input edge stat (smooth scenes
  get ~0 GM, textured get full). Predicts E0 holds >=0.988
  AND overall returns >=0.997 (both bars together, for real
  this time). Flat -> loss design exhausted, Phase C.

## 2026-09-24 — PRE-REGISTERED Phase A2 (edge-weighted GM)

- A1+gms-full+synthpool100 + per-image GM weight =
  clip(edge/median, 0.1, 2.0), MAE unweighted. Predicts E0
  holds >=0.988 AND overall returns >=0.997 (both bars
  together). Flat -> loss design exhausted, Phase C.

## 2026-09-24 — OUTCOME A2: E0 FIXED (0.994), overall 0.99682

- Best 0.99682 (ep64, 17/54). E0T0L0V0 0.99383 (was 0.982);
  exploration-2 0.9907 (was 0.919) — E0 bars PASS, mechanism
  (per-scene GM weighting resolves edge/smooth tension)
  CONFIRMED. Overall misses 0.997 by 0.0002 (within run
  noise of A1's 0.99702, stated).
- Scoreboard: A1 overall 0.99702 / E0 0.982; A2 overall
  0.99682 / E0 0.994. Each fixes what the other breaks.
- Next (Phase C, residue exists): rank-192 + edgeweight +
  synthpool — capacity for the union. Pre-registered: E0
  holds >=0.99 AND overall >=0.99702 (beats both parents).
  Flat -> accept nearest-equivalent, run Phase D certificate.

## 2026-09-24 — OUTCOME Phase C: both bars PASS (0.99726 / E0 0.994)

- Best ep84 0.99726 (24/54, min 0.98126). E0T0L0V0 0.99383;
  exploration-2 0.9984 (HOLDS — was 0.919 under A1).
  Pre-registered bars pass: overall >=0.99702 ✓, E0 >=0.99 ✓.
- Capacity for the union CONFIRMED (rank-192 + edgeweight +
  synthpool beats both parents on both axes). Rank-192 RRR
  init alone started at 0.99743 — capacity was the missing
  ingredient for the union all along; gradients added the
  last +0.0005 and the pass count (16 -> 24).
- Remaining: Phase D certificate (margin-gated parity per
  stratum — E0 0.9938 vs seed-calibrated margin ~0.0003 will
  NOT tie; the certificate documents nearest-equivalent).

## 2026-09-24 — OUTCOME Phase D: 7/31 certificate (nearest-equivalent)

- Margin m=0.000270 (seed range, blind, per rule): 7/31
  scenes tie; no stratum fully ties (best 2/3). Gaps sized
  per scene in certificate.json (E0 worst, expected).
- The certificate IS the nearest-equivalent documentation:
  parity holds on 7 scenes, near-parity (millipoints) on
  most reals, centipoint gaps on E0/dark. The margin rule
  was designed to resolve exactly this; it did.
- Gap-closure program CLOSED. Ledger: A1 0.99702 / B trade /
  AxB E0-best / A2 E0-fixed-0.994 / C 0.99726+both-bars / D
  7/31. Remaining: bank Phase-C best (encode) or new ideas.

## 2026-09-24 — Banked Phase-C best (0.99734 phi-parity)

- weights/student_backbone_r192-phaseC0.99726.npz (drop-in;
  8M student params phi-encoded): 12-scene subset 0.99734 /
  min 0.98688. Receipt adapt/runs/student_banked.json.
  Bank script now parametrized (BANK_CKPT/BANK_TAG env).

## 2026-09-24 — Banked Phase-C + upstream branch pushed

- Bank: weights/student_backbone_r192-phaseC0.99726.npz
  (drop-in, 41.7MB-ish class) at 0.99734/0.98688 phi-parity;
  script parametrized (BANK_CKPT/BANK_TAG).
- Upstream: `feature/baseline-gate` pushed to
  lostdemeter/adaptation_foundry (NOT merged): optional
  Experimenter baseline_gate + 5 BaselineGateTests (73/73
  green) + docs/proposals/margin-gates.md follow-up.
  PR itself needs opening (no gh CLI here) — compare URL:
  github.com/lostdemeter/adaptation_foundry/compare/main...feature/baseline-gate

## 2026-09-24 — Phase-C best live: 0.995 vs teacher (sanity PASSES)

- Highest-accuracy student (r192, 0.99726) on webcam:
  0.992-0.996 vs teacher (mean 0.9949), 0.990-0.996 vs HF,
  24ms/frame. Mixed-light room (bright curtain + dark
  corners + cables); student tracks teacher across regimes
  including dim regions. Best live numbers in program
  history (0.767 -> 0.975 -> 0.991 -> 0.995 across runs).
- Webcam arc corroborates the lab arc: fidelity climbs with
  the training program, dark extreme remains the known gap.

## 2026-09-24 — PRE-REGISTERED Phase E (detail-weighted loss)

- Webcam falloff: thin structures vanish (bottleneck low-pass +
  Sobel-L1 blind to 2px cables + top-10% mask drops exactly the
  detail-error pixels). Fix: DETAILW (per-pixel GM boost where
  teacher has fine structure, symmetric completion of edgew) +
  STRUCTMASK (never mask high-teacher-grad pixels).
- Config: Phase-C recipe (rank-192, deepsup, gms, synthpool100,
  edgew, cosine) + both flags. Predicts: E1T1* thin-structure
  scenes improve; exploration-2/E0 holds (no regression);
  overall >= Phase C 0.99726. Flat -> detail needs capacity,
  not supervision (rank up early maps).

## 2026-09-24 — OUTCOME Phase E: 0.99788, all bars pass

- Best ep84 0.99788 (27/54, min 0.98869). Failmap: E0T0L0V0
  0.99661 (exploration-2 0.9993 — FIXED from 0.919);
  E1T1L0V1 0.99864, E1T1L1V1 0.99942 (thin detail HELD);
  worst scenes audit-1 0.9887, gate-1 0.9910.
- All three pre-registered bars PASS: E1T1* improve ✓,
  exploration-2/E0 hold ✓, overall >= 0.99726 ✓ (0.99788).
  Detail-weighting + structmask is strictly additive over
  Phase C: nothing regressed anywhere. New program best on
  every axis simultaneously.

## 2026-09-24 — Webcam sanity Phase-E best: 0.918, scene-dependent

- Dark emissive desk scene (monitor glow, RGB LEDs, cables):
  student-vs-teacher 0.89-0.94 (mean 0.918), student-vs-HF
  0.86-0.90. Same accepted dark limitation as before (near-
  black + emissive colors absent from fit); NOT a regression
  — Phase-C live run hit 0.995 in brighter conditions. The
  webcam number tracks scene regime, as established.
- structmask/detailw did not target dark scenes; dark thread
  remains closed with limitation accepted.

## 2026-09-25 — PRE-REGISTERED Phase F (direct weight surgery)

- Weights backed up (~/dav2_weight_backup_20260925, 409M,
  sha256 manifest); tree clean + pushed. Edits run on live
  copies, rebuild never needed.
- F1 (residual-corrective edit): audit-1 (0.9887, worst) —
  teacher-vs-student block residuals at L0-2 maps, constrained
  least-squares delta per map (correct target, pin 60 fit
  scenes via cached covariances), re-factorized to rank-192.
  Predicts: audit-1 >=0.995, 54-gate mean unmoved.
- F2 (singular detail-gain knob): scale trailing singular
  values in early maps x1.5/x2. Predicts: thin-detail scenes
  improve slightly; overdone -> noise everywhere (min drops).
- Framework discussion only AFTER F1/F2 verdicts (no premature
  optimization — user directive).

## 2026-09-25 — OUTCOME F1: surgical edit FAILS (clean, informative)

- Residual-corrective deltas on audit-1, all lambdas: best
  lam=10 audit-1=0.9839 (< 0.98869 base), gate 0.99174;
  lam=0.01 destroys everything (audit-1 0.26, gate 0.51).
  Monotonic in lambda toward base, never above it.
  Prediction (audit-1 >=0.995) fails decisively.
- Mechanism (read off trunc table): |D|/|W| = 0.3-0.96 —
  the residual demands deltas nearly as large as the weights
  themselves. No SMALL corrective edit exists: audit-1's
  block outputs live far from fit scenes in activation space,
  so any delta big enough to fix it breaks the other 60
  (rank truncation, rel up to 0.39, compounds it).
- Finding: student errors are DISTRIBUTED, not locally
  editable within rank capacity. Consistent with the dense/
  irreducible theme: corr 0.989 hides weight-space distance.
  Framework implication (for later): edits need either full-
  rank freedom or a different basis than per-map deltas.

## 2026-09-25 — OUTCOME F2: detail-gain knob FAILS (trailing SVs ≠ detail)

- gain=1.25: mean 0.99687 (down 0.001); 1.5: 0.98661
  (gate scenes -0.12); 2.0: 0.95546 (hold-44 -0.30).
  Monotonic destruction with gain; no scene class benefits
  (top-5 improved are sub-millipoint noise except one hold).
  Prediction fails: trailing singular directions are NOT a
  detail reservoir — training suppressed them (or they are
  noise), and amplifying them amplifies damage.
- Phase F verdict: neither additive correction (F1) nor
  multiplicative reweighting (F2) edits the student usefully
  within rank-192. Weights->behavior is too entangled here
  for local post-hoc edits; the knobs that work are
  training-time (loss/data). Framework discussion now open.

## 2026-09-25 — Theory note drafted (convergence operator)

- adapt/CONVERGENCE_OPERATOR.md: survey/reshape/traverse as one
  operator C(W,F) with halting predicate F=∅; contractivity
  conjecture + ledger evidence; refusal semantics (3 instances);
  repertoire-ceiling; lattice-native move readings; dark-thread
  stress test (contracts F_gated; acceptance is operator
  behavior); 4 open questions. For upstream proposal only
  after in-repo review.

## 2026-09-25 — PRE-REGISTERED Phase G (bias opcodes, ADD_IMM)

- audit-1 attribution -> top-3 channels x 18 maps; bias steps
  x(1+/-{0.001,0.005,0.01}); keep iff 54-mean rises AND no
  scene below base floor. 324 trials, single sweep.
- Predicts: millipoint gains on a handful of ops, or clean
  refusal (zero keeps). Either maps the granularity where
  surgery becomes possible.
- Opcode brainstorm (future, unordered): weight-entry immediates;
  SV-scale (F2 precedent: destructive); attention-bias reroute;
  norm-scale tweaks; readout-mix coefficients; opcode COMPOSITION
  (kept-op chaining with re-gating) only if singles ever keep.

## 2026-09-25 — OUTCOME Phase G: clean refusal, knife-edge Pareto-tightness

- 0/324 ADD_IMM ops kept. Mean deltas +-1e-5 (bias moves at
  0.1-1% barely move the mean at all); but floors trigger
  almost everywhere (hold-23 breaks on 309/324 trials).
  Best trials (+1e-5) still floor 15-26 scenes.
- Mechanism: trained weights sit at a sharp joint optimum
  across scenes — every single-instruction direction hurts
  something while helping nothing. No slack exists for greedy
  single moves; the floor constraint (not the mean) is what
  refuses everything.
- Phase F+G verdict (all negative, all clean): F1 deltas too
  big (30-96% norms), F2 directions destructive, G moves too
  weak (+-1e-5) yet still floor-breaking. Post-hoc discrete
  ops look dead at every granularity tried. The language, if
  it exists, is writable only at training time (joint
  optimization preserves floors; isolated moves cannot).

## 2026-09-25 — PRE-REGISTERED H3 (loud-vs-quiet ablation, gatekeeper)

- Theory: dead channels carry cancellation info (§4.7: 42.4%
  energy via destructive interference; §6.7: dark-fringe
  removal collapses argmax). Test on Phase-E student L0-2
  GeLU: zero top-k vs bottom-k by energy (k=32,128;
  calibration on 6 fit scenes), 54-gate damage compared.
- Holographic prediction: quiet-k damage >= loud-k (or
  comparable despite ~0 energy). Null: loud >> quiet ->
  analogy strained, stop before H1/H2.

## 2026-09-25 — OUTCOME H3: null wins, stop (analogy strained)

- quiet-32: 0.99696 (drop 0.0009) vs loud-32: 0.99313 (drop
  0.0048); quiet-128: 0.99587 (drop 0.0020) vs loud-128:
  0.98701 (drop 0.0109). Damage ∝ energy removed (~5x ratio
  both ks); per-unit-energy damage comparable. Standard view
  holds — no holographic excess.
- Worse for transfer: our quiet-128 carries 3.0% of energy;
  the theory's dark fringe carried 42.4%. Different regimes,
  not just different outcomes. DAV2 MLP channels show none
  of the holographic energy signature.
- Per pre-registration: STOP before H1/H2. Caveat (kept
  narrow): this tests energy-ablation, not phase structure
  directly — the claim killed is the transferable prediction
  (dead channels matter disproportionately), not interference
  anywhere. F1/F2/G stand as re-read: locality failures, now
  without holographic cover.

## 2026-09-25 — PRE-REGISTERED trace run (sequentiality probe)

- Instrumented training (STUDENT_TRACE=1): per-scene corrs +
  per-layer/map factor drift from init at every gate eval.
  Phase-E recipe (rank-192 + all levers, cosine 100ep).
- Predicts: (a) scenes flip fail->pass in difficulty order
  (high-edge reals early, smooth E0 late); (b) layers
  stabilize in depth order (L0 reaches 90%-of-final drift
  before L2 — matches A-seq direction). Null: simultaneous
  flips, no layer order -> training is parallel; kill the
  sequentiality claim.

## 2026-09-25 — OUTCOME trace run (sequentiality probe): NULL, kill claim

- 21 rows, best 0.99757. Scene flips: 16 pass at init, then a
  1-2-per-gate trickle through ep89 with NO stratum ordering
  (median-first-pass meaningless at n=1-2 per stratum). Stuck
  scenes are ALL fixtures (gate-0..3, audit-0..3 never pass);
  only holds flip. No difficulty-ordered sequence.
- Layer drift: L0/L1/L2 all hit 90%-of-final at ep44 —
  identical. No depth order (magnitudes differ slightly:
  L2 0.046 > L0 0.034, deeper moves more, but simultaneously).
- Consolation finding (map level): mlp2 moves most in every
  layer (0.06-0.08), k-matrices least (0.017-0.026), q/v/proj/
  mlp1 middle. Routing (keys) is right from RRR init; gradient
  polish = output-mixing adjustments. Consistent across L0-2.
- Verdict per pre-registration: training has NO internal
  arrow at scene or layer level — parallel, not sequential.
  A-seq's advantage was conditioning, not order. C's reshape
  step stays atomic. Theory note updated accordingly.

## 2026-09-25 — Audit green: all three claims executable

- adapt/audit_operator.py ALL PASS: 7 turns classified (all
  5 ops used); contractivity (strict); 6 refusals w/
  mechanisms; live RRR-init gate 0.99138 (40 scenes,
  rank-192); polish bound (init jump 0.081 vs max gradient
  turn 0.012); remainders registered, 7/31 tied.
- The audit forced two corrections: (1) rank-128->192 turn
  REMOVED (double-counted inside A-seq numbers — caught by
  the gaps assertion, not by reading); (2) polish bound
  restated as init-vs-gradient comparison, not an absolute.
  Executable checks earn their keep against prose.

## 2026-09-25 — Opcodes status + quantify_remainders (hardening)

- Opcode program: F1 (deltas 30-96% norms, destroys), F2
  (trailing-SV scaling, monotonic destruction), G (0/324,
  knife-edge floors) — all negative with mechanisms. No
  single-move alphabet exists post-hoc; the reusable artifact
  is the diagnosis-to-retraining loop + floor-criterion.
- Hardening: quantify_remainders.py measures each registry
  gap across 3 checkpoints (r128_cos, r128_edgew, r192C)
  with bootstrap CI95 (10k resamples over scenes). Audit
  extended: bounds present + CI width sane + same 3-ckpt set.
