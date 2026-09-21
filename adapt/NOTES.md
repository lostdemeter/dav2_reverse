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
