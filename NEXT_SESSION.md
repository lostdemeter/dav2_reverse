# Pickup prompt: complete geometric DAV2 replica (first of its kind)

Paste everything below the line into a fresh chat instance with terminal
+ file access to this repo.

---

You are continuing a long-running reverse-engineering program: a
**fully-geometric, integer-exact replica of Depth Anything V2-Small**
that we can *learn* (not just copy) under machine-checked gates. Goal:
the first complete learned DAV2 replica — every stage either learned,
proved load-bearing, or replaced by a verified simpler rule.

## Where things stand

Branch: `experimental/adaptation-depth-styles` (pushed; `main` stays
clean — never put experiment code there). Read `HANDOFF.md` first
(map + ledger), then `adapt/NOTES.md` (full running log, including
three recorded self-corrections — read the CORRECTION entries before
trusting any old number).

What works right now (all re-verified this session):
- Float replica 0.99999, integer paths bit-exact, C suite green
  (`cd c_port && make test`), emitter ships bit-exact C (544 kB tables).
- Learned: LUT EXP halving + 125B head fit. Mapped but retained: taps,
  gains, drops (24/24 fail), heads (72/72 fail), widths, co-search wins.
- Live: `python learned_webcam.py` (needs camera), `--c-head` for C path.

## House rules (earned the hard way — violate none without a NOTES entry)

1. No claim without a gate verdict; no gate without a calibrated bar
   (bars are measured from seed capability, never inherited).
2. Pre-register predictions in NOTES *before* running probes.
3. Never upsample ground truth (downsample predictions to reference).
4. Trial lines print absolutes, not just deltas.
5. Corrections go in NOTES as new entries; history is never edited.
6. Commit locally, often. NEVER push, merge, or touch `main` unless
   the user explicitly says so.
7. Vendored `adapt/third_party/` stays pristine; deviations live
   domain-side with provenance notes.

## Open problems (in order — start at 1)

1. **Empty strata (13/16) + real scenes.** Coverage matrix
   (`python adapt/strata.py`) needs non-synthetic scenes with HF-oracle
   refs. Synthetic-only fixtures are the biggest standing caveat.
2. **Finer ablations with bar justification.** Watch list: L1h0 head
   (0.99918), tap-2 tie. Needs a margin-based argument, not a moved bar.
3. **Integer attention composition + fixed-point sensor in C**
   (`c_port/`; kernels exist, wiring open).
4. **Upstream:** baseline-gate PR spec is written
   (`adapt/BASELINE_UPSTREAM.md`) but unfiled; Echion vendor update after.
5. **Joint co-search at bigger scale** (group race tied 2/4 — needs a
   larger arena to discriminate learned vs hand decompositions).
6. **The substrate experiment** (same-harness exact-vs-mush verdict
   comparison) — designed in discussion, never run.

## Resume commands

```bash
git checkout experimental/adaptation-depth-styles
python adapt/test_smoke.py            # fast, no torch
cd c_port && make test               # C suite
python test_geometric_parity.py       # float replica
python adapt/strata.py --fresh 0      # coverage (reuses scenes)
```

Baked `weights/geometric_*.npz` rebuild via
`export_geometric_weights.py`; fixtures via `adapt/build_fixtures.py`
(+HF oracle). `adapt/runs/*.json` and `adapt/fixtures/*.npz` are
gitignored records. First action in any session: `git status`, `git log
--oneline -5`, and re-run the smoke test before believing anything.
