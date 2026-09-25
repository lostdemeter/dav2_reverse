# The Convergence Operator: diagnosis-to-retraining as one geometric operation

Status: DRAFT v2 theory note (dav2 program). Not yet proposed upstream.
Companion to phi_lattice §6 (navigation paradigm); proposes the
learning operator that paradigm implies but does not write down.

v2 changes: move alphabet enumerated (§5 rewritten); refusal
instances now six incl. H3 theory-stop; float-honesty correction
(§5); F admission criteria (§6); F* resting-state semantics (§7);
trace null — reshape stays atomic (§8).

---

## 1. Definition

Let W be weights on the φ-lattice (exact integer coordinates) and F
the *measured* arrival-failure set: gate verdicts per scene/stratum
(failmap, strata matrix, certificates). The convergence operator:

```
C(Wₙ, Fₙ) = (Wₙ₊₁, Fₙ₊₁)
```

where the reshape step is: **the minimal move from the allowed
repertoire that empties the largest failing subset, verified by
gates.** Halting predicate: F = ∅ (parity everywhere). The three
levels — survey (measure F), reshape (move W), traverse (gate =
arrival check) — are one operation converging to a fixed point,
not three activities.

This mirrors §6.6 one level up: where T moves states
(hₜ₊₁ = T(hₜ)) toward generation fixed points, C moves spaces
((W,F)ₙ₊₁ = C(W,F)ₙ) toward parity fixed points. Same fixed-point
mathematics; the navigator becomes the navigated.

## 2. Contractivity conjecture

**Claim:** every accepted turn strictly shrinks |F|; C is a
contraction in failure-set size, and the contraction is *enforced*,
not assumed — by the gate architecture (guards refuse non-shrinking
moves). Evidence (dav2 ledger, all pre-registered):

| Turn | Gate mean | E0 / worst scene | Verdict |
|---|---|---|---|
| First gradient run | 0.9928 | E0 ~0.92 | accepted, F shrinks |
| Phase A1 (multi-scale GM) | 0.99702 | E0 0.982 | accepted |
| Phase C (rank-192 + all) | 0.99726 | E0 0.994 | accepted |
| Phase E (detail + structmask) | 0.99788 | expl-2 0.919→0.9993 | accepted |
| Phase D certificate | — | 7/31 ties | F documented, not shrunk (terminal report) |

No accepted turn in program history ever grew F. Every guard in the
system (dissolution evidence-guard, margin refusals, EfficiencyRule
tie-handling, λ-sweep monotonicity) exists to enforce exactly this.

## 3. Refusal semantics (six instances, one rule)

When no repertoire move empties anything, C returns (W, F) unchanged.
Refusal is correct behavior — fixed-point detection at the current
repertoire — not failure:

1. **Dissolution guard:** unguarded regroup churned good structure;
   guarded refusals restored parity ("prevents harm; doesn't
   create wins").
2. **Margin refusals:** three binary full-ties refused promotion
   solely by `margin_regressed` — without the margin they promote
   as false wins (would have *grown* F while claiming to shrink).
3. **F1 λ-sweep:** audit-1 edit monotonic toward base (0.26 →
   0.984 < 0.98869) across λ. The operator declines to move;
   trunc table (|D|/|W| up to 0.96) shows why: the destination is
   not reachable by small moves — a wall, not a road.
4. **F2 SV scaling:** monotonic destruction (1.25× → −0.001,
   2.0× → −0.04 with a −0.30 crater). Trailing directions are not
   suppressed detail; scaling them scales damage.
5. **Phase G 0/324:** bias immediates move the mean ±1e-5 yet
   break floors almost everywhere. Sharp joint optimum, no slack
   for greedy single moves.
6. **H3 theory-stop:** loud-vs-quiet ablation favored the null
   (~5x damage ratio both ks) — work on H1/H2 declined *before*
   building. Second-order refusal: the operator governs ideas,
   halting construction on falsified imports.

## 4. Repertoire sets the ceiling

Same operator, different move-sets, different fixed points:
rank-128 converges to ~0.99688 territory; rank-192 (+edgeweight +
synthpool) converges to 0.99726 with E0 fixed. "Capacity" thereby
gets a geometric meaning: **the dimensionality of the subspace in
which C may move** — connecting directly to §2's finding that
intrinsic dimensionality < encoding dimensionality. C converges
within the intrinsic subspace; rank sets how much of it is
reachable. F2's destruction under trailing-SV amplification is the
same statement from below: directions outside the reachable set
cannot be recruited by scaling.

## 5. Move alphabet (lattice-native) + float-honesty correction

Without φ-coordinates, C is search-with-gates (true but shallow).
With them, the repertoire is an enumerable alphabet of lattice
motions — this is what makes "reshape" a defined operation rather
than a black box:

- **PROJECT** (RRR init): closed-form projection onto the measured
  activation subspace. ~40 scenes, 0.99 before epoch 0.
- **EXPAND** (rank): subspace dimension; sets the ceiling (§4).
- **REWEIGHT** (loss changes): alter which arrivals verify —
  edge-weighting, detail-boost, multi-scale GM (measured per
  stratum, not metaphorical).
- **COVER** (data changes): alter which strata exist to arrive
  at — synthpool, darkpool, holdout discipline.
- **VERIFY** (gates): arrival checks with integer-exact meters
  (corr, margins from seed variance, bit-exact C suite).

Honesty correction (v2): the gradient polish happens in *float*
factor space, not on the lattice. Quantitatively: lattice-native
init does ~0.99 of the work exact (projection), float gradients
add ~+0.004. The division is itself a finding — the geometric
part of training is closed-form; the float part is small
corrections. "Move the weights and move on the lattice are the
same sentence" holds for 99% of the result, not 100%.

"Move the weights and move on the lattice are the same sentence"
*only* because weights are lattice positions. That is the precise
sense in which the lattice upgrades the loop from methodology (§6.5's
checklist: represent → study → compare → understand → modify) to a
closed operator with a halting predicate.

## 6. F admission criteria

The dark thread plus H3 jointly imply: a regime enters F iff it is
both *measured* and *gated*. Dark-eval oscillated freely because it
was report-only, never a gate verdict; holography was refused *at
the door* (transfer falsified before construction). Ungated
observations may oscillate; unverified imports get refused before
entering. What counts as F is load-bearing: promoting dark to F
would have broken contraction with no repertoire move to restore
it. The acceptance, not just the fixes, is operator behavior.

## 7. Achievable fixed points (F* resting-state semantics)

F = ∅ is unreachable — dark accepted, 7/31 certified. Convergence
in practice means F* = irreducible remainder, and a remainder is a
legitimate resting state *iff* it carries a mechanism, not a shrug:
dark (regime competition, measured) counts; E0 did not count as
resting — it got fixed. This partially answers the certificate
question: partial-strata convergence is a converged state when the
remainder is characterized (gap sizes, mechanisms, per-scene
receipts in certificate.json), interim otherwise.

## 8. Reshape stays atomic (trace null, 2026-09-25)

Instrumented training (per-scene corrs + per-layer/map drift at
21 gates): scene flips show no stratum ordering (all fixtures stuck,
only holds flip, trickle through ep89); layers hit 90%-of-final
drift simultaneously (L0/L1/L2 all ep44). No internal arrow at
scene or layer level — A-seq's advantage was conditioning, not
order. C's reshape step stays atomic; no triangular decomposition.
Consolation finding: mlp2 moves most in every layer, k-matrices
least — routing is right from RRR init, polish adjusts mixing.

## 9. Open questions

1. Is refusal *detectable a priori* (required-delta vs pinned
   subspace, as F1 suggests) rather than only by attempted move?
2. Does contractivity hold across model families (second instance
   needed; generality unclaimed)?
3. Can repertoire be *grown* endogenously (the operator proposing
   its own move types — cf. the retired proposer, which blobbed)?
4. (Partially answered, §7): resting-state legitimacy criteria —
   mechanism-carrying remainders vs shrugs. Needs a second instance.
