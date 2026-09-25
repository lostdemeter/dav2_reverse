# The Convergence Operator: diagnosis-to-retraining as one geometric operation

Status: DRAFT theory note (dav2 program). Not yet proposed upstream.
Companion to phi_lattice §6 (navigation paradigm); proposes the
learning operator that paradigm implies but does not write down.

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

## 3. Refusal semantics (three instances, one rule)

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

## 5. Lattice-native reading of move types

Without φ-coordinates, C is search-with-gates (true but shallow).
With them, every move type is a lattice motion:

- **RRR init** = projection onto a measured activation subspace.
  Closed-form, ~40 scenes, 0.99 before epoch 0.
- **Rank** = subspace dimension (see §4).
- **Loss/data changes** = reshaping which paths exist in the
  navigated space (edge-weighting, detail-boost, regime coverage
  alter arrival verdicts per stratum — measured, not metaphorical).
- **Gates** = arrival checks with integer-exact meters (corr,
  margins from seed variance, bit-exact C suite).

"Move the weights" and "move on the lattice" are the same sentence
*only* because weights are lattice positions. That is the precise
sense in which the lattice upgrades the loop from methodology (§6.5's
checklist: represent → study → compare → understand → modify) to a
closed operator with a halting predicate.

## 6. Stress test: the dark thread (hardest case)

Dark-eval oscillated 0.82–0.92 across v3/fraction runs without
monotonic shrinkage — apparent contractivity violation. Resolution:
contractivity holds on the **gated** failure set (clean gate held
0.994–0.996 throughout; dark was report-only, never a gate
verdict). Ungated observations may oscillate freely. Refinement to
the conjecture: **C contracts F_gated; what counts as F is load-
bearing.** A regime enters F iff it is gated; the dark limitation
was *accepted* (documented, not fixed) precisely because promoting
it to F would have broken contraction with no repertoire move to
restore it. The acceptance, not just the fixes, is operator behavior.

## 7. Open questions

1. Is refusal *detectable a priori* (required-delta vs pinned
   subspace, as F1 suggests) rather than only by attempted move?
2. Does contractivity hold across model families (second instance
   needed; generality unclaimed)?
3. Can repertoire be *grown* endogenously (the operator proposing
   its own move types — cf. the retired proposer, which blobbed)?
4. Certificate semantics: Phase D's 7/31 under margin 0.00027 is a
   partial fixed point. Is "converged on subset S of strata" a
   legitimate resting state of C, or must F be global?
