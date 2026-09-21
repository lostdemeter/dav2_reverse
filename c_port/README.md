# c_port — portable C99 integer-only phi core (embedded artifact)

The Python/Numba paths prove the math; this directory is the shape of the
firmware: a portable C core with **zero floating point in the compute path**
(`phi_int.c`, `fib_solve.c`, `fixed_nn.c` — verify with
`grep -n "float\|double" phi_int.c fib_solve.c fixed_nn.c`, hits are
comments only), plus offline generators whose floats never ship.

## Kernel registry (the emitter's vocabulary)

| Kernel | File | Mirrors | Status |
|---|---|---|---|
| phi_add, from/to_fixed, tree head, softmax, GELU | `phi_int.c` | `geo_int` log-domain path | bit-exact 0/64 |
| fib_dot (exact counts + wide solve) | `fib_solve.c` | theory §11 shootout | corr 0.999846 |
| fixed_dot, rescale, head_fixed | `fixed_nn.c` | `fixed_dot_terms`, `_to_common_scale`, `_head_fixed_pixel` | bit-exact 0/64 |
| linear, layernorm, GELU-vec | `fixed_nn.c` | `int_linear_fixed`, `int_layernorm_affine`, `int_gelu_fixed` | bit-exact (incl. 384-wide) |
| attention composition | — | `int_attention_fixed` | open (kernels exist, composition unwired) |

Portability note: the one trap is floor semantics — Python `//`/`>>`
round toward −inf, C truncates. Every negative-capable site uses
`floor_div`/`floor_div128`/integer `isqrt` (Newton, no libm). Sums wrap
mod 2^64 via `uint64_t` accumulators, identical to numpy int64; the head
dot uses `__int128` (matches Python's unbounded ints at these sizes).
Bounds are documented in `fixed_nn.h`, not checked.

## What's inside

- `phi_int.c` — LUT-add accumulation, fixed-point bridge, head, softmax, GELU.
- `fib_solve.c` — Fibonacci-coefficient **exact** dots (theory §11): each
  product splits into exactly 2 nonzero integer coefficients in a
  1024-dim lattice basis; int64 count tables accumulate with zero
  rounding; one wide (`__int128`) solve + single re-encode finishes.
  Needs a 40-bit basis (an 18-bit basis dies on giant cancellation for
  negative-q bands — diagnosed, documented in `gen_luts.py`). Out of
  int64-Fib range falls back to sequential LUT-adds. Bound: T ≤ ~1e6 terms.
- `test_fib.c` — full head fib path on real vectors: max|dExp|=28,
  maxRel=1.8%, corr=0.999846 vs float refs (cross-tradition bounds —
  tree and fib round at different points, each ~2% worst-pixel vs
  float independently).

## Layout

```
c_port/
├── phi_int.h / phi_int.c   # the core: phi_add, from/to_fixed, head, softmax, gelu
├── fixed_nn.h / fixed_nn.c # fixed-point dots, rescale, head-fixed, linear, layernorm, gelu-vec
├── fib_solve.h / fib_solve.c # exact Fibonacci-coefficient dots
├── gen_luts.py             # offline: emits generated/luts.h (~2.6MB tables)
├── gen_vectors.py          # offline: emits generated/test_vectors.h (real features)
├── gen_fixed_vectors.py    # offline: emits generated/test_fixed_nn.h (kernel vectors)
├── test_phi_int.c          # host test (doubles ONLY here, for statistics)
├── test_fixed_nn.c         # host test for fixed_nn.c (bit-exact + corr)
├── Makefile                # make test
└── generated/              # gitignored build outputs
```

## Build & test (host)

```bash
cd c_port && make test
```

Expected: `head bit-exact mismatches: 0/64` + `C PORT: ALL PASS`, then
`fib head` cross-tradition numbers + `C FIB: ALL PASS`, then all-zero
mismatches + `C FIXED-NN: ALL PASS` (head_fixed corr 0.999851).

## Trimming for microcontrollers

`generated/` is gitignored and ~2.6MB — too fat for an MCU as-is. Trim by
keeping only the subranges your model touches (all indices are clipped, so
narrowing is safe; out-of-range falls back to domination/zero):

| Table | Full | Typical trim |
|-------|------|--------------|
| ADD/SUB | 4097×4B | keep d ≤ 2048 (near-identical magnitudes dominate anyway) |
| FRAC | 13313×4B | keep d ≤ 8192 |
| EXP/GELU | 262145×4B each | keep span ±8 → ±4 if logits bounded |
| FINE | 16384×4B | keep; 64KB is cheap |
| ABS | not shipped | convert weights offline, ship fixed-point directly |

With trimming + `phi_head_predict`-style kernels, the runtime core fits in
tens of KB of flash plus your baked weight tables — no FPU required.
