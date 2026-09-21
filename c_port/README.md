# c_port — portable C99 integer-only phi core (embedded artifact)

The Python/Numba paths prove the math; this directory is the shape of the
firmware: a portable C core with **zero floating point in the compute path**
(`phi_int.c` — verify with `grep -n "float\|double" phi_int.c`, hits are
comments only), plus offline generators whose floats never ship.

## Layout

```
c_port/
├── phi_int.h / phi_int.c   # the core: phi_add, from/to_fixed, head, softmax, gelu
├── gen_luts.py             # offline: emits generated/luts.h (~2.6MB tables)
├── gen_vectors.py          # offline: emits generated/test_vectors.h (real features)
├── test_phi_int.c          # host test (doubles ONLY here, for statistics)
├── Makefile                # make test
└── generated/              # gitignored build outputs
```

## Build & test (host)

```bash
cd c_port && make test
```

Expected: `head bit-exact mismatches: 0/64` then `C PORT: ALL PASS`.

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
