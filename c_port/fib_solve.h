/* fib_solve.h — Fibonacci-coefficient exact dot products (theory §11).
 *
 * Each product term s*PHI^(u/512) (u unbiased) splits EXACTLY:
 *   q, r = divmod(u, 512)
 *   PHI^(u/512) = F_q * PHI^((r+512)/512) + F_{q-1} * PHI^(r/512)
 * i.e. 2 nonzero integer coefficients in a 1024-dim lattice basis.
 * Accumulation into int64 count tables is EXACT (zero rounding); the
 * single solve at the end carries the only rounding — same error budget
 * as the theory's multi-limb solver, without any floats.
 *
 * Integer-only throughout (C99 + __int128 for wide sums, no libm).
 */
#ifndef FIB_SOLVE_H
#define FIB_SOLVE_H

#include <stdint.h>

#define FIB_K 512
#define FIB_BASIS 1024
#define FIB_QMIN -92
#define FIB_QMAX 92
#define FIB_RATIO_SHIFT 40

/* Exact dot of T product terms given as (sign, biased-exp) charges.
 * Returns phi triple. Falls back to FRAC-ratio weights when any band
 * leaves [-92, 92] (documented approximation, still integer-only).
 */
void fib_dot(const int8_t *s, const int32_t *e, int n,
             int8_t *so, int32_t *eo, uint8_t *zo);

#endif
