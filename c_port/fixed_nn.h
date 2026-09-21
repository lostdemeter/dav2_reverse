/* fixed_nn.h — fixed-point neural kernels, integer-only (C99).
 *
 * Companions to phi_int.h: everything here runs on absolute fixed-point
 * int64 (2^-14 units unless noted) with NO floating point. The single
 * portability trap in this file is floor semantics: Python's // and >>
 * round toward -inf, while C's / and >> on negatives are truncated /
 * implementation-defined. Every site that can see a negative operand
 * goes through floor_div/floor_shr below. The host test proves
 * bit-exactness vs the Python implementations.
 *
 * Bounds (documented, not checked): int64 accumulators hold test and
 * model sizes used here (384-wide rows, values within +-2^21); sums
 * wrap mod 2^64 exactly like numpy int64, via uint64_t accumulators.
 * head_fixed's dot uses __int128 (safe while T * term^2 < 2^126).
 */
#ifndef FIXED_NN_H
#define FIXED_NN_H

#include <stdint.h>

/* Portable floor division / floor right-shift (denominator/scale > 0). */
int64_t floor_div(int64_t a, int64_t b);

/* Integer square root (floor), Newton method. No libm. */
uint64_t int_isqrt(uint64_t x);

/* Fixed-point dot: terms (s,e)[n] -> triple via ONE accumulation
 * (to_fixed group at max scale, mod-2^64 sum, single re-encode).
 * Mirrors geo_int.fixed_dot_terms. */
void phi_fixed_dot(const int8_t *s, const int32_t *e, int n,
                   int8_t *so, int32_t *eo, uint8_t *zo);

/* Re-scale fixed-point vector q[n] from phi-scale m to m_star >= m with
 * rounding shift. Mirrors geo_int._to_common_scale. */
void phi_rescale(int64_t *q, int n, int32_t m, int32_t m_star);

/* Full fixed-point head dot for one pixel (center, dot, shift,
 * re-encode, +target mean). Mirrors geo_int._head_fixed_pixel's triple
 * (the Python helper decodes to float only for display).
 * fs/fe: [32] feature triple; ws/we: weights; ms/me: means. */
void phi_head_fixed(const int8_t *fs, const int32_t *fe,
                    const int8_t *ws, const int32_t *we,
                    const int8_t *ms, const int32_t *me,
                    int8_t tms, int32_t tme,
                    int8_t *so, int32_t *eo, uint8_t *zo);

/* Integer linear: Y = floor((X@W.T + 2^(F-1)) / 2^F) + b.
 * X:(N,Di) W:(Do,Di) b:(Do) fixed 2^-14. Mirrors int_linear_fixed. */
void nn_linear_fixed(const int64_t *X, const int64_t *W, const int64_t *b,
                     int64_t *Y, int N, int Di, int Do);

/* Integer LayerNorm+affine per row. X:(N,D) -> Y:(N,D), fixed 2^-14.
 * eps_i: integer epsilon already scaled (Python: int(eps*2^28+0.5)).
 * Mirrors int_layernorm_affine. */
void nn_layernorm_fixed(const int64_t *X, const int64_t *w, const int64_t *b,
                        int64_t *Y, int N, int D, int64_t eps_i);

/* Integer GELU over vector (bounded LUT + asymptotes). Mirrors int_gelu_fixed. */
void nn_gelu_vec(const int32_t *X, int32_t *Y, int n);

#endif
