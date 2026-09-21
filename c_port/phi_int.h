/* phi_int.h — portable C99 integer-only phi-arithmetic core.
 *
 * Runtime path: integer add/sub/compare/shift/XOR + table gather ONLY.
 * No float/double, no libm calls in phi_int.c (verify: `grep -n "float\|double\|exp\|log\|pow\|sqrt" phi_int.c` shows only comments).
 * All LUTs live in generated luts.h (offline floats, baked into firmware).
 *
 * Value representation: v = sign * PHI**((exp - 32768)/512), K=512.
 * Zero is explicit: z=1 (payloads ignored). Callers thread uint8_t masks.
 */
#ifndef PHI_INT_H
#define PHI_INT_H

#include <stdint.h>

#define PHI_K 512
#define PHI_BIAS 32768
#define PHI_MAX_EXP 65535
#define PHI_DMAX 4096
#define PHI_FRAC_CAP 13312
#define PHI_FIXED_F 18

/* ---- LUTs (generated/luts.h) ---- */
extern const int32_t PHI_ADD_LUT[4097];
extern const int32_t PHI_SUB_LUT[4097];
extern const int32_t PHI_FRAC_LUT[13313];
extern const int32_t PHI_COARSE_LUT[193];   /* index t+64, t in [-64,128] */
extern const int32_t PHI_FINE_LUT[16384];   /* 15-bit mantissa */
extern const int32_t PHI_EXP_LUT[262145];   /* 2^24 * e^(-d/2^14) */
extern const int32_t PHI_GELU_LUT[262145];  /* 2^14 * gelu((k-span)/2^14) */
#define PHI_GELU_SPAN (8 * 16384)

/* ---- core ops ---- */

/* Integer phi-add. Zero-aware via z flags. */
void phi_add(int8_t s1, int32_t e1, uint8_t z1,
             int8_t s2, int32_t e2, uint8_t z2,
             int8_t *so, int32_t *eo, uint8_t *zo);

/* Integer re-encode: (q, scale m, width f) -> triple. Bit ops + LUTs. */
void phi_from_fixed(int64_t q, int32_t m, int f,
                    int8_t *so, int32_t *eo, uint8_t *zo);

/* Fixed-point value of one triple at scale m (0 if z or beyond cap). */
int64_t phi_to_fixed(int8_t s, int32_t e, uint8_t z, int32_t m);

/* Depth head: depth = (feat-mean)@w + tm over 32 channels, integer-only.
 * fs/fe/fz: [32] feature triple; ws/we: weights; ms/me: means; tms/tme: target.
 */
void phi_head_predict(const int8_t *fs, const int32_t *fe, const uint8_t *fz,
                      const int8_t *ws, const int32_t *we,
                      const int8_t *ms, const int32_t *me,
                      int8_t tms, int32_t tme,
                      int8_t *so, int32_t *eo, uint8_t *zo);

/* Stable softmax over n absolute-fixed (2^-14 unit) scores -> (num, den).
 * Downstream: attn@V = sum(num[i]*vq[i]) / den, all integer. */
void phi_softmax(const int64_t *scores, int n, int64_t *num, int64_t *den);

/* GELU over fixed 2^-14 input -> 2^-14 output (bounded LUT + asymptotes). */
int32_t phi_gelu(int32_t x);

#endif
