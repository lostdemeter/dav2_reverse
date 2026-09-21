/* fib_solve.c — Fibonacci-coefficient exact dot products. See fib_solve.h.
 *
 * Design (theory §11, adapted to bounded C ints):
 *   Each product term s*PHI^(u/512), u = E-BIAS, splits EXACTLY with
 *   q, r = divmod(u, 512):
 *     PHI^(u/512) = F_q * PHI^((r+512)/512) + F_{q-1} * PHI^(r/512)
 *   i.e. 2 nonzero integer coefficients in a 1024-dim lattice basis.
 *   Counts accumulate EXACTLY in int64 tables (zero rounding during the
 *   whole accumulation — the theory's claim). Bands combine with exact
 *   int64 Fibonacci weights (FIBTAB, |q| <= 92) into one __int128 sum,
 *   then a SINGLE from_fixed-style encode (COARSE/FINE) produces the
 *   triple. One rounding total. Terms outside [-91, 92] fall back to
 *   sequential LUT-adds (documented, never triggers on real dots —
 *   verified by the host test on real weights).
 *
 * Integer-only (C99 + __int128 wide sums; no floats, no libm).
 */
#include <stdlib.h>
#include "fib_solve.h"
#include "phi_int.h"
#include "generated/luts.h"

static void divmod_u(int64_t u, int *q, int *r) {
    int64_t qq = u / FIB_K;
    int64_t rr = u % FIB_K;
    if (rr < 0) { rr += FIB_K; qq -= 1; }
    *q = (int)qq;
    *r = (int)rr;
}

/* K*log_phi(x) for positive __int128 x (COARSE/FINE mantissa method).
 * NOTE: raw integer in, no fixed scale assumed — callers subtract their
 * own scale (e.g. COARSE[18] for a 2^18-fixed total) exactly once. */
static int32_t klog_phi_u128(unsigned __int128 x) {
    int bl = 0;
    unsigned __int128 t = x;
    while (t) { bl++; t >>= 1; }
    int shift = bl - 15;
    uint64_t mant = shift >= 0 ? (uint64_t)(x >> shift) : (uint64_t)(x << (-shift));
    int tt = shift + 14;
    int32_t c;
    if (tt < -64) c = PHI_COARSE_LUT[0];
    else if (tt > 128) c = PHI_COARSE_LUT[192];
    else c = PHI_COARSE_LUT[tt + 64];
    return c + PHI_FINE_LUT[mant - 16384];
}

void fib_dot(const int8_t *s, const int32_t *e, int n,
             int8_t *so, int32_t *eo, uint8_t *zo) {
    int i;
    int qmin = 1000000, qmax = -1000000;
    int *qs, *rs;
    if (n <= 0) { *so = 1; *eo = 0; *zo = 1; return; }
    qs = (int *)malloc(sizeof(int) * (size_t)n);
    rs = (int *)malloc(sizeof(int) * (size_t)n);
    if (!qs || !rs) { free(qs); free(rs); *so = 1; *eo = 0; *zo = 1; return; }
    for (i = 0; i < n; i++) {
        int64_t u = (int64_t)e[i] - PHI_BIAS;
        int q, r;
        divmod_u(u, &q, &r);
        qs[i] = q;
        rs[i] = r;
        if (q < qmin) qmin = q;
        if (q > qmax) qmax = q;
    }

    /* Out-of-range bands, or a zero scale band (F_0 == 0 makes exact
     * ratios undefined) -> sequential LUT-add fallback (integer-only).
     * FIBTAB covers [-92, 92] as int64; the F_{q-1} side additionally
     * needs q >= -91. Real dots sit within +-~30 (host test asserts). */
    if (qmin < -91 || qmax > 92 || PHI_FIBTAB[qmax + 92] == 0) {
        int8_t acc_s = 1;
        int32_t acc_e = 0;
        uint8_t acc_z = 1;
        int first = 1;
        for (i = 0; i < n; i++) {
            if (first) {
                acc_s = s[i]; acc_e = e[i]; acc_z = 0;
                first = 0;
            } else {
                phi_add(acc_s, acc_e, acc_z, s[i], e[i], 0,
                        &acc_s, &acc_e, &acc_z);
            }
        }
        free(qs);
        free(rs);
        *so = acc_s; *eo = acc_e; *zo = acc_z;
        return;
    }

    /* Exact count tables over the observed span: int64, zero rounding. */
    {
        int span = qmax - qmin + 1;
        int64_t *c0, *ck;
        int64_t f_q, f_qm1;
        unsigned __int128 total = 0;
        int total_neg = 0;
        int qi, r;
        c0 = (int64_t *)calloc((size_t)span * FIB_BASIS, sizeof(int64_t));
        ck = (int64_t *)calloc((size_t)span * FIB_BASIS, sizeof(int64_t));
        if (!c0 || !ck) {
            free(c0); free(ck); free(qs); free(rs);
            *so = 1; *eo = 0; *zo = 1;
            return;
        }
        for (i = 0; i < n; i++) {
            int b = (qs[i] - qmin) * FIB_BASIS;
            c0[b + rs[i]] += (int64_t)s[i];
            ck[b + rs[i] + FIB_K] += (int64_t)s[i];
        }
        /* Combine bands with exact int64 Fibonacci weights. Basis is
         * 2^40-scale (required: giant canceling parts need the headroom;
         * products need __int128, bounded for T <= ~1e6 terms). */
        {
            unsigned __int128 pos = 0, neg = 0;
            for (qi = 0; qi < span; qi++) {
                int q = qmin + qi;
                f_q = PHI_FIBTAB[q + 92];
                f_qm1 = PHI_FIBTAB[q - 1 + 92];
                for (r = 0; r < FIB_K; r++) {
                    int64_t a = c0[qi * FIB_BASIS + r];
                    int64_t b = ck[qi * FIB_BASIS + r + FIB_K];
                    /* C0 part: a counts of F_{q-1} * BASIS[r] */
                    if (a != 0 && f_qm1 != 0) {
                        __int128 t = (__int128)a * f_qm1 * PHI_BASIS_FRAC[r];
                        if (t > 0) pos += (unsigned __int128)t;
                        else neg += (unsigned __int128)(-t);
                    }
                    /* CK part: b counts of F_q * BASIS[r+K] */
                    if (b != 0 && f_q != 0) {
                        __int128 t = (__int128)b * f_q * (int64_t)PHI_BASIS_FRAC[r + FIB_K];
                        if (t > 0) pos += (unsigned __int128)t;
                        else neg += (unsigned __int128)(-t);
                    }
                }
            }
            /* total is 2^40-scale absolute fixed-point. */
            if (pos >= neg) { total = pos - neg; total_neg = 0; }
            else { total = neg - pos; total_neg = 1; }
        }
        free(c0);
        free(ck);
        free(qs);
        free(rs);
        if (total == 0) { *so = 1; *eo = 0; *zo = 1; return; }
        {
            int32_t klp = klog_phi_u128(total);
            /* value = total / 2^40 -> e = BIAS + klp - COARSE[40]:
             * klp covers log_phi(total); subtract the 2^40 fixed scale
             * via the coarse table (t = 40 fits its [-64, 128] range). */
            int64_t ev = (int64_t)PHI_BIAS + (int64_t)klp - (int64_t)PHI_COARSE_LUT[40 + 64];
            if (ev < 0) ev = 0;
            if (ev > PHI_MAX_EXP) ev = PHI_MAX_EXP;
            *so = (int8_t)(total_neg ? -1 : 1);
            *eo = (int32_t)ev;
            *zo = 0;
        }
    }
}
