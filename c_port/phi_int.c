/* phi_int.c — portable C99 integer-only phi-arithmetic core.
 * See phi_int.h for the contract. No floating point below this line.
 */
#include "phi_int.h"
#include "generated/luts.h"

void phi_add(int8_t s1, int32_t e1, uint8_t z1,
             int8_t s2, int32_t e2, uint8_t z2,
             int8_t *so, int32_t *eo, uint8_t *zo) {
    if (z1 && z2) { *so = 1; *eo = 0; *zo = 1; return; }
    if (z2) { *so = s1; *eo = e1; *zo = z1; return; }
    if (z1) { *so = s2; *eo = e2; *zo = z2; return; }
    if (s1 == s2) {
        if (e1 >= e2) {
            int32_t d = e1 - e2;
            if (d > PHI_DMAX) { *so = s1; *eo = e1; }
            else { *so = s1; *eo = e1 + PHI_ADD_LUT[d]; }
        } else {
            int32_t d = e2 - e1;
            if (d > PHI_DMAX) { *so = s2; *eo = e2; }
            else { *so = s2; *eo = e2 + PHI_ADD_LUT[d]; }
        }
        if (*eo > PHI_MAX_EXP) *eo = PHI_MAX_EXP;
        if (*eo < 0) *eo = 0;
        *zo = 0;
        return;
    }
    if (e1 == e2) { *so = 1; *eo = 0; *zo = 1; return; }
    if (e1 > e2) {
        int32_t d = e1 - e2;
        if (d > PHI_DMAX) { *so = s1; *eo = e1; }
        else { *so = s1; *eo = e2 + PHI_SUB_LUT[d]; }
    } else {
        int32_t d = e2 - e1;
        if (d > PHI_DMAX) { *so = s2; *eo = e2; }
        else { *so = s2; *eo = e1 + PHI_SUB_LUT[d]; }
    }
    if (*eo > PHI_MAX_EXP) *eo = PHI_MAX_EXP;
    if (*eo < 0) *eo = 0;
    *zo = 0;
}

static int bit_length_u64(uint64_t a) {
    int n = 0;
    while (a) { n++; a >>= 1; }
    return n;
}

void phi_from_fixed(int64_t q, int32_t m, int f,
                    int8_t *so, int32_t *eo, uint8_t *zo) {
    if (q == 0) { *so = 1; *eo = 0; *zo = 1; return; }
    int8_t s = q > 0 ? 1 : -1;
    uint64_t a = q > 0 ? (uint64_t)q : (uint64_t)(-(q + 1)) + 1u;
    int shift = bit_length_u64(a) - 15;
    uint64_t mant = shift >= 0 ? (a >> shift) : (a << (-shift));
    int t = shift + 14 - f;
    int32_t e = m + PHI_COARSE_LUT[t + 64] + PHI_FINE_LUT[mant - 16384];
    if (e < 0) e = 0;
    if (e > PHI_MAX_EXP) e = PHI_MAX_EXP;
    *so = s; *eo = e; *zo = 0;
}

int64_t phi_to_fixed(int8_t s, int32_t e, uint8_t z, int32_t m) {
    int32_t d;
    if (z) return 0;
    d = m - e;
    if (d < 0) d = 0;
    if (d > PHI_FRAC_CAP) return 0;
    return (int64_t)s * (int64_t)PHI_FRAC_LUT[d];
}

void phi_head_predict(const int8_t *fs, const int32_t *fe, const uint8_t *fz,
                      const int8_t *ws, const int32_t *we,
                      const int8_t *ms, const int32_t *me,
                      int8_t tms, int32_t tme,
                      int8_t *so, int32_t *eo, uint8_t *zo) {
    int8_t acc_s = 1;
    int32_t acc_e = 0;
    uint8_t acc_z = 1;
    int first = 1;
    int c;
    for (c = 0; c < 32; c++) {
        int8_t cs, ps, ms_neg = (int8_t)-ms[c];
        int32_t ce, pe;
        uint8_t cz;
        phi_add(fs[c], fe[c], fz[c], ms_neg, me[c], 0, &cs, &ce, &cz);
        if (cz) continue;
        ps = (int8_t)(cs * ws[c]);
        pe = ce + we[c] - PHI_BIAS;
        if (pe < 0) pe = 0;
        if (pe > PHI_MAX_EXP) pe = PHI_MAX_EXP;
        if (first) {
            acc_s = ps; acc_e = pe; acc_z = 0;
            first = 0;
        } else {
            phi_add(acc_s, acc_e, acc_z, ps, pe, 0, &acc_s, &acc_e, &acc_z);
        }
    }
    if (first) {
        *so = tms; *eo = tme; *zo = 0;
        return;
    }
    phi_add(acc_s, acc_e, acc_z, tms, tme, 0, so, eo, zo);
}

void phi_softmax(const int64_t *scores, int n, int64_t *num, int64_t *den) {
    int64_t vmax, dsum = 0;
    int i;
    vmax = scores[0];
    for (i = 1; i < n; i++) if (scores[i] > vmax) vmax = scores[i];
    for (i = 0; i < n; i++) {
        int64_t d = vmax - scores[i];
        int32_t idx = d > 262144 ? 262144 : (int32_t)d;
        num[i] = (int64_t)PHI_EXP_LUT[idx];
        dsum += num[i];
    }
    *den = dsum == 0 ? 1 : dsum;
}

int32_t phi_gelu(int32_t x) {
    int32_t idx = x + PHI_GELU_SPAN;
    if (x < -PHI_GELU_SPAN) return 0;
    if (x > PHI_GELU_SPAN) return x;
    return PHI_GELU_LUT[idx];
}
