/* fixed_nn.c — fixed-point neural kernels. See fixed_nn.h.
 * Integer-only: no float/double, no libm calls below (isqrt is Newton's
 * method on integers). Uses phi_to_fixed/phi_from_fixed/phi_add/phi_gelu
 * from phi_int.c and the generated LUTs.
 */
#include "fixed_nn.h"
#include "phi_int.h"
#include "generated/luts.h"

#define NN_F 14
#define NN_ONE (1 << NN_F)
#define NN_HALF (1 << (NN_F - 1))
#define BR_F 18
#define BR_HALF (1 << (BR_F - 1))

/* Portable floor_div for int64 with b > 0 (C truncates; Python floors). */
int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q -= 1;
    return q;
}

/* Floor-divide __int128 numerator by positive int64 denominator. */
static int64_t floor_div128(__int128 num, int64_t den) {
    if (num >= 0) return (int64_t)(num / den);
    return (int64_t)(-(((-num) + (den - 1)) / den));
}

uint64_t int_isqrt(uint64_t x) {
    uint64_t r = 0, bit;
    if (x == 0) return 0;
    /* highest power of 4 <= x */
    bit = (uint64_t)1 << 62;
    while (bit > x) bit >>= 2;
    while (bit) {
        uint64_t trial = r + bit;
        r >>= 1;
        if (x >= trial) {
            x -= trial;
            r += bit;
        }
        bit >>= 2;
    }
    return r;
}

void phi_fixed_dot(const int8_t *s, const int32_t *e, int n,
                   int8_t *so, int32_t *eo, uint8_t *zo) {
    int32_t m = e[0];
    uint64_t total = 0;
    int i;
    for (i = 1; i < n; i++) if (e[i] > m) m = e[i];
    for (i = 0; i < n; i++) {
        int32_t d = m - e[i];
        int64_t qv = (d < 0 || d > PHI_FRAC_CAP) ? 0
            : (int64_t)s[i] * (int64_t)PHI_FRAC_LUT[d];
        total += (uint64_t)qv; /* wraps mod 2^64, identical to numpy int64 */
    }
    phi_from_fixed((int64_t)total, m, BR_F, so, eo, zo);
}

void phi_rescale(int64_t *q, int n, int32_t m, int32_t m_star) {
    int32_t d = m_star - m; /* precondition: d >= 0 */
    int64_t mult;
    int i;
    if (d <= 0) return;
    if (d > PHI_FRAC_CAP) {
        for (i = 0; i < n; i++) q[i] = 0;
        return;
    }
    mult = (int64_t)PHI_FRAC_LUT[d];
    for (i = 0; i < n; i++) {
        /* q* = floor((q*FRAC[d] + 2^(F-1)) / 2^F), floor for negatives. */
        __int128 num = (__int128)q[i] * mult + BR_HALF;
        __int128 den = (__int128)BR_HALF << 1; /* 2^F */
        q[i] = floor_div128(num, (int64_t)den);
    }
}

void phi_head_fixed(const int8_t *fs, const int32_t *fe,
                    const int8_t *ws, const int32_t *we,
                    const int8_t *ms, const int32_t *me,
                    int8_t tms, int32_t tme,
                    int8_t *so, int32_t *eo, uint8_t *zo) {
    /* Mirrors geo_int._head_fixed_pixel's triple (32 channels). */
    int64_t qf[32], qm[32], qw[32];
    int32_t mf, mm, mw, m;
    __int128 total = 0;
    int i;
    mf = fe[0];
    for (i = 1; i < 32; i++) if (fe[i] > mf) mf = fe[i];
    for (i = 0; i < 32; i++) qf[i] = phi_to_fixed(fs[i], fe[i], 0, mf);
    mm = me[0];
    for (i = 1; i < 32; i++) if (me[i] > mm) mm = me[i];
    for (i = 0; i < 32; i++) qm[i] = phi_to_fixed(ms[i], me[i], 0, mm);
    m = mf >= mm ? mf : mm;
    phi_rescale(qf, 32, mf, m);
    phi_rescale(qm, 32, mm, m);
    mw = we[0];
    for (i = 1; i < 32; i++) if (we[i] > mw) mw = we[i];
    for (i = 0; i < 32; i++) qw[i] = phi_to_fixed(ws[i], we[i], 0, mw);
    for (i = 0; i < 32; i++) {
        __int128 qc = (__int128)qf[i] - qm[i];
        total += qc * (__int128)qw[i];
    }
    {
        int64_t shifted = floor_div128(total + BR_HALF, (int64_t)BR_HALF << 1);
        int8_t s;
        int32_t e;
        uint8_t z;
        phi_from_fixed(shifted, m + mw - 32768, BR_F, &s, &e, &z);
        phi_add(s, e, z, tms, tme, 0, so, eo, zo);
    }
}

void nn_linear_fixed(const int64_t *X, const int64_t *W, const int64_t *b,
                     int64_t *Y, int N, int Di, int Do) {
    int n, o, k;
    for (n = 0; n < N; n++) {
        for (o = 0; o < Do; o++) {
            uint64_t acc = 0; /* wraps mod 2^64, identical to numpy int64 */
            for (k = 0; k < Di; k++)
                acc += (uint64_t)X[n * Di + k] * (uint64_t)W[o * Di + k];
            Y[n * Do + o] = floor_div128((__int128)(int64_t)acc + NN_HALF,
                                         (int64_t)NN_ONE) + b[o];
        }
    }
}

void nn_layernorm_fixed(const int64_t *X, const int64_t *w, const int64_t *b,
                        int64_t *Y, int N, int D, int64_t eps_i) {
    int n, d;
    int64_t halfd = D / 2;
    for (n = 0; n < N; n++) {
        const int64_t *row = X + n * D;
        int64_t *out = Y + n * D;
        int64_t sum = 0;
        int64_t mean, var, csum = 0, std;
        for (d = 0; d < D; d++) sum += row[d];
        mean = floor_div(sum + halfd, D);
        for (d = 0; d < D; d++) {
            int64_t c = row[d] - mean;
            csum += c * c;
        }
        var = floor_div(csum + halfd, D) + eps_i;
        std = (int64_t)int_isqrt((uint64_t)(var < 0 ? 0 : var));
        if (std == 0) {
            for (d = 0; d < D; d++) out[d] = b[d];
            continue;
        }
        {
            int64_t halfs = std / 2;
            for (d = 0; d < D; d++) {
                int64_t c = row[d] - mean;
                int64_t norm = floor_div(c * (int64_t)NN_ONE + halfs, std);
                out[d] = floor_div(norm * w[d] + NN_HALF, (int64_t)NN_ONE) + b[d];
            }
        }
    }
}

void nn_gelu_vec(const int32_t *X, int32_t *Y, int n) {
    int i;
    for (i = 0; i < n; i++) Y[i] = phi_gelu(X[i]);
}

#define EXP_LUT_N 262145

void nn_attn_scores(const int64_t *Q, const int64_t *K,
                    int64_t *S, int N, int hd) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            uint64_t acc = 0; /* wraps mod 2^64, identical to numpy int64 */
            for (k = 0; k < hd; k++)
                acc += (uint64_t)Q[i * hd + k] * (uint64_t)K[j * hd + k];
            /* (dot + 2^13) >> 14 floors (numpy >>); then //8 floors. */
            S[i * N + j] = floor_div(
                floor_div128((__int128)(int64_t)acc + NN_HALF,
                             (int64_t)NN_ONE), 8);
        }
    }
}

void nn_softmax_rows(const int64_t *S, int64_t *NUM, int64_t *DEN,
                      int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        const int64_t *row = S + i * cols;
        int64_t *num = NUM + i * cols;
        int64_t vmax = row[0];
        uint64_t den = 0;
        for (j = 1; j < cols; j++) if (row[j] > vmax) vmax = row[j];
        for (j = 0; j < cols; j++) {
            int64_t dd = vmax - row[j];
            if (dd < 0) dd = 0;
            if (dd > EXP_LUT_N - 1) dd = EXP_LUT_N - 1;
            num[j] = (int64_t)PHI_EXP_LUT[dd];
            den += (uint64_t)num[j]; /* wraps mod 2^64, like numpy */
        }
        DEN[i] = (den == 0) ? 1 : (int64_t)den;
    }
}

void nn_attn_av(const int64_t *NUM, const int64_t *DEN, const int64_t *V,
                int64_t *O, int rows, int terms, int hd) {
    int i, j, k;
    for (i = 0; i < rows; i++) {
        int64_t den = DEN[i] == 0 ? 1 : DEN[i];
        for (j = 0; j < hd; j++) {
            uint64_t acc = 0;
            for (k = 0; k < terms; k++)
                acc += (uint64_t)NUM[i * terms + k] * (uint64_t)V[k * hd + j];
            O[i * hd + j] = floor_div128((__int128)(int64_t)acc, den);
        }
    }
}

void nn_attention_fixed(const int64_t *X,
                        const int64_t *Wq, const int64_t *Wk,
                        const int64_t *Wv, const int64_t *Wp,
                        const int64_t *bq, const int64_t *bk,
                        const int64_t *bv, const int64_t *bp,
                        int64_t *O, int64_t *tmp, int N, int D, int heads) {
    int h, i, d, hd = D / heads;
    int64_t *Q = tmp;                    /* N*D */
    int64_t *K = Q + N * D;              /* N*D */
    int64_t *V = K + N * D;              /* N*D */
    int64_t *Oc = V + N * D;             /* N*D concat */
    int64_t *hq = Oc + N * D;            /* N*hd */
    int64_t *hk = hq + N * hd;           /* N*hd */
    int64_t *hv = hk + N * hd;           /* N*hd */
    int64_t *ho = hv + N * hd;           /* N*hd */
    int64_t *S = ho + N * hd;            /* N*N */
    int64_t *NUM = S + N * N;            /* N*N */
    int64_t *DEN = NUM + N * N;          /* N */
    nn_linear_fixed(X, Wq, bq, Q, N, D, D);
    nn_linear_fixed(X, Wk, bk, K, N, D, D);
    nn_linear_fixed(X, Wv, bv, V, N, D, D);
    for (h = 0; h < heads; h++) {
        for (i = 0; i < N; i++)
            for (d = 0; d < hd; d++) {
                hq[i * hd + d] = Q[i * D + h * hd + d];
                hk[i * hd + d] = K[i * D + h * hd + d];
                hv[i * hd + d] = V[i * D + h * hd + d];
            }
        nn_attn_scores(hq, hk, S, N, hd);
        nn_softmax_rows(S, NUM, DEN, N, N);
        nn_attn_av(NUM, DEN, hv, ho, N, N, hd);
        for (i = 0; i < N; i++)
            for (d = 0; d < hd; d++)
                Oc[i * D + h * hd + d] = ho[i * hd + d];
    }
    nn_linear_fixed(Oc, Wp, bp, O, N, D, D);
}

void nn_sensor_encode(const uint8_t *rgb, int64_t *X, int H, int W,
                      const int64_t *A, const int64_t *B, int K) {
    int i, c, n = H * W * 3;
    int64_t half = (int64_t)1 << (K - 1);
    int64_t step = (int64_t)1 << K;
    for (i = 0; i < n; i++) {
        c = i % 3;
        /* p*A+B+half fits easily (p<=255, |A|,|B| < 2^31); floor_div floors. */
        X[i] = floor_div((int64_t)rgb[i] * A[c] + B[c] + half, step);
    }
}

void nn_patch_tokens(const int64_t *X, const int64_t *Wp, const int64_t *bp,
                     const int64_t *cls, const int64_t *pos,
                     int64_t *T, int H, int W) {
    int ty, tx, y, x, c, d, ph = H / 14;
    int64_t patch[588];
    int64_t out[384];
    for (d = 0; d < 384; d++) T[d] = cls[d] + pos[d];
    for (ty = 0; ty < ph; ty++) {
        for (tx = 0; tx < ph; tx++) {
            int64_t *Trow = T + (1 + ty * ph + tx) * 384;
            const int64_t *Prow = pos + (1 + ty * ph + tx) * 384;
            for (y = 0; y < 14; y++)
                for (x = 0; x < 14; x++)
                    for (c = 0; c < 3; c++)
                        patch[(c * 14 + y) * 14 + x] =
                            X[((ty * 14 + y) * W + tx * 14 + x) * 3 + c];
            nn_linear_fixed(patch, Wp, bp, out, 1, 588, 384);
            for (d = 0; d < 384; d++) Trow[d] = out[d] + Prow[d];
        }
    }
}
