/* test_fixed_nn.c — host test for fixed-point/NN C kernels.
 *
 * Bit-exactness vs the Python implementations (integer ops have no
 * tolerance), plus a correlation check of the C head-fixed path vs the
 * float refs. Doubles appear ONLY in this harness for statistics.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fixed_nn.h"
#include "generated/test_fixed_nn.h"

static int fails = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } \
} while (0)

int main(void) {
    int i, c;

    /* 1. fixed dot */
    {
        int mism = 0;
        for (i = 0; i < FD_N; i++) {
            int8_t so;
            int32_t eo;
            uint8_t zo;
            phi_fixed_dot(FD_S + i * FD_T, FD_E + i * FD_T, FD_T, &so, &eo, &zo);
            if (so != FD_XS[i] || eo != FD_XE[i] || zo != FD_XZ[i]) mism++;
        }
        CHECK(mism == 0, "fixed_dot mismatches=%d/%d", mism, FD_N);
        printf("fixed_dot mismatches: %d/%d\n", mism, FD_N);
    }

    /* 2. rescale */
    {
        int mism = 0;
        for (i = 0; i < 4; i++) {
            int64_t q[16];
            for (c = 0; c < 16; c++) q[c] = RS_Q[i * 16 + c];
            phi_rescale(q, 16, RS_M[i], RS_MS[i]);
            for (c = 0; c < 16; c++)
                if (q[c] != RS_Y[i * 16 + c]) mism++;
        }
        CHECK(mism == 0, "rescale mismatches=%d/64", mism);
        printf("rescale mismatches: %d/64\n", mism);
    }

    /* 3. head_fixed bit-exact + corr vs float refs */
    {
        int mism = 0;
        double sumx = 0, sumy = 0, sumxx = 0, sumyy = 0, sumxy = 0;
        for (i = 0; i < 64; i++) {
            int8_t fs[32], so;
            int32_t fe[32], eo;
            uint8_t zo;
            for (c = 0; c < 32; c++) {
                fs[c] = HF_FS[i * 32 + c];
                fe[c] = HF_FE[i * 32 + c];
            }
            phi_head_fixed(fs, fe, HF_WS, HF_WE, HF_MS, HF_ME,
                           HF_TMS, HF_TME, &so, &eo, &zo);
            if (so != HF_XS[i] || eo != HF_XE[i]) mism++;
            {
                double got = (double)so * pow(1.618033988749895,
                    ((double)eo - 32768.0) / 512.0);
                double ref = HF_REF[i];
                sumx += got; sumy += ref;
                sumxx += got * got; sumyy += ref * ref; sumxy += got * ref;
            }
        }
        {
            double n = 64.0;
            double corr = (n * sumxy - sumx * sumy)
                / sqrt(fmax(1e-300, (n * sumxx - sumx * sumx) * (n * sumyy - sumy * sumy)));
            printf("head_fixed mismatches: %d/64 corr=%.6f\n", mism, corr);
            CHECK(mism == 0, "head_fixed bit-exact");
            CHECK(corr > 0.999, "head_fixed vs float refs");
        }
    }

    /* 4. linear bit-exact */
    {
        int mism = 0, k;
        int64_t Y[4 * 32];
        nn_linear_fixed(LN_X, LN_W, LN_B, Y, 4, 64, 32);
        for (k = 0; k < 4 * 32; k++) if (Y[k] != LN_Y[k]) mism++;
        CHECK(mism == 0, "linear mismatches=%d/128", mism);
        printf("linear mismatches: %d/128\n", mism);
    }

    /* 5. layernorm bit-exact (real 384 width) */
    {
        int mism = 0, k;
        int64_t Y[4 * 384];
        nn_layernorm_fixed(LM_X, LM_W, LM_B, Y, 4, 384, 268);
        for (k = 0; k < 4 * 384; k++) if (Y[k] != LM_Y[k]) mism++;
        CHECK(mism == 0, "layernorm mismatches=%d/1536", mism);
        printf("layernorm mismatches: %d/1536\n", mism);
    }

    /* 6. gelu bit-exact */
    {
        int mism = 0, k;
        int32_t Y[256];
        nn_gelu_vec(GE_X, Y, 256);
        for (k = 0; k < 256; k++) if ((int64_t)Y[k] != GE_Y[k]) mism++;
        CHECK(mism == 0, "gelu mismatches=%d/256", mism);
        printf("gelu mismatches: %d/256\n", mism);
    }

    /* 7. softmax rows bit-exact (incl. clip + uniform rows) */
    {
        int mism = 0, k;
        int64_t NUM[SM_N * SM_M], DEN[SM_N];
        nn_softmax_rows(SM_S, NUM, DEN, SM_N, SM_M);
        for (k = 0; k < SM_N * SM_M; k++) if (NUM[k] != SM_NUM[k]) mism++;
        for (k = 0; k < SM_N; k++) if (DEN[k] != SM_DEN[k]) mism++;
        CHECK(mism == 0, "softmax_rows mismatches=%d/%d", mism,
              SM_N * SM_M + SM_N);
        printf("softmax_rows mismatches: %d/%d\n", mism, SM_N * SM_M + SM_N);
    }

    /* 8. attention end-to-end bit-exact (real layer0 weights, N=8).
     * Scores/av have no standalone exact fns; covered here end-to-end. */
    {
        int mism = 0, k;
        static int64_t O[AN_N * AN_D];
        static int64_t TMP[NN_ATTN_TMP(AN_N, AN_D, AN_H)];
        nn_attention_fixed(AN_X, AN_WQ, AN_WK, AN_WV, AN_WP,
                           AN_BQ, AN_BK, AN_BV, AN_BP,
                           O, TMP, AN_N, AN_D, AN_H);
        for (k = 0; k < AN_N * AN_D; k++) if (O[k] != AN_O[k]) mism++;
        CHECK(mism == 0, "attention mismatches=%d/%d", mism, AN_N * AN_D);
        printf("attention mismatches: %d/%d\n", mism, AN_N * AN_D);
    }

    if (fails == 0) printf("C FIXED-NN: ALL PASS\n");
    else printf("C FIXED-NN: %d FAILURES\n", fails);
    return fails != 0;
}
