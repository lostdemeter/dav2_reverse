/* test_phi_int.c — host-side test of the integer core.
 *
 * NOTE: doubles appear ONLY in this test harness (parity statistics),
 * never in phi_int.c. The core under test is integer-only.
 */
#include <stdio.h>
#include <math.h>
#include "phi_int.h"
#include "generated/test_vectors.h"

static int fails = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } \
} while (0)

int main(void) {
    int8_t so;
    int32_t eo;
    uint8_t zo;
    int i, c;

    /* 1. phi_add unit properties */
    phi_add(1, 33000, 0, 1, 0, 1, &so, &eo, &zo);       /* x + 0 == x */
    CHECK(so == 1 && eo == 33000 && zo == 0, "add identity");
    phi_add(1, 33000, 0, -1, 33000, 0, &so, &eo, &zo);  /* x + (-x) == 0 */
    CHECK(zo == 1, "add cancel");
    phi_add(1, 1, 1, 1, 0, 1, &so, &eo, &zo);           /* 0 + 0 == 0 */
    CHECK(zo == 1, "add zero");

    /* 2. from_fixed spot checks: q=2^18 @ m=BIAS,F=18 -> ~1.0 -> exp BIAS */
    {
        int8_t s2;
        int32_t e2;
        uint8_t z2;
        phi_from_fixed(262144, 32768, 18, &s2, &e2, &z2);
        CHECK(s2 == 1 && !z2 && e2 > 32760 && e2 < 32776, "from_fixed(1.0) e=%d", e2);
        phi_from_fixed(0, 32768, 18, &s2, &e2, &z2);
        CHECK(z2 == 1, "from_fixed(0)");
    }

    /* 3. head vectors bit-exact vs Python integer path */
    {
        int8_t fs[32];
        uint8_t fz[32];
        int32_t fe[32];
        int mism = 0;
        for (i = 0; i < TV_N; i++) {
            for (c = 0; c < 32; c++) {
                fs[c] = TV_FS[i * 32 + c];
                fe[c] = TV_FE[i * 32 + c];
                fz[c] = 0;
            }
            phi_head_predict(fs, fe, fz, TV_WS, TV_WE, TV_MS, TV_ME,
                             TV_TMS, TV_TME, &so, &eo, &zo);
            if (so != TV_EXP_S[i] || eo != TV_EXP_E[i]) mism++;
        }
        CHECK(mism == 0, "head bit-exact mismatches=%d/%d", mism, TV_N);
        printf("head bit-exact mismatches: %d/%d\n", mism, TV_N);
    }

    /* 3b. batch head == per-pixel head AND matches Python vectors */
    {
        static int8_t bfs[64 * 32];
        static int32_t bfe[64 * 32];
        static uint8_t bfz[64 * 32];
        static int8_t bso[64];
        static int32_t beo[64];
        static uint8_t bzo[64];
        int mism = 0;
        for (i = 0; i < TV_N * 32; i++) {
            bfs[i] = TV_FS[i];
            bfe[i] = TV_FE[i];
            bfz[i] = 0;
        }
        phi_head_predict_batch(bfs, bfe, bfz, TV_WS, TV_WE, TV_MS, TV_ME,
                               TV_TMS, TV_TME, bso, beo, bzo, TV_N);
        for (i = 0; i < TV_N; i++)
            if (bso[i] != TV_EXP_S[i] || beo[i] != TV_EXP_E[i]) mism++;
        CHECK(mism == 0, "batch head mismatches=%d/%d", mism, TV_N);
        printf("batch head mismatches: %d/%d\n", mism, TV_N);
    }

    /* 4. softmax rows sum to ~1.0 (in double, harness only) */
    {
        int64_t scores[5] = { 32768, 16384, 0, -16384, -32768 };
        int64_t num[5], den;
        double sum = 0;
        phi_softmax(scores, 5, num, &den);
        for (i = 0; i < 5; i++) sum += (double)num[i] / (double)den;
        CHECK(sum > 0.999 && sum < 1.001, "softmax sums to %f", sum);
        CHECK(num[0] > num[1] && num[1] > num[2], "softmax ordering");
    }

    /* 5. gelu asymptotes + midpoint */
    CHECK(phi_gelu(-8 * 16384 - 100) == 0, "gelu far left");
    CHECK(phi_gelu(8 * 16384 + 100) == 8 * 16384 + 100, "gelu far right");
    {
        int32_t g0 = phi_gelu(0);
        CHECK(g0 > -20 && g0 < 20, "gelu(0)=%d", g0);
    }

    if (fails == 0) printf("C PORT: ALL PASS\n");
    else printf("C PORT: %d FAILURES\n", fails);
    return fails != 0;
}
