"""
Integer-only (no-FPU) phi-arithmetic core + depth head.

Runtime path uses ONLY integer ops (add/sub/compare/XOR + LUT gather).
Floats appear solely in offline LUT construction (baked into firmware
on embedded) and in final decode-for-display / parity checks.

Math (K=512 grid, value = sign * PHI**((exp-BIAS)/K)):
  mul: exp_a + exp_b - BIAS            (integer add)
  add (same sign): max + ADD_LUT[a-b]  (integer LUT gather)
  sub (diff sign): max + SUB_LUT[a-b], sign of larger (integer LUT)
  equal magnitude, opposite sign -> exact zero.

Ported from rlm_springboard/dav2_reverse_engineering/integer_phi_engine.py
(XOR+ADD engine, LUT-only decode) and phi_avx512.c (offline LUT build).
Full-ViT integer attention/softmax is staged future work (each bounded
nonlinearity becomes an integer LUT, as in quantized inference); this
module proves the accumulation core on the 32-wide depth head.
"""

import numpy as np
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)

K = 512
BIAS = 2 ** 16 // 2
MAX_EXP = 2 ** 16 - 1
DMAX = 4096  # exponent-diff range covered by add/sub LUTs


def build_add_lut(k: int = K, dmax: int = DMAX) -> np.ndarray:
    """Offline: same-sign correction c(d) for d = max-min in [0, dmax].

    PHI^a + PHI^b (a>=b) = PHI^a * (1 + PHI**(-d/K)) = PHI^((a + c(d))/K).
    """
    lut = np.zeros(dmax + 1, dtype=np.int32)
    for d in range(dmax + 1):
        # 1 + PHI**(-d/K); d=0 -> 2 (doubling ~ +737 levels at K=512)
        val = 1.0 + PHI ** (-d / k)
        lut[d] = int(round(k * np.log(val) / LN_PHI))
    return lut


def build_sub_lut(k: int = K, dmax: int = DMAX) -> np.ndarray:
    """Offline: different-sign correction for d = max-min in [1, dmax].

    |PHI^a - PHI^b| (a>b) = PHI^b * (PHI^(d/K) - 1); result exp = min + LUT[d].
    d=0 (equal magnitude) -> true zero, handled by special case (lut[0] unused).
    """
    lut = np.zeros(dmax + 1, dtype=np.int32)
    for d in range(1, dmax + 1):
        val = PHI ** (d / k) - 1.0
        if val <= 0:
            lut[d] = -10 ** 9
        else:
            lut[d] = int(round(k * np.log(val) / LN_PHI))
    return lut


def phi_add(s1: int, e1: int, s2: int, e2: int,
            add_lut: np.ndarray, sub_lut: np.ndarray):
    """Integer-only phi addition. Returns (sign, exp). (1, 0) means zero."""
    if s1 == s2:
        # same sign: result exp = max + LUT[max-min]; far apart -> larger dominates
        if e1 >= e2:
            d = e1 - e2
            return (s1, e1) if d > DMAX else (s1, e1 + int(add_lut[d]))
        d = e2 - e1
        return (s2, e2) if d > DMAX else (s2, e2 + int(add_lut[d]))
    # different signs -> subtract magnitudes: result = min + SUB_LUT[max-min]
    if e1 == e2:
        return 1, 0  # exact cancel -> zero (decoded only at display)
    if e1 > e2:
        d = e1 - e2
        return (s1, e1) if d > DMAX else (s1, e2 + int(sub_lut[d]))
    d = e2 - e1
    return (s2, e2) if d > DMAX else (s2, e1 + int(sub_lut[d]))


def _decode_exp(e: int, zero_exp: int = 0) -> float:
    """Display/parity only: exponent -> float. NOT part of runtime path."""
    if e == zero_exp:
        return 0.0
    return float(np.float32(PHI) ** ((np.float32(e) - BIAS) / K))


class IntegerPhiHead:
    """Depth head with integer-only runtime (32 MACs/pixel via LUT-adds)."""

    def __init__(self, compact_weights: Path):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from phi_compact import CompactPhiWeights
        cw = CompactPhiWeights.load(Path(compact_weights))
        w, fm, tm = cw.to_weights()
        # bake to integer (sign, exp) grids — offline step
        self.w_s = np.sign(w).astype(np.int8)
        self.w_s[self.w_s == 0] = 1
        self.w_e = np.round(K * np.log(np.abs(w) + 1e-15) / LN_PHI).astype(np.int32) + BIAS
        self.m_s = np.ones_like(self.w_s)  # feature means are positive (PHI2 format)
        self.m_e = np.round(K * np.log(np.abs(fm) + 1e-15) / LN_PHI).astype(np.int32) + BIAS
        self.tm_s = int(np.sign(tm)) or 1
        self.tm_e = int(round(K * np.log(abs(tm) + 1e-15) / LN_PHI)) + BIAS
        self.add_lut = build_add_lut()
        self.sub_lut = build_sub_lut()
        # float copies for parity checks only (never touched by int_predict)
        self.w_f = w.astype(np.float64)
        self.m_f = fm.astype(np.float64)
        self.tm_f = float(tm)

    @staticmethod
    def encode_features(feat: np.ndarray):
        """Sensor boundary: float features -> integer (sign, exp). Uses FPU.

        On FPU-free hardware the sensor front-end emits fixed-point ints
        and this step becomes an integer LUT over ADC codes (future work).
        """
        s = np.sign(feat).astype(np.int8)
        s[s == 0] = 1
        e = np.round(K * np.log(np.abs(feat) + 1e-15) / LN_PHI).astype(np.int32) + BIAS
        return s, np.clip(e, 0, MAX_EXP).astype(np.int32)

    def int_predict_pixel(self, fs: np.ndarray, fe: np.ndarray) -> tuple:
        """Integer-only: depth = (feat-mean)@w + tm. Returns (sign, exp)."""
        add_lut, sub_lut = self.add_lut, self.sub_lut
        # start accumulator at zero
        acc_s, acc_e = 1, 0
        first = True
        for c in range(32):
            # centered feature: feat - mean (integer phi_add, mean negated)
            cs, ce = phi_add(int(fs[c]), int(fe[c]),
                             int(-self.m_s[c]), int(self.m_e[c]),
                             add_lut, sub_lut)
            if ce == 0 and cs == 1:
                continue  # centered exactly zero -> no contribution
            # multiply by weight: exponent add (integer), sign XOR
            ps = cs * int(self.w_s[c])
            pe = int(ce) + int(self.w_e[c]) - BIAS
            pe = max(0, min(MAX_EXP, pe))
            if first:
                acc_s, acc_e = ps, pe
                first = False
            else:
                acc_s, acc_e = phi_add(acc_s, acc_e, ps, pe, add_lut, sub_lut)
        if first:
            return self.tm_s, self.tm_e
        return phi_add(acc_s, acc_e, self.tm_s, self.tm_e, add_lut, sub_lut)

    def int_predict(self, fs: np.ndarray, fe: np.ndarray) -> np.ndarray:
        """Integer-only over N pixels. fs/fe: [N,32] int. Returns float (decode for display)."""
        n = fs.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            s, e = self.int_predict_pixel(fs[i], fe[i])
            out[i] = _decode_exp(e) * s if e != 0 else 0.0
        return out

    def float_predict(self, feat: np.ndarray) -> np.ndarray:
        """Parity reference: float (feat-mean)@w + tm. NOT the integer path."""
        return (feat - self.m_f) @ self.w_f + self.tm_f


def main():
    """Prototype parity: integer head vs float head on real backbone features."""
    import torch
    from geo_depth import GeometricDepthAnythingV2
    from geo_head import GeometricHead

    device = torch.device('cpu')
    print("loading geometric pipeline for features (CPU)...")
    geo = GeometricDepthAnythingV2(device=device)
    head = IntegerPhiHead(BASE_W := (Path(__file__).parent / 'weights' / 'phi_weights_compact.bin'))

    # synthetic 518x518 input -> backbone+neck (float) -> head features
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.linspace(0, 1, 518, dtype=np.float32),
                         np.linspace(0, 1, 518, dtype=np.float32))
    rgb = np.stack([gx, gy, np.full((518, 518), 0.5, dtype=np.float32)], axis=-1)
    from geo_depth import preprocess
    pv = preprocess(rgb)
    with torch.no_grad():
        fmaps, ph, pw = geo.backbone.forward_stages(pv)
        fused = geo.neck(fmaps)
        _ = geo.head(fused, ph, pw)
        feat = geo.head.last_features.squeeze(0).permute(1, 2, 0).reshape(-1, 32).numpy()
    print(f"features: {feat.shape}")
    ref = head.float_predict(feat)
    fs, fe = IntegerPhiHead.encode_features(feat)
    got = head.int_predict(fs, fe)
    corr = float(np.corrcoef(ref.astype(np.float64), got)[0, 1])
    print(f"integer-head vs float-head corr={corr:.6f} "
          f"relMAE={np.mean(np.abs(ref-got))/(np.mean(np.abs(ref))+1e-9):.4f}")
    print("PASS" if corr > 0.999 else "FAIL")


if __name__ == '__main__':
    main()
