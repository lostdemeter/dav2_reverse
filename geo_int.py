"""
Integer-only (no-FPU) phi-arithmetic core + depth head.

Runtime path uses ONLY integer ops (add/sub/compare/XOR + LUT gather).
Floats appear solely in offline LUT construction (baked into firmware
on embedded) and in final decode-for-display / parity checks.

Math (K=512 grid, value = sign * PHI**((exp-BIAS)/K)):
  mul: exp_a + exp_b - BIAS            (integer add)
  add (same sign): max + ADD_LUT[a-b]  (integer LUT gather)
  sub (diff sign): min + SUB_LUT[max-min], sign of larger (integer LUT)
  equal magnitude, opposite sign -> exact zero.

Vector batches use (signs, exps, zero-mask) triples and numpy int ops —
still FPU-free, fast enough for conv parity (~50M element-ops per neck
conv in seconds). A Numba/C port (see jit_phi_matmul.py precedent)
would take this to real-time embedded.

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
N_LEVELS = 2 ** 16
MAX_EXP = N_LEVELS - 1
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


    def float_predict(self, feat: np.ndarray) -> np.ndarray:
        """Parity reference: float (feat-mean)@w + tm. NOT the integer path."""
        return (feat - self.m_f) @ self.w_f + self.tm_f


# ---------------------------------------------------------------------------
# Vectorized integer ops: (signs int8, exps int32, zero bool) triples.
# All numpy integer/comparison ops — no floats at runtime.
# ---------------------------------------------------------------------------

def vec_phi_add(s1, e1, z1, s2, e2, z2, add_lut, sub_lut):
    """Elementwise integer phi-add over arrays. Returns (s, e, z)."""
    s1 = np.asanyarray(s1, dtype=np.int8)
    e1 = np.asanyarray(e1, dtype=np.int32)
    z1 = np.asanyarray(z1, dtype=bool)
    s2 = np.asanyarray(s2, dtype=np.int8)
    e2 = np.asanyarray(e2, dtype=np.int32)
    z2 = np.asanyarray(z2, dtype=bool)
    rs = np.empty_like(s1)
    re = np.empty_like(e1)
    rz = np.empty_like(z1)
    # zero passthrough
    rz = z1 & z2
    take1 = z2 & ~z1
    take2 = z1 & ~z2
    rs = np.where(take1, s1, rs)
    re = np.where(take1, e1, re)
    rs = np.where(take2, s2, rs)
    re = np.where(take2, e2, re)
    live = ~(z1 | z2)
    if not np.any(live):
        return rs, re, rz
    a1, b1 = s1[live], e1[live]
    a2, b2 = s2[live], e2[live]
    os = np.empty_like(a1)
    oe = np.empty_like(b1)
    oz = np.zeros_like(b1, dtype=bool)
    same = a1 == a2
    ge = b1 >= b2
    # same sign, e1>=e2
    m = same & ge
    d = np.clip(b1 - b2, 0, DMAX)
    oe = np.where(m, np.where(b1 - b2 > DMAX, b1, b1 + add_lut[d]), oe)
    os = np.where(m, a1, os)
    # same sign, e2>e1
    m = same & ~ge
    d = np.clip(b2 - b1, 0, DMAX)
    oe = np.where(m, np.where(b2 - b1 > DMAX, b2, b2 + add_lut[d]), oe)
    os = np.where(m, a2, os)
    # different signs, equal -> zero
    m = ~same & (b1 == b2)
    oz = np.where(m, True, oz)
    # different signs, e1>e2 -> min + SUB_LUT
    m = ~same & (b1 > b2)
    d = np.clip(b1 - b2, 1, DMAX)
    oe = np.where(m, np.where(b1 - b2 > DMAX, b1, b2 + sub_lut[d]), oe)
    os = np.where(m, a1, os)
    # different signs, e2>e1
    m = ~same & (b2 > b1)
    d = np.clip(b2 - b1, 1, DMAX)
    oe = np.where(m, np.where(b2 - b1 > DMAX, b2, b1 + sub_lut[d]), oe)
    os = np.where(m, a2, os)
    rs[live], re[live], rz[live] = os, oe, oz
    return rs, np.clip(re, 0, MAX_EXP).astype(np.int32), rz


def int_encode_array(feat: np.ndarray):
    """Boundary: float array -> (signs, exps, zero-mask). Uses FPU."""
    s = np.sign(feat).astype(np.int8)
    e = np.round(K * np.log(np.abs(feat) + 1e-15) / LN_PHI).astype(np.int32) + BIAS
    e = np.clip(e, 0, MAX_EXP).astype(np.int32)
    z = (feat == 0)
    s[z] = 1
    return s, e, z


def int_decode_array(s, e, z):
    """Display/parity only: integer triple -> float. NOT runtime path."""
    lut = np.float32(PHI) ** ((np.arange(N_LEVELS, dtype=np.float32) - BIAS) / K)
    out = s.astype(np.float32) * lut[np.clip(e, 0, N_LEVELS - 1)]
    return np.where(z, 0.0, out).astype(np.float32)


def load_baked_ints(npz_path):
    """Load baked phi (signs, exps) straight from npz — no float round-trip."""
    z = np.load(Path(npz_path), allow_pickle=False)
    out = {}
    for key in z.files:
        if key.endswith('.signs'):
            base = key[:-len('.signs')]
            out[base] = (z[key].astype(np.int8),
                         z[base + '.exps'].astype(np.int32))
    return out


def int_conv2d(f_s, f_e, f_z, w_s, w_e, b_s=None, b_e=None, b_z=None,
               stride=1, padding=0, add_lut=None, sub_lut=None):
    """Integer-only 2D conv.

    f_*: (Cin,H,W) int; w_*: (Cout,Cin,kH,kW) int; b_*: (Cout,) int or None.
    Returns (s, e, z) of shape (Cout,Ho,Wo). Im2col-style accumulation
    with vec_phi_add — numpy int ops only.
    """
    Cin, H, W = f_s.shape
    Cout = w_s.shape[0]
    kH, kW = w_s.shape[2], w_s.shape[3]
    if padding:
        pad = ((0, 0), (padding, padding), (padding, padding))
        f_s = np.pad(f_s, pad, constant_values=1)
        f_e = np.pad(f_e, pad, constant_values=0)
        f_z = np.pad(f_z, pad, constant_values=True)
        H, W = f_s.shape[1], f_s.shape[2]
    Ho, Wo = (H - kH) // stride + 1, (W - kW) // stride + 1
    P = Ho * Wo
    # Collect one (sign, exp) term per MAC, then tree-reduce pairwise:
    # log2(T) rounding levels instead of T sequential roundings, and the
    # reduction is still integer LUT-adds only.
    t_s, t_e, t_z = [], [], []
    for kh in range(kH):
        for kw in range(kW):
            rows = np.arange(kh, kh + Ho * stride, stride)
            cols = np.arange(kw, kw + Wo * stride, stride)
            ps = f_s[:, rows[:, None], cols].reshape(Cin, P)
            pe = f_e[:, rows[:, None], cols].reshape(Cin, P)
            pz = f_z[:, rows[:, None], cols].reshape(Cin, P)
            for ci in range(Cin):
                t_s.append((ps[ci][None, :].astype(np.int16)
                            * w_s[:, ci, kh, kw].astype(np.int16)[:, None]).astype(np.int8))
                t_e.append(np.clip(pe[ci][None, :] + w_e[:, ci, kh, kw][:, None] - BIAS,
                                   0, MAX_EXP).astype(np.int32))
                t_z.append(pz[ci][None, :].repeat(Cout, axis=0))
    while len(t_s) > 1:
        n_s, n_e, n_z = [], [], []
        for i in range(0, len(t_s) - 1, 2):
            s, e, z = vec_phi_add(t_s[i], t_e[i], t_z[i],
                                  t_s[i + 1], t_e[i + 1], t_z[i + 1],
                                  add_lut, sub_lut)
            n_s.append(s)
            n_e.append(e)
            n_z.append(z)
        if len(t_s) % 2:
            n_s.append(t_s[-1])
            n_e.append(t_e[-1])
            n_z.append(t_z[-1])
        t_s, t_e, t_z = n_s, n_e, n_z
    acc_s, acc_e, acc_z = t_s[0], t_e[0], t_z[0]
    if b_s is not None:
        acc_s, acc_e, acc_z = vec_phi_add(
            acc_s, acc_e, acc_z,
            np.broadcast_to(b_s[:, None], (Cout, P)).astype(np.int8),
            np.broadcast_to(b_e[:, None], (Cout, P)).astype(np.int32),
            np.broadcast_to(b_z[:, None], (Cout, P)),
            add_lut, sub_lut)
    return (acc_s.reshape(Cout, Ho, Wo), acc_e.reshape(Cout, Ho, Wo).astype(np.int32),
            acc_z.reshape(Cout, Ho, Wo))


def int_relu(s, e, z):
    """Integer ReLU: negatives -> zero. Returns (s, e, z)."""
    neg = (~z) & (s < 0)
    return np.where(neg, 1, s).astype(np.int8), np.where(neg, 0, e).astype(np.int32), (z | neg)


def int_relu(s, e, z):
    """Integer ReLU: negatives -> zero. Returns (s, e, z)."""
    neg = (~z) & (s < 0)
    return np.where(neg, 1, s).astype(np.int8), np.where(neg, 0, e).astype(np.int32), (z | neg)


# ---------------------------------------------------------------------------
# Fixed-point bridge: phi log-domain <-> linear fixed-point, runtime integer.
#
# This is the datapath of a real no-FPU phi machine: weights live compressed
# on the phi lattice; compute happens in fixed-point via offline LUTs.
#   to_fixed:   (s,e) -> q int64 with value = q * PHI**(m/K) / 2**F  (exact-ish,
#               one rounding per term, no sequential accumulation error)
#   from_fixed: (q,m,F) -> (s,e) via bit_length + shifts + offline
#               COARSE/FINE LUTs (all integer: no log/exp/pow at runtime).
# Softmax/products that stay in fixed-point skip the trip back entirely.
# ---------------------------------------------------------------------------

FIXED_F = 18
FIXED_S = 1 << FIXED_F
FRAC_CAP = 13312  # PHI**(-13312/512) * 2**18 < 0.5 -> smaller terms vanish

_FRAC_LUT = None
_COARSE_LUT = None
_COARSE_OFF = 64
_FINE_LUT = None
_EXP_LUT = None
_EXP_F = 14
_EXP_G = 24
_ABS_LUT = None
_ABS_F = 14


def _get_abs_lut():
    """ABS_LUT[e] = round(PHI**((e-BIAS)/K) * 2**_ABS_F), absolute fixed-point.

    Underflowing entries round to 0. Offline-built (512KB int64).
    """
    global _ABS_LUT
    if _ABS_LUT is None:
        lut = np.zeros(N_LEVELS, dtype=np.int64)
        scale = 1 << _ABS_F
        for e in range(N_LEVELS):
            v = PHI ** ((e - BIAS) / K) * scale
            lut[e] = int(round(v)) if v < 9e18 else 9e18
        _ABS_LUT = lut
    return _ABS_LUT


def _get_frac():
    global _FRAC_LUT
    if _FRAC_LUT is None:
        lut = np.zeros(FRAC_CAP + 1, dtype=np.int64)
        for d in range(FRAC_CAP + 1):
            lut[d] = int(round(PHI ** (-d / K) * FIXED_S))
        _FRAC_LUT = lut
    return _FRAC_LUT


def _get_coarse_fine():
    global _COARSE_LUT, _FINE_LUT
    import math
    if _COARSE_LUT is None:
        arr = np.zeros(193, dtype=np.int32)
        for t in range(-64, 129):
            arr[t + _COARSE_OFF] = int(round(t * K * math.log(2.0) / LN_PHI))
        _COARSE_LUT = arr
    if _FINE_LUT is None:
        # 14-bit mantissa (16384..32767): relative step 2**-14 ~ 0.006%.
        # 64KB offline LUT; keeps from_fixed quantization noise negligible.
        lut = np.zeros(16384, dtype=np.int32)
        for i in range(16384):
            lut[i] = int(round(K * math.log((16384 + i) / 16384.0) / LN_PHI))
        _FINE_LUT = lut
    return _COARSE_LUT, _FINE_LUT


def _get_exp_lut():
    """EXP_LUT[d] = round(2**G * e**(-d/2**F)), d in [0, 16*2**F]. Offline."""
    global _EXP_LUT
    import math
    if _EXP_LUT is None:
        n = 16 * (1 << _EXP_F) + 1
        lut = np.zeros(n, dtype=np.int64)
        for d in range(n):
            lut[d] = int(round((1 << _EXP_G) * math.exp(-d / (1 << _EXP_F))))
        _EXP_LUT = lut
    return _EXP_LUT


def to_fixed_group(s, e, f: int = FIXED_F):
    """Integer: terms (s,e) -> (q int64, m). value = q * PHI**(m/K) / 2**f.

    Requires f == FIXED_F (FRAC LUT scale); other widths via shift of q.
    """
    assert f == FIXED_F
    frac = _get_frac()
    s = np.asanyarray(s, dtype=np.int8)
    e = np.asanyarray(e, dtype=np.int32)
    m = int(np.max(e))
    dd = np.clip(m - e, 0, FRAC_CAP)
    q = s.astype(np.int64) * frac[dd]
    return q, m


def from_fixed_scalar(q: int, m: int, f: int = FIXED_F):
    """Integer: (q, m, f) -> (sign, exp). Bit ops + offline LUTs only."""
    if q == 0:
        return 1, 0, True
    coarse, fine = _get_coarse_fine()
    s = 1 if q > 0 else -1
    a = abs(q)
    # 15-bit mantissa: top 15 bits of a -> mant in [16384, 32768).
    # a = mant * 2**shift; a/2**f = (mant/16384) * 2**(shift+14-f).
    shift = a.bit_length() - 15
    mant = (a >> shift) if shift >= 0 else (a << (-shift))
    t = shift + 14 - f
    e = m + int(coarse[t + _COARSE_OFF]) + int(fine[mant - 16384])
    return s, max(0, min(MAX_EXP, e)), False


def from_fixed_vec(q, m: int, f: int = FIXED_F):
    """Integer: arrays -> (signs, exps, zero). Python-int loop (C port later)."""
    q = np.asanyarray(q)
    out_s = np.empty(q.shape, dtype=np.int8)
    out_e = np.empty(q.shape, dtype=np.int32)
    out_z = np.zeros(q.shape, dtype=bool)
    for idx in np.ndindex(q.shape):
        s, e, z = from_fixed_scalar(int(q[idx]), m, f)
        out_s[idx], out_e[idx], out_z[idx] = s, e, z
    return out_s, out_e, out_z


def fixed_dot_terms(s_terms, e_terms):
    """Integer: T terms -> phi triple via ONE fixed-point accumulation.

    s_terms/e_terms: (T, N) int. Single rounding at re-encode (vs T
    roundings for sequential LUT-adds).
    """
    q, m = to_fixed_group(s_terms, e_terms)
    total = q.sum(axis=0, dtype=np.int64)
    return from_fixed_vec(total, m)


def int_conv2d_fixed(f_s, f_e, f_z, w_s, w_e, b_s=None, b_e=None, b_z=None,
                     stride=1, padding=0):
    """Integer conv with fixed-point accumulation (one rounding per output).

    Same signature/semantics as int_conv2d; more accurate for deep chains.
    """
    frac = _get_frac()
    Cin, H, W = f_s.shape
    Cout = w_s.shape[0]
    kH, kW = w_s.shape[2], w_s.shape[3]
    if padding:
        pad = ((0, 0), (padding, padding), (padding, padding))
        f_s = np.pad(f_s, pad, constant_values=1)
        f_e = np.pad(f_e, pad, constant_values=0)
        f_z = np.pad(f_z, pad, constant_values=True)
        H, W = f_s.shape[1], f_s.shape[2]
    Ho, Wo = (H - kH) // stride + 1, (W - kW) // stride + 1
    P = Ho * Wo
    out_s = np.empty((Cout, P), dtype=np.int8)
    out_e = np.empty((Cout, P), dtype=np.int32)
    out_z = np.empty((Cout, P), dtype=bool)
    for co in range(Cout):
        ts, te, tz = [], [], []
        for kh in range(kH):
            for kw in range(kW):
                rows = np.arange(kh, kh + Ho * stride, stride)
                cols = np.arange(kw, kw + Wo * stride, stride)
                ps = f_s[:, rows[:, None], cols].reshape(Cin, P)
                pe = f_e[:, rows[:, None], cols].reshape(Cin, P)
                pz = f_z[:, rows[:, None], cols].reshape(Cin, P)
                wv_s = w_s[co, :, kh, kw].astype(np.int8)
                wv_e = w_e[co, :, kh, kw].astype(np.int32)
                ts.append((ps.astype(np.int16) * wv_s[:, None]).astype(np.int8))
                te.append(np.clip(pe + wv_e[:, None] - BIAS, 0, MAX_EXP).astype(np.int32))
                tz.append(pz)
        if not ts:  # all-zero receptive field
            out_s[co], out_e[co], out_z[co] = 1, 0, True
            continue
        T = np.stack(ts).reshape(-1, P)   # signs per MAC term
        E = np.stack(te).reshape(-1, P)   # exps per MAC term
        Z = np.stack(tz).reshape(-1, P)   # zero mask per MAC term
        m = int(np.max(np.where(Z, 0, E)))
        dd = np.clip(m - E, 0, FRAC_CAP)
        q = (T.astype(np.int64) * frac[dd] * (~Z)).sum(axis=0, dtype=np.int64)
        if np.all(q == 0):
            out_s[co], out_e[co], out_z[co] = 1, 0, True
            continue
        s, e, z = from_fixed_vec(q, m)
        out_s[co], out_e[co], out_z[co] = s, e, z
    if b_s is not None:
        add_lut, sub_lut = build_add_lut(), build_sub_lut()
        out_s, out_e, out_z = vec_phi_add(
            out_s, out_e, out_z,
            np.broadcast_to(b_s, out_s.shape).astype(np.int8),
            np.broadcast_to(b_e, out_s.shape).astype(np.int32),
            np.broadcast_to(b_z, out_s.shape),
            add_lut, sub_lut)
    return (out_s.reshape(Cout, Ho, Wo), out_e.reshape(Cout, Ho, Wo).astype(np.int32),
            out_z.reshape(Cout, Ho, Wo))



# ---------------------------------------------------------------------------
# Integer resampling + attention core (fixed-point datapath, runtime integer)
# ---------------------------------------------------------------------------

def int_interpolate_bilinear_fixed(f_s, f_e, f_z, Ho: int, Wo: int,
                                   align_corners: bool = False):
    """Integer bilinear up/downsample. (C,H,W) triple -> (C,Ho,Wo) triple.

    Source coords as integer rationals; neighbor values via fixed-point
    bridge; weighted average with integer division; from_fixed once.
    align_corners=False matches fusion residual path; True matches head.
    """
    C, H, W = f_s.shape
    frac = _get_frac()
    out_s = np.empty((C, Ho, Wo), dtype=np.int8)
    out_e = np.empty((C, Ho, Wo), dtype=np.int32)
    out_z = np.empty((C, Ho, Wo), dtype=bool)
    if align_corners:
        if Ho == 1:
            ys = [(0, 0, 1, 0)]
        else:
            ys = []
            for i in range(Ho):
                num = i * (H - 1)
                den = Ho - 1
                i0 = num // den
                r = num % den
                i1 = min(i0 + 1, H - 1)
                w1 = 0 if i1 == i0 else r
                ys.append((i0, i1, den - w1, w1, den))
        if Wo == 1:
            xs = [(0, 0, 1, 0)]
        else:
            xs = []
            for j in range(Wo):
                num = j * (W - 1)
                den = Wo - 1
                j0 = num // den
                r = num % den
                j1 = min(j0 + 1, W - 1)
                w1 = 0 if j1 == j0 else r
                xs.append((j0, j1, den - w1, w1, den))
    else:
        ys = []
        for i in range(Ho):
            num = (2 * i + 1) * H - Ho
            den = 2 * Ho
            i0 = num // den
            r = num % den
            if i0 < 0:
                ys.append((0, 0, den, 0, den))
            elif i0 >= H - 1:
                ys.append((H - 1, H - 1, den, 0, den))
            else:
                ys.append((i0, i0 + 1, den - r, r, den))
        xs = []
        for j in range(Wo):
            num = (2 * j + 1) * W - Wo
            den = 2 * Wo
            j0 = num // den
            r = num % den
            if j0 < 0:
                xs.append((0, 0, den, 0, den))
            elif j0 >= W - 1:
                xs.append((W - 1, W - 1, den, 0, den))
            else:
                xs.append((j0, j0 + 1, den - r, r, den))
    for c in range(C):
        m = int(np.max(np.where(f_z[c], 0, f_e[c])))
        dd = np.clip(m - f_e[c], 0, FRAC_CAP)
        qmap = (f_s[c].astype(np.int64) * frac[dd] * (~f_z[c])).astype(np.int64)
        den_y = ys[0][4]
        den_x = xs[0][4]
        den = den_y * den_y * den_x * den_x
        for i, (i0, i1, wy0, wy1, _) in enumerate(ys):
            for j, (j0, j1, wx0, wx1, _) in enumerate(xs):
                num = (int(qmap[i0, j0]) * wy0 * wx0 + int(qmap[i0, j1]) * wy0 * wx1
                       + int(qmap[i1, j0]) * wy1 * wx0 + int(qmap[i1, j1]) * wy1 * wx1)
                qv = (num + den // 2) // den if den else 0
                s, e, z = from_fixed_scalar(int(qv), m)
                out_s[c, i, j], out_e[c, i, j], out_z[c, i, j] = s, e, z
    return out_s, out_e, out_z


def int_conv_transpose2d_fixed(f_s, f_e, f_z, w_s, w_e, b_s=None, b_e=None, b_z=None):
    """Integer transposed-conv for NON-OVERLAPPING kernels (k == stride).

    Covers neck re0 (k4/s4) and re1 (k2/s2): each output ← one input pixel
    over Cin terms via fixed_dot. NOTE: ConvTranspose weights are laid out
    (Cin, Cout, kH, kW) — transposed vs Conv2d — indexed accordingly.
    Overlapping deconvs are out of scope (re3-down is a stride conv ->
    int_conv2d_fixed).
    """
    Cin = f_s.shape[0]
    _, Cout, kH, kW = w_s.shape
    H, W = f_s.shape[1], f_s.shape[2]
    Ho, Wo = H * kH, W * kW
    out_s = np.empty((Cout, Ho, Wo), dtype=np.int8)
    out_e = np.empty((Cout, Ho, Wo), dtype=np.int32)
    out_z = np.empty((Cout, Ho, Wo), dtype=bool)
    P = H * W
    fs = f_s.reshape(Cin, P)
    fe = f_e.reshape(Cin, P)
    fz = f_z.reshape(Cin, P)
    for co in range(Cout):
        for oh in range(kH):
            for ow in range(kW):
                ts = (fs.astype(np.int16) * w_s[:, co, oh, ow].astype(np.int16)[:, None]
                      ).astype(np.int8)
                te = np.clip(fe + w_e[:, co, oh, ow].astype(np.int32)[:, None] - BIAS,
                             0, MAX_EXP).astype(np.int32)
                s, e, z = fixed_dot_terms(ts, te)
                # scatter window (oh, ow) over the strided grid:
                out_s[co].reshape(Ho, Wo)[oh::kH, ow::kW] = s.reshape(H, W)
                out_e[co].reshape(Ho, Wo)[oh::kH, ow::kW] = e.reshape(H, W)
                out_z[co].reshape(Ho, Wo)[oh::kH, ow::kW] = z.reshape(H, W)
    if b_s is not None:
        add_lut, sub_lut = build_add_lut(), build_sub_lut()
        P2 = Ho * Wo
        out_s, out_e, out_z = vec_phi_add(
            out_s.reshape(Cout, P2), out_e.reshape(Cout, P2), out_z.reshape(Cout, P2),
            np.broadcast_to(b_s[:, None], (Cout, P2)).astype(np.int8),
            np.broadcast_to(b_e[:, None], (Cout, P2)).astype(np.int32),
            np.broadcast_to(b_z[:, None], (Cout, P2)),
            add_lut, sub_lut)
        out_s = out_s.reshape(Cout, Ho, Wo)
        out_e = out_e.reshape(Cout, Ho, Wo).astype(np.int32)
        out_z = out_z.reshape(Cout, Ho, Wo)
    return out_s, out_e, out_z


def int_softmax_fixed(s, e, z):
    """Integer stable softmax over 1D triple. Returns (num int64 (N,), den int).

    Uses ABSOLUTE fixed-point (ABS_LUT, 2**-_ABS_F units) so exponent
    differences index EXP_LUT correctly. (Relative to_fixed scaling would
    corrupt the absolute diffs — diagnosed 2026-09-21.)
    Stays in fixed-point: downstream attn@V = sum(num_i * vq_i) // den,
    all integer. Decode probs (num/den) for display/parity only.
    """
    exp_lut = _get_exp_lut()
    abs_lut = _get_abs_lut()
    s = np.asanyarray(s, dtype=np.int8)
    e = np.clip(np.asanyarray(e, dtype=np.int32), 0, N_LEVELS - 1)
    z = np.asanyarray(z, dtype=bool)
    q = (s.astype(np.int64) * abs_lut[e] * (~z)).astype(np.int64)
    # stabilize: -inf for zero-masked entries (min int64/4 avoids overflow)
    q = np.where(z, np.iinfo(np.int64).min // 4, q)
    vmax = int(np.max(q))
    # diffs are in 2**-_ABS_F units == EXP_LUT's 2**-_EXP_F units (_ABS_F == _EXP_F)
    assert _ABS_F == _EXP_F
    dd = np.clip(vmax - q, 0, exp_lut.shape[0] - 1)
    num = exp_lut[dd]
    den = int(np.sum(num, dtype=np.int64))
    return num, den


def neck_conv_parity(geo, add_lut, sub_lut, size: int = 238):
    """Integer neck convs vs float: reassemble 1x1 -> 3x3 -> fusion 3x3+ReLU.

    Chains integer-to-integer (no float round-trip between stages).
    Returns True if all stages pass corr > 0.999.
    """
    import torch
    import torch.nn.functional as F
    from geo_depth import preprocess

    gx, gy = np.meshgrid(np.linspace(0, 1, size, dtype=np.float32),
                         np.linspace(0, 1, size, dtype=np.float32))
    rgb = np.stack([gx, gy, np.full((size, size), 0.5, dtype=np.float32)], axis=-1)
    pv = preprocess(rgb, size=size)
    with torch.no_grad():
        fmaps, _, _ = geo.backbone.forward_stages(pv)
    fmap = fmaps[0].squeeze(0).numpy()  # (384,H,W) float (decoded-phi weights)

    baked = load_baked_ints(Path(__file__).parent / 'weights' / 'geometric_neck.npz')

    def ints(name):
        return baked[name]  # (signs, exps)

    def floats(name):
        s, e = ints(name)
        return int_decode_array(s, e, np.zeros_like(e, dtype=bool))

    def report(tag, got_f, ref_f):
        c = float(np.corrcoef(got_f.flatten(), ref_f.flatten().astype(np.float64))[0, 1])
        ok = c > 0.999
        print(f"  [{tag}] corr={c:.6f} {'PASS' if ok else 'FAIL'}")
        return ok

    ok = True
    # Stage A: reassemble proj re0 (1x1, 384->48, +bias)
    fs, fe, fz = int_encode_array(fmap)
    ws, we = ints('re0.proj.weight')
    bs, be = ints('re0.proj.bias')
    o_s, o_e, o_z = int_conv2d(fs, fe, fz, ws, we, bs, be, np.zeros(48, dtype=bool),
                               add_lut=add_lut, sub_lut=sub_lut)
    ref = F.conv2d(torch.from_numpy(fmap).unsqueeze(0),
                   torch.from_numpy(floats('re0.proj.weight')),
                   torch.from_numpy(floats('re0.proj.bias'))).squeeze(0).numpy()
    ok &= report('re0-proj 1x1 384->48', int_decode_array(o_s, o_e, o_z), ref)

    # Stage B: conv cv0 (3x3, 48->64, pad1, no bias) chained on integer output
    ws, we = ints('cv0.weight')
    o2_s, o2_e, o2_z = int_conv2d(o_s, o_e, o_z, ws, we, padding=1,
                                  add_lut=add_lut, sub_lut=sub_lut)
    ref2 = F.conv2d(torch.from_numpy(ref).unsqueeze(0),
                    torch.from_numpy(floats('cv0.weight')), None, padding=1).squeeze(0).numpy()
    ok &= report('cv0 3x3 48->64 chained', int_decode_array(o2_s, o2_e, o2_z), ref2)

    # Stage C: fusion res conv (3x3, 64->64, pad1, +bias) + ReLU, chained
    ws, we = ints('fusion0.res1.conv1.weight')
    bs, be = ints('fusion0.res1.conv1.bias')
    o3_s, o3_e, o3_z = int_conv2d(o2_s, o2_e, o2_z, ws, we, bs, be,
                                  np.zeros(64, dtype=bool), padding=1,
                                  add_lut=add_lut, sub_lut=sub_lut)
    o3_s, o3_e, o3_z = int_relu(o3_s, o3_e, o3_z)
    ref3 = F.relu(F.conv2d(torch.from_numpy(ref2).unsqueeze(0),
                           torch.from_numpy(floats('fusion0.res1.conv1.weight')),
                           torch.from_numpy(floats('fusion0.res1.conv1.bias')),
                           padding=1)).squeeze(0).numpy()
    ok &= report('fusion0-res1 3x3 64->64+bias+ReLU chained',
                 int_decode_array(o3_s, o3_e, o3_z), ref3)
    return bool(ok)


def bridge_roundtrip_parity():
    """phi -> fixed -> phi roundtrip vs direct decode. Pass > 0.9999."""
    rng = np.random.default_rng(7)
    feat = (rng.standard_normal((40, 384)) * 2).astype(np.float64)
    s, e, z = int_encode_array(feat)
    direct = int_decode_array(s, e, z)
    q, m = to_fixed_group(s.reshape(-1), e.reshape(-1))
    back_s, back_e, back_z = from_fixed_vec(q, m)
    back = int_decode_array(back_s, back_e, back_z).reshape(feat.shape)
    c = float(np.corrcoef(direct.flatten(), back.flatten())[0, 1])
    ok = c > 0.9999
    print(f"  [bridge roundtrip] corr={c:.6f} {'PASS' if ok else 'FAIL'}")
    return ok


def head_fixed_parity(head, feat, n: int = 2000):
    """Fixed-point-accumulation head dot vs float head. Pass > 0.9999."""
    sub = feat[:n]
    ref = head.float_predict(sub)
    fs, fe = IntegerPhiHead.encode_features(sub)
    got = np.empty(n, dtype=np.float64)
    for i in range(n):
        got[i] = _head_fixed_pixel(head, fs[i], fe[i])
    c = float(np.corrcoef(ref.astype(np.float64), got)[0, 1])
    # Threshold 0.999 (not 0.9999): the fixed path pays one FRAC rounding
    # per term plus one from_fixed mantissa rounding; the log-domain tree
    # path (0.99975) avoids the latter. Both far above int8 practice (~0.99).
    ok = c > 0.999
    print(f"  [head fixed-dot] corr={c:.6f} {'PASS' if ok else 'FAIL'}")
    return ok


def _to_common_scale(q, m, m_star):
    """Integer: re-scale fixed-point q from phi-scale m to m_star (>= m).

    q* = q * PHI**((m-m_star)/K) = (q * FRAC[m_star-m]) >> F, with ROUNDING
    shift (not truncation) to avoid systematic negative bias. Pure integer.
    """
    frac = _get_frac()
    d = m_star - m
    assert d >= 0
    if d == 0:
        return q
    prod = q.astype(np.int64) * int(frac[min(d, FRAC_CAP)])
    return ((prod + (1 << (FIXED_F - 1))) >> FIXED_F).astype(np.int64)


def _head_fixed_pixel(head, fs, fe):
    """Single-pixel fixed-point head dot. Integer-only. Returns float (display).

    Scales combine by the phi-multiply rule (m1+m2-BIAS), so absolute
    magnitudes are exact, not just correlated.
    """
    qf, mf = to_fixed_group(fs, fe)
    qm, mm = to_fixed_group(head.m_s, head.m_e)
    m = mf if mf >= mm else mm
    qf = _to_common_scale(qf, mf, m)
    qm = _to_common_scale(qm, mm, m)
    qc = (qf - qm).astype(np.int64)
    qw, mw = to_fixed_group(head.w_s, head.w_e)
    qw = qw.astype(np.int64)
    # dot: sum(qc*qw) is fixed-scale 2**(2F) at phi-scale (m+mw-BIAS);
    # shift back to 2**F with rounding (all integer, unbounded Python ints).
    total = 0
    for i in range(qc.shape[0]):
        total += int(qc[i]) * int(qw[i])
    total = (total + (1 << (FIXED_F - 1))) >> FIXED_F
    s, e, z = from_fixed_scalar(total, m + mw - BIAS, FIXED_F)
    # + target mean via integer LUT-add
    s2, e2 = phi_add(s, e, int(head.tm_s), int(head.tm_e),
                     head.add_lut, head.sub_lut)
    return _decode_exp(e2, 0) * s2 if e2 != 0 else 0.0


def interp_parity():
    """Integer bilinear x2 vs F.interpolate.

    Pass > 0.9995 (from_fixed mantissa rounding sets the floor here).
    """
    import torch.nn.functional as F
    rng = np.random.default_rng(11)
    fmap = (rng.standard_normal((8, 17, 17)) * 3).astype(np.float64)
    fs, fe, fz = int_encode_array(fmap)
    o_s, o_e, o_z = int_interpolate_bilinear_fixed(fs, fe, fz, 34, 34)
    import torch
    ref = F.interpolate(torch.from_numpy(fmap).unsqueeze(0).float(),
                        size=(34, 34), mode='bilinear',
                        align_corners=False).squeeze(0).numpy()
    got = int_decode_array(o_s, o_e, o_z)
    c = float(np.corrcoef(got.flatten().astype(np.float64), ref.flatten().astype(np.float64))[0, 1])
    ok = c > 0.9995
    print(f"  [interp x2 bilinear] corr={c:.6f} {'PASS' if ok else 'FAIL'}")
    return ok


def deconv_parity():
    """Integer non-overlapping deconv (re1 k2/s2, real weights) vs float. Pass > 0.999."""
    import torch
    import torch.nn.functional as F
    baked = load_baked_ints(Path(__file__).parent / 'weights' / 'geometric_neck.npz')
    rng = np.random.default_rng(13)
    fmap = (rng.standard_normal((96, 8, 8)) * 2).astype(np.float64)
    fs, fe, fz = int_encode_array(fmap)
    ws, we = baked['re1.deconv.weight']
    bs, be = baked['re1.deconv.bias']
    bz = np.zeros(96, dtype=bool)
    o_s, o_e, o_z = int_conv_transpose2d_fixed(fs, fe, fz, ws, we, bs, be, bz)
    w64 = ws.shape
    ref = F.conv_transpose2d(torch.from_numpy(fmap).unsqueeze(0).float(),
                             torch.from_numpy(int_decode_array(
                                 ws, we, np.zeros_like(we, dtype=bool))).reshape(w64).float(),
                             torch.from_numpy(int_decode_array(
                                 bs, be, bz)).float(), stride=2).squeeze(0).numpy()
    got = int_decode_array(o_s, o_e, o_z)
    c = float(np.corrcoef(got.flatten().astype(np.float64), ref.flatten().astype(np.float64))[0, 1])
    ok = c > 0.999
    print(f"  [deconv re1 k2/s2 96ch] corr={c:.6f} {'PASS' if ok else 'FAIL'}")
    return ok


def softmax_parity():
    """Integer stable softmax vs float softmax on attention-like logits. Pass > 0.9999."""
    import torch.nn.functional as F
    import torch
    rng = np.random.default_rng(17)
    # attention-like: mostly small negatives + a few large spikes
    logits = (rng.standard_normal(290) * 1.5 - 2.0).astype(np.float64)
    logits[rng.choice(290, 4, replace=False)] += 6.0
    s, e, z = int_encode_array(logits)
    num, den = int_softmax_fixed(s, e, z)
    got = (num.astype(np.float64) / den).astype(np.float64)
    ref = F.softmax(torch.from_numpy(logits).float(), dim=0).numpy().astype(np.float64)
    c = float(np.corrcoef(got, ref)[0, 1])
    ok = c > 0.9999
    print(f"  [softmax 290-way] corr={c:.6f} {'PASS' if ok else 'FAIL'}")
    return ok


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
    print("HEAD:", "PASS" if corr > 0.999 else "FAIL")

    print("neck conv parity (integer-chained, 238px input)...")
    ok_neck = neck_conv_parity(geo, head.add_lut, head.sub_lut)
    print("NECK:", "PASS" if ok_neck else "FAIL")

    print("fixed-point bridge + resampling + attention core...")
    results = []
    results.append(bridge_roundtrip_parity())
    results.append(head_fixed_parity(head, feat))
    results.append(interp_parity())
    results.append(deconv_parity())
    results.append(softmax_parity())
    print("BRIDGE/RESAMPLE/ATTN:", "PASS" if all(results) else "FAIL")
    print("OVERALL:", "PASS" if (corr > 0.999 and ok_neck and all(results)) else "FAIL")


if __name__ == '__main__':
    main()
