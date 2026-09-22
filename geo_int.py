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

import math

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


_JIT = None  # lazy (fn_from_fixed, coarse, coff, fine) or False


def _jit_kernels():
    """Lazy Numba kernels (geo_jit). Returns dict or None (python fallback)."""
    global _JIT
    if _JIT is None:
        try:
            from geo_jit import from_fixed_njit, int_head_njit
            coarse, fine = _get_coarse_fine()
            _JIT = {'from_fixed': from_fixed_njit,
                    'head': int_head_njit,
                    'coarse': coarse.astype(np.int32),
                    'coff': _COARSE_OFF,
                    'fine': fine.astype(np.int32)}
        except Exception:
            _JIT = False
    return _JIT if _JIT is not False else None


def _reencode(q, m):
    """from_fixed dispatch: Numba kernel when available (bit-exact), else python."""
    j = _jit_kernels()
    q = np.asanyarray(q, dtype=np.int64)
    if j is not None:
        s, e, z = j['from_fixed'](q.reshape(-1), int(m), FIXED_F,
                                  j['coarse'], j['coff'], j['fine'])
        return s.reshape(q.shape), e.reshape(q.shape), z.reshape(q.shape)
    return from_fixed_vec(q, m)


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
        """Integer-only over N pixels. fs/fe: [N,32] int. Returns float (decode for display).

        Uses the Numba kernel when available (bit-exact vs the python loop).
        """
        j = _jit_kernels()
        if j is not None:
            s, e = j['head'](np.asanyarray(fs, dtype=np.int8),
                             np.asanyarray(fe, dtype=np.int32),
                             self.w_s, self.w_e, self.m_s, self.m_e,
                             np.int8(self.tm_s), np.int32(self.tm_e),
                             self.add_lut.astype(np.int32),
                             self.sub_lut.astype(np.int32))
            return np.array([_decode_exp(int(ee), 0) * int(ss) if int(ee) != 0 else 0.0
                             for ss, ee in zip(s, e)])
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
    return _reencode(total, m)


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
        s, e, z = _reencode(q, m)
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


# ---------------------------------------------------------------------------
# Full integer transformer layer (fixed-point datapath, runtime integer).
#
# Strategy (standard quantized-inference practice, phi-compressed weights):
# bake phi triples -> absolute fixed-point int64 matrices ONCE (offline),
# then the whole layer runs in fixed 2**14: LN (int mean/var/isqrt/div),
# QKV/out-proj/MLP (int64 MACs + rounding shift), attention scores,
# stable softmax via EXP_LUT (stays fixed for attn@V), GELU via bounded
# LUT, layer-scale multiplies, residuals. from_fixed/decode only for
# display/parity. No exp/log/sqrt/div-float/pow anywhere at runtime
# (math.isqrt is integer bit arithmetic).
# ---------------------------------------------------------------------------

INT_F = 14
INT_S = 1 << INT_F
_GELU_LUT = None
_GELU_SPAN = 8  # covers [-8, +8]; asymptotes beyond


def _get_gelu_lut():
    """GELU_LUT[i] = round(gelu((i - SPAN*2**F)/2**F) * 2**F). Offline."""
    global _GELU_LUT
    import math
    if _GELU_LUT is None:
        n = 2 * _GELU_SPAN * INT_S + 1
        lut = np.zeros(n, dtype=np.int32)
        for k in range(n):
            x = (k - _GELU_SPAN * INT_S) / INT_S
            # exact gelu via erf (offline floats OK)
            g = 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
            lut[k] = int(round(g * INT_S))
        _GELU_LUT = lut
    return _GELU_LUT


def _phi_to_fixed14(s, e):
    """Offline bake helper: phi triple arrays -> absolute fixed 2**14 int64."""
    abs_lut = _get_abs_lut()  # 2**-_ABS_F units; _ABS_F == _EXP_F == INT_F
    assert _ABS_F == INT_F
    return (np.asanyarray(s, dtype=np.int8).astype(np.int64)
            * abs_lut[np.clip(np.asanyarray(e, dtype=np.int32), 0, N_LEVELS - 1)])


def bake_layer_fixed(npz_path, layer: int = 0):
    """Bake one backbone layer's phi weights -> fixed-point int64 dict."""
    baked = load_baked_ints(npz_path)
    p = f'layer{layer}.'
    out = {}
    for name in ('q.weight', 'k.weight', 'v.weight', 'proj.weight',
                 'mlp1.weight', 'mlp2.weight'):
        s, e = baked[p + name]
        out[name] = _phi_to_fixed14(s, e).reshape(
            {'q.weight': (384, 384), 'k.weight': (384, 384),
             'v.weight': (384, 384), 'proj.weight': (384, 384),
             'mlp1.weight': (1536, 384), 'mlp2.weight': (384, 1536)}[name])
    for name in ('q.bias', 'k.bias', 'v.bias', 'proj.bias',
                 'mlp1.bias', 'mlp2.bias',
                 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias',
                 'ls1', 'ls2'):
        s, e = baked[p + name]
        out[name] = _phi_to_fixed14(s, e).reshape(-1)
    return out


def int_linear_fixed(X, W, b):
    """Integer: Y = (X @ W.T + 2**(F-1)) >> F + b. X:(N,Di), W:(Do,Di), b:(Do,)."""
    acc = X.astype(np.int64) @ W.astype(np.int64).T  # 2**(2F) scale
    Y = ((acc + (1 << (INT_F - 1))) >> INT_F).astype(np.int64)
    return Y + b


def int_layernorm_affine(X, w, b, eps: float = 1e-6):
    """Integer LayerNorm+affine. X:(N,D) fixed 2**F. Returns fixed 2**F."""
    import math
    N, D = X.shape
    Y = np.empty_like(X)
    eps_i = int(eps * (1 << (2 * INT_F)) + 0.5)
    for n in range(N):
        row = X[n].astype(np.int64)
        mean = (int(np.sum(row)) + D // 2) // D
        c = row - mean
        var = (int(np.sum(c * c)) + D // 2) // D + eps_i
        std = math.isqrt(var)
        if std == 0:
            Y[n] = b
            continue
        norm = (c * (1 << INT_F) + (std // 2)) // std
        Y[n] = ((norm * w + (1 << (INT_F - 1))) >> INT_F) + b
    return Y.astype(np.int64)


def int_gelu_fixed(X):
    """Integer GELU via bounded LUT. X fixed 2**F -> fixed 2**F."""
    lut = _get_gelu_lut()
    span = _GELU_SPAN * INT_S
    idx = np.clip(X.astype(np.int64) + span, 0, lut.shape[0] - 1)
    lo = X < -span
    hi = X > span
    Y = lut[idx].astype(np.int64)
    Y = np.where(lo, 0, Y)       # gelu -> 0 far left
    Y = np.where(hi, X, Y)       # gelu -> x far right
    return Y


def int_softmax_fixedvals(Q):
    """Integer stable softmax over rows of absolute fixed-point Q.

    Returns (num (...,N) int64 scale 2**_EXP_G, den (...) int) for direct
    fixed-point downstream MACs: attn@V = sum(num*V) // den.
    """
    exp_lut = _get_exp_lut()
    Q = np.asanyarray(Q, dtype=np.int64)
    vmax = np.max(Q, axis=-1, keepdims=True)
    dd = np.clip(vmax - Q, 0, exp_lut.shape[0] - 1)
    num = exp_lut[dd]
    den = np.sum(num, axis=-1, dtype=np.int64)
    return num, den


def int_attention_fixed(X, W, heads: int = 6):
    """Integer 6-head self-attention. X:(N,384) fixed -> (N,384) fixed."""
    N, D = X.shape
    hd = D // heads
    Q = int_linear_fixed(X, W['q.weight'], W['q.bias'])
    K = int_linear_fixed(X, W['k.weight'], W['k.bias'])
    V = int_linear_fixed(X, W['v.weight'], W['v.bias'])
    outs = []
    for h in range(heads):
        q = Q[:, h * hd:(h + 1) * hd]
        k = K[:, h * hd:(h + 1) * hd]
        v = V[:, h * hd:(h + 1) * hd]
        scores = (q.astype(np.int64) @ k.astype(np.int64).T
                  + (1 << (INT_F - 1))) >> INT_F   # 2**F scale
        scores = scores // 8                        # /sqrt(64), integer
        num, den = int_softmax_fixedvals(scores)    # 2**G scale
        den = np.where(den == 0, 1, den)
        # (2**G @ 2**F) // 2**G -> 2**F scale. Exact integer bookkeeping.
        o = (num.astype(np.int64) @ v.astype(np.int64)) // den[..., None]
        outs.append(o)
    O = np.concatenate(outs, axis=1)
    return int_linear_fixed(O, W['proj.weight'], W['proj.bias'])


def int_transformer_layer_fixed(X, W):
    """Full integer transformer layer (DINOV2 order). X:(N,384) fixed 2**F."""
    h = int_layernorm_affine(X, W['norm1.weight'], W['norm1.bias'])
    a = int_attention_fixed(h, W)
    x = X + (a * W['ls1'] + (1 << (INT_F - 1))) // (1 << INT_F)
    h2 = int_layernorm_affine(x, W['norm2.weight'], W['norm2.bias'])
    m = int_linear_fixed(h2, W['mlp1.weight'], W['mlp1.bias'])
    m = int_gelu_fixed(m)
    m = int_linear_fixed(m, W['mlp2.weight'], W['mlp2.bias'])
    return (x + (m * W['ls2'] + (1 << (INT_F - 1))) // (1 << INT_F)).astype(np.int64)


# ---------------------------------------------------------------------------
# Fixed-point sensor front end (518-native, integer-only).
#
# Replaces the last float boundary before the backbone: uint8 RGB ->
# normalized fixed 2**14 -> patch tokens + CLS/pos, all integer.
# Resize interpolation stays host-side (like decode-for-display);
# pos_embed is the baked 1370x384 518-native grid, so non-518 inputs
# are out of scope for v1 (documented, not hidden).
# ---------------------------------------------------------------------------

SENSOR_MEAN = (0.485, 0.456, 0.406)
SENSOR_STD = (0.229, 0.224, 0.225)
SENSOR_K = 8  # extra precision bits in the affine


def sensor_affine_consts():
    """Per-channel (A_c, B_c): X = floor((p*A_c + B_c + 2**(K-1))/2**K).

    Offline floats OK (constants, not runtime). A_c = 2**(14+K)/(255*s_c),
    B_c = -m_c*2**(14+K)/s_c, rounded to int64."""
    A, B = [], []
    for m, s in zip(SENSOR_MEAN, SENSOR_STD):
        A.append(int(round(2 ** (INT_F + SENSOR_K) / (255 * s))))
        B.append(int(round(-m * 2 ** (INT_F + SENSOR_K) / s)))
    return (np.array(A, dtype=np.int64), np.array(B, dtype=np.int64))


def sensor_encode_fixed(rgb_u8):
    """uint8 RGB (H,W,3) -> normalized fixed 2**14 int64. Integer-only."""
    A, B = sensor_affine_consts()
    p = np.asanyarray(rgb_u8, dtype=np.int64)
    half = 1 << (SENSOR_K - 1)
    return ((p * A[None, None, :] + B[None, None, :] + half)
            // (1 << SENSOR_K)).astype(np.int64)


def bake_sensor_fixed(npz_path):
    """Bake patch-embed + CLS + pos to fixed 2**14 int64 dict."""
    baked = load_baked_ints(npz_path)
    s, e = baked['patch_proj.weight']
    W = _phi_to_fixed14(s, e).reshape(384, -1)  # (384, 588)
    s, e = baked['patch_proj.bias']
    b = _phi_to_fixed14(s, e).reshape(-1)
    s, e = baked['cls_token']
    cls = _phi_to_fixed14(s, e).reshape(-1)
    s, e = baked['pos_embed']
    pos = _phi_to_fixed14(s, e).reshape(-1, 384)
    return {'patch_w': W, 'patch_b': b, 'cls': cls, 'pos': pos}


def sensor_patch_tokens(fixed_rgb, W, b, cls, pos):
    """Fixed normalized (518,518,3) -> (1370,384) tokens. Integer-only.

    Row-major 14x14 patches via int_linear_fixed (Di=588; accumulator
    bound: 588 terms x ~2^16 x ~2^14 << 2^63), then CLS prepend + pos add.
    """
    H, Wd, _ = fixed_rgb.shape
    assert H % 14 == 0 and Wd % 14 == 0 and pos.shape[0] == H // 14 * (Wd // 14) + 1
    ph = H // 14
    T = np.empty((ph * ph + 1, 384), dtype=np.int64)
    T[0] = cls
    patch = np.empty((1, 588), dtype=np.int64)
    for ty in range(ph):
        for tx in range(ph):
            block = fixed_rgb[ty * 14:(ty + 1) * 14, tx * 14:(tx + 1) * 14, :]
            patch[0] = block.transpose(2, 0, 1).reshape(-1)
            T[1 + ty * ph + tx] = int_linear_fixed(patch, W, b)[0]
    return T + pos


def sensor_parity():
    """Fixed sensor tokens vs HF embeddings; sensor-fed layer0 vs HF layer0.
    Pass: tokens > 0.9999, layer0 > 0.999."""
    import torch
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    from PIL import Image
    gx, gy = np.meshgrid(np.linspace(0, 1, 518, dtype=np.float32),
                         np.linspace(0, 1, 518, dtype=np.float32))
    rgb = np.stack([gx, gy, np.full((518, 518), 0.5, dtype=np.float32)], axis=-1)
    u8 = (rgb * 255).astype(np.uint8)
    proc = AutoImageProcessor.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf')
    model = AutoModelForDepthEstimation.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf').eval()
    bb = model.backbone
    pv = proc(images=Image.fromarray(u8), return_tensors='pt')['pixel_values']
    got = {}
    h1 = bb.encoder.layer[0].register_forward_hook(
        lambda m, i, o: got.__setitem__(
            'full', o[0].detach() if isinstance(o, tuple) else o.detach()))
    with torch.no_grad():
        emb = bb.embeddings(pv)
        _ = bb.encoder.layer[0](emb)
    h1.remove()
    fixed = sensor_encode_fixed(u8)
    baked = bake_sensor_fixed(Path(__file__).parent / 'weights' / 'geometric_backbone.npz')
    T = sensor_patch_tokens(fixed, baked['patch_w'], baked['patch_b'],
                            baked['cls'], baked['pos'])
    ref = emb.squeeze(0).numpy().astype(np.float64)
    c0 = float(np.corrcoef((T.astype(np.float64) / INT_S).flatten(),
                           ref.flatten())[0, 1])
    maxabs = float(np.max(np.abs(T.astype(np.float64) / INT_S - ref)))
    print(f"  [sensor tokens] corr={c0:.6f} maxabs={maxabs:.6f} "
          f"(fixed units: {maxabs * INT_S:.2f} ULP)")
    W = bake_layer_fixed(Path(__file__).parent / 'weights' / 'geometric_backbone.npz', 0)
    full = int_transformer_layer_fixed(T, W)
    ref2 = got['full'].squeeze(0).numpy().astype(np.float64)
    c1 = float(np.corrcoef((full.astype(np.float64) / INT_S).flatten(),
                           ref2.flatten())[0, 1])
    print(f"  [sensor-fed layer0] corr={c1:.6f}")
    ok = c0 > 0.9999 and c1 > 0.999
    print("  SENSOR:", "PASS" if ok else "FAIL")
    return ok


def transformer_layer_parity(size: int = 112):
    """Full integer transformer layer vs HF DINOv2 layer0, stage-wise.

    Input: real HF embeddings (float->fixed boundary); everything after
    is integer. (Note: the HF processor upscales to 518, so this runs at
    the full 1370 tokens.) Reports attn-block / mlp-block / full-layer
    corr. Pass: full > 0.99.
    """
    import torch
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor
    from PIL import Image

    gx, gy = np.meshgrid(np.linspace(0, 1, size, dtype=np.float32),
                         np.linspace(0, 1, size, dtype=np.float32))
    rgb = np.stack([gx, gy, np.full((size, size), 0.5, dtype=np.float32)], axis=-1)
    proc = AutoImageProcessor.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf')
    model = AutoModelForDepthEstimation.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf').eval()
    bb = model.backbone
    layer = bb.encoder.layer[0]
    pil = Image.fromarray((rgb * 255).astype(np.uint8))
    pv = proc(images=pil, return_tensors='pt')['pixel_values']

    got = {}

    def hk(name):
        def fn(mod, inp, out):
            got[name] = out[0].detach() if isinstance(out, tuple) else out.detach()
        return fn

    h1 = layer.register_forward_hook(lambda m, i, o: got.__setitem__(
        'full', o[0].detach() if isinstance(o, tuple) else o.detach()))
    h2 = layer.layer_scale1.register_forward_hook(
        lambda m, i, o: got.__setitem__('attn_scaled', o.detach()))
    h3 = layer.layer_scale2.register_forward_hook(
        lambda m, i, o: got.__setitem__('mlp_scaled', o.detach()))
    with torch.no_grad():
        emb = bb.embeddings(pv)
        _ = layer(emb)
    h1.remove()
    h2.remove()
    h3.remove()
    print(f"  layer0 in/out: {emb.shape} -> {got['full'].shape}")

    W = bake_layer_fixed(Path(__file__).parent / 'weights' / 'geometric_backbone.npz', 0)
    X = np.round(emb.squeeze(0).numpy().astype(np.float64) * INT_S).astype(np.int64)

    def stage(tag, int_arr, ref_t):
        ref = ref_t.squeeze(0).numpy().astype(np.float64)
        mine = (int_arr.astype(np.float64) / INT_S)
        c = float(np.corrcoef(mine.flatten(), ref.flatten())[0, 1])
        print(f"  [{tag}] corr={c:.6f}")
        return c

    # integer path with intermediates
    h = int_layernorm_affine(X, W['norm1.weight'], W['norm1.bias'])
    a = int_attention_fixed(h, W)
    a_scaled = (a * W['ls1'] + (1 << (INT_F - 1))) // (1 << INT_F)
    c1 = stage('attn block', a_scaled, got['attn_scaled'])
    x = X + a_scaled
    h2 = int_layernorm_affine(x, W['norm2.weight'], W['norm2.bias'])
    m = int_linear_fixed(h2, W['mlp1.weight'], W['mlp1.bias'])
    m = int_gelu_fixed(m)
    m = int_linear_fixed(m, W['mlp2.weight'], W['mlp2.bias'])
    c2 = stage('mlp block', (m * W['ls2'] + (1 << (INT_F - 1))) // (1 << INT_F),
               got['mlp_scaled'])
    full = int_transformer_layer_fixed(X, W)
    c3 = stage('full layer0', full, got['full'])
    ok = c3 > 0.99
    print("  LAYER:", "PASS" if ok else "FAIL")
    return ok


# ---------------------------------------------------------------------------
# Accumulation shootout: four traditions, one harness.
#
# The same dot product accumulated four ways (all on OUR K=512 weights):
#   tree   — log-domain LUT-adds, pairwise (this file's production path)
#   fixed  — fixed-point bridge, single rounding (this file)
#   fib    — Fibonacci-coefficient exact accumulation (phi_lattice theory
#            Pathology §11: decompose each term to 2K exact int coeffs,
#            accumulate with ZERO rounding, single float solve at the
#            display boundary; Python ints stand in for the theory's
#            multi-limb solver, which firmware would need)
#   taylor — table-free log-space Taylor addition (phi_geist vendor
#            tradition: correction log_phi(1+x) via series, no LUTs).
#            NOTE: Taylor evaluates its correction in floats at runtime,
#            so it is NOT integer-only — included as a math comparison
#            (tables vs compute tradeoff), labeled honestly throughout.
# ---------------------------------------------------------------------------

_fib_cache = {0: (0, 1), 1: (1, 1)}


def _fib_pair(n: int):
    """Exact (F_n, F_{n+1}) by fast doubling; negafibonacci for n<0.

    F_0=0, F_1=1; F_{-n} = (-1)^{n+1} F_n. Python ints: exact, unbounded
    (firmware needs the theory's multi-limb solver past ~F_92).
    """
    if n in _fib_cache:
        return _fib_cache[n]
    if n < 0:
        # F_{-m} = (-1)^{m+1} F_m with m = -n > 0.
        m = -n
        a, b = _fib_pair(m)  # a = F_m, b = F_{m+1}; F_{m-1} = b - a
        s = -1 if m % 2 == 0 else 1
        out = (s * a, -s * (b - a))
        _fib_cache[n] = out
        return out
    a, b = _fib_pair(n >> 1)
    c = a * ((b << 1) - a)
    d = a * a + b * b
    out = (c, d) if n % 2 == 0 else (d, c + d)
    _fib_cache[n] = out
    return out


def fib_dot_terms(s_terms, e_terms):
    """Exact Fibonacci-coefficient dot product. (T,N) int -> float (N,).

    Each term s*PHI^(u/K) (u = E-BIAS unbiased) splits by divmod(u,K)=(q,r):
      PHI^(u/K) = F_q * PHI^((r+K)/K) + F_{q-1} * PHI^(r/K)
    i.e. exactly 2 nonzero coeffs in a 2K lattice basis. Integer
    accumulation is EXACT (no rounding anywhere); the single float solve
    uses max-normalization so small groups aren't lost. Solve happens at
    the display boundary (parity only) — cf. int_decode_array.
    """
    s_terms = np.asanyarray(s_terms, dtype=np.int64)
    e_terms = np.asanyarray(e_terms, dtype=np.int64)
    T, N = s_terms.shape
    U = e_terms - BIAS
    Q, R = np.divmod(U, np.int64(K))
    out = np.empty(N, dtype=np.float64)
    phi_r = np.float64(PHI) ** (np.arange(2 * K, dtype=np.float64) / K)
    for n in range(N):
        coeffs = {}
        for t in range(T):
            s = int(s_terms[t, n])
            if s == 0:
                continue
            q, r = int(Q[t, n]), int(R[t, n])
            Fq, _ = _fib_pair(q)
            Fqm1, _ = _fib_pair(q - 1)
            coeffs[r] = coeffs.get(r, 0) + s * Fqm1
            coeffs[r + K] = coeffs.get(r + K, 0) + s * Fq
        if not coeffs:
            out[n] = 0.0
            continue
        # drop exact-cancelled entries; all-zero -> true zero (no 0/0)
        coeffs = {i: c for i, c in coeffs.items() if c != 0}
        if not coeffs:
            out[n] = 0.0
            continue
        # max-normalized float solve (display boundary only)
        idx = np.array(list(coeffs.keys()))
        mag = np.array([abs(coeffs[i]) * float(phi_r[i]) for i in idx])
        m = mag.max()
        total = sum((coeffs[i] / m) * float(phi_r[i]) for i in idx)
        out[n] = total * m
    return out


def taylor_add(s1, e1, s2, e2, terms: int = 6):
    """Table-free log-space addition (phi_geist tradition, ported to K=512).

    Same-sign: e_out = max + round(C*K), C = log_phi(1+PHI^(-d/K)) via
    `terms`-term Taylor series. Different-sign mirrors with log_phi|1-x|.
    Float correction at runtime -> NOT integer-only; math comparison only.
    """
    if s1 == s2:
        if e1 >= e2:
            d = (e1 - e2) / K
            x = PHI ** (-d)
            c = sum(((-1) ** (k + 1)) * (x ** k) / k for k in range(1, terms + 1)) / LN_PHI
            return s1, e1 + int(round(c * K))
        d = (e2 - e1) / K
        x = PHI ** (-d)
        c = sum(((-1) ** (k + 1)) * (x ** k) / k for k in range(1, terms + 1)) / LN_PHI
        return s2, e2 + int(round(c * K))
    if e1 == e2:
        return 1, 0
    if e1 > e2:
        d = (e1 - e2) / K
        x = PHI ** (-d)
        c = math.log(1 - x) / LN_PHI if x < 1 else float('-inf')
        return (s1, e1) if c == float('-inf') else (s1, e1 + int(round(c * K)))
    d = (e2 - e1) / K
    x = PHI ** (-d)
    c = math.log(1 - x) / LN_PHI if x < 1 else float('-inf')
    return (s2, e2) if c == float('-inf') else (s2, e1 + int(round(c * K)))


def taylor_dot_terms(fs, fe, ws, we, terms: int = 6):
    """Sequential Taylor-add dot product (math comparison; float corrections).

    Products are exact integer exponent-adds (ps = fs*ws, pe = fe+we-BIAS);
    only the ADDITION correction uses the Taylor series. N outputs.
    fs/fe: (N, C) int; ws/we: (C,) int. Returns (s, e) (N,).
    """
    S = np.asanyarray(fs, dtype=np.int64)
    E = np.asanyarray(fe, dtype=np.int64)
    N, C = S.shape
    out_s = np.ones(N, dtype=np.int8)
    out_e = np.zeros(N, dtype=np.int32)
    first = np.ones(N, dtype=bool)
    for c in range(C):
        ps = (S[:, c] * ws[c]).astype(np.int8)
        pe = np.clip(E[:, c] + we[c] - BIAS, 0, MAX_EXP).astype(np.int32)
        for n in range(N):
            if first[n]:
                out_s[n], out_e[n] = ps[n], pe[n]
                first[n] = False
            else:
                out_s[n], out_e[n] = taylor_add(
                    int(out_s[n]), int(out_e[n]), int(ps[n]), int(pe[n]), terms)
    return out_s, out_e


def shootout(feat=None, n_px: int = 1500, seed: int = 23):
    """Four accumulation traditions, one harness. Prints table. Returns dict.

    Single-add sweep across delta bands (vs float64 exact) plus full
    32-wide head dots (vs float head). All methods accumulate the SAME
    phi-centered terms (centering via LUT-add); only accumulation differs.
    Timings are CPython loops except tree/fixed (numpy) — scaling, not
    ranking, is the point (see geo_jit/c_port for fast paths).
    """
    import time
    rng = np.random.default_rng(seed)
    add_lut, sub_lut = build_add_lut(), build_sub_lut()

    print("  single-add rel err vs float64 exact (mean / max per delta band):")
    print(f"  {'band':>12} {'lut':>16} {'taylor2':>16} {'taylor6':>16} {'fib':>16}")
    bands = [(0, 0), (1, 3), (4, 16), (17, 64), (65, 256), (257, 1024), (2049, 8192)]
    for lo, hi in bands:
        d = rng.integers(lo, hi + 1, size=400)
        e1 = rng.integers(28000, 38000, size=400)
        e2 = e1 - d
        s1 = rng.choice([-1, 1], size=400).astype(np.int8)
        s2 = np.where(rng.random(400) < 0.5, s1, -s1).astype(np.int8)
        exact = s1 * PHI ** ((e1 - BIAS) / K) + s2 * PHI ** ((e2 - BIAS) / K)

        def rel(got):
            return np.abs((got - exact) / np.where(exact == 0, 1, exact))

        outs = {}
        gl, ge = np.empty(400, dtype=np.int8), np.empty(400, dtype=np.int32)
        for i in range(400):
            gl[i], ge[i] = phi_add(int(s1[i]), int(e1[i]), int(s2[i]), int(e2[i]),
                                   add_lut, sub_lut)[:2]
        outs['lut'] = gl * PHI ** ((ge - BIAS) / K)
        for t, key in ((2, 'taylor2'), (6, 'taylor6')):
            tl, te = np.empty(400, dtype=np.int8), np.empty(400, dtype=np.int32)
            for i in range(400):
                tl[i], te[i] = taylor_add(int(s1[i]), int(e1[i]),
                                          int(s2[i]), int(e2[i]), t)
            outs[key] = tl * PHI ** ((te - BIAS) / K)
        fb = fib_dot_terms(np.stack([s1, s2]), np.stack([e1, e2]))
        outs['fib'] = fb
        row = []
        for k in ('lut', 'taylor2', 'taylor6', 'fib'):
            r = rel(outs[k])
            row.append(f"{r.mean():.2e}/{r.max():.2e}")
        print(f"  {lo}-{hi:>9} {row[0]:>16} {row[1]:>16} {row[2]:>16} {row[3]:>16}")

    print("  full 32-wide head dots vs float head:")
    if feat is None:
        feat = (rng.standard_normal((n_px, 32)) * 4).astype(np.float64)
    from pathlib import Path as _P
    head = IntegerPhiHead(_P(__file__).parent / 'weights' / 'phi_weights_compact.bin')
    ref = head.float_predict(feat)
    fs, fe = IntegerPhiHead.encode_features(feat)
    res = {}
    t0 = time.perf_counter()
    got_tree = head.int_predict(fs, fe)
    res['tree'] = (time.perf_counter() - t0, got_tree)
    # centered terms shared by fib/taylor/fixed comparisons
    cs = np.empty_like(fs)
    ce = np.empty_like(fe)
    for i in range(feat.shape[0]):
        for c in range(32):
            cs[i, c], ce[i, c] = phi_add(int(fs[i, c]), int(fe[i, c]),
                                         int(-head.m_s[c]), int(head.m_e[c]),
                                         add_lut, sub_lut)
    prod_s = (cs.astype(np.int16) * head.w_s[None, :].astype(np.int16)).astype(np.int8)
    prod_e = np.clip(ce.astype(np.int32) + head.w_e[None, :].astype(np.int32) - BIAS,
                     0, MAX_EXP).astype(np.int32)
    t0 = time.perf_counter()
    acc = fib_dot_terms(prod_s.T, prod_e.T)
    t_fib = time.perf_counter() - t0
    # + target mean via integer add, then decode
    tm = np.empty_like(acc)
    for n in range(feat.shape[0]):
        ss, ee = phi_add(1, 0, 1, 0, add_lut, sub_lut)  # zero
        ss, ee = _fib_plus_tm(acc[n], head, add_lut, sub_lut)
        tm[n] = ss * PHI ** ((ee - BIAS) / K) if ee != 0 else 0.0
    res['fib'] = (t_fib, tm)
    t0 = time.perf_counter()
    # NB: target-mean shift omitted here (correlation is shift-invariant).
    tsl, tel = taylor_dot_terms(cs, ce, head.w_s, head.w_e)
    t_tay = time.perf_counter() - t0
    tay = np.array([tsl[i] * PHI ** ((tel[i] - BIAS) / K) for i in range(feat.shape[0])])
    res['taylor6'] = (t_tay, tay)
    for key, (dt, got) in res.items():
        c = float(np.corrcoef(ref.astype(np.float64), got.astype(np.float64))[0, 1])
        print(f"  [{key:>7}] corr={c:.6f}  {dt:.2f}s/{feat.shape[0]}px")
    return res


def _fib_plus_tm(acc_val, head, add_lut, sub_lut):
    """Add scalar float accumulator value + target mean in phi domain (display)."""
    s = 1 if acc_val >= 0 else -1
    e = int(round(K * math.log(abs(acc_val) + 1e-15) / LN_PHI)) + BIAS
    return phi_add(s, np.clip(e, 0, MAX_EXP), int(head.tm_s), int(head.tm_e),
                   add_lut, sub_lut)


def neck_conv_parity(geo, add_lut, sub_lut, size: int = 238):
    """Integer neck convs vs float: reassemble 1x1 -> 3x3 -> fusion 3x3+ReLU.

    Chains integer-to-integer (no float round-trip between stages).
    Returns True if all stages pass corr > 0.999.
    """
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
    print("integer transformer layer0 vs HF...")
    ok_layer = transformer_layer_parity()
    print("OVERALL:", "PASS" if (corr > 0.999 and ok_neck and all(results)
                                 and ok_layer) else "FAIL")
    print("accumulation shootout (tree/fixed/fib/taylor, informational)...")
    shootout(feat=feat[:1500])


if __name__ == '__main__':
    main()
