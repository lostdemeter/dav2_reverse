"""
Numba-JIT port of the integer (no-FPU) hot loops.

Same integer ops as geo_int.py (add/sub/compare/shift/LUT gather) —
bit-exact identical results, 50-100x faster by removing Python-loop
overhead. Compiled with cache=True (first import compiles, ~10-30s).

Ported hot spots (profiled 2026-09-21):
  from_fixed_vec  1.5 us/el Python  -> from_fixed_njit
  int head loop  28 us/px Python     -> int_head_njit
numpy vec_phi_add is already ~40ns/el; left as is.

The C port (gcc, phi_avx512.c precedent) stays future work; Numba
gives most of the win with none of the build pain.
"""

import numpy as np
from numba import njit, prange

K_JIT = 512
BIAS_JIT = 32768
MAX_EXP_JIT = 65535
DMAX_JIT = 4096


@njit(cache=True)
def _bit_length(a):
    n = 0
    while a:
        n += 1
        a >>= np.int64(1)
    return n


@njit(cache=True)
def from_fixed_njit(q, m, f, coarse, coff, fine):
    """Integer re-encode: (q int64[:], scale m, width f) -> (s, e, z).

    Bit ops + offline LUTs only. Bit-exact match of from_fixed_vec.
    """
    n = q.shape[0]
    s = np.empty(n, dtype=np.int8)
    e = np.empty(n, dtype=np.int32)
    z = np.empty(n, dtype=np.bool_)
    for i in range(n):
        qi = q[i]
        if qi == 0:
            s[i] = np.int8(1)
            e[i] = np.int32(0)
            z[i] = True
            continue
        s[i] = np.int8(1) if qi > 0 else np.int8(-1)
        a = qi if qi > 0 else -qi
        shift = _bit_length(a) - 15
        mant = (a >> np.int64(shift)) if shift >= 0 else (a << np.int64(-shift))
        t = shift + 14 - f
        ev = np.int64(m) + np.int64(coarse[t + coff]) + np.int64(fine[mant - 16384])
        if ev < 0:
            ev = 0
        if ev > MAX_EXP_JIT:
            ev = MAX_EXP_JIT
        e[i] = np.int32(ev)
        z[i] = False
    return s, e, z


@njit(cache=True)
def _phi_add_njit(s1, e1, s2, e2, add_lut, sub_lut):
    """Scalar integer phi-add. Mirrors geo_int.phi_add exactly."""
    if s1 == s2:
        if e1 >= e2:
            d = e1 - e2
            if d > DMAX_JIT:
                return s1, e1
            return s1, e1 + add_lut[d]
        d = e2 - e1
        if d > DMAX_JIT:
            return s2, e2
        return s2, e2 + add_lut[d]
    if e1 == e2:
        return np.int8(1), np.int32(0)
    if e1 > e2:
        d = e1 - e2
        if d > DMAX_JIT:
            return s1, e1
        return s1, e2 + sub_lut[d]
    d = e2 - e1
    if d > DMAX_JIT:
        return s2, e2
    return s2, e1 + sub_lut[d]


@njit(parallel=True, cache=True)
def int_head_njit(fs, fe, w_s, w_e, m_s, m_e, tm_s, tm_e, add_lut, sub_lut):
    """Integer depth head over N pixels. fs/fe:(N,32); returns (s,e) (N,).

    depth = (feat-mean)@w + tm, all integer LUT-adds. Bit-exact match of
    IntegerPhiHead.int_predict_pixel.
    """
    n = fs.shape[0]
    out_s = np.empty(n, dtype=np.int8)
    out_e = np.empty(n, dtype=np.int32)
    for i in prange(n):
        acc_s = np.int8(1)
        acc_e = np.int32(0)
        first = True
        for c in range(32):
            cs, ce = _phi_add_njit(fs[i, c], fe[i, c],
                                   -m_s[c], m_e[c], add_lut, sub_lut)
            if ce == 0 and cs == 1:
                continue
            ps = cs * w_s[c]
            pe = ce + w_e[c] - BIAS_JIT
            if pe < 0:
                pe = 0
            if pe > MAX_EXP_JIT:
                pe = MAX_EXP_JIT
            if first:
                acc_s, acc_e = ps, pe
                first = False
            else:
                acc_s, acc_e = _phi_add_njit(acc_s, acc_e, ps, pe,
                                             add_lut, sub_lut)
        if first:
            out_s[i], out_e[i] = tm_s, tm_e
        else:
            out_s[i], out_e[i] = _phi_add_njit(acc_s, acc_e, tm_s, tm_e,
                                               add_lut, sub_lut)
    return out_s, out_e


def parity_vs_python(n: int = 2000, seed: int = 5):
    """Bit-exactness vs geo_int Python loops + corr vs float head."""
    import time
    from pathlib import Path
    from geo_int import (IntegerPhiHead, from_fixed_vec, to_fixed_group,
                         _get_coarse_fine, _COARSE_OFF, FIXED_F)
    rng = np.random.default_rng(seed)
    feat = (rng.standard_normal((n, 32)) * 4).astype(np.float64)
    head = IntegerPhiHead(Path(__file__).parent / 'weights' / 'phi_weights_compact.bin')
    fs, fe = IntegerPhiHead.encode_features(feat)

    # 1. head loop bit-exactness
    t0 = time.perf_counter()
    ref_s = np.empty(n, dtype=np.int8)
    ref_e = np.empty(n, dtype=np.int32)
    for i in range(n):
        ref_s[i], ref_e[i] = head.int_predict_pixel(fs[i], fe[i])
    t_py = time.perf_counter() - t0
    add = head.add_lut.astype(np.int32)
    sub = head.sub_lut.astype(np.int32)
    t0 = time.perf_counter()  # warmup compile
    got_s, got_e = int_head_njit(fs, fe, head.w_s, head.w_e, head.m_s, head.m_e,
                                 np.int8(head.tm_s), np.int32(head.tm_e), add, sub)
    t0 = time.perf_counter()
    got_s, got_e = int_head_njit(fs, fe, head.w_s, head.w_e, head.m_s, head.m_e,
                                 np.int8(head.tm_s), np.int32(head.tm_e), add, sub)
    t_jit = time.perf_counter() - t0
    exact = bool(np.array_equal(ref_s, got_s) and np.array_equal(ref_e, got_e))
    print(f"  [head bit-exact] {exact}  python {t_py*1e6/n:.1f}us/px -> "
          f"jit {t_jit*1e6/n:.1f}us/px ({t_py/max(t_jit,1e-9):.0f}x)")

    # 2. from_fixed bit-exactness + speed
    q = (rng.standard_normal(n * 8) * 2**20).astype(np.int64)
    t0 = time.perf_counter()
    ps, pe, pz = from_fixed_vec(q, 33000)
    t_py = time.perf_counter() - t0
    coarse, fine = _get_coarse_fine()
    t0 = time.perf_counter()
    js, je, jz = from_fixed_njit(q, 33000, FIXED_F, coarse.astype(np.int32),
                                 _COARSE_OFF, fine.astype(np.int32))
    t0 = time.perf_counter()
    js, je, jz = from_fixed_njit(q, 33000, FIXED_F, coarse.astype(np.int32),
                                 _COARSE_OFF, fine.astype(np.int32))
    t_jit = time.perf_counter() - t0
    exact2 = bool(np.array_equal(ps, js) and np.array_equal(pe, je)
                  and np.array_equal(pz, jz))
    print(f"  [from_fixed bit-exact] {exact2}  python {t_py/len(q)*1e6:.2f}us/el -> "
          f"jit {t_jit/len(q)*1e6:.3f}us/el ({t_py/max(t_jit,1e-9):.0f}x)")

    # 3. end-to-end corr vs float (through jit path)
    from geo_int import _decode_exp
    got_f = np.array([_decode_exp(int(ee), 0) * int(ss) if int(ee) != 0 else 0.0
                      for ss, ee in zip(got_s, got_e)])
    ref_f = head.float_predict(feat)
    c = float(np.corrcoef(ref_f.astype(np.float64), got_f)[0, 1])
    print(f"  [jit head vs float] corr={c:.6f}")
    # Random features are adversarial (uncorrelated 32-d vectors maximize
    # LUT-rounding stress); on real backbone features the same integer path
    # measures 0.999751 (see geo_int.main). Threshold 0.998 here.
    ok = exact and exact2 and c > 0.998
    print("JIT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if parity_vs_python() else 1)
