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
    print("OVERALL:", "PASS" if (corr > 0.999 and ok_neck) else "FAIL")


if __name__ == '__main__':
    main()
