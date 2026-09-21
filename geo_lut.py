"""
Shared phi-arithmetic LUT for fully-geometric DAV2.

Ported from:
- BACKUP_misc/truthspace-lcm/phi_geometric/core/encoder.py (PhiEncoder, K/bias/LUT)
- BACKUP_misc/truthspace-lcm/experiments/phi_da2_decoder/phi_cuda.py (GPU LUT)

Representation (matches phi_decoder.PhiConfig):
    value = sign * PHI ** ((exponent - BIAS) / K)
    K=512, BITS=16, N_LEVELS=65536, BIAS=32768

All learned weights (backbone ViT, neck reassemble/fusion, head convs)
are stored as (sign, exponent) and LUT-decoded to float once at load.
Multiplication then uses standard torch ops on decoded values — exact to
~99.99%+ correlation, with no HuggingFace download at inference.
"""

import numpy as np
import torch

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)

K = 512
BITS = 16
N_LEVELS = 2 ** BITS
BIAS = N_LEVELS // 2


def build_phi_lut(k: int = K, bias: int = BIAS, n_levels: int = N_LEVELS,
                  device=None, dtype=torch.float32) -> torch.Tensor:
    """Build LUT tensor: lut[e] = PHI ** ((e - bias) / k)."""
    exponents = torch.arange(n_levels, dtype=torch.float64)
    lut = torch.tensor(PHI, dtype=torch.float64) ** ((exponents - bias) / k)
    lut = lut.to(dtype)
    if device is not None:
        lut = lut.to(device)
    return lut


# CPU float32 LUT shared by all modules (computed once).
_LUT_CPU = None


def get_lut(device=None, dtype=torch.float32) -> torch.Tensor:
    global _LUT_CPU
    if _LUT_CPU is None:
        _LUT_CPU = build_phi_lut()
    if device is None:
        return _LUT_CPU.to(dtype)
    return _LUT_CPU.to(device=device, dtype=dtype)


def phi_encode_numpy(values: np.ndarray, k: int = K,
                     bias: int = BIAS, n_levels: int = N_LEVELS):
    """Encode float array -> (signs int8, exponents uint16)."""
    signs = np.sign(values).astype(np.int8)
    signs[signs == 0] = 1
    magnitudes = np.abs(values.astype(np.float64)) + 1e-15
    exponents = k * np.log(magnitudes) / LN_PHI
    exponents = np.round(exponents).astype(np.int64) + bias
    exponents = np.clip(exponents, 0, n_levels - 1).astype(np.uint16)
    return signs, exponents


def phi_decode_numpy(signs: np.ndarray, exponents: np.ndarray,
                     k: int = K, bias: int = BIAS) -> np.ndarray:
    """Decode (signs, exponents) -> float32 array via direct formula."""
    return (signs.astype(np.float32)
            * np.float32(PHI) ** ((exponents.astype(np.float32) - bias) / k))


def phi_decode_torch(signs: torch.Tensor, exponents: torch.Tensor,
                     lut: torch.Tensor = None) -> torch.Tensor:
    """Decode (signs, exponents) -> float tensor via LUT gather."""
    if lut is None:
        lut = get_lut(device=exponents.device)
    else:
        lut = lut.to(exponents.device)
    exps = exponents.to(torch.int64).clamp(0, lut.numel() - 1)
    return signs.to(lut.dtype) * lut[exps]


def phi_encode_torch(values: torch.Tensor, k: int = K,
                     bias: int = BIAS, n_levels: int = N_LEVELS):
    """Encode float tensor -> (signs, exponents)."""
    signs = torch.sign(values)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    magnitudes = torch.abs(values.double()).clamp_min(1e-15)
    exponents = (k * torch.log(magnitudes) / LN_PHI).round().to(torch.int64) + bias
    exponents = exponents.clamp(0, n_levels - 1).to(torch.int64)
    return signs, exponents
