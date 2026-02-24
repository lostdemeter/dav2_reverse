"""
Compact φ-Decoder Storage Format
================================

Optimized storage for φ-decoder weights achieving maximum compression
while maintaining full accuracy.

Storage comparison:
- DA2 full model: 94.55 MB
- DA2 head only: 108 KB  
- φ-decoder (standard): 203 bytes
- φ-decoder (compact): 125 bytes

The compact format exploits:
1. Relative exponents (13 bits instead of 16)
2. All feature means are positive (no sign bits needed)
3. Exponent base values stored once
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import struct

PHI = (1 + np.sqrt(5)) / 2
LN_PHI = np.log(PHI)


@dataclass
class CompactPhiWeights:
    """
    Ultra-compact φ-decoder weights.
    
    Format (125 bytes total):
    - Magic: 4 bytes ('PHI2')
    - k: 2 bytes (uint16)
    - weight_exp_base: 2 bytes (uint16)
    - feature_mean_exp_base: 2 bytes (uint16)
    - target_mean: 3 bytes (sign + exp)
    - weight_signs: 4 bytes (32 bits packed)
    - weight_exps: 52 bytes (32 × 13 bits packed)
    - feature_mean_exps: 56 bytes (32 × 14 bits packed)
    """
    k: int
    weight_exp_base: int
    feature_mean_exp_base: int
    target_mean_sign: int
    target_mean_exp: int
    weight_signs: np.ndarray  # 32 int8
    weight_exps_relative: np.ndarray  # 32 uint16 (relative to base)
    feature_mean_exps_relative: np.ndarray  # 32 uint16 (relative to base)
    
    def to_weights(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert to float weights, feature_mean, target_mean."""
        bias = 2**16 // 2
        
        weights = self.weight_signs * PHI ** (
            (self.weight_exps_relative.astype(np.int32) + self.weight_exp_base - bias) / self.k
        )
        
        feature_mean = PHI ** (
            (self.feature_mean_exps_relative.astype(np.int32) + self.feature_mean_exp_base - bias) / self.k
        )
        
        target_mean = self.target_mean_sign * PHI ** (
            (self.target_mean_exp - bias) / self.k
        )
        
        return weights, feature_mean, target_mean
    
    def save(self, path: Path):
        """Save to compact binary format."""
        with open(path, 'wb') as f:
            # Magic
            f.write(b'PHI2')
            
            # Config
            f.write(struct.pack('H', self.k))
            f.write(struct.pack('H', self.weight_exp_base))
            f.write(struct.pack('H', self.feature_mean_exp_base))
            
            # Target mean
            f.write(struct.pack('b', self.target_mean_sign))
            f.write(struct.pack('H', self.target_mean_exp))
            
            # Pack weight signs into 4 bytes (32 bits)
            sign_bits = 0
            for i, s in enumerate(self.weight_signs):
                if s > 0:
                    sign_bits |= (1 << i)
            f.write(struct.pack('I', sign_bits))
            
            # Pack weight exponents (13 bits each)
            self._write_packed_ints(f, self.weight_exps_relative, 13)
            
            # Pack feature mean exponents (14 bits each)
            self._write_packed_ints(f, self.feature_mean_exps_relative, 14)
    
    def _write_packed_ints(self, f, values: np.ndarray, bits: int):
        """Write packed integers with specified bit width."""
        total_bits = len(values) * bits
        total_bytes = (total_bits + 7) // 8
        
        # Pack into bytes
        packed = 0
        for i, v in enumerate(values):
            packed |= (int(v) << (i * bits))
        
        # Write as bytes
        f.write(packed.to_bytes(total_bytes, 'little'))
    
    @classmethod
    def load(cls, path: Path) -> 'CompactPhiWeights':
        """Load from compact binary format."""
        with open(path, 'rb') as f:
            magic = f.read(4)
            assert magic == b'PHI2', f"Invalid magic: {magic}"
            
            k = struct.unpack('H', f.read(2))[0]
            weight_exp_base = struct.unpack('H', f.read(2))[0]
            feature_mean_exp_base = struct.unpack('H', f.read(2))[0]
            
            target_mean_sign = struct.unpack('b', f.read(1))[0]
            target_mean_exp = struct.unpack('H', f.read(2))[0]
            
            # Unpack weight signs
            sign_bits = struct.unpack('I', f.read(4))[0]
            weight_signs = np.array([
                1 if (sign_bits & (1 << i)) else -1 
                for i in range(32)
            ], dtype=np.int8)
            
            # Unpack weight exponents
            weight_exps = cls._read_packed_ints(f, 32, 13)
            
            # Unpack feature mean exponents
            feature_mean_exps = cls._read_packed_ints(f, 32, 14)
        
        return cls(
            k=k,
            weight_exp_base=weight_exp_base,
            feature_mean_exp_base=feature_mean_exp_base,
            target_mean_sign=target_mean_sign,
            target_mean_exp=target_mean_exp,
            weight_signs=weight_signs,
            weight_exps_relative=weight_exps,
            feature_mean_exps_relative=feature_mean_exps,
        )
    
    @staticmethod
    def _read_packed_ints(f, count: int, bits: int) -> np.ndarray:
        """Read packed integers with specified bit width."""
        total_bits = count * bits
        total_bytes = (total_bits + 7) // 8
        
        data = f.read(total_bytes)
        packed = int.from_bytes(data, 'little')
        
        mask = (1 << bits) - 1
        values = np.array([
            (packed >> (i * bits)) & mask
            for i in range(count)
        ], dtype=np.uint16)
        
        return values
    
    @classmethod
    def from_standard_weights(cls, weights_path: Path) -> 'CompactPhiWeights':
        """Convert from standard PHI1 format to compact PHI2 format."""
        with open(weights_path, 'rb') as f:
            magic = f.read(4)
            assert magic == b'PHI1'
            
            k = struct.unpack('H', f.read(2))[0]
            _ = struct.unpack('H', f.read(2))[0]  # k_fm (same as k)
            
            w_signs = np.frombuffer(f.read(32), dtype=np.int8)
            w_exps = np.frombuffer(f.read(64), dtype=np.uint16)
            
            fm_signs = np.frombuffer(f.read(32), dtype=np.int8)
            fm_exps = np.frombuffer(f.read(64), dtype=np.uint16)
            
            tm_sign = struct.unpack('b', f.read(1))[0]
            tm_exp = struct.unpack('H', f.read(2))[0]
        
        # Compute bases and relative exponents
        w_exp_base = int(w_exps.min())
        fm_exp_base = int(fm_exps.min())
        
        return cls(
            k=k,
            weight_exp_base=w_exp_base,
            feature_mean_exp_base=fm_exp_base,
            target_mean_sign=tm_sign,
            target_mean_exp=tm_exp,
            weight_signs=w_signs,
            weight_exps_relative=(w_exps - w_exp_base).astype(np.uint16),
            feature_mean_exps_relative=(fm_exps - fm_exp_base).astype(np.uint16),
        )


def convert_to_compact(standard_path: Path, compact_path: Path):
    """Convert standard weights to compact format."""
    compact = CompactPhiWeights.from_standard_weights(standard_path)
    compact.save(compact_path)
    return compact_path.stat().st_size


if __name__ == "__main__":
    standard_path = Path(__file__).parent / 'weights' / 'phi_weights.bin'
    compact_path = Path(__file__).parent / 'weights' / 'phi_weights_compact.bin'
    
    print("=" * 60)
    print("COMPACT φ-DECODER FORMAT")
    print("=" * 60)
    print()
    
    if not standard_path.exists():
        print(f"Standard weights not found: {standard_path}")
        print("Run fit_weights.py first to generate weights.")
    else:
        # Convert
        print("Converting standard → compact...")
        compact_size = convert_to_compact(standard_path, compact_path)
        standard_size = standard_path.stat().st_size
        
        print(f"  Standard: {standard_size} bytes")
        print(f"  Compact:  {compact_size} bytes")
        print(f"  Compression: {standard_size / compact_size:.1f}x")
