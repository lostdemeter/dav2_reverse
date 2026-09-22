#!/usr/bin/env python3
"""Resonant bloom filter for genotype identifiers (stdlib only).

Mirrors Echion's echion/generation/bloom.py addressing math exactly —
bytearray + zeta-zero hash family (phase = (gamma * key) mod 2pi), O(k)
check, honest false-positive behavior — keyed here on config identifier
hex digests instead of text. Same lineage, new domain.

Role in the search loop (this turn): OBSERVER only. It tracks visited
space in constant memory beside the exact cache, reporting coverage
estimates and measured false positives, with zero false negatives
asserted (every cache hit must test positive). Selection still uses the
exact cache; the bloom proves itself as an instrument before it is ever
trusted with decisions. Cross-run persistence (save/load) is implemented
and tested for the day visited sets outgrow memory.
"""
import hashlib
import math

# First Riemann zeros (public constants; shared with Echion's bloom and
# our styles.py zeta fingerprints — one family, three uses).
ZEROS = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
]
TWO_PI = 2 * math.pi


def _stable_key(token: str) -> int:
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


class ResonantBloom:
    def __init__(self, n_hashes: int = 5, n_bits: int = 2**16):
        if n_hashes < 1 or n_bits < 8:
            raise ValueError("Need >=1 hashes and >=8 bits")
        if n_hashes > len(ZEROS):
            raise ValueError(f"Only {len(ZEROS)} zeta zeros bundled; got {n_hashes}")
        self.k = n_hashes
        self.m = n_bits
        self.bits = bytearray((n_bits + 7) // 8)
        self.count = 0

    def _positions(self, token: str):
        key = _stable_key(token)
        for g in ZEROS[:self.k]:
            yield int(((g * key) % TWO_PI) / TWO_PI * self.m) % self.m

    def add(self, token: str):
        for bit in self._positions(token):
            self.bits[bit // 8] |= 1 << (bit % 8)
        self.count += 1

    def check(self, token: str) -> bool:
        return all(self.bits[bit // 8] & (1 << (bit % 8))
                   for bit in self._positions(token))

    def estimated_count(self) -> float:
        """Cardinality estimate from fill fraction (standard formula)."""
        ones = sum(bin(b).count("1") for b in self.bits)
        frac = ones / self.m
        if frac >= 1.0:
            return float("inf")
        return -self.m / self.k * math.log(1.0 - frac)

    def expected_fp(self, n: int) -> float:
        """Theoretical FP rate after n inserts (for the report, not gating)."""
        return (1.0 - math.exp(-self.k * n / self.m)) ** self.k

    def save(self, path):
        import json
        with open(path, "wb") as f:
            f.write(self.bits)
        with open(str(path) + ".meta.json", "w") as f:
            json.dump({"k": self.k, "m": self.m, "count": self.count}, f)

    @classmethod
    def load(cls, path):
        import json
        meta = json.load(open(str(path) + ".meta.json"))
        obj = cls(n_hashes=meta["k"], n_bits=meta["m"])
        with open(path, "rb") as f:
            obj.bits = bytearray(f.read())
        obj.count = meta["count"]
        return obj
