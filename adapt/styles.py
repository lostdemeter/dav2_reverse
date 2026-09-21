#!/usr/bin/env python3
"""Styles-as-algorithms: genotypes, fingerprints, and GA operators.

A STYLE is a DSL genotype (here: LUT-width configs) plus a behavior
fingerprint. The fingerprint ladder is deliberate (Echion's own rule):
  rung 1 (baseline): sha256 of the canonical behavior record.
  rung 2 (zeta): phases from Riemann-zero spacings over the same record,
    mirroring Echion gates.phase(): phase = (gamma * cell) mod 2pi.
Rung 2 earns its place only if it niches better than rung 1 — measured,
not assumed (see ga_search diversity report).

Genotype (width DSL): {"frac_cap", "exp_span", "dmax", "accum"} validated
by graduate.validate_widths. Crossover is uniform per-key; mutation
resamples one key from its domain (mirroring neighbor_widths moves).
"""
import hashlib
import json
import math
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))
sys.path.insert(0, str(ADAPT / 'third_party'))

# First Riemann zeros (public constants; cf. Echion docs/FINGERPRINT.md
# which measures gamma_4=30.425, gamma_6=37.586, gamma_12=56.446).
ZETAS = (14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178)
TWO_PI = 2 * math.pi

DOMAINS = {"frac_cap": (13312, 8192, 4096, 2048),
           "exp_span": (16, 8, 4),
           "dmax": (4096, 1024),
           "accum": ("tree", "fixed")}
KEYS = ("frac_cap", "exp_span", "dmax", "accum")


def random_genotype(rng):
    """Uniform random genotype. rng: random.Random (seeded by caller)."""
    import graduate
    cfg = {k: rng.choice(DOMAINS[k]) for k in KEYS}
    return graduate.validate_widths(cfg)


def behavior_record(n_correct, mean_corrs, nbytes):
    """Canonical behavior record: what the gates saw, nothing else."""
    return json.dumps({"correct": list(n_correct),
                       "means": [round(float(m), 6) for m in mean_corrs],
                       "bytes": int(nbytes)}, sort_keys=True)


def sha_fingerprint(record):
    """Rung 1: avalanche hash of the behavior record (null hypothesis)."""
    return hashlib.sha256(record.encode()).hexdigest()


def zeta_fingerprint(record, nzeros=6):
    """Rung 2: zeta-spaced phases of the behavior record.

    cell = int(sha(record), 16); phase_k = (gamma_k * cell) mod 2pi,
    quantized to 12 bits. Returns tuple of ints (the style signature).
    """
    cell = int(hashlib.sha256(record.encode()).hexdigest()[:12], 16)
    return tuple(int(((ZETAS[k] * cell) % TWO_PI) / TWO_PI * 4096)
                 for k in range(min(nzeros, len(ZETAS))))


def zeta_distance(a, b):
    """Mean circular distance between two zeta signatures (0..2048)."""
    ds = []
    for x, y in zip(a, b):
        d = abs(x - y) % 4096
        ds.append(min(d, 4096 - d))
    return sum(ds) / len(ds)


def crossover(rng, a, b):
    """Uniform per-key crossover. Returns new validated genotype."""
    import graduate
    cfg = {k: rng.choice((a[k], b[k])) for k in KEYS}
    return graduate.validate_widths(cfg)


def mutate(rng, cfg):
    """Resample one uniformly-chosen key from its domain."""
    import graduate
    k = rng.choice(KEYS)
    out = dict(cfg)
    out[k] = rng.choice(DOMAINS[k])
    return graduate.validate_widths(out)


def dominates(p, q):
    """p dominates q: >= correct AND <= bytes, strictly better in one."""
    return ((p[0] >= q[0] and p[1] <= q[1])
            and (p[0] > q[0] or p[1] < q[1]))


def pareto_front(scores):
    """Indices of the nondominated set. scores: list[(correct, bytes)]."""
    front = []
    for i, p in enumerate(scores):
        if not any(dominates(q, p) for j, q in enumerate(scores) if j != i):
            front.append(i)
    return front


# Level-2: fingerprints OF fronts (patterns of patterns). A front record
# bottoms out in measurements only: member config hashes + their measured
# (correct, bytes) + hypervolume. No vibes cross this line.
STYLE_GROUPS = (("frac_cap", "dmax"), ("exp_span", "accum"))


def hypervolume_2d(points, ref_correct=0, ref_bytes=None):
    """Hypervolume of (correct, bytes) points vs reference. Higher = better.

    Sort by correct desc; accumulate (c_i - c_next) * (ref_bytes - b_i)
    over points with bytes < ref_bytes. Deterministic integer/float mix
    rounded for record stability."""
    if ref_bytes is None:
        return 0.0
    pts = sorted(((c, b) for c, b in points if b < ref_bytes),
                 key=lambda t: -t[0])
    hv, prev_c = 0.0, ref_correct
    for c, b in pts:
        if c > prev_c:
            hv += (c - prev_c) * (ref_bytes - b)
            prev_c = c
    return hv


def front_record(items, ref_bytes):
    """Canonical record of a Pareto front. items: list[(cfg, correct, bytes)]."""
    members = sorted((config_fingerprint(cfg), c, b) for cfg, c, b in items)
    return json.dumps({"members": members,
                       "hypervolume": round(hypervolume_2d(
                           [(c, b) for _, c, b in members], ref_bytes=ref_bytes), 3)},
                      sort_keys=True)


def config_fingerprint(cfg):
    """Stable hash of a genotype (identity for front records)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def front_fingerprint(items, ref_bytes):
    """Level-2 signature: sha + zeta of the front record. The convergence
    signal — a stable front hash across generations means search settled."""
    rec = front_record(items, ref_bytes)
    return sha_fingerprint(rec), zeta_fingerprint(rec)


def style_crossover(rng, a, b):
    """Group-level crossover: whole parameter groups move together.

    G1={frac_cap,dmax} (add/sub-side sizes), G2={exp_span,accum}
    (bridge/softmax side). Child takes one group from each parent —
    co-adapted pairs travel together, unlike uniform per-key mixing."""
    import graduate
    take_a_first = rng.random() < 0.5
    cfg = {}
    for gi, group in enumerate(STYLE_GROUPS):
        src = a if (gi == 0) == take_a_first else b
        for k in group:
            cfg[k] = src[k]
    return graduate.validate_widths(cfg)
