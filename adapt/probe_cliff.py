#!/usr/bin/env python3
"""Active identification on cliff stratum E1T1L0V1 (high edge + texture).

Passive scenes entangle every statistic; parametric stimuli vary ONE
input statistic at a time so response curves attribute the failure.
Probe-validity guard: sweeps interpolate synthetic<->natural (gratings
over real-scene bases); smooth corr variation = valid probe, cliff
jumps = mechanism boundary (also reported, distinctly).

Families (168px, deterministic):
  A texture sweep: smooth base + sinusoidal grating, frequency x amplitude
  B edge sweep: gradient + N discs (edge count), fixed texture
  C cliff corner: high edge AND high texture jointly
Each stimulus: profile stats + HF oracle ref + seed-pipeline depth.
Writes adapt/runs/cliff_sweep.json. Next step reads the curves, not tea.
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np

SIZE = 168


def base_gradient(seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    g = yy / (SIZE - 1)
    tint = rng.uniform(0.85, 1.0, 3).astype(np.float32)
    return np.clip(g[..., None] * tint, 0, 1).astype(np.float32)


def stim_texture(freq, amp, seed=0):
    """Family A: grating texture over smooth base."""
    rgb = base_gradient(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    grating = np.sin(2 * np.pi * freq * xx / SIZE) * np.sin(2 * np.pi * freq * yy / SIZE)
    out = rgb + (amp * grating)[..., None] * np.array([1.0, 0.95, 0.9], np.float32)
    return np.clip(out, 0, 1).astype(np.float32)


def stim_edges(n_discs, radius=14, seed=0):
    """Family B: edge count via discs on gradient."""
    rng = np.random.default_rng(seed)
    rgb = base_gradient(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    for _ in range(n_discs):
        cy, cx = rng.uniform(0, SIZE, 2)
        disc = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius ** 2
        rgb[disc] = rng.uniform(0.05, 0.4, 3).astype(np.float32)
    return rgb


def stim_corner(freq, amp, n_discs, seed=0):
    """Family C: cliff corner — texture AND edges jointly."""
    rgb = stim_texture(freq, amp, seed=seed)
    rng = np.random.default_rng(seed + 999)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    for _ in range(n_discs):
        cy, cx = rng.uniform(0, SIZE, 2)
        disc = ((yy - cy) ** 2 + (xx - cx) ** 2) < 12 ** 2
        rgb[disc] = rng.uniform(0.05, 0.4, 3).astype(np.float32)
    return rgb


def main():
    import torch
    from run_search import load_shared
    from harden_anchors import oracle_refs
    from strata import profile
    import copy
    from depth_adapter import run_pipeline, corr, SEED_CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}", flush=True)
    shared = load_shared(device)

    stimuli = []
    for freq in (2, 4, 8, 12, 16, 24):
        for amp in (0.05, 0.15, 0.30):
            stimuli.append((f"A-f{freq}-a{amp}", stim_texture(freq, amp)))
    for n in (0, 1, 2, 4, 8, 12):
        stimuli.append((f"B-n{n}", stim_edges(n)))
    for freq, amp, n in ((8, 0.15, 2), (12, 0.15, 4), (16, 0.30, 4),
                         (12, 0.30, 8), (24, 0.30, 8)):
        stimuli.append((f"C-f{freq}-a{amp}-n{n}", stim_corner(freq, amp, n)))

    rgbs = [s for _, s in stimuli]
    refs = oracle_refs(rgbs)
    cfg = copy.deepcopy(dict(SEED_CONFIG))
    rows = []
    for (name, rgb), ref in zip(stimuli, refs):
        pred = run_pipeline(shared, cfg, [rgb])[0]
        c = corr(pred, ref)
        p = profile(rgb)
        rows.append({"stim": name, "corr": round(float(c), 5),
                     "edge": round(p["edge"], 4), "texture": round(p["texture"], 4),
                     "lumspread": round(p["lumspread"], 4)})
        print(f"{name:18s} corr={c:.5f} edge={p['edge']:.4f} "
              f"tex={p['texture']:.4f}", flush=True)
    out = ADAPT / 'runs' / 'cliff_sweep.json'
    out.write_text(__import__('json').dumps(rows, indent=1))
    print(f"wrote {out} ({len(rows)} stimuli)")


if __name__ == '__main__':
    main()
