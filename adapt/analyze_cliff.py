#!/usr/bin/env python3
"""Localize cliff errors: where (in edge-distance) does seed depth fail?

For B-n12 stimulus + every E1T1L0V1 natural scene: seed depth vs HF
oracle, error map vs edge map + distance-to-edge bands. Decides the
rule shape (halo/refinement vs global bias) or kills the approach.
Writes adapt/runs/cliff_localize.json.
"""
import sys
from pathlib import Path

ADAPT = Path(__file__).parent
REPO = ADAPT.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(ADAPT))

import numpy as np


def main():
    import torch
    from scipy.ndimage import sobel, distance_transform_edt
    from run_search import load_shared
    from depth_adapter import run_pipeline, corr, SEED_CONFIG
    from harden_anchors import oracle_refs
    from probe_cliff import stim_edges
    from strata import profile
    import copy, json
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    shared = load_shared(device)
    med = json.load(open(ADAPT / 'runs' / 'strata.json'))['medians']

    def st_of(p):
        return ''.join(f"{s[0].upper()}{int(p[s] >= med[s])}" for s in
                       ('edge', 'texture', 'lumspread', 'vertical'))

    items = [('B-n12', stim_edges(12))]
    z = np.load(REPO / 'adapt' / 'fixtures' / 'scenes.npz', allow_pickle=False)
    for role in ('exploration', 'gate', 'retention'):
        r = z[f'{role}_rgb'].astype(np.float32) / 255.0
        for i in range(len(r)):
            if st_of(profile(r[i])) == 'E1T1L0V1':
                items.append((f'{role}-{i}', r[i]))
    ze = np.load(REPO / 'adapt' / 'fixtures' / 'strata_extra.npz', allow_pickle=False)
    re_ = ze['rgb'].astype(np.float32) / 255.0
    for i in range(len(re_)):
        if st_of(profile(re_[i])) == 'E1T1L0V1':
            items.append((f'extra-{i}', re_[i]))

    cfg = copy.deepcopy(dict(SEED_CONFIG))
    out = []
    for cid, rgb in items:
        ref = oracle_refs([rgb])[0].astype(np.float64)
        pred = run_pipeline(shared, cfg, [rgb])[0].astype(np.float64)
        if pred.shape != ref.shape:
            import cv2
            pred = cv2.resize(pred, (ref.shape[1], ref.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        err = np.abs(pred - ref)
        gray = rgb[..., :3].mean(axis=-1)
        emag = np.sqrt(sobel(gray, axis=0) ** 2 + sobel(gray, axis=1) ** 2)
        if emag.shape != ref.shape:
            import cv2
            emag = cv2.resize(emag, (ref.shape[1], ref.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        eth = np.percentile(emag, 90)
        edist = distance_transform_edt(emag < eth)
        bands, band_err = {}, {}
        for lo, hi, tag in ((0, 2, 'edge0-2'), (2, 5, 'edge2-5'), (5, 1e9, 'far5+')):
            m = (edist >= lo) & (edist < hi)
            bands[tag] = float(m.mean())
            band_err[tag] = float(err[m].mean()) if m.any() else 0.0
        c = float(np.corrcoef(pred.flatten(), ref.flatten())[0, 1])
        ce = float(np.corrcoef(err.flatten(), emag.flatten())[0, 1])
        row = {"scene": cid, "corr": round(c, 5), "corr_err_edge": round(ce, 3),
               "bands": bands, "band_err": {k: round(v, 5) for k, v in band_err.items()}}
        out.append(row)
        print(f"{cid:16s} corr={c:.5f} corr(err,edge)={ce:+.3f} "
              f"err@0-2px={band_err['edge0-2']:.5f} @2-5px={band_err['edge2-5']:.5f} "
              f"@far={band_err['far5+']:.5f}", flush=True)
    (ADAPT / 'runs' / 'cliff_localize.json').write_text(json.dumps(out, indent=1))
    print("wrote adapt/runs/cliff_localize.json")


if __name__ == '__main__':
    main()
