#!/usr/bin/env python3
"""Analyze a training trace: is there a sequential component?

Reads adapt/runs/trace.jsonl (STUDENT_TRACE=1 run). Reports:
(a) first-pass epoch per scene (first gate with corr>=0.999), ordered
    by scene and grouped by stratum — do strata flip in order?
(b) per-layer drift trajectories — do layers stabilize in depth
    order (L0 first)? Reports per-layer drift at init/early/late
    plus the epoch each layer reaches 90% of its final drift.
(c) per-map drift ranking at end — qkv vs proj vs mlp order?
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

ADAPT = Path(__file__).parent
BAR = 0.999


def ep_of(tag):
    return -1 if tag == "init" else int(tag[2:])


def main():
    rows = [json.loads(l) for l in
            open(ADAPT / 'runs' / 'trace.jsonl')]
    rows.sort(key=lambda r: ep_of(r["tag"]))
    print(f"{len(rows)} gate rows: {[r['tag'] for r in rows]}")

    # (a) first-pass epoch per scene
    firstpass = {}
    for r in rows:
        ep = ep_of(r["tag"])
        for cid, st, c in r["scenes"]:
            if c >= BAR and cid not in firstpass:
                firstpass[cid] = (ep, st)
    ever = {cid for cid, _, _ in rows[0]["scenes"]}
    never = sorted(ever - set(firstpass))
    print(f"\nfirst-pass: {len(firstpass)}/{len(ever)} scenes ever pass")
    by_ep = defaultdict(list)
    for cid, (ep, st) in firstpass.items():
        by_ep[ep].append((cid, st))
    for ep in sorted(by_ep):
        sts = defaultdict(int)
        for _, st in by_ep[ep]:
            sts[st] += 1
        print(f"  ep{ep}: {len(by_ep[ep])} scenes "
              + str(dict(sts)))
    print(f"  never pass ({len(never)}): {never}")
    # stratum ordering: median first-pass ep per stratum
    st_eps = defaultdict(list)
    for cid, (ep, st) in firstpass.items():
        st_eps[st].append(ep)
    print("  median first-pass ep per stratum:")
    for st in sorted(st_eps):
        v = sorted(st_eps[st])
        print(f"    {st}: median={v[len(v)//2]} n={len(v)}")

    # (b) layer drift trajectories
    print("\nper-layer drift (init -> mid -> final):")
    layers = sorted(rows[0]["drift_layer"], key=int)
    finals = {li: rows[-1]["drift_layer"][li] for li in layers}
    mid = rows[len(rows)//2]
    for li in layers:
        print(f"  L{li}: init=0 mid({mid['tag']})="
              f"{mid['drift_layer'][li]:.4f} final={finals[li]:.4f}")
    print("  epoch each layer reaches 90% of final drift:")
    for li in layers:
        tgt = 0.9 * finals[li]
        hit = next((r["tag"] for r in rows
                    if r["drift_layer"][li] >= tgt), "never")
        print(f"    L{li}: {hit}")

    # (c) per-map final drift ranking
    print("\nper-map final drift (high = moved most):")
    fm = rows[-1]["drift_map"]
    for k in sorted(fm, key=lambda k: -fm[k]):
        print(f"  {k}: {fm[k]:.4f}")


if __name__ == '__main__':
    main()
