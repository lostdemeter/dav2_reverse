"""
One-time export: bake HF DAV2-Small weights -> phi-encoded npz.

Needs: transformers, torch, numpy, internet (HF download ~94MB).
Writes (local only, gitignored unless forced):
  weights/geometric_backbone.npz  (~22M params phi-encoded)
  weights/geometric_neck.npz      (~2.7M params)
  weights/geometric_head.npz      (~27K params)

Run once:
    python export_geometric_weights.py
Then inference (geo_depth.py) needs no transformers/HF.
"""

from pathlib import Path

from geo_backbone import export_from_hf as export_bb
from geo_neck import export_from_hf as export_neck
from geo_head import export_from_hf as export_head

BASE = Path(__file__).parent / 'weights'


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    print("Exporting backbone (22M params, may take a few minutes)...")
    p1 = export_bb(BASE / 'geometric_backbone.npz')
    print("  saved", p1, f"{p1.stat().st_size / 1e6:.1f} MB")
    print("Exporting neck...")
    p2 = export_neck(BASE / 'geometric_neck.npz')
    print("  saved", p2, f"{p2.stat().st_size / 1e6:.1f} MB")
    print("Exporting head...")
    p3 = export_head(BASE / 'geometric_head.npz')
    print("  saved", p3, f"{p3.stat().st_size / 1e3:.1f} KB")
    print("Done. Inference now runs without HuggingFace.")


if __name__ == '__main__':
    main()
