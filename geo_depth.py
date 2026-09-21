"""
End-to-end fully-geometric Depth Anything V2 (Small).

Assembles:
  GeometricDinov2Backbone (geo_backbone) -> 4x [B,384,ph,pw]
  GeometricNeck (geo_neck)               -> 4x fused [B,64,...]
  GeometricHead (geo_head)               -> depth [B,H,W]

No `transformers` import at inference. Weights come from
weights/geometric_{backbone,neck,head}.npz baked once by
export_geometric_weights.py (which does need transformers + HF).

Preprocessing replicates DPTImageProcessor for 518x518 inputs:
  x = (img/255 - mean) / std, mean=(0.485,0.456,0.406),
  std=(0.229,0.224,0.225). Arbitrary sizes are resized keeping
  aspect ratio with longest side 518 and padded/cropped to a
  multiple of 14 (same ensure_multiple_of=14 rule).
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from geo_backbone import GeometricDinov2Backbone
from geo_neck import GeometricNeck
from geo_head import GeometricHead

BASE = Path(__file__).parent
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(rgb: np.ndarray, size: int = 518) -> torch.Tensor:
    """RGB float [0,1] HxWx3 -> pixel_values [1,3,H',W'] normalized."""
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    nh -= nh % 14
    nw -= nw % 14
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).resize((nw, nh), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


class GeometricDepthAnythingV2(torch.nn.Module):
    def __init__(self, weights_dir: Path = None, device=None):
        super().__init__()
        d = Path(weights_dir or BASE / 'weights')
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.backbone = GeometricDinov2Backbone(d / 'geometric_backbone.npz', device=self.device)
        self.neck = GeometricNeck(d / 'geometric_neck.npz', device=self.device)
        self.head = GeometricHead(d / 'geometric_head.npz', device=self.device)
        self.eval()

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values [B,3,H,W] -> depth [B,H,W]."""
        pv = pixel_values.to(self.device)
        B, _, H, W = pv.shape
        fmaps, ph, pw = self.backbone.forward_stages(pv)
        fused = self.neck(fmaps)
        depth = self.head(fused, ph, pw)
        return depth

    @torch.no_grad()
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """RGB float [0,1] -> depth HxW float32."""
        pv = preprocess(rgb)
        with torch.no_grad():
            d = self.forward(pv)
        return d.squeeze(0).cpu().numpy()
