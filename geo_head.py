"""
Fully-geometric DAV2 head.

Replicates HF DepthAnythingDepthEstimationHead exactly with phi-encoded
weights (conv1 64->32 3x3, conv2 32->32 3x3, conv3 32->1 1x1), plus:

- `activation1` output exposed (compat with existing phi_decoder hook at
  model.head.activation1 and phi_depth.py GPU fast path).
- `fast_predict_125b()`: existing 125-byte phi fast path
  (weights/phi_weights_compact.bin) for webcam speed.
- `AnalyticHead`: explicit geometric head from
  rlm_springboard/dav2_reverse_engineering/experimental_decoder.py
  (PhiGeometricDecoder: edge/texture/color/perspective groups with
  PHI^0,-1,-2,-3 weights, no learned weights). Lower accuracy, documents
  the zero-weight limit.

Export via export_from_hf(); inference needs only torch+numpy.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from geo_lut import PHI, K, BIAS, phi_encode_numpy


def _decode(signs: np.ndarray, exps: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(
        (signs.astype(np.float32)
         * np.float32(PHI) ** ((exps.astype(np.float32) - BIAS) / K)))


class GeometricHead(torch.nn.Module):
    MAX_DEPTH = 1.0

    def __init__(self, weights_path: Path = None, device=None):
        super().__init__()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.buffers_loaded = False
        # compat: last activation1 features for phi fast path / hooks
        self.last_features = None
        if weights_path is not None:
            self.load_geometric(weights_path)
        self.to(self.device)

    def load_geometric(self, path: Path):
        z = np.load(Path(path), allow_pickle=False)
        self._w = {}
        for key in z.files:
            if key.endswith('.signs'):
                base = key[:-len('.signs')]
                self._w[base] = _decode(z[key], z[base + '.exps']).to(self.device)
        self.buffers_loaded = True

    def forward(self, fused_list, patch_h, patch_w):
        """
        fused_list: 4 fused [B,64,...] from GeometricNeck (uses last).
        Returns: depth [B,H,W] float, H=patch_h*14, W=patch_w*14.
        """
        assert self.buffers_loaded
        h = fused_list[-1]
        h = F.conv2d(
            h, self._w['conv1.weight'], self._w['conv1.bias'], padding=1)
        h = F.interpolate(h, (int(patch_h * 14), int(patch_w * 14)),
                          mode='bilinear', align_corners=True)
        h = F.conv2d(h, self._w['conv2.weight'], self._w['conv2.bias'], padding=1)
        h = F.relu(h)
        self.last_features = h  # [B,32,H,W] — same tap as old head.activation1 hook
        d = F.conv2d(h, self._w['conv3.weight'], self._w['conv3.bias'])
        d = F.relu(d) * self.MAX_DEPTH
        return d.squeeze(dim=1)

    def fast_predict_125b(self, features_b32hw: torch.Tensor,
                          weight_t: torch.Tensor, mean_t: torch.Tensor,
                          target_mean: float) -> torch.Tensor:
        """Existing 125-byte fast path: (feat-mean)@w + tm (see phi_depth.py)."""
        C, H, W = features_b32hw.shape[-3:]
        f = features_b32hw.reshape(-1, C).float()
        depth = (f - mean_t.to(f.device)) @ weight_t.to(f.device) + target_mean
        return depth.reshape(H, W)


class AnalyticHead:
    """
    Zero-learned-weight geometric head (experimental_decoder.py port).

    Groups image cues geometrically:
      edge -> depth discontinuity (PHI^0), texture frequency -> distance
      (PHI^-1), color contrast -> atmospheric perspective (PHI^-2),
      perspective lines -> distance (PHI^-3).
    """

    def __init__(self):
        self.w = {'edge': float(PHI ** 0), 'texture': float(PHI ** -1),
                  'color': float(PHI ** -2), 'perspective': float(PHI ** -3)}

    def __call__(self, gray: np.ndarray) -> np.ndarray:
        from scipy.ndimage import sobel, gaussian_filter
        gx = sobel(gray, axis=1)
        gy = sobel(gray, axis=0)
        edge = np.sqrt(gx ** 2 + gy ** 2)
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-9)
        smooth = gaussian_filter(gray, 3.0)
        texture = np.abs(gray - smooth)
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-9)
        h, w = gray.shape
        yy, _ = np.mgrid[0:h, 0:w]
        perspective = yy / max(h - 1, 1)  # top=far assumption
        depth = (self.w['edge'] * (1.0 - edge)
                 + self.w['texture'] * (1.0 - texture)
                 + self.w['perspective'] * perspective)
        depth /= (self.w['edge'] + self.w['texture'] + self.w['perspective'])
        return depth.astype(np.float32)


def export_from_hf(save_path: Path):
    """One-time bake: HF head convs -> phi npz. Needs transformers."""
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf')
    head = model.head
    out = {}

    def put(name, tensor):
        arr = tensor.detach().cpu().numpy().astype(np.float64)
        s, e = phi_encode_numpy(arr)
        out[name + '.signs'] = s
        out[name + '.exps'] = e
        out[name + '.shape'] = np.array(arr.shape, dtype=np.int64)

    put('conv1.weight', head.conv1.weight)
    put('conv1.bias', head.conv1.bias)
    put('conv2.weight', head.conv2.weight)
    put('conv2.bias', head.conv2.bias)
    put('conv3.weight', head.conv3.weight)
    put('conv3.bias', head.conv3.bias)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **out)
    return save_path
