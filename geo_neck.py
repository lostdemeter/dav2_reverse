"""
Fully-geometric DPT neck for DAV2 (reassemble + fusion).

Replicates HF DepthAnythingNeck exactly, with phi-encoded weights:
- reassemble 4x: 1x1 proj 384->C, resize (deconv x4 / deconv x2 /
  identity / stride-2 conv), C in [48,96,192,384]
- convs 4x: 3x3 C->64 pad1 (no bias in HF)
- fusion 4x: 1x1 proj 64->64 + 2x pre-act residual (ReLU,3x3,ReLU,3x3),
  bilinear x2 upsample, reversed-order fusion (small -> large)

Plus phi-weighted multiscale fusion factors (from
BACKUP_misc/.../da2_multiscale_phi.py `fusion_exponents`):
  fused_pred = sum(phi**e_i * layer_pred_i) / sum(...)
Default exponents are zeros (= uniform) so the module is an exact
replication; tune later for geometric emphasis.

Export via export_from_hf(); inference needs only torch+numpy.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from geo_lut import PHI, K, BIAS, phi_encode_numpy

NECK_DIMS = (48, 96, 192, 384)


def _decode(signs: np.ndarray, exps: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(
        (signs.astype(np.float32)
         * np.float32(PHI) ** ((exps.astype(np.float32) - BIAS) / K)))


class GeometricNeck(torch.nn.Module):
    def __init__(self, weights_path: Path = None, device=None,
                 fusion_exponents=None):
        super().__init__()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.buffers_loaded = False
        # phi fusion weights (da2_multiscale_phi style); None -> exact replication
        if fusion_exponents is None:
            self.fusion_w = None
        else:
            e = torch.tensor(list(fusion_exponents), dtype=torch.float32)
            w = torch.tensor([PHI ** float(x) for x in e])
            self.fusion_w = (w / w.sum()).to(self.device)
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
            elif key.endswith(('.shape', '.exps')):
                continue
        for key in z.files:
            if not (key.endswith('.signs') or key.endswith('.exps') or key.endswith('.shape')):
                arr = z[key]
                if arr.dtype in (np.float32, np.float64):
                    self._w[key] = torch.from_numpy(arr.astype(np.float32)).to(self.device)
        self.buffers_loaded = True

    def _g(self, name):
        return self._w[name]

    def _residual(self, x, prefix):
        r = x
        x = F.relu(x)
        x = F.conv2d(x, self._g(prefix + '.conv1.weight'), self._g(prefix + '.conv1.bias'), padding=1)
        x = F.relu(x)
        x = F.conv2d(x, self._g(prefix + '.conv2.weight'), self._g(prefix + '.conv2.bias'), padding=1)
        return x + r

    def _fusion_layer(self, hidden, residual, size, li):
        p = f'fusion{li}.'
        if residual is not None:
            if hidden.shape != residual.shape:
                residual = F.interpolate(residual, size=(hidden.shape[2], hidden.shape[3]),
                                         mode='bilinear', align_corners=False)
            hidden = hidden + self._residual(residual, p + 'res1')
        hidden = self._residual(hidden, p + 'res2')
        if size is None:
            hidden = F.interpolate(hidden, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            hidden = F.interpolate(hidden, size=size, mode='bilinear', align_corners=True)
        hidden = F.conv2d(hidden, self._g(p + 'proj.weight'), self._g(p + 'proj.bias'))
        return hidden

    def forward(self, feature_maps):
        """
        feature_maps: list of 4 [B,384,ph,pw] from GeometricDinov2Backbone.
        Returns: list of 4 fused [B,64,...] (last = highest resolution).
        """
        assert self.buffers_loaded
        feats = []
        for i, fm in enumerate(feature_maps):
            C = NECK_DIMS[i]
            x = F.conv2d(fm, self._g(f're{i}.proj.weight'), self._g(f're{i}.proj.bias'))
            if i == 0:
                x = F.conv_transpose2d(x, self._g('re0.deconv.weight'), self._g('re0.deconv.bias'), stride=4)
            elif i == 1:
                x = F.conv_transpose2d(x, self._g('re1.deconv.weight'), self._g('re1.deconv.bias'), stride=2)
            elif i == 3:
                x = F.conv2d(x, self._g('re3.down.weight'), self._g('re3.down.bias'),
                             stride=2, padding=1)
            # i==2 identity
            x = F.conv2d(x, self._g(f'cv{i}.weight'), None, padding=1)
            feats.append(x)

        rev = feats[::-1]
        fused_list = []
        fused = None
        for idx in range(4):
            hs = rev[idx]
            if fused is None:
                size = rev[idx + 1].shape[2:] if idx < 3 else None
                fused = self._fusion_layer(hs, None, size, idx)
            else:
                size = rev[idx + 1].shape[2:] if idx < 3 else None
                fused = self._fusion_layer(fused, hs, size, idx)
            fused_list.append(fused)
        return fused_list


def export_from_hf(save_path: Path):
    """One-time bake: HF neck weights -> phi npz. Needs transformers."""
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf')
    neck = model.neck
    out = {}

    def put(name, tensor):
        if tensor is None:
            return
        arr = tensor.detach().cpu().numpy().astype(np.float64)
        s, e = phi_encode_numpy(arr)
        out[name + '.signs'] = s
        out[name + '.exps'] = e
        out[name + '.shape'] = np.array(arr.shape, dtype=np.int64)

    for i, layer in enumerate(neck.reassemble_stage.layers):
        put(f're{i}.proj.weight', layer.projection.weight)
        put(f're{i}.proj.bias', layer.projection.bias)
        if i in (0, 1):
            put(f're{i}.deconv.weight', layer.resize.weight)
            put(f're{i}.deconv.bias', layer.resize.bias)
        elif i == 3:
            put('re3.down.weight', layer.resize.weight)
            put('re3.down.bias', layer.resize.bias)
    for i, conv in enumerate(neck.convs):
        put(f'cv{i}.weight', conv.weight)
    for i, layer in enumerate(neck.fusion_stage.layers):
        put(f'fusion{i}.proj.weight', layer.projection.weight)
        put(f'fusion{i}.proj.bias', layer.projection.bias)
        for j, rl in enumerate((layer.residual_layer1, layer.residual_layer2), start=1):
            put(f'fusion{i}.res{j}.conv1.weight', rl.convolution1.weight)
            put(f'fusion{i}.res{j}.conv1.bias', rl.convolution1.bias)
            put(f'fusion{i}.res{j}.conv2.weight', rl.convolution2.weight)
            put(f'fusion{i}.res{j}.conv2.bias', rl.convolution2.bias)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **out)
    return save_path
