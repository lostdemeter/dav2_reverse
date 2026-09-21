"""
Fully-geometric DINOv2 ViT-S backbone for DAV2.

Ported from:
- BACKUP_misc/.../geometric_colorizer_v15_attention.py
  (GeometricAttentionLayer + GeometricDINOv2 forward order:
   norm1 -> QKV -> 6-head softmax -> proj -> layer_scale1 residual ->
   norm2 -> MLP(GELU) -> layer_scale2 residual, final layernorm)
- rlm_springboard/dav2_reverse_engineering/phi_transformer.py
  (PhiLinear phi-encode/store, LUT-decode for F.linear)

All weights stored as phi (sign, exponent) via geo_lut (K=512),
LUT-decoded once at load into float buffers. No `transformers`
import at inference — weights come from weights/geometric_backbone.npz
baked by export_geometric_weights.py.

Backbone spec (Depth-Anything-V2-Small-hf):
  hidden=384, layers=12, heads=6, patch=14, out stages 3/6/9/12.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from geo_lut import PHI, K, BIAS, N_LEVELS, phi_encode_numpy, get_lut

HIDDEN = 384
LAYERS = 12
HEADS = 6
HEAD_DIM = HIDDEN // HEADS
PATCH = 14
# 1-indexed stages used by DAV2 neck
OUT_STAGES = (3, 6, 9, 12)


def _decode(signs: np.ndarray, exps: np.ndarray) -> torch.Tensor:
    vals = (signs.astype(np.float32)
            * np.float32(PHI) ** ((exps.astype(np.float32) - BIAS) / K))
    return torch.from_numpy(vals)


class GeometricDinov2Backbone(torch.nn.Module):
    """ViT-S backbone with phi-decoded (baked) weights."""

    def __init__(self, weights_path: Path = None, device=None):
        super().__init__()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.buffers_loaded = False
        if weights_path is not None:
            self.load_geometric(weights_path)
        self.to(self.device)

    def load_geometric(self, path: Path):
        """Load baked phi weights (signs+exponents) and decode to buffers."""
        z = np.load(Path(path), allow_pickle=False)
        self._w = {}
        for key in z.files:
            if key.endswith('.signs'):
                base = key[:-len('.signs')]
                signs = z[key]
                exps = z[base + '.exps']
                self._w[base] = _decode(signs, exps).to(self.device)
            elif key.endswith('.shape'):
                continue
        # also load 1-D shapes / scalars stored directly
        for key in z.files:
            if not (key.endswith('.signs') or key.endswith('.exps') or key.endswith('.shape')):
                arr = z[key]
                if arr.dtype in (np.float32, np.float64):
                    self._w[key] = torch.from_numpy(arr.astype(np.float32)).to(self.device)
        self.buffers_loaded = True

    def _g(self, name: str) -> torch.Tensor:
        return self._w[name]

    def forward_stages(self, pixel_values: torch.Tensor, taps=None, bgains=None):
        """
        Args: pixel_values [B,3,H,W] float32.
              taps: 1-indexed layer numbers to tap (default OUT_STAGES).
              bgains: 24 phi-exponent ints (attn0,mlp0,attn1,mlp1,...) scaling
                block outputs before the layer-scale residual add. None = exact.
                Uniform weight scaling would be absorbed by LayerNorm; these
                output gains survive it by changing residual mixing ratios.
        Returns: list of feature maps [B,384,H/14,W/14] in tap order,
                 plus patch_h, patch_w.
        """
        assert self.buffers_loaded, "call load_geometric() first"
        want = tuple(taps) if taps is not None else OUT_STAGES
        assert all(1 <= t <= LAYERS for t in want), f"taps out of range: {want}"
        if bgains is not None:
            assert len(bgains) == 2 * LAYERS, f"bgains needs {2 * LAYERS} ints"
        B, _, H, W = pixel_values.shape
        x = pixel_values.to(self.device)
        ph, pw = H // PATCH, W // PATCH

        # Patch embed: Conv2d 3->384 k14 s14
        w = self._g('patch_proj.weight')  # [384,3,14,14]
        b = self._g('patch_proj.bias')
        x = F.conv2d(x, w, b, stride=PATCH)  # [B,384,ph,pw]
        x = x.flatten(2).transpose(1, 2)  # [B,N,384], N=ph*pw

        # CLS + pos embed (stored shapes may be squeezed; normalize to 3D)
        cls_t = self._g('cls_token')
        if cls_t.dim() == 1:
            cls_t = cls_t.view(1, 1, -1)
        elif cls_t.dim() == 2:
            cls_t = cls_t.unsqueeze(0) if cls_t.shape[0] == 1 else cls_t.view(1, 1, -1)
        cls = cls_t.expand(B, -1, -1)  # [B,1,384]
        x = torch.cat([cls, x], dim=1)  # [B,N+1,384]
        pos = self._g('pos_embed')
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)  # [1,1370,384]
        if pos.shape[1] != x.shape[1]:
            # bicubic interpolate patch part (matches v15 interpolate_pos_encoding)
            cls_pos = pos[:, :1, :]
            patch_pos = pos[:, 1:, :].reshape(1, 37, 37, HIDDEN).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(ph, pw), mode='bicubic', align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, HIDDEN)
            pos = torch.cat([cls_pos, patch_pos], dim=1)
        x = x + pos

        stages = {}
        for li in range(LAYERS):
            p = f'layer{li}.'
            D = HIDDEN
            # attention block
            h = F.layer_norm(x, (D,), self._g(p + 'norm1.weight'), self._g(p + 'norm1.bias'))
            q = F.linear(h, self._g(p + 'q.weight'), self._g(p + 'q.bias'))
            k_ = F.linear(h, self._g(p + 'k.weight'), self._g(p + 'k.bias'))
            v = F.linear(h, self._g(p + 'v.weight'), self._g(p + 'v.bias'))
            q = q.view(B, -1, HEADS, HEAD_DIM).transpose(1, 2)
            k_ = k_.view(B, -1, HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(B, -1, HEADS, HEAD_DIM).transpose(1, 2)
            attn = (q @ k_.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
            attn = attn.softmax(dim=-1)
            o = (attn @ v).transpose(1, 2).contiguous().view(B, -1, D)
            o = F.linear(o, self._g(p + 'proj.weight'), self._g(p + 'proj.bias'))
            if bgains is not None and bgains[2 * li]:
                o = o * float(PHI ** bgains[2 * li])
            ls1 = self._g(p + 'ls1')
            # layer_scale lambda is [384] — broadcast over B,N
            x = x + o * ls1
            # MLP block
            h2 = F.layer_norm(x, (D,), self._g(p + 'norm2.weight'), self._g(p + 'norm2.bias'))
            h2 = F.linear(h2, self._g(p + 'mlp1.weight'), self._g(p + 'mlp1.bias'))
            h2 = F.gelu(h2)
            h2 = F.linear(h2, self._g(p + 'mlp2.weight'), self._g(p + 'mlp2.bias'))
            if bgains is not None and bgains[2 * li + 1]:
                h2 = h2 * float(PHI ** bgains[2 * li + 1])
            ls2 = self._g(p + 'ls2')
            x = x + h2 * ls2
            if (li + 1) in want:
                stages[li + 1] = x

        # final layernorm (HF apply_layernorm=True)
        norm_w = self._w.get('final_norm.weight')
        norm_b = self._w.get('final_norm.bias')
        fmaps = []
        for s in want:
            h = stages[s]
            if norm_w is not None:
                h = F.layer_norm(h, (HIDDEN,), norm_w, norm_b)
            patch = h[:, 1:, :]  # drop CLS
            fmap = patch.reshape(B, ph, pw, HIDDEN).permute(0, 3, 1, 2).contiguous()
            fmaps.append(fmap)
        return fmaps, ph, pw


def export_from_hf(save_path: Path):
    """One-time bake: HF DINOv2 weights -> phi (signs,exps) npz. Needs transformers."""
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(
        'depth-anything/Depth-Anything-V2-Small-hf')
    bb = model.backbone
    out = {}

    def put(name, tensor: torch.Tensor):
        arr = tensor.detach().cpu().numpy().astype(np.float64)
        s, e = phi_encode_numpy(arr)
        out[name + '.signs'] = s
        out[name + '.exps'] = e
        out[name + '.shape'] = np.array(arr.shape, dtype=np.int64)

    emb = bb.embeddings
    put('patch_proj.weight', emb.patch_embeddings.projection.weight)
    put('patch_proj.bias', emb.patch_embeddings.projection.bias)
    put('cls_token', emb.cls_token)  # keep [1,1,384]
    put('pos_embed', emb.position_embeddings)  # keep [1,1370,384]
    for li, layer in enumerate(bb.encoder.layer):
        p = f'layer{li}.'
        put(p + 'norm1.weight', layer.norm1.weight)
        put(p + 'norm1.bias', layer.norm1.bias)
        put(p + 'q.weight', layer.attention.attention.query.weight)
        put(p + 'q.bias', layer.attention.attention.query.bias)
        put(p + 'k.weight', layer.attention.attention.key.weight)
        put(p + 'k.bias', layer.attention.attention.key.bias)
        put(p + 'v.weight', layer.attention.attention.value.weight)
        put(p + 'v.bias', layer.attention.attention.value.bias)
        put(p + 'proj.weight', layer.attention.output.dense.weight)
        put(p + 'proj.bias', layer.attention.output.dense.bias)
        put(p + 'ls1', layer.layer_scale1.lambda1)
        put(p + 'norm2.weight', layer.norm2.weight)
        put(p + 'norm2.bias', layer.norm2.bias)
        put(p + 'mlp1.weight', layer.mlp.fc1.weight)
        put(p + 'mlp1.bias', layer.mlp.fc1.bias)
        put(p + 'mlp2.weight', layer.mlp.fc2.weight)
        put(p + 'mlp2.bias', layer.mlp.fc2.bias)
        put(p + 'ls2', layer.layer_scale2.lambda1)
    put('final_norm.weight', bb.layernorm.weight)
    put('final_norm.bias', bb.layernorm.bias)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, **out)
    return save_path
