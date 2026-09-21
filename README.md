# dav2_reverse: Fully-Geometric Depth Anything V2

A complete reverse-engineering of Depth Anything V2 (Small) into
φ-arithmetic: **backbone + neck + head**, every weight stored as
`sign × φ^((exponent − 32768) / 512)`, runnable on GPU, on CPU with no
GPU at all, and — via the integer datapath — on hardware with **no
floating-point unit**.

It started with a 125-byte decoder head. It ended with the whole model.

![HF vs geometric parity — input | HF depth | geometric depth | |error|, corr=0.9999](docs/geometric_parity.png)

*Same input through HF `Depth-Anything-V2-Small-hf` and our geometric
pipeline. Correlation 0.9999 — and the geometric side runs the full
model, not just the head.*

## What is this?

Depth Anything V2 is a monocular depth model: a DINOv2 ViT backbone
(22M params) feeds a DPT neck (2.7M, multiscale reassemble + fusion)
which feeds a small convolutional head (27K) that outputs depth. We
rebuilt all three stages geometrically:

1. **Phi-encoded replica** (`geo_*`) — every weight baked from the HF
   model into golden-ratio-lattice form. Multiplication becomes exponent
   addition; a shared LUT decodes. Bit-near-exact vs HF, no
   `transformers`, no download at inference.
2. **Integer datapath** (`geo_int.py`, `geo_jit.py`) — the same math with
   zero floating point at runtime: integer LUT-adds, fixed-point MACs,
   LUT softmax/GELU. Proven stage by stage against HF activations.
3. **Embedded artifact** (`c_port/`) — the integer core as portable C99
   with offline-generated tables and bit-exact host tests.

The original 125-byte head demo (`phi_depth.py`) is preserved below as
the lightweight path: stock HF backbone/neck with only the final `32→1`
layer replaced by 125 bytes of φ-weights.

## Results

Measured 2026-09-21 (CUDA fp32 unless noted). Correlation vs HF unless noted.

| Check | Result |
|-------|--------|
| Full pipeline, synthetic gradient | 0.999995 |
| Full pipeline, synthetic checker | 0.999998 |
| Full pipeline, demo scene | 0.999914 |
| Full pipeline, real webcam frame (shared input) | 0.999990 |
| Live webcam, fp16 GPU | ~95–100 FPS, ≈0.986–0.999 |
| Live webcam, CPU-only (`--cpu`, 364px) | ~15 FPS, no GPU |
| Integer transformer layer0 vs HF (1370 tokens) | 0.999996 (attn 0.999999, MLP 0.999999) |
| Integer neck convs, chained (1×1 → 3×3 → 3×3+ReLU) | 0.999920 / 0.999877 / 0.999663 |
| Integer deconv / softmax / bridge roundtrip | 0.999970 / 1.000000 / 1.000000 |
| Numba hot loops (bit-exact) | head 368×, re-encode 130×, interp up to 120× |
| C99 core host test | bit-exact 0/64, ALL PASS |

## Quick start

```bash
git clone git@github.com:lostdemeter/dav2_reverse.git && cd dav2_reverse
python3 -m venv venv && source venv/bin/activate   # recommended (PEP 668 systems require it)
pip install -r requirements.txt
python export_geometric_weights.py   # one-time: HF download ~94MB, bakes phi weights locally
python demo_geometric.py             # HF -> geometric figure, expect corr > 0.999 (no webcam needed)
python test_geometric_parity.py      # gradient + checker parity suite
python geo_webcam.py                 # live fully-geometric webcam (below)
python geo_int.py                    # integer-datapath parity suite (takes a few minutes, CPU)
cd c_port && make test               # C core host test
```

The baked weights (`weights/geometric_*.npz`, ~47MB) are **not** in git —
each user builds them with the export script. Nothing large is ever pushed.

Webcam controls (`phi_depth.py` and `geo_webcam.py`):

| Key | Action |
|-----|--------|
| M | Cycle colormap (magma → viridis → plasma → inferno → turbo) |
| S | Save frame to `./captures/` (gitignored, never uploaded) |
| X | Quit |

```bash
python geo_webcam.py --cpu              # CPU-only, ~15 FPS
python geo_webcam.py --cpu --size 518   # full res, slower
python geo_webcam.py --frames 5 --compare-hf   # headless check + HF parity
```

## How it works

A φ-value is `sign × φ^(exponent / 512)` with φ = (1+√5)/2. On this
lattice, **multiplication is integer exponent addition** and addition is
an integer LUT over the exponent difference
(`φ^a + φ^b = φ^(max + LUT[max−min])`). Weights cluster tightly on the
lattice (peak near `φ^−9`), so the encoding is near-lossless and the
runtime needs no FPU — only add/sub/compare/shift/XOR plus table
gathers, with all tables baked offline.

### Backbone — DINOv2 ViT-S (`geo_backbone.py`)

12 layers, hidden 384, 6 heads, patch 14, out stages 3/6/9/12 (22M
params). Pre-norm → QKV → 6-head softmax → proj → layer-scale residual
→ MLP/GELU → layer-scale residual, final layernorm. Patch embed, QKV,
proj, MLP, norms, layer-scales, CLS token and position embeddings are
all baked as phi (sign, exponent) pairs and LUT-decoded once at load.
Bicubic position-embedding interpolation handles non-518 inputs.

### Neck — DPT reassemble + fusion (`geo_neck.py`)

4× reassemble (1×1 proj 384→48/96/192/384, then ×4 deconv / ×2 deconv /
identity / stride-2 conv), 4× 3×3 convs to 64ch, then 4 fusion stages in
reverse order (small→large), each a 1×1 proj plus 2× pre-act residual
blocks with bilinear ×2 upsample (2.7M params). All conv weights are
phi-baked; interpolation stays analytic. An optional `fusion_exponents`
argument does φ-weighted multiscale fusion (`φ^e_i / Σφ^e_j`); default
is uniform, i.e. exact replication.

### Head — depth decoder (`geo_head.py`)

Conv 64→32 3×3, bilinear upsample to full res, conv 32→32 3×3, ReLU,
conv 32→1 1×1, ReLU×max_depth (27K params, all phi-baked). Two extra
modes: `fast_predict_125b()` reuses the 125-byte compact weights for
webcam speed, and `AnalyticHead` (edge/texture/color/perspective cues
with `φ^0,−1,−2,−3` weights, zero learned parameters) shows the
no-learning limit.

### Integer datapath (`geo_int.py`, `geo_jit.py`, `c_port/`)

Two runtimes, same math. The **log-domain** path accumulates with
integer LUT-adds and tree reduction (log₂(T) rounding levels instead of
T). The **fixed-point bridge** converts groups to int64 once
(`to_fixed`), accumulates exactly, and re-encodes once (`from_fixed`,
15-bit mantissa, offline FRAC/COARSE/FINE LUTs) — this carries
resampling, deconvs, softmax (which stays fixed-point through attn@V),
LayerNorm (int mean/var/`isqrt`/division) and GELU (bounded LUT). The
full integer transformer layer matches HF layer0 at 0.999996.
`geo_jit.py` compiles the hot loops with Numba (bit-exact, 100×+).
`c_port/` is the firmware-shaped artifact: portable C99, zero floats in
the core (verified by grep), offline LUT/vector generators, `make test`
with bit-exact host checks, and an MCU trimming guide.

Honest boundaries: float→int encoding at the sensor/embeddings input
and final decode-for-display still use floats — on FPU-free hardware the
front end emits fixed-point ints with an integer encode LUT. Overlapping
deconv/stride-conv on the fixed canvas reuses the same machinery and is
not yet wired. Thresholds in the test suites are set where each path's
quantization floor actually is (documented inline), all far above
typical int8 practice (~0.99).

### The 125-byte head (`phi_decoder.py`, `phi_compact.py`)

The project origin: DA2's final layer is a linear `32→1` projection, and
the whole projection fits in 125 bytes of φ-quantized weights
(`PHI2` format: relative exponents, packed signs) at 99.99% correlation.
`phi_depth.py` runs this against the stock HF backbone at 33+ FPS;
`fit_weights.py` re-derives the weights from any image folder (optional
— pre-fitted weights are committed).

## Repository structure

```
dav2_reverse/
├── README.md              # This file
├── LICENSE                # GPLv3
├── requirements.txt       # numpy, opencv, torch, transformers, Pillow, scipy, numba
├── geo_lut.py             # Shared φ-LUT (sign×φ^(exp/K) encode/decode)
├── geo_backbone.py        # Geometric DINOv2 ViT-S + export_from_hf()
├── geo_neck.py            # Geometric DPT reassemble+fusion + export_from_hf()
├── geo_head.py            # Geometric head + AnalyticHead + export_from_hf()
├── geo_depth.py           # End-to-end geometric DAV2 (no transformers)
├── geo_webcam.py          # Live fully-geometric webcam (GPU/CPU/headless)
├── geo_int.py             # Integer-only datapath + parity suites
├── geo_jit.py             # Numba-JIT integer hot loops (bit-exact)
├── c_port/                # Portable C99 integer core + LUT gens + host test
│   ├── phi_int.h / phi_int.c   # zero-float core
│   ├── gen_luts.py / gen_vectors.py  # offline generators (floats stay here)
│   ├── test_phi_int.c / Makefile     # `make test` → bit-exact 0/64, ALL PASS
│   └── generated/              # GITIGNORED build outputs
├── export_geometric_weights.py  # One-time HF->phi bake (builds gitignored npz)
├── test_geometric_parity.py     # Float pipeline parity (corr > 0.999)
├── demo_geometric.py            # HF->geometric demo figure (no webcam)
├── phi_depth.py           # Head-only realtime demo (HF backbone + 125-byte head)
├── phi_decoder.py         # φ-arithmetic decoder core
├── phi_compact.py         # Compact 125-byte storage format
├── fit_weights.py         # Optional: re-fit 125-byte weights from images
├── weights/
│   ├── phi_weights.bin / phi_weights_compact.bin  # 203/125 bytes, committed
│   └── geometric_*.npz    # ~47MB baked phi weights, GITIGNORED, build locally
└── docs/
    ├── demo.png               # Head-only demo screenshot
    └── geometric_parity.png   # HF vs geometric parity figure
```

`captures/` (webcam snapshots), `c_port/generated/`, and
`weights/geometric_*.npz` are gitignored and never uploaded.

## Requirements

- Python 3.8+, USB webcam for the live demos
- GPU recommended (`~95–100 FPS` fp16); CPU works (`~15 FPS` at 364px)
- gcc + standard C library for `c_port` (host test only)

## License

GPLv3 — see [LICENSE](LICENSE).

## Credits

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for the original model
- Part of the [TruthSpace Geometric LCM](https://github.com/lostdemeter/truthspace-lcm) research project
