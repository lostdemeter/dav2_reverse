"""
Live webcam test for the FULLY-GEOMETRIC DAV2 pipeline.

Runs every frame through:
  GeometricDinov2Backbone -> GeometricNeck -> GeometricHead
(phi-encoded weights + LUT, baked in weights/geometric_*.npz —
 no `transformers` / HF download at inference).

Learned integer-head mode (--int-head) swaps the final float conv3 for
the graduated integer head (adapt/runs/graduation.json widths): backbone
+neck+head-convs stay float to produce 32ch features, then
IntegerPhiHead (tree LUT-adds, JIT) predicts depth with zero FPU in the
accumulation. This is the adaptation-foundry learned artifact running live.

Usage:
    python geo_webcam.py                  # interactive GUI (M/S/X like phi_depth.py)
    python geo_webcam.py --frames 5       # headless: capture 5 frames, save to captures/
    python geo_webcam.py --frames 5 --compare-hf   # also run HF baseline for parity
    python geo_webcam.py --int-head --frames 5     # learned integer head live test

Controls (interactive): [M] colormap  [S] save  [X] quit
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from geo_depth import GeometricDepthAnythingV2

COLORMAPS = [
    (cv2.COLORMAP_MAGMA, 'magma'),
    (cv2.COLORMAP_VIRIDIS, 'viridis'),
    (cv2.COLORMAP_PLASMA, 'plasma'),
    (cv2.COLORMAP_INFERNO, 'inferno'),
    (cv2.COLORMAP_TURBO, 'turbo'),
]


def colorize(depth: np.ndarray, cmap) -> np.ndarray:
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax > dmin:
        norm = ((depth - dmin) / (dmax - dmin) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(depth, dtype=np.uint8)
    return cv2.applyColorMap(norm, cmap)


def main():
    ap = argparse.ArgumentParser(description='Fully-geometric DAV2 webcam test')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--frames', type=int, default=0,
                    help='0 = interactive GUI, N = headless capture N frames')
    ap.add_argument('--compare-hf', action='store_true',
                    help='also run HF baseline on captured frames for parity')
    ap.add_argument('--no-fp16', action='store_true')
    ap.add_argument('--cpu', action='store_true',
                    help='force CPU-only (no GPU needed, slower, smaller default size)')
    ap.add_argument('--size', type=int, default=None,
                    help='input long side (default 518, use 364/336 on CPU for speed)')
    ap.add_argument('--int-head', action='store_true',
                    help='use learned integer head (graduated widths) instead of float conv3')
    ap.add_argument('--c-head', action='store_true',
                    help='run the integer head in compiled C (implies --int-head); firmware-shaped path')
    ap.add_argument('--widths-json', type=str, default='adapt/runs/graduation.json',
                    help='graduation incumbent JSON (default: adapt/runs/graduation.json)')
    args = ap.parse_args()
    if args.c_head:
        args.int_head = True

    if args.size is None:
        args.size = 364 if args.cpu else 518
    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    use_fp16 = (not args.no_fp16) and device.type == 'cuda'
    print(f"device: {device}  fp16: {use_fp16}")

    print("loading fully-geometric pipeline (backbone+neck+head, no HF)...")
    geo = GeometricDepthAnythingV2(device=device)
    if use_fp16:
        for mod in (geo.backbone, geo.neck, geo.head):
            for k, v in mod._w.items():
                mod._w[k] = v.half()
    geo.eval()
    print("ready.")

    int_head = None
    int_cfg = None
    cheat = None  # ctypes C-head binding when --c-head
    if args.int_head:
        import json as _json
        import geo_int as _G
        from geo_int import IntegerPhiHead
        wpath = Path(args.widths_json)
        if wpath.exists():
            int_cfg = _json.load(open(wpath))['incumbent']['config']
            print(f"learned widths: {int_cfg}")
        else:
            int_cfg = {"frac_cap": 2048, "exp_span": 8, "dmax": 4096, "accum": "tree"}
            print(f"no {wpath}, using fallback widths {int_cfg}")
        if int_cfg.get("accum", "tree") != "tree":
            print("warning: live int-head supports tree accum only "
                  f"(graduated={int_cfg.get('accum')}); using tree tables")
        # Patch global DMAX used by phi_add clipping + rebuild head LUTs.
        # frac_cap/exp_span are fixed-bridge widths (no-op for tree) — logged
        # for provenance; the byte savings they represent ship in the tables.
        _G.DMAX = int(int_cfg["dmax"])
        int_head = IntegerPhiHead(Path(__file__).parent / 'weights' / 'phi_weights_compact.bin')
        int_head.add_lut = _G.build_add_lut(dmax=int(int_cfg["dmax"]))
        int_head.sub_lut = _G.build_sub_lut(dmax=int(int_cfg["dmax"]))
        print(f"integer head ready (dmax={int_cfg['dmax']}, "
              f"tables ~{(int_cfg['dmax'] + 1) * 8 / 1024:.0f}kB add+sub, "
              f"JIT={'yes' if _G._jit_kernels() is not None else 'no (python fallback)'})")
        if args.c_head:
            import ctypes
            import subprocess
            import numpy as _np
            cdir = Path(__file__).parent / 'c_port'
            lib = cdir / 'generated' / 'libphi_head.so'
            if not lib.exists():
                print("compiling C head library (one-time)...")
                subprocess.run(['gcc', '-O2', '-std=c99', '-shared', '-fPIC',
                                str(cdir / 'phi_int.c'), str(cdir / 'generated' / 'luts.c'),
                                '-o', str(lib)], check=True)
            so = ctypes.CDLL(str(lib))
            fn = so.phi_head_predict_batch
            I8 = _np.ctypeslib.ndpointer(dtype=_np.int8, flags='C_CONTIGUOUS')
            I32 = _np.ctypeslib.ndpointer(dtype=_np.int32, flags='C_CONTIGUOUS')
            U8 = _np.ctypeslib.ndpointer(dtype=_np.uint8, flags='C_CONTIGUOUS')
            fn.argtypes = [I8, I32, U8, I8, I32, I8, I32,
                           ctypes.c_int8, ctypes.c_int32, I8, I32, U8, ctypes.c_int]
            fn.restype = None
            import geo_lut as _gl
            cheat = {"fn": fn, "ws": np.ascontiguousarray(int_head.w_s, dtype=np.int8),
                     "we": np.ascontiguousarray(int_head.w_e, dtype=np.int32),
                     "ms": np.ascontiguousarray(int_head.m_s, dtype=np.int8),
                     "me": np.ascontiguousarray(int_head.m_e, dtype=np.int32),
                     "tms": int(int_head.tm_s), "tme": int(int_head.tm_e),
                     "lut": _gl.get_lut().to(torch.float32).cpu().numpy()}
            print(f"C head ready: {lib} (batch, one call/frame)")

    hf = hf_proc = None
    if args.compare_hf:
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        from PIL import Image
        print("loading HF baseline for comparison...")
        hf_proc = AutoImageProcessor.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf')
        hf = AutoModelForDepthEstimation.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf').to(device).eval()
        if use_fp16:
            hf = hf.half()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"error: could not open camera {args.camera}")
        raise SystemExit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    outdir = Path('./captures')
    outdir.mkdir(exist_ok=True)

    def process(rgb_float):
        """RGB float [0,1] -> geometric depth float32."""
        from geo_depth import preprocess
        pv = preprocess(rgb_float, size=args.size)
        if use_fp16:
            pv = pv.half()
        with torch.no_grad():
            d = geo.forward(pv).squeeze(0).float().cpu().numpy()
        return d

    def process_int(rgb_float):
        """Learned integer head: float backbone/neck/convs -> 32ch features,
        then IntegerPhiHead tree LUT-adds (JIT). Returns
        (int_depth, float_depth, float_ms, int_ms)."""
        import geo_int as _G
        from geo_depth import preprocess
        pv = preprocess(rgb_float, size=args.size)
        if use_fp16:
            pv = pv.half()
        t0 = time.perf_counter()
        with torch.no_grad():
            pv_d = pv.to(geo.device)
            fmaps, ph, pw = geo.backbone.forward_stages(pv_d)
            fused = geo.neck(fmaps)
            float_depth = geo.head(fused, ph, pw).squeeze(0).float().cpu().numpy()
            feat = geo.head.last_features.squeeze(0).float().permute(1, 2, 0)
            H, W, _ = feat.shape
            flat = feat.reshape(-1, 32).cpu().numpy()
        t_float = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        fs, fe = _G.IntegerPhiHead.encode_features(flat)
        fs = np.ascontiguousarray(fs, dtype=np.int8)
        fe = np.ascontiguousarray(fe, dtype=np.int32)
        t_enc = (time.perf_counter() - t1) * 1000
        if cheat is not None:
            n = fs.shape[0]
            fz = np.zeros((n, 32), dtype=np.uint8)
            so = np.empty(n, dtype=np.int8)
            eo = np.empty(n, dtype=np.int32)
            zo = np.empty(n, dtype=np.uint8)
            t1 = time.perf_counter()
            cheat["fn"](fs, fe, fz, cheat["ws"], cheat["we"], cheat["ms"],
                        cheat["me"], cheat["tms"], cheat["tme"],
                        so, eo, zo, n)
            t_head = (time.perf_counter() - t1) * 1000
            lut = cheat["lut"]
            int_flat = (so.astype(np.float32)
                        * lut[np.clip(eo, 0, 65535)]).astype(np.float32)
            int_flat = np.where(zo == 1, 0.0, int_flat)
        else:
            t1 = time.perf_counter()
            int_flat = int_head.int_predict(fs, fe)
            t_head = (time.perf_counter() - t1) * 1000
        int_depth = int_flat.reshape(H, W).astype(np.float32)
        t_int = t_enc + t_head  # (decode is numpy-vectorized, ~ms; folded into head)
        return int_depth, float_depth, t_float, t_int

    if args.frames > 0:
        # headless capture test
        corrs = []
        int_corrs = []
        t_floats = []
        t_ints = []
        for i in range(args.frames):
            ret, frame = cap.read()
            if not ret:
                print("error: could not read frame")
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t0 = time.perf_counter()
            if int_head is not None:
                depth, float_depth, t_f, t_i = process_int(rgb)
                t_floats.append(t_f)
                t_ints.append(t_i)
                c_if = float(np.corrcoef(depth.flatten().astype(np.float64),
                                         float_depth.flatten().astype(np.float64))[0, 1])
                int_corrs.append(c_if)
            else:
                depth = process(rgb)
            ms = (time.perf_counter() - t0) * 1000
            if int_head is not None:
                fcol = colorize(float_depth, COLORMAPS[0][0])
                fcol = cv2.resize(fcol, (frame.shape[1], frame.shape[0]))
                panels = [frame, fcol, colorize(depth, COLORMAPS[0][0])]
                panels[2] = cv2.resize(panels[2], (frame.shape[1], frame.shape[0]))
                cv2.imwrite(str(outdir / f'geo_webcam_{i}_combined.png'),
                            np.hstack(panels))
            else:
                colored = colorize(depth, COLORMAPS[0][0])
                colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))
                cv2.imwrite(str(outdir / f'geo_webcam_{i}_combined.png'),
                            np.hstack([frame, colored]))
            line = f"frame {i}: geo {ms:.0f}ms ({1000 / ms:.1f} FPS) depth{depth.shape}"
            if int_head is not None:
                line += (f"  [float {t_f:.0f}ms | int-head {t_i:.0f}ms]"
                         f"  int-vs-float corr={int_corrs[-1]:.6f}")
            if hf is not None:
                from PIL import Image
                inputs = hf_proc(images=Image.fromarray((rgb * 255).astype(np.uint8)),
                                 return_tensors='pt')
                pv = inputs['pixel_values'].to(device)
                if use_fp16:
                    pv = pv.half()
                with torch.no_grad():
                    base = hf(pixel_values=pv).predicted_depth.squeeze().float().cpu().numpy()
                if base.shape != depth.shape:
                    import cv2 as _cv
                    depth_r = _cv.resize(depth, (base.shape[1], base.shape[0]))
                else:
                    depth_r = depth
                c = float(np.corrcoef(base.flatten().astype(np.float64),
                                      depth_r.flatten().astype(np.float64))[0, 1])
                corrs.append(c)
                line += f"  HF parity corr={c:.6f}"
            print(line, flush=True)
        if corrs:
            print(f"mean HF parity corr: {np.mean(corrs):.6f}")
        if int_corrs:
            print(f"mean int-vs-float corr: {np.mean(int_corrs):.6f}")
            print(f"mean float-stage: {np.mean(t_floats):.0f}ms | "
                  f"mean int-head: {np.mean(t_ints):.0f}ms")
        cap.release()
        print(f"saved to {outdir}/geo_webcam_*_combined.png")
        return

    # interactive GUI
    cmap_idx = 0
    win = []
    print("\ncontrols: [M] colormap  [S] save  [X] quit\n")
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if int_head is not None:
            depth, float_depth, _tf, _ti = process_int(rgb)
            fcol = colorize(float_depth, COLORMAPS[cmap_idx][0])
            fcol = cv2.resize(fcol, (frame.shape[1], frame.shape[0]))
            icol = colorize(depth, COLORMAPS[cmap_idx][0])
            icol = cv2.resize(icol, (frame.shape[1], frame.shape[0]))
            cv2.putText(fcol, "float geo (backbone+neck+head)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(icol, "learned int-head (graduated widths)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            view = np.hstack([frame, fcol, icol])
            label = "geo-DAV2 learned: camera | float | int-head"
        else:
            depth = process(rgb)
            colored = colorize(depth, COLORMAPS[cmap_idx][0])
            colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))
            view = np.hstack([frame, colored])
            label = "geo-DAV2 (backbone+neck+head)"
        dt = time.perf_counter() - t0
        win.append(dt)
        if len(win) > 30:
            win.pop(0)
        fps = len(win) / sum(win)
        cv2.putText(view, f"FPS: {fps:.1f} | {COLORMAPS[cmap_idx][1]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(view, label,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('geo-DAV2', view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('x'), ord('X')):
            break
        elif key in (ord('m'), ord('M')):
            cmap_idx = (cmap_idx + 1) % len(COLORMAPS)
        elif key in (ord('s'), ord('S')):
            ts = int(time.time())
            cv2.imwrite(str(outdir / f'geo_webcam_{ts}_combined.png'), view)
            print(f"saved {outdir}/geo_webcam_{ts}_combined.png")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
