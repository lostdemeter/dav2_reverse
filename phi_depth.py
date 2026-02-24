"""
phi-depth: Real-Time Depth Estimation from Webcam
==================================================

Captures frames from a USB webcam, runs them through Depth Anything V2's
backbone, replaces DA2's decoder head with a 125-byte φ-arithmetic decoder,
and displays the original camera feed side-by-side with a colorized depth map.

Performance optimizations:
- FP16 backbone (half precision on GPU)
- torch.compile for backbone + neck
- GPU-native φ-prediction (no CPU round-trip)

Usage:
    python phi_depth.py
    python phi_depth.py --camera 1
    python phi_depth.py --no-compile    # skip torch.compile if it causes issues

Controls:
    M - Cycle colormap (magma → viridis → plasma → inferno → turbo)
    S - Save current frame pair to ./captures/
    X - Quit
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

from phi_decoder import PhiDecoder, PhiConfig
from phi_compact import CompactPhiWeights

PHI = (1 + np.sqrt(5)) / 2


class PhiDepthCamera:
    """Real-time depth estimation using the φ-arithmetic decoder."""

    COLORMAPS = [
        (cv2.COLORMAP_MAGMA, 'magma'),
        (cv2.COLORMAP_VIRIDIS, 'viridis'),
        (cv2.COLORMAP_PLASMA, 'plasma'),
        (cv2.COLORMAP_INFERNO, 'inferno'),
        (cv2.COLORMAP_TURBO, 'turbo'),
    ]

    def __init__(self, weights_path: Path, use_fp16: bool = True,
                 use_compile: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        print(f"Device: {self.device}")

        # Load DA2 backbone
        print("Loading DA2 backbone (first run downloads ~94 MB)...")
        self.processor = AutoImageProcessor.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf'
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            'depth-anything/Depth-Anything-V2-Small-hf'
        ).to(self.device)
        self.model.eval()

        # FP16 for speed
        if self.use_fp16:
            print("Converting backbone to FP16...")
            self.model = self.model.half()

        # torch.compile for speed
        if use_compile and self.device.type == 'cuda' and hasattr(torch, 'compile'):
            print("Compiling backbone with torch.compile...")
            try:
                self.model.backbone = torch.compile(
                    self.model.backbone, mode='reduce-overhead', fullgraph=False
                )
                self.model.neck = torch.compile(
                    self.model.neck, mode='reduce-overhead', fullgraph=False
                )
            except Exception as e:
                print(f"  torch.compile failed ({e}), continuing without it")

        # Load φ-decoder weights as GPU tensors
        self._load_weights(weights_path)

        # Register feature extraction hook (stays on GPU, no detach needed)
        self.captured_features = None
        def hook(module, input, output):
            self.captured_features = output
        self.hook_handle = self.model.head.activation1.register_forward_hook(hook)

        # Colormap state
        self.colormap_idx = 0

        # Warmup (triggers JIT compilation and CUDA kernel caching)
        if self.device.type == 'cuda':
            print("Warming up GPU pipeline...")
            self._warmup()

        print("Ready!")

    def _load_weights(self, weights_path: Path):
        """Load weights from PHI1 or PHI2 format into GPU tensors."""
        config = PhiConfig(k_weights=512, bits_weights=16)

        # Detect format
        with open(weights_path, 'rb') as f:
            magic = f.read(4)

        if magic == b'PHI2':
            compact = CompactPhiWeights.load(weights_path)
            w, fm, tm = compact.to_weights()
            weight_np = w.astype(np.float32)
            mean_np = fm.astype(np.float32)
            target_mean = float(tm)
            print(f"Loaded compact φ-weights ({weights_path.stat().st_size} bytes)")
        else:
            decoder = PhiDecoder(config)
            decoder.load_weights(weights_path)
            lut = np.array(
                [PHI ** ((e - config.bias_weights) / config.k_weights)
                 for e in range(config.n_levels_weights)], dtype=np.float64
            )
            ww = decoder.weights
            weight_np = (lut[ww.weights.exponents] * ww.weights.signs).astype(np.float32)
            mean_np = (lut[ww.feature_mean.exponents] * ww.feature_mean.signs).astype(np.float32)
            target_mean = float(ww.target_mean.to_float())
            print(f"Loaded standard φ-weights ({weights_path.stat().st_size} bytes)")

        # Store as GPU tensors for fast prediction
        self.weight_t = torch.from_numpy(weight_np).to(self.device)
        self.mean_t = torch.from_numpy(mean_np).to(self.device)
        self.target_mean = target_mean

    def _warmup(self):
        """Warmup pass to trigger JIT compilation and CUDA caching."""
        dtype = torch.float16 if self.use_fp16 else torch.float32
        dummy = torch.randn(1, 3, 518, 518, device=self.device, dtype=dtype)
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(dummy)

    @torch.no_grad()
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a BGR frame and return a uint8 depth map."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        inputs = self.processor(images=pil_image, return_tensors='pt')
        if self.use_fp16:
            inputs = {k: v.to(self.device).half() for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward through backbone (captures features via hook)
        _ = self.model(**inputs)

        # φ-prediction entirely on GPU
        features = self.captured_features.squeeze()  # (32, H, W), stays on GPU
        if features.dtype == torch.float16:
            features = features.float()
        C, H, W = features.shape
        features = features.permute(1, 2, 0).reshape(-1, C)  # (H*W, 32)

        # Linear prediction: depth = (features - mean) @ weights + target_mean
        depth = (features - self.mean_t) @ self.weight_t + self.target_mean

        # Normalize to 0-255 on GPU
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max > depth_min:
            depth = ((depth - depth_min) / (depth_max - depth_min) * 255)
        else:
            depth = torch.zeros_like(depth)

        # Only transfer the final uint8 to CPU
        return depth.reshape(H, W).to(torch.uint8).cpu().numpy()

    def run(self, camera_id: int = 0):
        """Run the real-time depth visualization loop."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        captures_dir = Path('./captures')

        fps_window = []

        print()
        print("Controls: [M] colormap  [S] save  [X] quit")
        print()

        while True:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            depth = self.process_frame(frame)

            # Apply colormap
            cmap, cmap_name = self.COLORMAPS[self.colormap_idx]
            depth_colored = cv2.applyColorMap(depth, cmap)
            depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))

            # FPS calculation (rolling window of last 30 frames)
            frame_time = time.perf_counter() - t_start
            fps_window.append(frame_time)
            if len(fps_window) > 30:
                fps_window.pop(0)
            fps = len(fps_window) / sum(fps_window)

            # HUD overlay on depth side
            cv2.putText(depth_colored, f"FPS: {fps:.1f}  |  {cmap_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_colored, "phi-depth (125 bytes)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_colored, "[M] colormap  [S] save  [X] quit",
                        (10, depth_colored.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Side-by-side display
            combined = np.hstack([frame, depth_colored])
            cv2.imshow('phi-depth', combined)

            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('x') or key == ord('X'):
                break
            elif key == ord('m') or key == ord('M'):
                self.colormap_idx = (self.colormap_idx + 1) % len(self.COLORMAPS)
                print(f"Colormap: {self.COLORMAPS[self.colormap_idx][1]}")
            elif key == ord('s') or key == ord('S'):
                captures_dir.mkdir(exist_ok=True)
                timestamp = int(time.time())
                path_rgb = captures_dir / f'phi_depth_{timestamp}_rgb.png'
                path_depth = captures_dir / f'phi_depth_{timestamp}_depth.png'
                path_combined = captures_dir / f'phi_depth_{timestamp}_combined.png'
                cv2.imwrite(str(path_rgb), frame)
                cv2.imwrite(str(path_depth), depth_colored)
                cv2.imwrite(str(path_combined), combined)
                print(f"Saved to {captures_dir}/phi_depth_{timestamp}_*.png")

        cap.release()
        cv2.destroyAllWindows()
        self.hook_handle.remove()

        if fps_window:
            print(f"Average FPS: {len(fps_window) / sum(fps_window):.1f}")


def find_weights() -> Path:
    """Find the best available weight file."""
    base = Path(__file__).parent / 'weights'
    compact = base / 'phi_weights_compact.bin'
    standard = base / 'phi_weights.bin'
    if compact.exists():
        return compact
    if standard.exists():
        return standard
    raise FileNotFoundError(
        f"No weight files found in {base}/\n"
        "Run fit_weights.py first, or download pre-fitted weights."
    )


def main():
    parser = argparse.ArgumentParser(
        description='phi-depth: Real-time depth estimation from webcam'
    )
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to weight file (default: auto-detect in weights/)')
    parser.add_argument('--no-fp16', action='store_true',
                        help='Disable FP16 (use full precision)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile')
    args = parser.parse_args()

    weights_path = Path(args.weights) if args.weights else find_weights()

    print("=" * 55)
    print("  phi-depth: Real-Time Depth Estimation from Webcam")
    print("=" * 55)
    print()

    try:
        app = PhiDepthCamera(
            weights_path,
            use_fp16=not args.no_fp16,
            use_compile=not args.no_compile,
        )
        app.run(camera_id=args.camera)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
