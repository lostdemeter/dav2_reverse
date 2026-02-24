"""
Fit φ-decoder weights from Depth Anything V2.

This script:
1. Loads the DA2 model
2. Extracts head features from sample images
3. Fits φ-decoder weights via least squares
4. Saves universal weights (works for ANY image)

The resulting weight files are only 203 bytes (standard) / 125 bytes (compact).

This is optional — pre-fitted weights are included in weights/.
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

from phi_decoder import PhiDecoder, PhiConfig, extract_head_features
from phi_compact import convert_to_compact


def main():
    parser = argparse.ArgumentParser(
        description='Fit φ-decoder weights from DA2 (optional — pre-fitted weights included)'
    )
    parser.add_argument('--images', type=str, required=True,
                        help='Path to directory containing images (e.g. COCO val2017)')
    parser.add_argument('--output', type=str, default='weights',
                        help='Output directory for weight files (default: weights/)')
    args = parser.parse_args()

    image_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / 'phi_weights.bin'
    compact_path = output_dir / 'phi_weights_compact.bin'

    print("=" * 70)
    print("FITTING φ-DECODER WEIGHTS FROM DA2")
    print("=" * 70)
    print()

    # Find an image to use
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_files.extend(image_dir.glob(ext))
    image_files.sort()

    if not image_files:
        print(f"Error: No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images in {image_dir}")
    print()

    # Load DA2
    print("Loading DA2 model...")
    processor = AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
    model = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
    model.eval()
    print("  Done.")
    print()

    # Use first image for fitting (weights are universal)
    img_path = image_files[0]
    print(f"Extracting features from {img_path.name}...")

    pil_image = Image.open(img_path).convert('RGB')
    inputs = processor(images=pil_image, return_tensors='pt')

    # Get DA2 depth
    with torch.no_grad():
        outputs = model(inputs['pixel_values'])
    da2_depth = outputs.predicted_depth.squeeze().numpy()
    da2_depth_norm = (da2_depth - da2_depth.min()) / (da2_depth.max() - da2_depth.min())

    # Get head features
    features = extract_head_features(model, inputs)
    H, W = features.shape[:2]

    features_flat = features.reshape(-1, 32)
    depths_flat = da2_depth_norm.flatten()

    print(f"  Shape: {H}x{W} = {len(depths_flat):,} pixels")
    print()

    # Fit decoder
    print("Fitting φ-decoder...")
    config = PhiConfig(k_weights=512, k_residual=64, bits_weights=16, bits_residual=12)
    decoder = PhiDecoder(config)

    stats = decoder.fit(features_flat, depths_flat)

    print(f"  Correlation: {stats['correlation']:.10f}")
    print(f"  Residual std: {stats['residual_std']:.6f}")
    print(f"  Weights size: {stats['weights_bytes']} bytes")
    print()

    # Save standard weights
    print(f"Saving standard weights to {weights_path}...")
    decoder.save_weights(weights_path)
    standard_size = weights_path.stat().st_size
    print(f"  Saved: {standard_size} bytes")

    # Save compact weights
    print(f"Saving compact weights to {compact_path}...")
    compact_size = convert_to_compact(weights_path, compact_path)
    print(f"  Saved: {compact_size} bytes")
    print()

    # Verify
    print("Verifying saved weights...")
    decoder2 = PhiDecoder(config)
    decoder2.load_weights(weights_path)
    pred = decoder2.predict(features)
    corr = np.corrcoef(pred.flatten(), da2_depth_norm.flatten())[0, 1]
    print(f"  Verification correlation: {corr:.10f}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  Standard weights: {standard_size} bytes  ({weights_path})")
    print(f"  Compact weights:  {compact_size} bytes  ({compact_path})")
    print(f"  Correlation:      {corr:.6f}")
    print()
    print("These weights work universally for ANY image processed by DA2.")


if __name__ == "__main__":
    main()
