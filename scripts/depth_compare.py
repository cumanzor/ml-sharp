#!/usr/bin/env python3
"""
depth_compare.py - Compare monocular depth estimators on a single image.

Runs Depth Pro, Marigold LCM, and Depth Anything V2 sequentially,
saves raw depth maps (.npy), colorized visualizations (.png),
and a side-by-side comparison image.

Usage:
    python depth_compare.py --input path/to/image.jpg --output depth_output/
"""

import argparse
import gc
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clear_cuda():
    """Free CUDA memory between model runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_used_mb():
    """Return current CUDA memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def save_depth(depth: np.ndarray, out_dir: str) -> np.ndarray:
    """Save raw .npy and colorized .png depth map. Returns the RGB visualization."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "depth.npy"), depth)

    # Normalize to [0, 1] for visualization
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min > 1e-6:
        depth_norm = (depth - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth)

    # Colorize with inferno
    cmap = plt.get_cmap("inferno")
    depth_rgba = cmap(depth_norm)
    depth_rgb = (depth_rgba[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(depth_rgb).save(os.path.join(out_dir, "depth_vis.png"))
    return depth_rgb


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_depth_pro(image_path: str):
    """Apple Depth Pro — metric monocular depth (meters)."""
    print("\n========== Depth Pro ==========")
    try:
        import depth_pro
    except ImportError:
        print("  SKIPPED: 'depth_pro' not installed.")
        print("  Install with: pip install git+https://github.com/apple/ml-depth-pro.git")
        return None, 0.0

    # Ensure checkpoint exists (auto-download from HuggingFace Hub)
    ckpt_path = Path("checkpoints/depth_pro.pt")
    if not ckpt_path.exists():
        print("  Downloading Depth Pro checkpoint from HuggingFace...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="apple/DepthPro",
            filename="depth_pro.pt",
            local_dir="checkpoints",
        )
        print("  Checkpoint downloaded.")

    print(f"  Loading model... (VRAM: {vram_used_mb():.0f} MB)")
    config = depth_pro.depth_pro.DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=str(ckpt_path),
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )
    model, transform = depth_pro.create_model_and_transforms(
        config=config,
        device=torch.device("cuda"),
        precision=torch.float16,
    )
    model.eval()
    print(f"  Model loaded.   (VRAM: {vram_used_mb():.0f} MB)")

    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)

    start = time.time()
    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px)
    elapsed = time.time() - start

    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    print(f"  Inference: {elapsed:.2f}s")
    print(f"  Shape: {depth.shape}  Range: [{depth.min():.3f}, {depth.max():.3f}] m")

    del model, prediction
    clear_cuda()
    return depth, elapsed


def run_marigold(image_path: str):
    """Marigold LCM — diffusion-based relative depth (sharp boundaries)."""
    print("\n========== Marigold LCM ==========")
    try:
        from diffusers import MarigoldDepthPipeline
    except ImportError:
        print("  SKIPPED: 'diffusers' not installed.")
        print("  Install with: pip install diffusers transformers accelerate")
        return None, 0.0

    print(f"  Loading model... (VRAM: {vram_used_mb():.0f} MB)")
    pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0",
        variant="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")
    print(f"  Model loaded.   (VRAM: {vram_used_mb():.0f} MB)")

    image = Image.open(image_path).convert("RGB")

    start = time.time()
    output = pipe(
        image,
        ensemble_size=1,
    )
    elapsed = time.time() - start

    # output.prediction: numpy array (default output_type="np"), shape (1, H, W) or (batch, 1, H, W)
    depth = np.asarray(output.prediction).squeeze()  # → (H, W)
    print(f"  Inference: {elapsed:.2f}s")
    print(f"  Shape: {depth.shape}  Range: [{depth.min():.3f}, {depth.max():.3f}]")

    del pipe, output
    clear_cuda()
    return depth, elapsed


def run_depth_anything_v2(image_path: str):
    """Depth Anything V2 Large — metric depth (meters) via local checkpoint."""
    print("\n========== Depth Anything V2 (Metric) ==========")

    DA_V2_ROOT = Path("C:/AI/DepthAnythingV2")
    # Pick checkpoint: outdoor (80m) for general use, indoor (20m) for close-ups.
    ckpt = DA_V2_ROOT / "checkpoints" / "depth_anything_v2_metric_vkitti_vitl.pth"
    max_depth = 80.0
    if not ckpt.exists():
        ckpt = DA_V2_ROOT / "checkpoints" / "depth_anything_v2_metric_hypersim_vitl.pth"
        max_depth = 20.0
    if not ckpt.exists():
        print("  SKIPPED: No DA V2 metric checkpoint found.")
        return None, 0.0

    # Import from the local DA V2 repo.
    import sys
    if str(DA_V2_ROOT / "metric_depth") not in sys.path:
        sys.path.insert(0, str(DA_V2_ROOT / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    import cv2

    print(f"  Loading model (max_depth={max_depth}m)... (VRAM: {vram_used_mb():.0f} MB)")
    model_cfg = {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    model = DepthAnythingV2(**{**model_cfg, 'max_depth': max_depth})
    model.load_state_dict(torch.load(str(ckpt), map_location='cpu'))
    model = model.to('cuda').eval()
    print(f"  Model loaded.   (VRAM: {vram_used_mb():.0f} MB)")

    raw_image = cv2.imread(image_path)

    start = time.time()
    with torch.no_grad():
        depth = model.infer_image(raw_image, input_size=518)
    elapsed = time.time() - start

    print(f"  Inference: {elapsed:.2f}s")
    print(f"  Shape: {depth.shape}  Range: [{depth.min():.3f}, {depth.max():.3f}] m")

    del model
    clear_cuda()
    return depth, elapsed


# ---------------------------------------------------------------------------
# Comparison image
# ---------------------------------------------------------------------------

def create_comparison(original_path: str, results: list, output_path: str):
    """Create a labeled side-by-side comparison image."""
    original = Image.open(original_path).convert("RGB")
    valid = [(name, vis, t) for name, vis, t in results if vis is not None]

    n_cols = 1 + len(valid)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    for i, (name, vis, elapsed) in enumerate(valid):
        axes[i + 1].imshow(vis)
        axes[i + 1].set_title(f"{name}\n({elapsed:.1f}s)", fontsize=13, fontweight="bold")
        axes[i + 1].axis("off")

    plt.tight_layout(pad=1.0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nComparison saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "depth_pro":      ("Depth Pro",        run_depth_pro,          "depth_pro"),
    "marigold":       ("Marigold LCM",     run_marigold,           "marigold"),
    "depth_anything": ("Depth Anything V2", run_depth_anything_v2, "depth_anything"),
}


def main():
    parser = argparse.ArgumentParser(description="Compare monocular depth estimators")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", default="depth_output", help="Output directory")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Which models to run (default: all)",
    )
    args = parser.parse_args()

    # Validate input
    if not os.path.isfile(args.input):
        print(f"Error: input image not found: {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Save a copy of the original
    original = Image.open(args.input).convert("RGB")
    original.save(os.path.join(args.output, "original.png"))

    print("=" * 50)
    print("Depth Estimation Comparison")
    print("=" * 50)
    print(f"  Input:  {args.input}  ({original.size[0]}x{original.size[1]})")
    print(f"  Output: {args.output}")
    if torch.cuda.is_available():
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM:   {total:.1f} GB total")
    else:
        print("  GPU:    NONE (running on CPU — will be slow)")
    print(f"  Models: {', '.join(args.models)}")

    # Run each model sequentially
    results = []
    for key in args.models:
        name, runner, subdir = MODEL_REGISTRY[key]
        try:
            depth, elapsed = runner(args.input)
        except Exception as e:
            print(f"\n  ERROR running {name}: {e}")
            import traceback
            traceback.print_exc()
            depth, elapsed = None, 0.0
            clear_cuda()

        if depth is not None:
            vis = save_depth(depth, os.path.join(args.output, subdir))
            results.append((name, vis, elapsed))
            print(f"  Saved → {args.output}/{subdir}/")
        else:
            results.append((name, None, 0.0))

    # Create comparison
    if any(vis is not None for _, vis, _ in results):
        create_comparison(args.input, results, os.path.join(args.output, "comparison.png"))
    else:
        print("\nNo models ran successfully — no comparison generated.")

    print("\nDone!")


if __name__ == "__main__":
    main()
