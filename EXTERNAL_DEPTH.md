# SHARP Custom: External Depth Map Injection

Custom modifications to Apple's [SHARP](https://github.com/apple/ml-sharp) single-image-to-3DGS pipeline, allowing external monocular depth estimators to replace SHARP's built-in DINOv2+DPT depth.

**Recommended depth source: Apple Depth Pro** (`--depth-format metric`). Produces the best results across people, pets, and vehicles. DA V2 metric is a viable alternative for outdoor scenes but Depth Pro is consistently better for close subjects.

## Problem

SHARP's internal depth estimator (DINOv2-Large + DPT head) produces flat, mushy depth on human subjects — limbs, clothing layers, and body contours lack 3D separation. Side-by-side comparison showed **Apple Depth Pro** captures significantly more body contour detail, producing metric depth in meters with no normalization needed.

## Approach

SHARP's pipeline:
```
image → monodepth (disparity → metric depth) → initializer → Gaussian base values
                                                                    ↓
                                            encoder features → delta prediction → final Gaussians
```

The monodepth model produces **two outputs**: (1) a depth tensor, and (2) DINOv2 encoder features (4 feature levels). The encoder features encode spatial structure (edges, semantics), not depth values — they're reused by the Gaussian decoder via skip connections.

**We keep the encoder features and only swap the depth tensor.** The monodepth model still runs every time; we just discard its depth output when an external depth map is provided.

### The 2-layer issue

SHARP predicts 2 depth layers (front/back surfaces, shape `(B, 2, H, W)`). External estimators give 1 layer.

**Ratio mode (default)**: The external depth is passed as a single layer `(1, 1, H, W)`. Inside the predictor, the back surface is derived from SHARP's internal depth front/back ratio — preserving the model's learned geometry for the hidden back side. This eliminates the "ghost duplication" artifact seen with fixed offsets.

**Fixed mode** (`--back-surface-factor > 1.0`): Falls back to a constant multiplier if specified. Not recommended — causes visible duplication of surfaces.

## Files Modified

### `patches/predictor.py` — Model forward pass

**Source**: `ml-sharp/src/sharp/models/predictor.py`

Changes to `RGBGaussianPredictor.forward()`:
1. Added `external_depth: torch.Tensor | None = None` parameter
2. Stores `self._internal_monodepth` before any swap (for `--save-internal-depth`)
3. If `external_depth` has 1 channel, expands to 2 using internal depth's per-pixel front/back ratio
4. Replaces `monodepth` with the (now 2-layer) external depth

```python
self._internal_monodepth = monodepth
if external_depth is not None:
    if external_depth.shape[1] == 1 and monodepth.shape[1] == 2:
        ratio = monodepth[:, 1:2] / monodepth[:, 0:1].clamp(min=1e-4)
        external_depth = torch.cat(
            [external_depth, external_depth * ratio], dim=1
        )
    monodepth = external_depth
```

### `patches/predict.py` — CLI + depth loading

**Source**: `ml-sharp/src/sharp/cli/predict.py`

New CLI options:
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--external-depth` | PATH | None | `.npy` file or directory of `.npy` files (matched by image stem) |
| `--depth-format` | Choice | `auto` | `metric`, `relative`, `inverse-relative`, or `auto` |
| `--back-surface-factor` | float | 1.0 | 1.0 = ratio mode (recommended). >1.0 = fixed multiplier |
| `--depth-scale` | float | 1.0 | Multiply depth values. <1.0 = closer, >1.0 = farther |
| `--depth-offset` | float | 0.0 | Add meters after scaling. Positive = farther, negative = closer |
| `--save-internal-depth` | flag | false | Save SHARP's own depth as `<stem>_internal_depth.npy` |

New functions:
- **`_resolve_depth_path()`** — Resolves per-image depth: file → use for all; directory → match `<stem>.npy`. Falls back to internal depth with a warning if no match.
- **`_prepare_external_depth()`** — Loads `.npy`, squeezes to 2D, converts format to metric meters, applies scale/offset, clamps to [0.001, 100], resizes to 1536x1536. Returns 1-layer (ratio mode) or 2-layer (fixed mode).

### `scripts/depth_compare.py` — Depth map generator

Changes from upstream:
- Added `pillow_heif` registration for HEIC support
- Replaced DA V2 HuggingFace relative model with **local metric checkpoint** (`C:\AI\DepthAnythingV2\checkpoints\depth_anything_v2_metric_vkitti_vitl.pth`, 80m outdoor). Outputs meters directly — no more `inverse-relative` heuristics.

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_batch.sh` | **Recommended**: Depth Pro → SHARP, timestamped output |
| `scripts/run_batch.ps1` | PowerShell version of the above |
| `scripts/run_depthpro_sharp.sh` | Batch: Depth Pro → SHARP, flat output naming |
| `scripts/run_3way_compare.sh` | A/B compare: Depth Pro vs DA V2 on a single image |

## Usage

### Batch processing (recommended)

**PowerShell:**
```powershell
& C:\Users\cuman\sharp-custom\scripts\run_batch.ps1
```

**Git Bash:**
```bash
bash /c/Users/cuman/sharp-custom/scripts/run_batch.sh
```

Processes all images in `shared/inputs/`, outputs to `shared/outputs/<timestamp>/`.

### Single image

```bash
conda run -n sharp sharp predict \
  -i image.jpg -o output/ \
  --external-depth depth_pro.npy \
  --depth-format metric
```

### Per-image depth tuning

```bash
# Scene feels too far — pull 20% closer
conda run -n sharp sharp predict \
  -i photo.jpg -o out/ \
  --external-depth depth.npy --depth-format metric \
  --depth-scale 0.8

# Scene feels too close — push 30% farther
conda run -n sharp sharp predict \
  -i photo.jpg -o out/ \
  --external-depth depth.npy --depth-format metric \
  --depth-scale 1.3

# Fine nudge — everything 0.5m closer
conda run -n sharp sharp predict \
  -i photo.jpg -o out/ \
  --external-depth depth.npy --depth-format metric \
  --depth-offset -0.5

# Combine scale + offset
conda run -n sharp sharp predict \
  -i photo.jpg -o out/ \
  --external-depth depth.npy --depth-format metric \
  --depth-scale 0.9 --depth-offset -0.3
```

### Batch with depth directory

```bash
conda run -n sharp sharp predict \
  -i inputs/ -o outputs/ \
  --external-depth depth_npy_dir/ \
  --depth-format metric
```

### Save SHARP's internal depth for debugging

```bash
conda run -n sharp sharp predict \
  -i image.jpg -o output/ \
  --external-depth depth_pro.npy \
  --depth-format metric \
  --save-internal-depth
```

## Depth Source Comparison

| Source | Output | Quality | Best for |
|--------|--------|---------|----------|
| **Depth Pro** | Metric meters | **Best overall** | People, pets, close objects, vehicles |
| DA V2 Metric (outdoor) | Metric meters, max 80m | Good | Outdoor scenes with distant objects |
| DA V2 Metric (indoor) | Metric meters, max 20m | Good | Indoor close-ups |
| ~~Marigold LCM~~ | Relative [0,1] | Poor | Dropped — not competitive |
| ~~DA V2 HuggingFace~~ | Relative disparity | Poor | Replaced by metric model |

## Restoring Changes

To apply patches to a fresh SHARP install:
```bash
cp patches/predictor.py  ml-sharp/src/sharp/models/predictor.py
cp patches/predict.py    ml-sharp/src/sharp/cli/predict.py
```

## Environment

- Windows 11, RTX 4070 Super (12GB), CUDA
- Conda env: `sharp` (SHARP, Depth Pro, pillow-heif, opencv-python-headless, diffusers, transformers)
- SHARP internal resolution: 1536x1536
- SHARP depth layers: 2 (front + back surface)
- Output: 3DGS `.ply` (~63 MB per image)
- DA V2 metric checkpoints: `C:\AI\DepthAnythingV2\checkpoints\`

## Known Limitations

- **Feature-depth mismatch**: Encoder features come from SHARP's DINOv2 while depth comes from an external model. Acceptable because encoder features encode spatial structure, not depth — the depth signal enters through the `[image, 1/depth]` skip connection input.
- **Interpolation blur**: Resizing external depth to 1536x1536 via bilinear may soften sharp edges slightly.
- **No EXIF focal length**: Most test images lack EXIF focal length data — SHARP defaults to 30mm equivalent. This affects 3D scale but not relative depth quality.
