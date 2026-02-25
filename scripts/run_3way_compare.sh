#!/usr/bin/env bash
set -e

CONDA="/c/Users/cuman/miniconda3/condabin/conda.bat"
IMG="/c/Users/cuman/shared/inputs/IMG_2011.jpeg"
OUTBASE="/c/Users/cuman/shared/outputs"
DEPTHDIR="$OUTBASE/_depth_maps"
TMPDIR="$OUTBASE/_tmp_ply"

STEM=$(basename "${IMG%.*}")

# Step 1: Generate depth maps
echo "=== Generating depth maps ==="
$CONDA run -n sharp python "$(dirname "$0")/depth_compare.py" \
  --input "$IMG" --output "$DEPTHDIR" --models depth_pro depth_anything

# Step 2: Run SHARP with each depth source
echo ""
echo "=== 1/2 Depth Pro (metric, ratio mode) ==="
rm -rf "$TMPDIR"
$CONDA run -n sharp sharp predict \
  -i "$IMG" -o "$TMPDIR" \
  --external-depth "$DEPTHDIR/depth_pro/depth.npy" \
  --depth-format metric
mv "$TMPDIR/${STEM}.ply" "$OUTBASE/${STEM}_depth-pro.ply"

echo ""
echo "=== 2/2 Depth Anything V2 (inverse-relative, ratio mode) ==="
rm -rf "$TMPDIR"
$CONDA run -n sharp sharp predict \
  -i "$IMG" -o "$TMPDIR" \
  --external-depth "$DEPTHDIR/depth_anything/depth.npy" \
  --depth-format inverse-relative
mv "$TMPDIR/${STEM}.ply" "$OUTBASE/${STEM}_depth-anything-v2.ply"

# Cleanup
rm -rf "$TMPDIR"

echo ""
echo "=== Done! ==="
ls -lh "$OUTBASE"/${STEM}_*.ply
