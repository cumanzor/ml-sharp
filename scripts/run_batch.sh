#!/usr/bin/env bash
set -e

CONDA="/c/Users/cuman/miniconda3/condabin/conda.bat"
INPUT_DIR="/c/Users/cuman/shared/inputs"
OUTPUT_ROOT="/c/Users/cuman/shared/outputs"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$OUTPUT_ROOT/$TIMESTAMP"
DEPTH_DIR="$OUTPUT_DIR/_depth_npy"
TMPDIR="$OUTPUT_DIR/_tmp"

mkdir -p "$DEPTH_DIR"

# Collect images
IMAGES=()
for img in "$INPUT_DIR"/*.{jpg,png,jpeg,heic,HEIC,JPG,PNG,JPEG}; do
    [ -f "$img" ] && IMAGES+=("$img")
done

if [ ${#IMAGES[@]} -eq 0 ]; then
    echo "No images found in $INPUT_DIR"
    exit 1
fi

echo "============================================"
echo "  SHARP + Depth Pro Batch"
echo "  $(date)"
echo "  Input:  $INPUT_DIR (${#IMAGES[@]} images)"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

# Step 1: Generate Depth Pro depth maps
echo ""
echo "--- Step 1: Depth Pro depth estimation ---"
for img in "${IMAGES[@]}"; do
    stem=$(basename "${img%.*}")
    echo "  Depth Pro: $stem"
    rm -rf "$TMPDIR"
    $CONDA run -n sharp python /c/Users/cuman/sharp-custom/scripts/depth_compare.py \
        --input "$img" \
        --output "$TMPDIR" \
        --models depth_pro
    cp "$TMPDIR/depth_pro/depth.npy" "$DEPTH_DIR/${stem}.npy"
done
rm -rf "$TMPDIR"

# Step 2: Run SHARP with Depth Pro depth
echo ""
echo "--- Step 2: SHARP 3DGS generation ---"
for img in "${IMAGES[@]}"; do
    stem=$(basename "${img%.*}")
    npy="$DEPTH_DIR/${stem}.npy"
    [ -f "$npy" ] || { echo "  SKIP $stem (no depth map)"; continue; }

    echo "  SHARP: $stem"
    rm -rf "$TMPDIR"
    $CONDA run -n sharp sharp predict \
        -i "$img" \
        -o "$TMPDIR" \
        --external-depth "$npy" \
        --depth-format metric
    mv "$TMPDIR/${stem}.ply" "$OUTPUT_DIR/${stem}.ply"
done
rm -rf "$TMPDIR"

# Cleanup internal files
rm -rf "$DEPTH_DIR"

echo ""
echo "============================================"
echo "  Done! $(date)"
echo "  Output: $OUTPUT_DIR"
echo "============================================"
ls -lh "$OUTPUT_DIR"/*.ply 2>/dev/null || echo "  No .ply files generated"
