#!/usr/bin/env bash
set -e

CONDA="/c/Users/cuman/miniconda3/condabin/conda.bat"
INPUT_DIR="/c/Users/cuman/shared/inputs"
OUTPUT_DIR="/c/Users/cuman/shared/outputs"
DEPTH_DIR="$OUTPUT_DIR/_depth_pro_npy"
TEMP_DIR="$OUTPUT_DIR/_tmp"

mkdir -p "$DEPTH_DIR" "$OUTPUT_DIR"

# Step 1: Generate Depth Pro depth maps for each image
echo "============================================"
echo "Step 1: Generating Depth Pro depth maps"
echo "============================================"
for img in "$INPUT_DIR"/*.{jpg,png,jpeg,heic,HEIC,JPG,PNG,JPEG}; do
    [ -f "$img" ] || continue
    stem=$(basename "${img%.*}")
    echo ""
    echo "--- Depth Pro: $stem ---"
    rm -rf "$TEMP_DIR"
    $CONDA run -n sharp python "$(dirname "$0")/depth_compare.py" \
        --input "$img" \
        --output "$TEMP_DIR" \
        --models depth_pro
    cp "$TEMP_DIR/depth_pro/depth.npy" "$DEPTH_DIR/${stem}.npy"
    echo "  -> $DEPTH_DIR/${stem}.npy"
done
rm -rf "$TEMP_DIR"

echo ""
echo "Depth maps generated:"
ls -l "$DEPTH_DIR/"

# Step 2: Run SHARP per image, output flat files as <stem>_depth-pro.ply
echo ""
echo "============================================"
echo "Step 2: Running SHARP with Depth Pro depth"
echo "============================================"
for img in "$INPUT_DIR"/*.{jpg,png,jpeg,heic,HEIC,JPG,PNG,JPEG}; do
    [ -f "$img" ] || continue
    stem=$(basename "${img%.*}")
    npy="$DEPTH_DIR/${stem}.npy"
    [ -f "$npy" ] || { echo "  SKIP $stem (no depth map)"; continue; }

    echo ""
    echo "--- SHARP: $stem ---"
    rm -rf "$TEMP_DIR"
    $CONDA run -n sharp sharp predict \
        -i "$img" \
        -o "$TEMP_DIR" \
        --external-depth "$npy" \
        --depth-format metric
    mv "$TEMP_DIR/${stem}.ply" "$OUTPUT_DIR/${stem}_depth-pro.ply"
done
rm -rf "$TEMP_DIR"

echo ""
echo "============================================"
echo "Done! Output .ply files:"
echo "============================================"
ls -lh "$OUTPUT_DIR"/*_depth-pro.ply 2>/dev/null || echo "No .ply files found"
