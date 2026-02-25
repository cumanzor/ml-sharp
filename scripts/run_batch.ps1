$CONDA = "C:\Users\cuman\miniconda3\condabin\conda.bat"
$INPUT_DIR = "C:\Users\cuman\shared\inputs"
$OUTPUT_ROOT = "C:\Users\cuman\shared\outputs"

$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$OUTPUT_DIR = "$OUTPUT_ROOT\$TIMESTAMP"
$DEPTH_DIR = "$OUTPUT_DIR\_depth_npy"
$TMPDIR = "$OUTPUT_DIR\_tmp"
$SCRIPT_DIR = $PSScriptRoot

New-Item -ItemType Directory -Force -Path $DEPTH_DIR | Out-Null

# Collect images
$IMAGES = Get-ChildItem -Path $INPUT_DIR -File | Where-Object {
    $_.Extension -match '\.(jpg|jpeg|png|heic)$'
}

if ($IMAGES.Count -eq 0) {
    Write-Host "No images found in $INPUT_DIR"
    exit 1
}

Write-Host "============================================"
Write-Host "  SHARP + Depth Pro Batch"
Write-Host "  $(Get-Date)"
Write-Host "  Input:  $INPUT_DIR ($($IMAGES.Count) images)"
Write-Host "  Output: $OUTPUT_DIR"
Write-Host "============================================"

# Step 1: Generate Depth Pro depth maps
Write-Host ""
Write-Host "--- Step 1: Depth Pro depth estimation ---"
foreach ($img in $IMAGES) {
    $stem = $img.BaseName
    Write-Host "  Depth Pro: $stem"
    if (Test-Path $TMPDIR) { Remove-Item -Recurse -Force $TMPDIR }
    cmd /c "$CONDA run -n sharp python `"$SCRIPT_DIR\depth_compare.py`" --input `"$($img.FullName)`" --output `"$TMPDIR`" --models depth_pro"
    if (-not (Test-Path "$TMPDIR\depth_pro\depth.npy")) {
        Write-Host "  ERROR: Depth Pro failed for $stem — skipping"
        continue
    }
    Copy-Item "$TMPDIR\depth_pro\depth.npy" "$DEPTH_DIR\$stem.npy"
}
if (Test-Path $TMPDIR) { Remove-Item -Recurse -Force $TMPDIR }

# Step 2: Run SHARP with Depth Pro depth
Write-Host ""
Write-Host "--- Step 2: SHARP 3DGS generation ---"
foreach ($img in $IMAGES) {
    $stem = $img.BaseName
    $npy = "$DEPTH_DIR\$stem.npy"
    if (-not (Test-Path $npy)) {
        Write-Host "  SKIP $stem (no depth map)"
        continue
    }

    Write-Host "  SHARP: $stem"
    if (Test-Path $TMPDIR) { Remove-Item -Recurse -Force $TMPDIR }
    cmd /c "$CONDA run -n sharp sharp predict -i `"$($img.FullName)`" -o `"$TMPDIR`" --external-depth `"$npy`" --depth-format metric"
    if (Test-Path "$TMPDIR\$stem.ply") {
        Move-Item "$TMPDIR\$stem.ply" "$OUTPUT_DIR\$stem.ply"
    } else {
        Write-Host "  ERROR: SHARP failed for $stem — skipping"
    }
}
if (Test-Path $TMPDIR) { Remove-Item -Recurse -Force $TMPDIR }

# Cleanup
if (Test-Path $DEPTH_DIR) { Remove-Item -Recurse -Force $DEPTH_DIR }

Write-Host ""
Write-Host "============================================"
Write-Host "  Done! $(Get-Date)"
Write-Host "  Output: $OUTPUT_DIR"
Write-Host "============================================"
Get-ChildItem "$OUTPUT_DIR\*.ply" -ErrorAction SilentlyContinue |
    Format-Table Name, @{N="Size(MB)";E={[math]::Round($_.Length/1MB,1)}} -AutoSize
