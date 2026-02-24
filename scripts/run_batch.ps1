$ErrorActionPreference = "Stop"

$CONDA = "C:\Users\cuman\miniconda3\condabin\conda.bat"
$INPUT_DIR = "C:\Users\cuman\shared\inputs"
$OUTPUT_ROOT = "C:\Users\cuman\shared\outputs"

$TIMESTAMP = Get-Date -Format "yyyyMMdd_HHmmss"
$OUTPUT_DIR = "$OUTPUT_ROOT\$TIMESTAMP"
$DEPTH_DIR = "$OUTPUT_DIR\_depth_npy"
$TMPDIR = "$OUTPUT_DIR\_tmp"

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
    & $CONDA run -n sharp python C:\Users\cuman\sharp-custom\scripts\depth_compare.py `
        --input "$($img.FullName)" `
        --output "$TMPDIR" `
        --models depth_pro
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
    & $CONDA run -n sharp sharp predict `
        -i "$($img.FullName)" `
        -o "$TMPDIR" `
        --external-depth "$npy" `
        --depth-format metric
    Move-Item "$TMPDIR\$stem.ply" "$OUTPUT_DIR\$stem.ply"
}
if (Test-Path $TMPDIR) { Remove-Item -Recurse -Force $TMPDIR }

# Cleanup
Remove-Item -Recurse -Force $DEPTH_DIR

Write-Host ""
Write-Host "============================================"
Write-Host "  Done! $(Get-Date)"
Write-Host "  Output: $OUTPUT_DIR"
Write-Host "============================================"
Get-ChildItem "$OUTPUT_DIR\*.ply" | Format-Table Name, @{N="Size(MB)";E={[math]::Round($_.Length/1MB,1)}} -AutoSize
