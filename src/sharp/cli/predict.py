"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    unproject_gaussians,
)

from .render import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or containing a list of images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians and renderings.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for checkpoint.",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
@click.option(
    "--external-depth",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to a .npy depth map, or a directory of .npy files matched by image stem.",
)
@click.option(
    "--depth-format",
    type=click.Choice(["auto", "metric", "relative", "inverse-relative"], case_sensitive=False),
    default="auto",
    help="Format of the external depth map. 'metric' = meters, 'relative' = [0,1], "
    "'inverse-relative' = large disparity-like values (e.g. Depth Anything V2). "
    "'auto' attempts heuristic detection.",
)
@click.option(
    "--back-surface-factor",
    type=float,
    default=1.0,
    help="Back surface mode. Default 1.0 = derive back layer from SHARP's internal "
    "front/back ratio (recommended). Values > 1.0 override with a fixed multiplier.",
)
@click.option(
    "--depth-scale",
    type=float,
    default=1.0,
    help="Multiply external depth values by this factor. <1.0 = closer, >1.0 = farther. "
    "Useful for per-image fine-tuning (e.g. 0.8 pulls scene 20%% closer).",
)
@click.option(
    "--depth-offset",
    type=float,
    default=0.0,
    help="Add a constant offset (in meters) to external depth after scaling. "
    "Positive = push farther, negative = pull closer.",
)
@click.option(
    "--save-internal-depth",
    is_flag=True,
    default=False,
    help="Save SHARP's own internal depth as .npy alongside the .ply (for comparison/debugging).",
)
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    device: str,
    verbose: bool,
    external_depth: Path | None,
    depth_format: str,
    back_surface_factor: float,
    depth_scale: float,
    depth_offset: float,
    save_internal_depth: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    extensions = io.get_supported_image_extensions()

    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Processing %d valid image files.", len(image_paths))

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Using device %s", device)

    if with_rendering and device != "cuda":
        LOGGER.warning("Can only run rendering with gsplat on CUDA. Rendering is disabled.")
        with_rendering = False

    # Load or download checkpoint
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)

    if external_depth is not None and external_depth.is_file() and len(image_paths) > 1:
        LOGGER.warning(
            "Single depth map provided for %d images — the same depth will be "
            "used for all. Pass a directory of .npy files for per-image depth.",
            len(image_paths),
        )

    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path)
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )

        # Resolve per-image depth path.
        depth_path = _resolve_depth_path(external_depth, image_path) if external_depth else None

        gaussians = predict_image(
            gaussian_predictor, image, f_px, torch.device(device),
            external_depth_path=depth_path,
            depth_format=depth_format,
            back_surface_factor=back_surface_factor,
            depth_scale=depth_scale,
            depth_offset=depth_offset,
        )

        if save_internal_depth:
            depth_npy = gaussian_predictor._internal_monodepth.cpu().numpy()
            depth_npy_path = output_path / f"{image_path.stem}_internal_depth.npy"
            np.save(str(depth_npy_path), depth_npy)
            LOGGER.info("Saved internal depth to %s", depth_npy_path)

        LOGGER.info("Saving 3DGS to %s", output_path)
        save_ply(gaussians, f_px, (height, width), output_path / f"{image_path.stem}.ply")

        if with_rendering:
            output_video_path = (output_path / image_path.stem).with_suffix(".mp4")
            LOGGER.info("Rendering trajectory to %s", output_video_path)

            metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")
            render_gaussians(gaussians, metadata, output_video_path)


def _resolve_depth_path(external_depth: Path, image_path: Path) -> Path | None:
    """Resolve the depth .npy file for a given image."""
    if external_depth.is_file():
        return external_depth
    # Directory mode: look for <stem>.npy matching the image name.
    candidate = external_depth / f"{image_path.stem}.npy"
    if candidate.exists():
        return candidate
    LOGGER.warning(
        "No depth map found for %s (expected %s) — using internal depth.",
        image_path.name, candidate,
    )
    return None


def _prepare_external_depth(
    depth_path: Path,
    depth_format: str,
    back_surface_factor: float,
    internal_shape: tuple[int, int],
    device: torch.device,
    depth_scale: float = 1.0,
    depth_offset: float = 0.0,
) -> torch.Tensor:
    """Load an external depth map and prepare it for SHARP's predictor.

    Returns a (1, 1, H, W) or (1, 2, H, W) tensor depending on back_surface_factor.
    """
    raw = np.load(str(depth_path)).astype(np.float32)

    # Squeeze to 2D regardless of how the .npy was saved.
    raw = raw.squeeze()
    if raw.ndim == 3:
        # Multi-channel remains — take first slice from the smallest axis.
        raw = np.take(raw, 0, axis=int(np.argmin(raw.shape)))

    LOGGER.info(
        "Loaded external depth: shape=%s, range=[%.4f, %.4f]",
        raw.shape, raw.min(), raw.max(),
    )

    # Auto-detect format from value range.
    if depth_format == "auto":
        vmin, vmax = float(raw.min()), float(raw.max())
        if vmax <= 1.5 and vmin >= -0.1:
            depth_format = "relative"
        elif vmax > 50:
            depth_format = "inverse-relative"
        else:
            depth_format = "metric"
        LOGGER.info("Auto-detected depth format: %s", depth_format)

    # Convert to metric depth (meters).
    if depth_format == "metric":
        depth = raw
    elif depth_format == "relative":
        # Map [0, 1] → [1, 10] metres — a reasonable portrait range.
        depth = raw * 9.0 + 1.0
    elif depth_format == "inverse-relative":
        # DA V2 outputs large disparity-like values; invert to get depth.
        safe = np.clip(raw, 1e-3, None)
        depth = 1.0 / safe
        # Normalise so the median depth maps to ~3 m (typical subject distance).
        median_val = float(np.median(depth[depth > 0]))
        if median_val > 0:
            depth = depth * (3.0 / median_val)
    else:
        raise ValueError(f"Unknown depth format: {depth_format}")

    # Apply per-run scale and offset adjustments.
    if depth_scale != 1.0 or depth_offset != 0.0:
        depth = depth * depth_scale + depth_offset
        LOGGER.info(
            "Depth adjusted: scale=%.3f, offset=%.3f → range=[%.4f, %.4f]",
            depth_scale, depth_offset, depth.min(), depth.max(),
        )

    depth = np.clip(depth, 1e-3, 100.0)

    # To tensor → resize to SHARP's internal resolution.
    depth_t = torch.from_numpy(depth).float().to(device)[None, None]  # (1, 1, H, W)
    depth_t = F.interpolate(
        depth_t, size=internal_shape, mode="bilinear", align_corners=True,
    )

    # Return single-layer depth (1, 1, H, W). The predictor will expand to
    # 2 layers using the internal model's learned front/back ratio, unless
    # back_surface_factor is set to override with a fixed multiplier.
    if back_surface_factor != 1.0:
        depth_2layer = torch.cat([depth_t, depth_t * back_surface_factor], dim=1)
        LOGGER.info(
            "External depth ready (fixed 2-layer, factor=%.3f): shape=%s, range=[%.4f, %.4f]",
            back_surface_factor, list(depth_2layer.shape),
            depth_2layer.min().item(), depth_2layer.max().item(),
        )
        return depth_2layer

    LOGGER.info(
        "External depth ready (1-layer, ratio mode): shape=%s, range=[%.4f, %.4f]",
        list(depth_t.shape), depth_t.min().item(), depth_t.max().item(),
    )
    return depth_t


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    external_depth_path: Path | None = None,
    depth_format: str = "auto",
    back_surface_factor: float = 1.05,
    depth_scale: float = 1.0,
    depth_offset: float = 0.0,
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Prepare external depth if provided.
    ext_depth_tensor = None
    if external_depth_path is not None:
        ext_depth_tensor = _prepare_external_depth(
            external_depth_path, depth_format, back_surface_factor,
            internal_shape, device,
            depth_scale=depth_scale, depth_offset=depth_offset,
        )

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    gaussians_ndc = predictor(image_resized_pt, disparity_factor, external_depth=ext_depth_tensor)

    LOGGER.info("Running postprocessing.")
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians
