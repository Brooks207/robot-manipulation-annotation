"""Depth storage implementation using 16-bit PNG."""

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from ..types import DepthResult
from .base import DepthWriter

logger = logging.getLogger(__name__)


class PNG16DepthWriter(DepthWriter):
    """
    Write depth maps as 16-bit PNG with sidecar JSON metadata.

    Layout: <output_dir>/depth/<camera_name>/episode_<idx>/frame_<idx>.png
    """

    def __init__(self, output_dir: Path):
        """
        Initialize depth writer.

        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.depth_dir = self.output_dir / "depth"

    def write_depth(
        self,
        episode_idx: int,
        frame_idx: int,
        depth: DepthResult,
        camera_name: str,
        config: dict,
    ) -> None:
        """
        Write depth map as 16-bit PNG with metadata.

        Args:
            episode_idx: Episode index
            frame_idx: Frame index
            depth: Depth result
            camera_name: Camera name (e.g., "observation.images.top")
            config: Storage config (unused here)
        """
        # Create directory structure
        camera_dir = self.depth_dir / camera_name / f"episode_{episode_idx:06d}"
        camera_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        png_path = camera_dir / f"frame_{frame_idx:06d}.png"
        json_path = camera_dir / f"frame_{frame_idx:06d}.json"

        # Normalize depth to uint16 range
        depth_array = depth.depth
        if depth.depth_type == "metric":
            # Metric depth: scale to mm and clip to uint16 range
            # Assume max depth of 65 meters (65000 mm fits in uint16)
            depth_mm = (depth_array * 1000).astype(np.float32)
            depth_mm = np.clip(depth_mm, 0, 65535)
            depth_uint16 = depth_mm.astype(np.uint16)
        else:
            # Relative depth: normalize to [0, 65535]
            depth_norm = (depth_array - depth_array.min()) / (
                depth_array.max() - depth_array.min() + 1e-8
            )
            depth_uint16 = (depth_norm * 65535).astype(np.uint16)

        # Write PNG
        img = Image.fromarray(depth_uint16, mode="I;16")
        img.save(png_path)

        # Write metadata JSON
        metadata = {
            "depth_type": depth.depth_type,
            "scale": float(depth.scale),
            "original_min": float(depth_array.min()),
            "original_max": float(depth_array.max()),
            "encoding": "uint16_png",
        }

        if depth.depth_type == "metric":
            metadata["units"] = "millimeters"
            metadata["conversion_note"] = "Divide pixel value by 1000 to get meters"
        else:
            metadata["conversion_note"] = "Relative depth, normalize to [0, 1] range"

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(
            f"Wrote depth for episode {episode_idx}, frame {frame_idx}, camera {camera_name}"
        )

    def is_frame_annotated(
        self, episode_idx: int, frame_idx: int, camera_name: str
    ) -> bool:
        """
        Check if frame already has depth annotation.

        Args:
            episode_idx: Episode index
            frame_idx: Frame index
            camera_name: Camera name

        Returns:
            True if both PNG and JSON exist and are valid
        """
        camera_dir = self.depth_dir / camera_name / f"episode_{episode_idx:06d}"
        png_path = camera_dir / f"frame_{frame_idx:06d}.png"
        json_path = camera_dir / f"frame_{frame_idx:06d}.json"

        if not (png_path.exists() and json_path.exists()):
            return False

        # Validate JSON structure
        try:
            with open(json_path) as f:
                metadata = json.load(f)
                required_keys = ["depth_type", "scale", "encoding"]
                if not all(k in metadata for k in required_keys):
                    logger.warning(f"Invalid metadata in {json_path}")
                    return False
        except Exception as e:
            logger.warning(f"Could not validate {json_path}: {e}")
            return False

        return True
