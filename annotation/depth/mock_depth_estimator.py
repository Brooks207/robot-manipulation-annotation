"""Mock depth estimator for testing pipeline before Depth Anything 3 deployment."""

import logging

import numpy as np

from ..types import DepthResult
from .base import DepthEstimator

logger = logging.getLogger(__name__)


class MockDepthEstimator(DepthEstimator):
    """
    Mock depth estimator that generates random but valid depth maps.

    Used for testing pipeline orchestration before Depth Anything 3 deployment.
    """

    def estimate_depth(self, frame: np.ndarray, config: dict) -> DepthResult | None:
        """
        Generate mock depth map.

        Args:
            frame: RGB image, shape (H, W, 3)
            config: Depth config

        Returns:
            Mock DepthResult
        """
        H, W = frame.shape[:2]

        # Generate random depth with some structure
        # Use frame content as seed for determinism
        seed = int(frame.mean()) % 2**32
        rng = np.random.RandomState(seed=seed)

        # Generate smooth-ish depth map (radial gradient + noise)
        y, x = np.ogrid[:H, :W]
        center_y, center_x = H // 2, W // 2

        # Radial distance from center
        dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        dist_norm = dist / dist.max()

        # Base depth: closer at center, farther at edges
        base_depth = 1.0 + dist_norm * 3.0  # 1-4 meters

        # Add noise
        noise = rng.randn(H, W) * 0.1
        depth = base_depth + noise
        depth = np.clip(depth, 0.1, 10.0)  # Valid range

        output_metric = config.get("output_metric", True)

        if output_metric:
            depth_type = "metric"
            scale = 1.0
        else:
            # Normalize to relative depth
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth_type = "relative"
            scale = float(depth.max())

        logger.debug(f"MockDepthEstimator: generated {depth_type} depth map")

        return DepthResult(
            depth=depth.astype(np.float32),
            depth_type=depth_type,
            scale=scale,
        )
