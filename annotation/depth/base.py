"""Base interface for depth estimation layer."""

from abc import ABC, abstractmethod

import numpy as np

from ..types import DepthResult


class DepthEstimator(ABC):
    """Abstract base for estimating depth from video frames."""

    @abstractmethod
    def estimate_depth(self, frame: np.ndarray, config: dict) -> DepthResult | None:
        """
        Estimate depth for a single frame.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8
            config: Depth estimation configuration dict

        Returns:
            DepthResult object, or None on failure (logged internally).
        """
        pass
