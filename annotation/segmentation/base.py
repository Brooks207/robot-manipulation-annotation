"""Base interface for segmentation layer."""

from abc import ABC, abstractmethod

import numpy as np

from ..types import InstanceMask


class Segmenter(ABC):
    """Abstract base for segmenting objects in video frames."""

    @abstractmethod
    def segment_frame(
        self, frame: np.ndarray, queries: list[str], config: dict
    ) -> list[InstanceMask]:
        """
        Segment objects in a single frame based on text queries.

        Args:
            frame: RGB image, shape (H, W, 3), dtype uint8
            queries: List of object queries (e.g., ["red cup", "robot hand"])
            config: Segmentation configuration dict

        Returns:
            List of InstanceMask objects. Returns empty list on failure (logged internally).
        """
        pass
