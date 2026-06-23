"""Mock segmenter for testing pipeline before SAM 3 deployment."""

import logging

import numpy as np

from ..types import InstanceMask
from .base import Segmenter

logger = logging.getLogger(__name__)


class MockSegmenter(Segmenter):
    """
    Mock segmenter that generates random but valid masks.

    Used for testing pipeline orchestration before SAM 3 deployment.
    """

    def segment_frame(
        self, frame: np.ndarray, queries: list[str], config: dict
    ) -> list[InstanceMask]:
        """
        Generate mock segmentation masks.

        Args:
            frame: RGB image, shape (H, W, 3)
            queries: Object queries
            config: Segmentation config

        Returns:
            List of mock InstanceMask objects
        """
        H, W = frame.shape[:2]
        masks = []

        # Generate 1-3 instances per query
        rng = np.random.RandomState(seed=hash(tuple(queries)) % 2**32)

        for query in queries:
            num_instances = rng.randint(1, 4)  # 1-3 instances

            for instance_idx in range(num_instances):
                # Generate random blob mask
                mask = self._generate_random_mask(H, W, rng)

                # Compute bbox from mask
                bbox = self._mask_to_bbox(mask)

                # Random confidence score
                score = rng.uniform(0.6, 0.95)

                masks.append(
                    InstanceMask(
                        category=query,
                        mask=mask,
                        score=float(score),
                        bbox=bbox,
                    )
                )

        logger.debug(f"MockSegmenter: generated {len(masks)} instances for {len(queries)} queries")

        return masks

    def _generate_random_mask(self, H: int, W: int, rng: np.random.RandomState) -> np.ndarray:
        """
        Generate a random blob-like mask.

        Args:
            H: Image height
            W: Image width
            rng: Random state

        Returns:
            Boolean mask of shape (H, W)
        """
        # Random center and size
        center_y = rng.randint(H // 4, 3 * H // 4)
        center_x = rng.randint(W // 4, 3 * W // 4)
        radius_y = rng.randint(20, H // 4)
        radius_x = rng.randint(20, W // 4)

        # Create elliptical mask
        y, x = np.ogrid[:H, :W]
        mask = ((y - center_y) / radius_y) ** 2 + ((x - center_x) / radius_x) ** 2 <= 1

        return mask

    def _mask_to_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int]:
        """
        Compute bounding box from mask.

        Args:
            mask: Boolean mask

        Returns:
            (x, y, w, h) bounding box
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return (0, 0, 0, 0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1))
