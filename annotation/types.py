"""Core types for the annotation pipeline."""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class InstanceMask:
    """Single segmentation instance result."""

    category: str
    mask: np.ndarray  # bool, shape (H, W)
    score: float
    bbox: tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class DepthResult:
    """Depth estimation result for a single frame."""

    depth: np.ndarray  # float32, shape (H, W), values in meters or normalized
    depth_type: Literal["metric", "relative"]
    scale: float  # scale factor applied (1.0 if metric, else normalization factor)
