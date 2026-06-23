"""Base interface for storage layer."""

from abc import ABC, abstractmethod
from pathlib import Path

from ..types import DepthResult, InstanceMask


class MaskWriter(ABC):
    """Abstract base for persisting segmentation masks."""

    @abstractmethod
    def write_masks(
        self,
        episode_idx: int,
        frame_idx: int,
        masks: list[InstanceMask],
        config: dict,
    ) -> None:
        """
        Write segmentation masks for a single frame.

        Args:
            episode_idx: Episode index
            frame_idx: Frame index within episode
            masks: List of instance masks to write
            config: Storage configuration dict
        """
        pass

    @abstractmethod
    def is_frame_annotated(self, episode_idx: int, frame_idx: int) -> bool:
        """
        Check if a frame already has mask annotations (for idempotency).

        Args:
            episode_idx: Episode index
            frame_idx: Frame index within episode

        Returns:
            True if frame is already annotated and valid
        """
        pass


class DepthWriter(ABC):
    """Abstract base for persisting depth maps."""

    @abstractmethod
    def write_depth(
        self,
        episode_idx: int,
        frame_idx: int,
        depth: DepthResult,
        camera_name: str,
        config: dict,
    ) -> None:
        """
        Write depth map for a single frame.

        Args:
            episode_idx: Episode index
            frame_idx: Frame index within episode
            depth: Depth result to write
            camera_name: Camera name (e.g., "observation.images.top")
            config: Storage configuration dict
        """
        pass

    @abstractmethod
    def is_frame_annotated(
        self, episode_idx: int, frame_idx: int, camera_name: str
    ) -> bool:
        """
        Check if a frame already has depth annotation (for idempotency).

        Args:
            episode_idx: Episode index
            frame_idx: Frame index within episode
            camera_name: Camera name

        Returns:
            True if frame is already annotated and valid
        """
        pass
