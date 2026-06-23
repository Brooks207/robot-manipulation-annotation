"""Mask storage implementation using COCO compressed RLE format."""

import logging
from pathlib import Path

import pandas as pd
import pycocotools.mask as mask_util

from ..types import InstanceMask
from .base import MaskWriter

logger = logging.getLogger(__name__)


class ParquetMaskWriter(MaskWriter):
    """
    Write segmentation masks to Parquet using COCO compressed RLE format.

    Each instance is stored as one row with compressed RLE encoding.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize mask writer.

        Args:
            output_dir: Directory where masks.parquet will be written
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_path = self.output_dir / "masks.parquet"

        # In-memory buffer for batch writes
        self.buffer: list[dict] = []
        self.buffer_size = 100  # Write every N instances

        # Load existing data if present
        self.existing_frames: set[tuple[int, int]] = set()
        if self.parquet_path.exists():
            self._load_existing_frames()

    def _load_existing_frames(self) -> None:
        """Load existing annotated frames for idempotency check."""
        try:
            df = pd.read_parquet(self.parquet_path, columns=["episode_idx", "frame_idx"])
            self.existing_frames = set(
                zip(df["episode_idx"].tolist(), df["frame_idx"].tolist())
            )
            logger.info(f"Loaded {len(self.existing_frames)} existing annotated frames")
        except Exception as e:
            logger.warning(f"Could not load existing masks: {e}")

    def write_masks(
        self,
        episode_idx: int,
        frame_idx: int,
        masks: list[InstanceMask],
        config: dict,
    ) -> None:
        """
        Write masks for a single frame.

        Args:
            episode_idx: Episode index
            frame_idx: Frame index
            masks: List of instance masks
            config: Storage config (unused here)
        """
        if not masks:
            logger.debug(f"No masks to write for episode {episode_idx}, frame {frame_idx}")
            return

        for instance_id, mask in enumerate(masks):
            # Encode mask to COCO RLE
            rle = mask_util.encode(
                mask.mask.astype("uint8", order="F")
            )  # Fortran order required

            # Decode bytes to string for storage
            rle_counts = rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"]

            # Compute area
            area = int(mask.mask.sum())

            # Add to buffer
            self.buffer.append(
                {
                    "episode_idx": episode_idx,
                    "frame_idx": frame_idx,
                    "instance_id": instance_id,
                    "category": mask.category,
                    "rle_counts": rle_counts,
                    "rle_size": tuple(rle["size"]),  # (H, W)
                    "area": area,
                    "bbox": mask.bbox,  # (x, y, w, h)
                    "score": mask.score,
                }
            )

        # Mark frame as annotated
        self.existing_frames.add((episode_idx, frame_idx))

        # Flush buffer if needed
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Write buffered masks to parquet."""
        if not self.buffer:
            return

        new_df = pd.DataFrame(self.buffer)

        if self.parquet_path.exists():
            # Append to existing parquet
            existing_df = pd.read_parquet(self.parquet_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_parquet(self.parquet_path, index=False)
        else:
            # Create new parquet
            new_df.to_parquet(self.parquet_path, index=False)

        logger.debug(f"Flushed {len(self.buffer)} mask instances to {self.parquet_path}")
        self.buffer.clear()

    def is_frame_annotated(self, episode_idx: int, frame_idx: int) -> bool:
        """
        Check if frame already has mask annotations.

        Args:
            episode_idx: Episode index
            frame_idx: Frame index

        Returns:
            True if frame is annotated
        """
        return (episode_idx, frame_idx) in self.existing_frames

    def finalize(self) -> None:
        """Flush remaining buffer and finalize storage."""
        self._flush_buffer()
        logger.info(f"Finalized mask storage at {self.parquet_path}")
