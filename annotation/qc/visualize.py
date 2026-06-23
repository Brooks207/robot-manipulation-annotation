"""Quality control visualization for annotated data."""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from PIL import Image

matplotlib.use("Agg")  # Non-interactive backend

logger = logging.getLogger(__name__)


def visualize_annotations(
    dataset,
    mask_parquet_path: Path,
    depth_base_dir: Path,
    output_dir: Path,
    camera_name: str,
    num_frames_per_episode: int = 5,
    colormap: str = "turbo",
) -> None:
    """
    Generate QC visualizations for random frames per episode.

    Args:
        dataset: Dataset object
        mask_parquet_path: Path to masks.parquet
        depth_base_dir: Base directory for depth PNGs
        output_dir: Where to save visualizations
        camera_name: Camera name to visualize
        num_frames_per_episode: Number of frames to sample per episode
        colormap: Matplotlib colormap for depth visualization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating QC visualizations in {output_dir}")

    # Load masks
    if not mask_parquet_path.exists():
        logger.warning(f"Mask file not found: {mask_parquet_path}")
        masks_df = None
    else:
        masks_df = pd.read_parquet(mask_parquet_path)
        logger.info(f"Loaded {len(masks_df)} mask instances")

    # Process each episode
    num_episodes = len(dataset)
    for episode_idx in range(num_episodes):
        try:
            _visualize_episode(
                dataset,
                episode_idx,
                masks_df,
                depth_base_dir,
                output_dir,
                camera_name,
                num_frames_per_episode,
                colormap,
            )
        except Exception as e:
            logger.error(f"Failed to visualize episode {episode_idx}: {e}")

    logger.info(f"QC visualizations complete: {output_dir}")


def _visualize_episode(
    dataset,
    episode_idx: int,
    masks_df: pd.DataFrame | None,
    depth_base_dir: Path,
    output_dir: Path,
    camera_name: str,
    num_frames: int,
    colormap: str,
) -> None:
    """Visualize sampled frames from one episode."""
    episode = dataset.get_episode(episode_idx)
    output_episode_idx = int(episode.get("episode_index", episode_idx))
    total_frames = episode["num_frames"]
    selected_frame_indices = episode.get("frame_indices")

    if selected_frame_indices:
        frame_indices = list(selected_frame_indices)[:num_frames]
    else:
        # Sample frame indices
        rng = np.random.RandomState(seed=episode_idx)
        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = sorted(rng.choice(total_frames, num_frames, replace=False))

    logger.info(
        f"Episode {output_episode_idx}: visualizing {len(frame_indices)} frames: {frame_indices}"
    )

    for frame_idx in frame_indices:
        try:
            _visualize_frame(
                episode,
                output_episode_idx,
                frame_idx,
                masks_df,
                depth_base_dir,
                output_dir,
                camera_name,
                colormap,
            )
        except Exception as e:
            logger.error(
                f"Failed to visualize episode {episode_idx}, frame {frame_idx}: {e}"
            )


def _visualize_frame(
    episode: dict,
    episode_idx: int,
    frame_idx: int,
    masks_df: pd.DataFrame | None,
    depth_base_dir: Path,
    output_dir: Path,
    camera_name: str,
    colormap: str,
) -> None:
    """Visualize one frame with masks and depth overlays."""
    # Get RGB frame
    rgb_frame = episode["frames"][camera_name][frame_idx]
    H, W = rgb_frame.shape[:2]

    # Create figure with 3 subplots: RGB, RGB+masks, depth
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: Original RGB
    axes[0].imshow(rgb_frame)
    axes[0].set_title(f"RGB (episode {episode_idx}, frame {frame_idx})")
    axes[0].axis("off")

    # Subplot 2: RGB + mask overlay
    axes[1].imshow(rgb_frame)

    if masks_df is not None:
        # Get masks for this frame
        frame_masks = masks_df[
            (masks_df["episode_idx"] == episode_idx)
            & (masks_df["frame_idx"] == frame_idx)
        ]

        if len(frame_masks) > 0:
            # Generate distinct colors for each instance
            colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(frame_masks)))

            for idx, row in frame_masks.iterrows():
                # Decode RLE
                rle = {
                    "counts": row["rle_counts"].encode("utf-8")
                    if isinstance(row["rle_counts"], str)
                    else row["rle_counts"],
                    "size": list(row["rle_size"]),
                }
                mask_binary = mask_util.decode(rle).astype(bool)

                # Overlay mask with transparency
                color = colors[idx % len(colors)]
                overlay = np.zeros((H, W, 4))
                overlay[mask_binary] = (*color[:3], 0.5)  # RGBA with alpha=0.5
                axes[1].imshow(overlay)

                # Draw bbox
                x, y, w, h = row["bbox"]
                rect = plt.Rectangle(
                    (x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"
                )
                axes[1].add_patch(rect)

                # Add label
                axes[1].text(
                    x,
                    y - 5,
                    f"{row['category']} ({row['score']:.2f})",
                    color="white",
                    fontsize=8,
                    bbox=dict(facecolor=color[:3], alpha=0.7, pad=2),
                )

    axes[1].set_title(f"Masks ({len(frame_masks) if masks_df is not None else 0} instances)")
    axes[1].axis("off")

    # Subplot 3: Depth
    depth_path = (
        depth_base_dir
        / camera_name
        / f"episode_{episode_idx:06d}"
        / f"frame_{frame_idx:06d}.png"
    )

    if depth_path.exists():
        depth_img = Image.open(depth_path)
        depth_array = np.array(depth_img)

        # Normalize for visualization
        depth_vis = (depth_array - depth_array.min()) / (
            depth_array.max() - depth_array.min() + 1e-8
        )

        axes[2].imshow(depth_vis, cmap=colormap)
        axes[2].set_title("Depth")
    else:
        axes[2].text(
            0.5,
            0.5,
            "Depth not available",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title("Depth (missing)")

    axes[2].axis("off")

    # Save figure
    output_path = output_dir / f"qc_ep{episode_idx:06d}_frame{frame_idx:06d}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    logger.debug(f"Saved QC visualization: {output_path}")
