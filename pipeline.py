"""Main annotation pipeline orchestration."""

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from annotation.config import PipelineConfig
from annotation.depth.base import DepthEstimator
from annotation.discovery.base import ObjectDiscoverer
from annotation.segmentation.base import Segmenter
from annotation.storage.base import DepthWriter, MaskWriter

logger = logging.getLogger(__name__)


class AnnotationPipeline:
    """Orchestrates the four-layer annotation pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        discoverer: ObjectDiscoverer,
        segmenter: Segmenter | None,
        depth_estimator: DepthEstimator | None,
        mask_writer: MaskWriter | None,
        depth_writer: DepthWriter | None,
    ):
        """
        Initialize pipeline with config and layer implementations.

        Args:
            config: Pipeline configuration
            discoverer: Object discovery implementation
            segmenter: Segmentation implementation (None in dry-run)
            depth_estimator: Depth estimation implementation (None in dry-run)
            mask_writer: Mask storage implementation (None in dry-run)
            depth_writer: Depth storage implementation (None in dry-run)
        """
        self.config = config
        self.discoverer = discoverer
        self.segmenter = segmenter
        self.depth_estimator = depth_estimator
        self.mask_writer = mask_writer
        self.depth_writer = depth_writer
        self.dataset = None

        # Setup checkpoint directory (stage-aware so segmentation and depth
        # resume independently and never clobber each other's progress file).
        self.checkpoint_dir = self._resolve_checkpoint_dir()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self.stats = {
            "total_episodes": 0,
            "processed_episodes": 0,
            "skipped_episodes": 0,
            "failed_episodes": 0,
            "total_frames": 0,
            "processed_frames": 0,
            "failed_frames": defaultdict(list),  # {episode_idx: [frame_idx, ...]}
            "start_time": None,
            "end_time": None,
        }

        # Dry-run discovery cache
        self.discovery_results: list[dict[str, Any]] = []

        # Sampling manifest: one row per sampled frame, mapping it back to the
        # original video frame_idx and its subtask. Lets downstream align mask
        # and depth outputs by (episode_idx, frame_idx) and group by subtask.
        self.sampling_manifest: list[dict[str, Any]] = []

    def _resolve_checkpoint_dir(self) -> Path:
        """
        Pick the checkpoint directory for the active stage.

        Each stage owns its own checkpoint so segmentation and depth resume
        independently. Explicit per-stage / global overrides take precedence;
        otherwise checkpoints live next to the stage output dir.
        """
        config = self.config
        if config.stage == "segmentation" and config.segmentation_checkpoint_dir:
            return config.segmentation_checkpoint_dir
        if config.stage == "depth" and config.depth_checkpoint_dir:
            return config.depth_checkpoint_dir
        if config.checkpoint_dir:
            return config.checkpoint_dir
        return self.output_dir / ".checkpoints"

    @property
    def output_dir(self) -> Path:
        """Stage-aware output directory; falls back to the shared output_dir."""
        storage = self.config.storage
        if self.config.stage == "segmentation" and storage.segmentation_output_dir:
            return storage.segmentation_output_dir
        if self.config.stage == "depth" and storage.depth_output_dir:
            return storage.depth_output_dir
        return storage.output_dir

    def load_checkpoint(self) -> set[int]:
        """
        Load checkpoint to determine which episodes have been processed.

        Returns:
            Set of completed episode indices
        """
        checkpoint_file = self.checkpoint_dir / "completed_episodes.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                data = json.load(f)
                completed = set(data.get("completed_episodes", []))
                logger.info(f"Loaded checkpoint: {len(completed)} completed episodes")
                return completed
        return set()

    def save_checkpoint(self, completed_episodes: set[int]) -> None:
        """
        Save checkpoint with completed episode indices.

        Args:
            completed_episodes: Set of completed episode indices
        """
        checkpoint_file = self.checkpoint_dir / "completed_episodes.json"
        with open(checkpoint_file, "w") as f:
            json.dump(
                {
                    "completed_episodes": sorted(completed_episodes),
                    "timestamp": time.time(),
                },
                f,
                indent=2,
            )

    def save_dry_run_results(self) -> None:
        """Save dry-run discovery results to JSONL."""
        if not self.config.dry_run:
            return

        output_file = self.config.storage.output_dir / "discovery_queries.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for result in self.discovery_results:
                f.write(json.dumps(result) + "\n")

        logger.info(f"Saved discovery results to {output_file}")

    def run(self) -> dict[str, Any]:
        """
        Run the annotation pipeline.

        Returns:
            Dictionary with execution statistics and failure report
        """
        self.stats["start_time"] = time.time()
        logger.info("Starting annotation pipeline")
        logger.info(f"Config: dry_run={self.config.dry_run}")

        # Load checkpoint
        completed_episodes = self.load_checkpoint()

        # Load dataset
        try:
            dataset = self._load_dataset()
            self.dataset = dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        self.stats["total_episodes"] = len(dataset)
        logger.info(f"Dataset contains {len(dataset)} episodes")

        # Process episodes
        for episode_idx in range(len(dataset)):
            checkpoint_idx = episode_idx
            if hasattr(dataset, "episode_index_at"):
                checkpoint_idx = int(dataset.episode_index_at(episode_idx))

            if checkpoint_idx in completed_episodes:
                logger.info(f"Episode {checkpoint_idx}: already completed, skipping")
                self.stats["skipped_episodes"] += 1
                continue

            try:
                self._process_episode(episode_idx, dataset)
                completed_episodes.add(checkpoint_idx)
                self.save_checkpoint(completed_episodes)
                self.stats["processed_episodes"] += 1

            except Exception as e:
                logger.error(f"Episode {checkpoint_idx}: FAILED - {e}", exc_info=True)
                self.stats["failed_episodes"] += 1
                # Continue to next episode (failure isolation)

        # Save dry-run results if applicable
        if self.config.dry_run:
            self.save_dry_run_results()

        # Persist the sampling manifest for downstream alignment
        self._save_sampling_manifest()

        # Finalize
        self.stats["end_time"] = time.time()
        self._log_summary()

        return self.stats

    def _load_dataset(self) -> Any:
        """
        Load LeRobot v3.0 dataset.

        Returns:
            Dataset object (implementation TBD based on LeRobot API)
        """
        dataset_type = self.config.dataset_type
        if dataset_type == "auto":
            has_lerobot_meta = (
                self.config.dataset_path / "meta" / "episodes"
            ).exists()
            dataset_type = "lerobot_v3" if has_lerobot_meta else "mock"

        if dataset_type == "lerobot_v3":
            from annotation.lerobot_v3_dataset import LeRobotV3Dataset

            return LeRobotV3Dataset(
                dataset_path=self.config.dataset_path,
                camera_names=self.config.camera_names,
                instruction_config=asdict(self.config.discovery),
                episode_indices=self.config.episode_indices,
                frame_indices=self.config.frame_indices,
                sample_frames_per_episode=self.config.sample_frames_per_episode,
                sampling_config=asdict(self.config.sampling),
                max_episodes=self.config.max_episodes,
                max_frames_per_episode=self.config.max_frames_per_episode,
                load_frames=not self.config.dry_run,
            )

        logger.warning("Using mock dataset loader")
        from annotation.mock_dataset import MockLeRobotDataset

        return MockLeRobotDataset(self.config.dataset_path)

    def _process_episode(self, episode_idx: int, dataset: Any) -> None:
        """
        Process a single episode through the pipeline.

        Args:
            episode_idx: Episode index
            dataset: Dataset object
        """
        # Get episode data
        episode = dataset.get_episode(episode_idx)
        output_episode_idx = int(episode.get("episode_index", episode_idx))
        logger.info(f"Episode {output_episode_idx}: starting")

        instruction = episode.get("instruction", "")
        num_frames = episode["num_frames"]
        frame_indices = episode.get("frame_indices", list(range(num_frames)))
        self.stats["total_frames"] += len(frame_indices)
        self._record_sampling_manifest(output_episode_idx, frame_indices, episode)

        # Layer 1: Discovery
        discovery_config = asdict(self.config.discovery)
        discovery_config["episode_idx"] = output_episode_idx
        queries = self.discoverer.discover_objects(instruction, discovery_config)
        logger.info(
            f"Episode {output_episode_idx}: discovered {len(queries)} objects: {queries}"
        )

        if self.config.dry_run:
            # Dry-run: only record discovery results
            self.discovery_results.append(
                {
                    "episode_idx": output_episode_idx,
                    "instruction": instruction,
                    "queries": queries,
                }
            )
            return

        # Full pipeline: process each frame
        for frame_idx in frame_indices:
            try:
                self._process_frame(output_episode_idx, frame_idx, queries, episode)
                self.stats["processed_frames"] += 1
            except Exception as e:
                logger.error(
                    f"Episode {output_episode_idx}, frame {frame_idx}: FAILED - {e}",
                    exc_info=False,
                )
                self.stats["failed_frames"][output_episode_idx].append(frame_idx)
                # Continue to next frame (failure isolation)

        logger.info(
            f"Episode {output_episode_idx}: completed "
            f"({len(frame_indices) - len(self.stats['failed_frames'].get(output_episode_idx, []))}/"
            f"{len(frame_indices)} selected frames succeeded)"
        )

    def _record_sampling_manifest(
        self, episode_idx: int, frame_indices: list[int], episode: dict
    ) -> None:
        """Record one manifest row per sampled frame (subtask + original idx)."""
        frame_metadata = episode.get("frame_metadata", {})
        for frame_idx in frame_indices:
            meta = frame_metadata.get(frame_idx, {})
            self.sampling_manifest.append(
                {
                    "episode_idx": episode_idx,
                    "frame_idx": frame_idx,
                    "subtask_index": meta.get("subtask_index"),
                    "atomic_skill": meta.get("atomic_skill", ""),
                    "subtask": meta.get("subtask", ""),
                    "timestamp": meta.get("timestamp"),
                }
            )

    def _save_sampling_manifest(self) -> None:
        """Persist the sampling manifest as parquet under the stage output dir."""
        if not self.sampling_manifest:
            return
        import pandas as pd

        manifest_path = self.output_dir / "sampling_manifest.parquet"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.sampling_manifest).to_parquet(manifest_path, index=False)
        logger.info(
            "Saved sampling manifest (%d frames) to %s",
            len(self.sampling_manifest),
            manifest_path,
        )

    def _process_frame(
        self, episode_idx: int, frame_idx: int, queries: list[str], episode: dict
    ) -> None:
        """
        Process a single frame through segmentation, depth, and storage.

        Args:
            episode_idx: Episode index
            frame_idx: Frame index
            queries: Object queries from discovery
            episode: Episode data dictionary
        """
        # Check if already annotated (idempotency). Only the products of the
        # stages actually running gate the skip, so a depth-only run is not
        # blocked from ever skipping just because masks were never written.
        if not self.config.storage.overwrite:
            mask_done = (
                self.mask_writer is None
                or self.mask_writer.is_frame_annotated(episode_idx, frame_idx)
            )
            depth_done = self.depth_estimator is None or (
                self.depth_writer is not None
                and all(
                    self.depth_writer.is_frame_annotated(
                        episode_idx, frame_idx, cam_name
                    )
                    for cam_name in self.config.camera_names
                )
            )
            if mask_done and depth_done:
                return  # Frame already annotated for every active stage

        # Get frame image for primary camera
        # TODO: Handle multi-camera properly
        camera_name = self.config.camera_names[0]
        frame = episode["frames"][camera_name][frame_idx]

        # Layer 2: Segmentation
        if self.segmenter:
            masks = self.segmenter.segment_frame(
                frame, queries, asdict(self.config.segmentation)
            )
            if self.mask_writer:
                self.mask_writer.write_masks(
                    episode_idx, frame_idx, masks, asdict(self.config.storage)
                )

        # Layer 3: Depth (independent of segmentation)
        if self.depth_estimator:
            for camera_name in self.config.camera_names:
                frame_cam = episode["frames"][camera_name][frame_idx]
                depth_result = self.depth_estimator.estimate_depth(
                    frame_cam, asdict(self.config.depth)
                )
                if depth_result and self.depth_writer:
                    self.depth_writer.write_depth(
                        episode_idx,
                        frame_idx,
                        depth_result,
                        camera_name,
                        asdict(self.config.storage),
                    )

    def _log_summary(self) -> None:
        """Log execution summary statistics."""
        duration = self.stats["end_time"] - self.stats["start_time"]
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total time: {duration:.2f}s")
        logger.info(
            f"Episodes: {self.stats['processed_episodes']}/"
            f"{self.stats['total_episodes']} processed"
        )
        logger.info(f"          {self.stats['skipped_episodes']} skipped (checkpoint)")
        logger.info(f"          {self.stats['failed_episodes']} failed")

        if not self.config.dry_run:
            logger.info(
                f"Frames:   {self.stats['processed_frames']}/"
                f"{self.stats['total_frames']} processed"
            )
            failed_frame_count = sum(
                len(frames) for frames in self.stats["failed_frames"].values()
            )
            logger.info(f"          {failed_frame_count} failed")

            if failed_frame_count > 0:
                logger.info("\nFailed frames by episode:")
                for ep_idx, frame_list in sorted(
                    self.stats["failed_frames"].items()
                ):
                    logger.info(f"  Episode {ep_idx}: {len(frame_list)} frames")

        if self.config.dry_run:
            logger.info(
                f"\nDiscovery results saved to: "
                f"{self.config.storage.output_dir}/discovery_queries.jsonl"
            )

        logger.info("=" * 60)


def load_config_from_yaml(config_path: Path) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        PipelineConfig instance
    """
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Allow stage-specific configs (e.g. segmentation-only) to omit unrelated sections.
    # Missing sections are filled with dataclass defaults; explicit configs are unaffected.
    config_dict.setdefault("discovery", {})
    config_dict.setdefault("segmentation", {})
    config_dict.setdefault("depth", {})
    config_dict.setdefault("sampling", {})
    config_dict.setdefault("qc", {})

    # Convert string paths to Path objects
    config_dict["dataset_path"] = Path(config_dict["dataset_path"])
    config_dict["storage"]["output_dir"] = Path(config_dict["storage"]["output_dir"])

    if config_dict["storage"].get("segmentation_output_dir"):
        config_dict["storage"]["segmentation_output_dir"] = Path(
            config_dict["storage"]["segmentation_output_dir"]
        )

    if config_dict["storage"].get("depth_output_dir"):
        config_dict["storage"]["depth_output_dir"] = Path(
            config_dict["storage"]["depth_output_dir"]
        )

    if config_dict.get("checkpoint_dir"):
        config_dict["checkpoint_dir"] = Path(config_dict["checkpoint_dir"])

    if config_dict.get("segmentation_checkpoint_dir"):
        config_dict["segmentation_checkpoint_dir"] = Path(
            config_dict["segmentation_checkpoint_dir"]
        )

    if config_dict.get("depth_checkpoint_dir"):
        config_dict["depth_checkpoint_dir"] = Path(config_dict["depth_checkpoint_dir"])

    if config_dict["qc"].get("output_dir"):
        config_dict["qc"]["output_dir"] = Path(config_dict["qc"]["output_dir"])

    if config_dict["discovery"].get("vocab_file"):
        config_dict["discovery"]["vocab_file"] = Path(
            config_dict["discovery"]["vocab_file"]
        )

    if config_dict["discovery"].get("qwen_model_path"):
        config_dict["discovery"]["qwen_model_path"] = Path(
            config_dict["discovery"]["qwen_model_path"]
        )

    if config_dict["discovery"].get("instruction_join_path"):
        config_dict["discovery"]["instruction_join_path"] = Path(
            config_dict["discovery"]["instruction_join_path"]
        )

    if config_dict["segmentation"].get("model_path"):
        config_dict["segmentation"]["model_path"] = Path(
            config_dict["segmentation"]["model_path"]
        )

    if config_dict["depth"].get("model_path"):
        config_dict["depth"]["model_path"] = Path(config_dict["depth"]["model_path"])

    # Build nested config
    from annotation.config import (
        DepthConfig,
        DiscoveryConfig,
        PipelineConfig,
        QCConfig,
        SamplingConfig,
        SegmentationConfig,
        StorageConfig,
    )

    discovery = DiscoveryConfig(**config_dict.pop("discovery"))
    segmentation = SegmentationConfig(**config_dict.pop("segmentation"))
    depth = DepthConfig(**config_dict.pop("depth"))
    sampling = SamplingConfig(**config_dict.pop("sampling", {}))
    storage = StorageConfig(**config_dict.pop("storage"))
    qc = QCConfig(**config_dict.pop("qc"))

    # Default per-stage output dirs to distinct subdirectories so a
    # segmentation-only and a depth-only run never write to the same place.
    if storage.segmentation_output_dir is None:
        storage.segmentation_output_dir = storage.output_dir / "segmentation"
    if storage.depth_output_dir is None:
        storage.depth_output_dir = storage.output_dir / "depth_stage"

    return PipelineConfig(
        discovery=discovery,
        segmentation=segmentation,
        depth=depth,
        sampling=sampling,
        storage=storage,
        qc=qc,
        **config_dict,
    )
