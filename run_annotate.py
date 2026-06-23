#!/usr/bin/env python3
"""Run full annotation pipeline."""

import argparse
import logging
import sys
from pathlib import Path

from pipeline import AnnotationPipeline, load_config_from_yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from annotation.depth.depth_anything3 import DepthAnything3Estimator
from annotation.depth.mock_depth_estimator import MockDepthEstimator
from annotation.discovery.factory import create_discoverer
from annotation.qc.visualize import visualize_annotations
from annotation.segmentation.mock_segmenter import MockSegmenter
from annotation.segmentation.sam3 import SAM3Segmenter
from annotation.storage.depth_writer import PNG16DepthWriter
from annotation.storage.mask_writer import ParquetMaskWriter


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Main entry point for full annotation."""
    parser = argparse.ArgumentParser(description="Run full annotation pipeline")
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock segmenter and depth estimator (for testing)",
    )
    parser.add_argument(
        "--stage",
        choices=["segmentation", "depth", "both"],
        default=None,
        help="Which stage to run. Overrides config.stage. Only the selected "
        "stage's model is loaded (segmentation=SAM3, depth=DA3).",
    )
    args = parser.parse_args()

    # Load config
    config = load_config_from_yaml(args.config)

    # CLI --stage overrides config.stage
    if args.stage is not None:
        config.stage = args.stage
    run_segmentation = config.stage in ("segmentation", "both")
    run_depth = config.stage in ("depth", "both")

    # Setup logging
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)

    # Stage-aware output dir: each stage writes to its own directory so two
    # operators can run segmentation and depth independently without clobbering.
    stage_output_dir = config.storage.output_dir
    if config.stage == "segmentation" and config.storage.segmentation_output_dir:
        stage_output_dir = config.storage.segmentation_output_dir
    elif config.stage == "depth" and config.storage.depth_output_dir:
        stage_output_dir = config.storage.depth_output_dir

    logger.info("=" * 60)
    logger.info("ANNOTATION PIPELINE - FULL MODE")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Dataset: {config.dataset_path}")
    logger.info(f"Cameras: {config.camera_names}")
    logger.info(f"Stage: {config.stage} (segmentation={run_segmentation}, depth={run_depth})")
    logger.info(f"Output: {stage_output_dir}")
    logger.info(f"Use mock models: {args.use_mock}")
    logger.info("=" * 60)

    discoverer = create_discoverer(config.discovery.__dict__)

    # Create segmenter only when the segmentation stage runs, so a depth-only
    # run never loads SAM3 (no wasted VRAM / download).
    segmenter = None
    if run_segmentation:
        if args.use_mock or config.segmentation.model_path is None:
            logger.info("Using MockSegmenter")
            segmenter = MockSegmenter()
        else:
            logger.info("Using SAM3Segmenter")
            segmenter = SAM3Segmenter(
                model_path=config.segmentation.model_path,
                config=config.segmentation.__dict__,
            )

    # Create depth estimator only when the depth stage runs, so a
    # segmentation-only run never loads DA3.
    depth_estimator = None
    if run_depth:
        if args.use_mock or config.depth.model_path is None:
            logger.info("Using MockDepthEstimator")
            depth_estimator = MockDepthEstimator()
        else:
            logger.info("Using DepthAnything3Estimator")
            depth_estimator = DepthAnything3Estimator(
                model_path=config.depth.model_path, config=config.depth.__dict__
            )

    # Create writers (only for active stages)
    mask_writer = ParquetMaskWriter(output_dir=stage_output_dir) if run_segmentation else None
    depth_writer = PNG16DepthWriter(output_dir=stage_output_dir) if run_depth else None

    # Create pipeline
    pipeline = AnnotationPipeline(
        config=config,
        discoverer=discoverer,
        segmenter=segmenter,
        depth_estimator=depth_estimator,
        mask_writer=mask_writer,
        depth_writer=depth_writer,
    )

    # Run annotation
    try:
        stats = pipeline.run()

        # Finalize storage
        if mask_writer is not None:
            mask_writer.finalize()

        # Run QC if enabled. Paths point at this stage's output dir; the QC
        # routine degrades gracefully when only masks or only depth exist.
        if config.qc.enabled:
            logger.info("Running QC visualization...")
            qc_output_dir = (
                config.qc.output_dir
                if config.qc.output_dir
                else stage_output_dir / "qc"
            )

            dataset = pipeline.dataset
            if dataset is None:
                raise RuntimeError("Pipeline dataset unavailable for QC")

            visualize_annotations(
                dataset=dataset,
                mask_parquet_path=stage_output_dir / "masks.parquet",
                depth_base_dir=stage_output_dir / "depth",
                output_dir=qc_output_dir,
                camera_name=config.camera_names[0],
                num_frames_per_episode=config.qc.num_frames_per_episode,
                colormap=config.qc.colormap,
            )
            logger.info(f"QC visualizations saved to: {qc_output_dir}")

        logger.info("=" * 60)
        logger.info("ANNOTATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Annotation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
