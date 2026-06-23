#!/usr/bin/env python3
"""Run annotation pipeline in dry-run mode (discovery only)."""

import argparse
import logging
import sys
from pathlib import Path

from pipeline import AnnotationPipeline, load_config_from_yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from annotation.discovery.factory import create_discoverer


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Main entry point for dry-run."""
    parser = argparse.ArgumentParser(
        description="Run annotation pipeline in dry-run mode (discovery only)"
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    config = load_config_from_yaml(args.config)

    # Ensure dry_run is True
    if not config.dry_run:
        print("WARNING: Config has dry_run=false, forcing to true for this script")
        config.dry_run = True

    # Setup logging
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("ANNOTATION PIPELINE - DRY RUN MODE")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Dataset: {config.dataset_path}")
    logger.info(f"Discovery mode: {config.discovery.mode}")
    logger.info(f"Discovery extractor: {config.discovery.extractor}")
    logger.info(f"Output: {config.storage.output_dir}")
    logger.info("=" * 60)

    discoverer = create_discoverer(config.discovery.__dict__)

    # Create pipeline (no segmenter, depth estimator, or writers in dry-run)
    pipeline = AnnotationPipeline(
        config=config,
        discoverer=discoverer,
        segmenter=None,
        depth_estimator=None,
        mask_writer=None,
        depth_writer=None,
    )

    # Run
    try:
        stats = pipeline.run()
        logger.info("Dry-run completed successfully")

        # Print discovery results summary
        output_file = config.storage.output_dir / "discovery_queries.jsonl"
        if output_file.exists():
            logger.info(f"\nDiscovery results saved to: {output_file}")
            logger.info("Review the queries before running full annotation.")

        sys.exit(0)

    except Exception as e:
        logger.error(f"Dry-run failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
