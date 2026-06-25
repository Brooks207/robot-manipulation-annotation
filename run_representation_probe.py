#!/usr/bin/env python3
"""Linear probing: does adding depth improve a simple object-category classifier?

For every annotated mask instance (episode_idx, frame_idx, instance_id, category):
  - RGB feature  = per-channel mean/std/median of pixels inside the mask (6 dims)
  - depth feature = mean/std/min/max of metric depth inside the mask (4 dims)

A *linear* classifier (logistic regression) is trained on RGB-only vs RGB+depth
features and evaluated with stratified k-fold cross-validation, so the comparison
measures whether the *features themselves* are more separable -- not whether a
stronger classifier can paper over weak features. This mirrors the standard
"frozen encoder + linear probe" protocol used in representation-learning papers
(e.g. DeFM, arXiv:2601.18923; DepthCues, arXiv:2411.17385).

Must run where the original LeRobot v3 dataset (RGB video frames) is available --
typically the server, not a laptop that only has masks.parquet/depth PNGs brought
back as portfolio evidence.

Usage:
    python run_representation_probe.py \
        --dataset-path ./data/libero \
        --masks outputs/libero_smoke/segmentation/masks.parquet \
        --depth-dir outputs/libero_smoke/depth_stage/depth \
        --camera-name observation.images.image \
        --out outputs/representation_probe
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).parent))
from annotation.lerobot_v3_dataset import LeRobotV3Dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def decode_mask(row: pd.Series) -> np.ndarray:
    """Decode a COCO-RLE mask row back into a boolean (H, W) array."""
    rle = {
        "counts": row["rle_counts"].encode("utf-8")
        if isinstance(row["rle_counts"], str)
        else row["rle_counts"],
        "size": list(row["rle_size"]),
    }
    return mask_util.decode(rle).astype(bool)


def rgb_features(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel mean/std/median of RGB pixels inside the mask. 9 dims.

    Deliberately simple appearance statistics, not a learned embedding (e.g. SAM3
    vision-encoder features) -- this keeps the probe transparent and reproducible
    without re-running SAM3 with feature hooks. A reasonable v1; swapping in real
    encoder embeddings is the natural next iteration.
    """
    pixels = frame[mask]  # (N, 3)
    if pixels.size == 0:
        return np.zeros(9, dtype=np.float32)
    return np.concatenate(
        [pixels.mean(axis=0), pixels.std(axis=0), np.median(pixels, axis=0)]
    ).astype(np.float32)


def depth_features(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean/std/min/max of metric depth (meters) inside the mask. 4 dims.

    DA3 processes frames at its own internal resolution (process_res, e.g. 504x504 --
    see README "Key technical note"), which generally differs from the mask/RGB frame
    resolution (e.g. 512x512). Resize depth to the mask's resolution before indexing so
    every pixel of `mask` has a corresponding depth value.
    """
    if depth_m.shape != mask.shape:
        depth_m = np.array(
            Image.fromarray(depth_m).resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)
        )
    values = depth_m[mask]
    if values.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array(
        [values.mean(), values.std(), values.min(), values.max()], dtype=np.float32
    )


def load_depth_meters(depth_dir: Path, camera_name: str, episode_idx: int, frame_idx: int) -> np.ndarray | None:
    """Load a depth PNG + JSON and return metric depth in meters, or None if missing."""
    base = depth_dir / camera_name / f"episode_{episode_idx:06d}" / f"frame_{frame_idx:06d}"
    png_path, json_path = base.with_suffix(".png"), base.with_suffix(".json")
    if not png_path.exists() or not json_path.exists():
        return None
    meta = json.loads(json_path.read_text())
    if meta.get("depth_type") != "metric":
        logger.warning("Frame %s/%d: depth_type=%s, not metric -- skipping", camera_name, frame_idx, meta.get("depth_type"))
        return None
    raw = np.array(Image.open(png_path), dtype=np.float32)
    # Per QUICKSTART/README convention: uint16 PNG encodes millimeters.
    return raw / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--camera-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-class-count", type=int, default=3,
                         help="Drop categories with fewer than this many instances.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    masks_df = pd.read_parquet(args.masks)
    logger.info("Loaded %d mask instances, %d categories", len(masks_df),
                masks_df["category"].nunique())

    # Drop categories too rare for any reasonable cross-validation.
    counts = masks_df["category"].value_counts()
    keep_categories = counts[counts >= args.min_class_count].index
    dropped = counts[counts < args.min_class_count]
    if len(dropped):
        logger.warning("Dropping categories with <%d instances: %s",
                        args.min_class_count, dropped.to_dict())
    masks_df = masks_df[masks_df["category"].isin(keep_categories)].reset_index(drop=True)

    # Only fetch the exact frames each episode's masks actually reference -- avoids
    # decoding every frame in (potentially 1000+-frame) episodes via ffmpeg, and
    # each episode's sampled frame_idx set differs (length-dependent), so one
    # global frame_indices list across all episodes would be wrong.
    needed_frames: dict[int, list[int]] = defaultdict(list)
    for episode_idx, frame_idx in masks_df[["episode_idx", "frame_idx"]].drop_duplicates().itertuples(index=False):
        needed_frames[episode_idx].append(frame_idx)

    frame_cache: dict[tuple[int, int], np.ndarray] = {}
    rgb_X, depthcat_X, y, skipped = [], [], [], 0

    for (episode_idx, frame_idx), group in masks_df.groupby(["episode_idx", "frame_idx"]):
        cache_key = (episode_idx, frame_idx)
        if cache_key not in frame_cache:
            try:
                episode_dataset = LeRobotV3Dataset(
                    dataset_path=args.dataset_path,
                    camera_names=[args.camera_name],
                    instruction_config={"instruction_source": "none"},
                    episode_indices=[episode_idx],
                    frame_indices=sorted(needed_frames[episode_idx]),
                    load_frames=True,
                )
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Episode %d not loadable (%s); skipping its %d masks",
                                episode_idx, exc, len(masks_df[masks_df["episode_idx"] == episode_idx]))
                for f in needed_frames[episode_idx]:
                    frame_cache[(episode_idx, f)] = None
                frame = None
            else:
                episode = episode_dataset.get_episode(0)
                for f in needed_frames[episode_idx]:
                    frame_cache[(episode_idx, f)] = episode["frames"][args.camera_name].get(f)
            frame = frame_cache.get(cache_key)
        else:
            frame = frame_cache[cache_key]
        if frame is None:
            logger.warning("No RGB frame for episode %d frame %d; skipping its %d masks",
                            episode_idx, frame_idx, len(group))
            skipped += len(group)
            continue

        depth_m = load_depth_meters(args.depth_dir, args.camera_name, episode_idx, frame_idx)
        if depth_m is None:
            logger.warning("No depth for episode %d frame %d; skipping its %d masks",
                            episode_idx, frame_idx, len(group))
            skipped += len(group)
            continue

        for _, row in group.iterrows():
            mask = decode_mask(row)
            rgb_X.append(rgb_features(frame, mask))
            depthcat_X.append(depth_features(depth_m, mask))
            y.append(row["category"])

    if not y:
        logger.error("No usable mask instances (frame/depth alignment failed). Aborting.")
        sys.exit(1)

    rgb_X = np.stack(rgb_X)
    depth_X = np.stack(depthcat_X)
    rgbd_X = np.concatenate([rgb_X, depth_X], axis=1)
    y = np.array(y)

    logger.info("Usable instances: %d (skipped %d) across %d categories",
                len(y), skipped, len(set(y)))

    min_class_count = pd.Series(y).value_counts().min()
    n_splits = max(2, min(5, int(min_class_count)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=2000)

    results = {}
    for name, X in [("rgb_only", rgb_X), ("rgb_plus_depth", rgbd_X)]:
        scores = cross_val_score(clf, X, y, cv=cv)
        results[name] = {"mean": float(scores.mean()), "std": float(scores.std()),
                          "folds": scores.tolist()}
        logger.info("%s: %.3f +/- %.3f (n_splits=%d)", name, scores.mean(), scores.std(), n_splits)

    report = {
        "n_instances": int(len(y)),
        "n_skipped": int(skipped),
        "n_splits": n_splits,
        "category_counts": pd.Series(y).value_counts().to_dict(),
        "results": results,
    }
    (args.out / "probe_results.json").write_text(json.dumps(report, indent=2))

    delta = results["rgb_plus_depth"]["mean"] - results["rgb_only"]["mean"]
    md = [
        "# Representation Probe: RGB-only vs RGB+depth (linear classifier)",
        "",
        f"Linear logistic-regression probe, {n_splits}-fold stratified cross-validation, "
        f"on {len(y)} mask instances ({skipped} skipped for missing frame/depth alignment).",
        "",
        "| Feature set | Accuracy (mean +/- std) |",
        "|---|---|",
        f"| RGB only (9-dim color stats) | {results['rgb_only']['mean']:.3f} +/- {results['rgb_only']['std']:.3f} |",
        f"| RGB + depth (13-dim) | {results['rgb_plus_depth']['mean']:.3f} +/- {results['rgb_plus_depth']['std']:.3f} |",
        "",
        f"**Delta: {delta:+.3f}** ({'depth helped' if delta > 0.01 else 'no clear benefit from depth' if abs(delta) <= 0.01 else 'depth hurt'}).",
        "",
        "## Category counts (after dropping rare categories)",
        "",
        "```",
        pd.Series(y).value_counts().to_string(),
        "```",
        "",
        "## Caveats",
        "",
        f"- Small N ({len(y)} instances) -- single-dataset, single-run result, not a "
        "statistically powered claim.",
        "- Features are simple per-mask RGB/depth pixel statistics, not learned encoder "
        "embeddings -- a reasonable first probe, not the final word on representation quality.",
        "- Classifier is intentionally linear (logistic regression): the point is to test "
        "whether the *features* are separable, not whether a stronger classifier can "
        "compensate for weak features.",
    ]
    (args.out / "probe_report.md").write_text("\n".join(md))
    logger.info("Wrote %s and %s", args.out / "probe_results.json", args.out / "probe_report.md")


if __name__ == "__main__":
    main()
