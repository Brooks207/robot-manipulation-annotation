#!/usr/bin/env python3
"""Rigorous representation probe: episode-level cross-validation.

The prior probes (v1/v2/v3) used StratifiedKFold at the frame level, which
allows frames from the same episode to appear in both train and test. Because
frames in the same episode share the same scene, lighting, and object layout,
this leaks scene-specific memorization into the accuracy estimate -- inflating
all feature sets equally but not measuring generalization.

This script uses GroupKFold(groups=episode_idx) so the test set always contains
*complete episodes* unseen during training. This is the minimum bar for claiming
the features generalize beyond a single recording session.

Requires having run the annotation pipeline on >= 5 distinct episodes that share
the same object categories (so each fold has meaningful train and test classes).
Run after collecting outputs from run_annotate.py on a larger episode set.

Usage:
    python run_representation_probe_rigorous.py \\
        --dataset-path ./data/libero \\
        --masks outputs/libero_large/segmentation/masks.parquet \\
        --depth-dir outputs/libero_large/depth_stage/depth \\
        --camera-name observation.images.image \\
        --probe-cache outputs/libero_large/probe_features.npz \\
        --vla-features outputs/libero_large/vla_features.npz \\
        --out outputs/representation_probe_rigorous
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from annotation.lerobot_v3_dataset import LeRobotV3Dataset  # noqa: E402
from run_representation_probe import (  # noqa: E402
    decode_mask,
    depth_features,
    load_depth_meters,
    rgb_features,
)
from run_representation_probe_richer_depth import (  # noqa: E402
    DEPTH_RICH_DIM,
    depth_features_rich,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_rgb_features(
    dataset_path: Path,
    camera_name: str,
    masks_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Extract per-instance RGB features from video frames.

    Returns (rgb_X, y, meta) where meta contains episode_idx, frame_idx,
    instance_id for each row.
    """
    from collections import defaultdict

    needed: dict[int, list[int]] = defaultdict(list)
    for ep, fr in masks_df[["episode_idx", "frame_idx"]].drop_duplicates().itertuples(index=False):
        needed[int(ep)].append(int(fr))

    frame_cache: dict[tuple[int, int], np.ndarray | None] = {}
    rgb_X, y_list, meta = [], [], []

    for (ep_idx, fr_idx), group in masks_df.groupby(["episode_idx", "frame_idx"]):
        key = (int(ep_idx), int(fr_idx))
        if key not in frame_cache:
            try:
                ds = LeRobotV3Dataset(
                    dataset_path=dataset_path,
                    camera_names=[camera_name],
                    instruction_config={"instruction_source": "none"},
                    episode_indices=[int(ep_idx)],
                    frame_indices=sorted(needed[int(ep_idx)]),
                    load_frames=True,
                )
                ep = ds.get_episode(0)
                for f in needed[int(ep_idx)]:
                    frame_cache[(int(ep_idx), f)] = ep["frames"][camera_name].get(f)
            except Exception as exc:
                logger.warning("Episode %d failed (%s); skipping.", ep_idx, exc)
                for f in needed[int(ep_idx)]:
                    frame_cache[(int(ep_idx), f)] = None

        frame = frame_cache.get(key)
        if frame is None:
            continue

        for _, row in group.iterrows():
            mask = decode_mask(row)
            rgb_X.append(rgb_features(frame, mask))
            y_list.append(row["category"])
            meta.append({
                "episode_idx": int(ep_idx),
                "frame_idx": int(fr_idx),
                "instance_id": int(row["instance_id"]),
            })

    return np.stack(rgb_X), np.array(y_list), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path,
                         help="LeRobot v3 dataset path (needed if no --probe-cache).")
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--camera-name", required=True)
    parser.add_argument("--probe-cache", type=Path,
                         help="Existing probe_features.npz; skips RGB extraction if provided.")
    parser.add_argument("--vla-features", type=Path,
                         help="Optional vla_features.npz from run_vla_feature_extract.py.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-episodes-per-class", type=int, default=3,
                         help="Drop categories that appear in fewer than this many distinct episodes.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    masks_df = pd.read_parquet(args.masks)

    # Load or extract RGB features.
    if args.probe_cache and args.probe_cache.exists():
        logger.info("Loading pre-extracted RGB features from %s", args.probe_cache)
        cache = np.load(args.probe_cache, allow_pickle=True)
        rgb_X = cache["rgb_X"]
        y = cache["y"]
        episode_idx = cache["episode_idx"]
        frame_idx = cache["frame_idx"]
        instance_id = cache["instance_id"]
        meta = [
            {"episode_idx": int(e), "frame_idx": int(f), "instance_id": int(i)}
            for e, f, i in zip(episode_idx, frame_idx, instance_id)
        ]
    else:
        if args.dataset_path is None:
            raise ValueError("--dataset-path required when no --probe-cache is given.")
        logger.info("Extracting RGB features from video frames...")
        rgb_X, y, meta = extract_rgb_features(args.dataset_path, args.camera_name, masks_df)
        episode_idx = np.array([m["episode_idx"] for m in meta])
        frame_idx = np.array([m["frame_idx"] for m in meta])
        instance_id = np.array([m["instance_id"] for m in meta])
        np.savez(
            args.out / "probe_features.npz",
            rgb_X=rgb_X, y=y,
            episode_idx=episode_idx, frame_idx=frame_idx, instance_id=instance_id,
        )
        # Note: depth_rich_X is extracted later (after filtering) and saved separately.

    n_total = len(y)
    logger.info("Total instances before filtering: %d across %d categories, %d episodes",
                n_total, len(set(y)), len(set(episode_idx)))

    # Drop categories that do not appear in enough distinct episodes.
    # A category that lives in only 1-2 episodes cannot be evaluated by
    # episode-level CV -- any test fold that contains its only episode
    # will have nothing to train on for that class.
    cat_episode_counts: dict[str, set[int]] = {}
    for cat, ep in zip(y, episode_idx):
        cat_episode_counts.setdefault(cat, set()).add(int(ep))

    keep = {cat for cat, eps in cat_episode_counts.items()
            if len(eps) >= args.min_episodes_per_class}
    dropped = {cat: len(eps) for cat, eps in cat_episode_counts.items() if cat not in keep}
    if dropped:
        logger.warning(
            "Dropping %d categories with < %d distinct episodes: %s",
            len(dropped), args.min_episodes_per_class, dropped,
        )

    valid = np.array([cat in keep for cat in y])
    rgb_X = rgb_X[valid]
    y = y[valid]
    episode_idx = episode_idx[valid]
    meta = [m for m, v in zip(meta, valid) if v]
    logger.info("After episode-coverage filter: %d instances, %d categories, %d episodes",
                len(y), len(set(y)), len(set(episode_idx)))

    # Re-extract depth features for the filtered set.
    masks_indexed = masks_df.set_index(["episode_idx", "frame_idx", "instance_id"])
    depth_cache: dict[tuple[int, int], np.ndarray | None] = {}
    depth_rich_X = []
    for m in meta:
        ep, fr, inst = m["episode_idx"], m["frame_idx"], m["instance_id"]
        row = masks_indexed.loc[(ep, fr, inst)]
        mask = decode_mask(row)
        ck = (ep, fr)
        if ck not in depth_cache:
            depth_cache[ck] = load_depth_meters(args.depth_dir, args.camera_name, ep, fr)
        dm = depth_cache[ck]
        depth_rich_X.append(
            depth_features_rich(dm, mask) if dm is not None else np.zeros(DEPTH_RICH_DIM, dtype=np.float32)
        )
    depth_rich_X = np.stack(depth_rich_X)

    # Save filtered features (RGB + depth) for local ablations and cross-dataset merge.
    np.savez(
        args.out / "all_features.npz",
        X_rgb=rgb_X, X_depth=depth_rich_X, y=y,
        episode_idx=episode_idx,
        frame_idx=np.array([m["frame_idx"] for m in meta]),
        instance_id=np.array([m["instance_id"] for m in meta]),
    )
    logger.info("Saved filtered features → %s/all_features.npz", args.out)

    feature_sets: dict[str, np.ndarray] = {
        "rgb_only": rgb_X,
        "depth_only": depth_rich_X,
        "rgb_plus_depth_rich": np.concatenate([rgb_X, depth_rich_X], axis=1),
    }

    # Optionally add SigLIP features.
    if args.vla_features and args.vla_features.exists():
        vla = np.load(args.vla_features, allow_pickle=True)
        siglip_all = vla["siglip_X"]
        # Align: vla_features is indexed the same way as the original probe_features.npz.
        # Build a lookup by (episode_idx, frame_idx, instance_id).
        vla_ep = vla["episode_idx"]
        vla_fr = vla["frame_idx"]
        vla_inst = vla["instance_id"]
        vla_lookup = {
            (int(vla_ep[i]), int(vla_fr[i]), int(vla_inst[i])): siglip_all[i]
            for i in range(len(vla_ep))
        }
        siglip_X = np.stack([
            vla_lookup.get((m["episode_idx"], m["frame_idx"], m["instance_id"]),
                           np.full(siglip_all.shape[1], np.nan))
            for m in meta
        ])
        n_missing = int(np.isnan(siglip_X).any(axis=1).sum())
        if n_missing:
            logger.warning("%d SigLIP features missing after filtering; those rows excluded.", n_missing)
        if not np.isnan(siglip_X).any():
            feature_sets["siglip_only"] = siglip_X
            feature_sets["siglip_plus_depth_rich"] = np.concatenate([siglip_X, depth_rich_X], axis=1)

    # Episode-level cross-validation.
    # GroupKFold ensures no episode appears in both train and test.
    unique_episodes = sorted(set(episode_idx))
    n_episodes = len(unique_episodes)
    # Use LeaveOneGroupOut when <= 10 episodes (gives maximum test granularity),
    # else GroupKFold with min(10, n_episodes) splits.
    if n_episodes <= 10:
        cv = LeaveOneGroupOut()
        cv_name = f"LeaveOneEpisodeOut (n_episodes={n_episodes})"
    else:
        n_splits = min(10, n_episodes)
        cv = GroupKFold(n_splits=n_splits)
        cv_name = f"GroupKFold(n_splits={n_splits}, n_episodes={n_episodes})"

    logger.info("CV strategy: %s", cv_name)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    results: dict[str, dict] = {}
    confusion_by_name: dict[str, dict] = {}
    for name, X in feature_sets.items():
        scores = cross_val_score(clf, X, y, groups=episode_idx, cv=cv,
                                 scoring="balanced_accuracy")
        results[name] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "folds": scores.tolist(),
            "n_features": int(X.shape[1]),
        }
        oof_pred = cross_val_predict(clf, X, y, groups=episode_idx, cv=cv)
        labels = sorted(set(y))
        confusion_by_name[name] = {
            "labels": labels,
            "matrix": confusion_matrix(y, oof_pred, labels=labels).tolist(),
        }
        logger.info("%s (%d-dim): %.3f +/- %.3f  [%s]",
                    name, X.shape[1], scores.mean(), scores.std(),
                    ", ".join(f"{s:.3f}" for s in scores.tolist()))

    # Hypothesis check: drawer ↔ cabinet.
    pair_lines: list[str] = []
    labels = confusion_by_name["rgb_only"]["labels"]
    a, b = "bottom drawer", "cabinet"
    if a in labels and b in labels:
        ia, ib = labels.index(a), labels.index(b)
        for name in feature_sets:
            cm = np.array(confusion_by_name[name]["matrix"])
            confused = int(cm[ia, ib] + cm[ib, ia])
            pair_lines.append(f"- {name}: {confused} out-of-episode confusions")
    else:
        pair_lines.append("(drawer/cabinet pair not present after episode-coverage filtering)")

    report = {
        "n_instances": int(len(y)),
        "n_episodes": n_episodes,
        "episode_ids": [int(e) for e in unique_episodes],
        "cv_strategy": cv_name,
        "category_counts": Counter(y.tolist()),
        "category_episode_counts": {k: len(v) for k, v in cat_episode_counts.items() if k in keep},
        "results": results,
        "confusion_matrices": confusion_by_name,
    }
    (args.out / "probe_rigorous_results.json").write_text(json.dumps(report, indent=2))

    base = results["rgb_only"]["mean"]
    md = [
        "# Representation Probe (Rigorous): Episode-Level Cross-Validation",
        "",
        f"**CV strategy:** {cv_name}. Test sets contain complete episodes unseen "
        "during training — no within-scene memorization.",
        "",
        f"**N:** {len(y)} instances, {n_episodes} episodes, {len(set(y))} categories "
        f"(categories with < {args.min_episodes_per_class} distinct episodes dropped).",
        "",
        "| Feature set | Dims | Balanced Accuracy (mean ± std) | Δ vs rgb_only |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        delta = r["mean"] - base
        md.append(
            f"| {name} | {r['n_features']} | {r['mean']:.3f} ± {r['std']:.3f} | {delta:+.3f} |"
        )
    md += [
        "",
        "## Hypothesis check: `bottom drawer` vs `cabinet`",
        "",
        "(Out-of-*episode* confusion — the test episode's objects were never "
        "seen in any training fold.)",
        "",
        *pair_lines,
        "",
        "## Category distribution",
        "",
        "| Category | Instances | Distinct episodes |",
        "|---|---|---|",
    ]
    for cat in sorted(keep):
        n = int((y == cat).sum())
        n_ep = len(cat_episode_counts[cat])
        md.append(f"| {cat} | {n} | {n_ep} |")
    md += [
        "",
        "## Caveats",
        "",
        "- Single dataset (LIBERO), single pipeline run — directional evidence.",
        "- Categories with narrow episode coverage dropped; results reflect "
        "the subset of categories present across multiple distinct episodes.",
        "- Features are hand-engineered pixel/depth statistics and SigLIP "
        "pooler output — not task-conditioned learned representations.",
    ]
    (args.out / "probe_rigorous_report.md").write_text("\n".join(md))
    logger.info("Wrote %s", args.out / "probe_rigorous_report.md")


if __name__ == "__main__":
    main()
