#!/usr/bin/env python3
"""Iteration 2 of the representation probe: richer, shape-aware depth features.

`run_representation_probe.py` found only a modest, fold-unstable RGB-vs-RGB+depth
gap (+1.5pp), and the specific hypothesis that depth would disambiguate a visually
similar but geometrically different pair ("bottom drawer" vs "cabinet") was NOT
supported: confusion dropped only 17 -> 16 out of ~73 instances of that pair.

The likely cause: the v1 depth feature (mean/std/min/max -- 4 numbers) is a pure
aggregate that throws away spatial structure. A drawer's front face and a flat
cabinet face can have very similar *average* depth and spread while having very
different *shape* (a protruding/non-planar surface vs. a flat one).

This script re-extracts depth features from the same masks with three additions:
  - a normalized depth histogram (captures distribution shape, e.g. bimodality)
  - depth-gradient magnitude statistics (captures edges / surface roughness)
  - planar-fit residual (RMS deviation from the best-fit plane -- directly tests
    "is this a flat surface", the actual shape cue that should separate a flush
    cabinet face from a protruding drawer front)

Runs entirely LOCALLY: reuses the already-extracted RGB features from
`probe_features.npz` (no need to re-fetch video frames from the server) and reads
the depth PNGs already brought back as portfolio evidence.

Usage:
    python run_representation_probe_richer_depth.py \
        --features outputs/representation_probe/probe_features.npz \
        --masks outputs/libero_smoke/segmentation/masks.parquet \
        --depth-dir outputs/libero_smoke/depth_stage/depth \
        --camera-name observation.images.image \
        --out outputs/representation_probe_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from run_representation_probe import decode_mask, load_depth_meters  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEPTH_RICH_DIM = 27  # 5 + 2 + 8 + 6 + 4 + 2


def depth_features_rich(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """27-dim geometric depth features within the masked instance region.

    Group 1 — Basic distribution (5): mean, std, min, max, median
    Group 2 — RANSAC planarity (2): residual_rms, inlier_ratio
    Group 3 — Depth-HOG (8): gradient orientation histogram, magnitude-weighted
    Group 4 — Distribution shape (6): iqr, pct10, pct90, skewness, kurtosis, entropy
    Group 5 — Quadrant spatial means (4): TL, TR, BL, BR mean depth
    Group 6 — Laplacian curvature (2): mean |∇²d|, std ∇²d within mask
    """
    if depth_m.shape != mask.shape:
        depth_m = np.array(
            Image.fromarray(depth_m).resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)
        )
    depth_m = depth_m.astype(np.float64)
    values = depth_m[mask]
    if values.size < 9:
        return np.zeros(DEPTH_RICH_DIM, dtype=np.float32)

    # Group 1 — Basic distribution (5-dim)
    g1 = np.array([values.mean(), values.std(), values.min(), values.max(), np.median(values)])

    # Group 2 — RANSAC planarity (2-dim)
    ys, xs = np.where(mask)
    g2 = _ransac_planarity(xs.astype(np.float64), ys.astype(np.float64), values)

    # Group 3 — Depth gradient orientation histogram (8-dim)
    g3 = _depth_hog(depth_m, mask)

    # Group 4 — Distribution shape (6-dim)
    g4 = _distribution_shape(values)

    # Group 5 — Quadrant spatial means (4-dim)
    g5 = _quadrant_means(depth_m, mask)

    # Group 6 — Laplacian curvature (2-dim)
    g6 = _laplacian_features(depth_m, mask)

    return np.concatenate([g1, g2, g3, g4, g5, g6]).astype(np.float32)


def _ransac_planarity(
    xs: np.ndarray,
    ys: np.ndarray,
    values: np.ndarray,
    n_trials: int = 100,
) -> np.ndarray:
    """RANSAC plane fit to (x, y, depth) points.

    Threshold = 5% of depth range (scale-invariant for metric and relative depth).
    Returns (residual_rms_of_inliers, inlier_ratio).
    """
    n = len(values)
    threshold = max(1e-6, np.ptp(values) * 0.05)
    A_full = np.stack([xs, ys, np.ones(n)], axis=1)

    # Least-squares as baseline
    try:
        coeffs_ls, *_ = np.linalg.lstsq(A_full, values, rcond=None)
        res_ls = np.abs(values - A_full @ coeffs_ls)
        best_inliers = res_ls < threshold
        best_res = res_ls
    except Exception:
        best_inliers = np.ones(n, dtype=bool)
        best_res = np.zeros(n)

    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        idx = rng.choice(n, 3, replace=False)
        A_s = A_full[idx]
        try:
            coeffs, *_ = np.linalg.lstsq(A_s, values[idx], rcond=None)
        except Exception:
            continue
        res = np.abs(values - A_full @ coeffs)
        inliers = res < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_res = res

    inlier_ratio = float(best_inliers.mean())
    inlier_res = best_res[best_inliers] if best_inliers.any() else best_res
    rms = float(np.sqrt(np.mean(inlier_res ** 2)))
    return np.array([rms, inlier_ratio])


def _depth_hog(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """8-bin gradient orientation histogram, magnitude-weighted."""
    gy, gx = np.gradient(depth_m)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)[mask]
    orientation = np.arctan2(gy[mask], gx[mask]) + np.pi  # [0, 2π]
    hog, _ = np.histogram(orientation, bins=8, range=(0.0, 2 * np.pi), weights=magnitude)
    total = hog.sum()
    if total > 0:
        hog = hog / total
    return hog.astype(np.float32)


def _distribution_shape(values: np.ndarray) -> np.ndarray:
    """6-dim: iqr, pct10, pct90, skewness, excess_kurtosis, entropy (32-bin)."""
    pct10, pct25, pct75, pct90 = np.percentile(values, [10, 25, 75, 90])
    iqr = float(pct75 - pct25)

    mu, sigma = values.mean(), values.std()
    if sigma > 1e-10:
        z = (values - mu) / sigma
        skewness = float(np.mean(z ** 3))
        kurtosis = float(np.mean(z ** 4) - 3.0)
    else:
        skewness, kurtosis = 0.0, 0.0

    vmin, vmax = values.min(), values.max()
    if vmax > vmin:
        hist, _ = np.histogram(values, bins=32, range=(vmin, vmax))
        p = hist / (hist.sum() + 1e-10)
        entropy = float(-np.sum(p * np.log(p + 1e-10)))
    else:
        entropy = 0.0

    return np.array([iqr, float(pct10), float(pct90), skewness, kurtosis, entropy])


def _quadrant_means(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """4-dim: mean depth in TL, TR, BL, BR quadrants of the mask bounding box."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros(4)
    cy = (ys.min() + ys.max()) / 2.0
    cx = (xs.min() + xs.max()) / 2.0
    quads = [
        mask & (np.arange(depth_m.shape[0])[:, None] < cy) & (np.arange(depth_m.shape[1])[None, :] < cx),
        mask & (np.arange(depth_m.shape[0])[:, None] < cy) & (np.arange(depth_m.shape[1])[None, :] >= cx),
        mask & (np.arange(depth_m.shape[0])[:, None] >= cy) & (np.arange(depth_m.shape[1])[None, :] < cx),
        mask & (np.arange(depth_m.shape[0])[:, None] >= cy) & (np.arange(depth_m.shape[1])[None, :] >= cx),
    ]
    global_mean = float(depth_m[mask].mean())
    return np.array([
        float(depth_m[q].mean()) if q.any() else global_mean
        for q in quads
    ])


def _laplacian_features(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """2-dim: mean |∇²d| and std of ∇²d within mask (Laplacian curvature)."""
    gy, gx = np.gradient(depth_m)
    gyy, _ = np.gradient(gy)
    _, gxx = np.gradient(gx)
    lap = (gxx + gyy)[mask]
    return np.array([float(np.mean(np.abs(lap))), float(np.std(lap))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True,
                         help="probe_features.npz from run_representation_probe.py")
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--camera-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    saved = np.load(args.features, allow_pickle=True)
    rgb_X, y = saved["rgb_X"], saved["y"]
    keys = list(zip(saved["episode_idx"].tolist(), saved["frame_idx"].tolist(),
                     saved["instance_id"].tolist()))
    logger.info("Loaded %d pre-extracted RGB feature vectors", len(y))

    masks_df = pd.read_parquet(args.masks)
    masks_df = masks_df.set_index(["episode_idx", "frame_idx", "instance_id"])

    depth_basic_X, depth_rich_X = [], []
    depth_cache: dict[tuple[int, int], np.ndarray] = {}
    for episode_idx, frame_idx, instance_id in keys:
        row = masks_df.loc[(episode_idx, frame_idx, instance_id)]
        mask = decode_mask(row)
        cache_key = (episode_idx, frame_idx)
        if cache_key not in depth_cache:
            depth_cache[cache_key] = load_depth_meters(
                args.depth_dir, args.camera_name, episode_idx, frame_idx
            )
        depth_m = depth_cache[cache_key]
        feat = depth_features_rich(depth_m, mask)
        depth_basic_X.append(feat[:4])
        depth_rich_X.append(feat)

    depth_basic_X = np.stack(depth_basic_X)
    depth_rich_X = np.stack(depth_rich_X)

    feature_sets = {
        "rgb_only": rgb_X,
        "rgb_plus_depth_basic": np.concatenate([rgb_X, depth_basic_X], axis=1),
        "rgb_plus_depth_rich": np.concatenate([rgb_X, depth_rich_X], axis=1),
    }

    min_class_count = pd.Series(y).value_counts().min()
    n_splits = max(2, min(5, int(min_class_count)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    # Standardize first: features here span very different scales (RGB 0-255,
    # depth in meters, gradient magnitude, histogram densities) -- an unscaled
    # linear model either fails to converge or lets large-scale features
    # dominate the decision boundary for reasons unrelated to information content.
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    results, confusion_by_name = {}, {}
    for name, X in feature_sets.items():
        scores = cross_val_score(clf, X, y, cv=cv)
        results[name] = {"mean": float(scores.mean()), "std": float(scores.std()),
                          "folds": scores.tolist()}
        oof_pred = cross_val_predict(clf, X, y, cv=cv)
        labels = sorted(set(y))
        confusion_by_name[name] = {
            "labels": labels, "matrix": confusion_matrix(y, oof_pred, labels=labels).tolist()
        }
        logger.info("%s: %.3f +/- %.3f", name, scores.mean(), scores.std())

    pair_lines = []
    labels = confusion_by_name["rgb_only"]["labels"]
    a, b = "bottom drawer", "cabinet"
    if a in labels and b in labels:
        ia, ib = labels.index(a), labels.index(b)
        for name in feature_sets:
            cm = np.array(confusion_by_name[name]["matrix"])
            confused = cm[ia, ib] + cm[ib, ia]
            pair_lines.append(f"- {name}: {a!r} <-> {b!r} confused {confused} times (out-of-fold)")

    report = {
        "n_instances": int(len(y)), "n_splits": n_splits,
        "results": results, "confusion_matrices": confusion_by_name,
    }
    (args.out / "probe_v2_results.json").write_text(json.dumps(report, indent=2))

    base_mean = results["rgb_only"]["mean"]
    md = [
        "# Representation Probe v2: richer (shape-aware) depth features",
        "",
        f"Same {len(y)} mask instances, RGB features reused from v1 "
        "(`probe_features.npz` -- no server round-trip needed). Depth features "
        "re-extracted with histogram + gradient + planarity terms (15-dim) "
        "alongside the original 4-dim mean/std/min/max.",
        "",
        "| Feature set | Accuracy (mean +/- std) | Delta vs RGB-only |",
        "|---|---|---|",
    ]
    for name in feature_sets:
        r = results[name]
        delta = r["mean"] - base_mean
        md.append(f"| {name} | {r['mean']:.3f} +/- {r['std']:.3f} | {delta:+.3f} |")
    md += [
        "",
        "## Hypothesis check: `bottom drawer` vs `cabinet` (visually similar, "
        "geometrically different)",
        "",
        *pair_lines,
        "",
        "## Caveats",
        "",
        f"- Same small-N ({len(y)} instances) and single-dataset limitations as v1.",
        "- Planarity/gradient features assume the mask's depth values are reasonably "
        "complete; very small or heavily occluded masks get noisier shape estimates.",
    ]
    (args.out / "probe_v2_report.md").write_text("\n".join(md))
    logger.info("Wrote %s", args.out / "probe_v2_report.md")


if __name__ == "__main__":
    main()
