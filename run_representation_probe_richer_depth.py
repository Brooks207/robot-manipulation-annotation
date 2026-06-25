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

N_HIST_BINS = 8


def depth_features_rich(depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Shape-aware depth features inside the mask. 4 + 8 + 2 + 1 = 15 dims.

    - basic: mean/std/min/max (same as v1, kept for continuity)
    - histogram: 8-bin density histogram over THIS mask's own [min, max] depth
      range -- a per-instance-normalized shape descriptor (distinguishes e.g. a
      bimodal depth distribution -- near edge + far recessed face -- from a
      unimodal flat surface), independent of the object's absolute distance.
    - gradient: mean/std of local depth-gradient magnitude inside the mask --
      near-zero for a flat surface, higher for a protruding/uneven one.
    - planarity: RMS residual of the best-fit plane to the mask's (x, y, depth)
      points -- the most direct test of "is this region flat".
    """
    if depth_m.shape != mask.shape:
        depth_m = np.array(
            Image.fromarray(depth_m).resize((mask.shape[1], mask.shape[0]), Image.BILINEAR)
        )
    values = depth_m[mask]
    if values.size < 4:
        return np.zeros(4 + N_HIST_BINS + 2 + 1, dtype=np.float32)

    basic = np.array([values.mean(), values.std(), values.min(), values.max()],
                      dtype=np.float32)

    vmin, vmax = values.min(), values.max()
    if vmax > vmin:
        hist, _ = np.histogram(values, bins=N_HIST_BINS, range=(vmin, vmax), density=True)
    else:
        hist = np.zeros(N_HIST_BINS)
    hist = hist.astype(np.float32)

    gy, gx = np.gradient(depth_m.astype(np.float64))
    grad_mag = np.sqrt(gx**2 + gy**2)[mask]
    grad_feat = np.array([grad_mag.mean(), grad_mag.std()], dtype=np.float32)

    ys, xs = np.where(mask)
    design = np.stack([xs, ys, np.ones_like(xs, dtype=np.float64)], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, values.astype(np.float64), rcond=None)
    residual = values.astype(np.float64) - design @ coeffs
    planarity = np.array([np.sqrt(np.mean(residual**2))], dtype=np.float32)

    return np.concatenate([basic, hist, grad_feat, planarity])


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
