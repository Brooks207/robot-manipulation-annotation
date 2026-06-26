#!/usr/bin/env python3
"""Representation probe v3: compare hand-crafted features vs frozen SigLIP encoder.

Extends the v2 probe (run_representation_probe_richer_depth.py) by adding two more
feature sets using frozen SigLIP-SO400M features (the same vision encoder used by
OpenVLA) extracted per-instance by run_vla_feature_extract.py.

Feature sets compared:
  rgb_only              (9-dim)    hand-crafted per-channel color stats
  rgb_plus_depth_rich   (24-dim)   color stats + shape-aware depth (histogram/gradient/planarity)
  siglip_only           (1152-dim) frozen SigLIP pooler_output (instance crop → 384×384)
  siglip_plus_depth     (1167-dim) SigLIP + shape-aware depth

The key question: does a VLA-scale vision encoder already capture the geometric
structure that our hand-crafted depth features encode, or does explicit metric depth
still add signal even on top of a 400M-parameter encoder?

Runs entirely locally -- no video frames needed (SigLIP features from vla_features.npz,
RGB features from probe_features.npz, depth re-extracted from local depth PNGs).

Usage:
    python run_representation_probe_vla.py \\
        --probe-cache    outputs/representation_probe/probe_features.npz \\
        --vla-features   outputs/vla_features.npz \\
        --masks          outputs/libero_smoke/segmentation/masks.parquet \\
        --depth-dir      outputs/libero_smoke/depth_stage/depth \\
        --camera-name    observation.images.image \\
        --out            outputs/representation_probe_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from run_representation_probe import decode_mask, load_depth_meters  # noqa: E402
from run_representation_probe_richer_depth import depth_features_rich  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-cache", type=Path, required=True,
                         help="probe_features.npz from run_representation_probe.py")
    parser.add_argument("--vla-features", type=Path, required=True,
                         help="vla_features.npz from run_vla_feature_extract.py")
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--depth-dir", type=Path, required=True)
    parser.add_argument("--camera-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Load hand-crafted features from v1 cache.
    probe = np.load(args.probe_cache, allow_pickle=True)
    rgb_X = probe["rgb_X"]
    y = probe["y"]
    keys = list(zip(probe["episode_idx"].tolist(), probe["frame_idx"].tolist(),
                     probe["instance_id"].tolist()))
    logger.info("Probe cache: %d instances, %d categories", len(y), len(set(y)))

    # Load SigLIP features; verify alignment.
    vla = np.load(args.vla_features, allow_pickle=True)
    siglip_X = vla["siglip_X"]
    assert len(siglip_X) == len(y), (
        f"vla_features.npz has {len(siglip_X)} rows but probe_features.npz has {len(y)}"
    )
    n_missing = int(np.isnan(siglip_X).any(axis=1).sum())
    if n_missing:
        logger.warning("%d SigLIP feature vectors are NaN (missing instances); "
                       "those rows will be dropped.", n_missing)
    valid = ~np.isnan(siglip_X).any(axis=1)
    if not valid.all():
        rgb_X = rgb_X[valid]
        siglip_X = siglip_X[valid]
        y = y[valid]
        keys = [k for k, v in zip(keys, valid) if v]
        logger.info("After dropping NaN SigLIP rows: %d instances remain.", len(y))

    # Re-extract shape-aware depth features locally from depth PNGs.
    masks_df = pd.read_parquet(args.masks)
    masks_df = masks_df.set_index(["episode_idx", "frame_idx", "instance_id"])

    depth_rich_X = []
    depth_cache: dict[tuple[int, int], np.ndarray | None] = {}
    for episode_idx, frame_idx, instance_id in keys:
        row = masks_df.loc[(episode_idx, frame_idx, instance_id)]
        mask = decode_mask(row)
        ck = (episode_idx, frame_idx)
        if ck not in depth_cache:
            depth_cache[ck] = load_depth_meters(
                args.depth_dir, args.camera_name, episode_idx, frame_idx
            )
        depth_m = depth_cache[ck]
        feat = depth_features_rich(depth_m, mask) if depth_m is not None else np.zeros(15, dtype=np.float32)
        depth_rich_X.append(feat)
    depth_rich_X = np.stack(depth_rich_X)

    feature_sets: dict[str, np.ndarray] = {
        "rgb_only": rgb_X,
        "rgb_plus_depth_rich": np.concatenate([rgb_X, depth_rich_X], axis=1),
        "siglip_only": siglip_X,
        "siglip_plus_depth_rich": np.concatenate([siglip_X, depth_rich_X], axis=1),
    }

    min_class_count = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_class_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    results: dict[str, dict] = {}
    confusion_by_name: dict[str, dict] = {}
    for name, X in feature_sets.items():
        scores = cross_val_score(clf, X, y, cv=cv)
        results[name] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "folds": scores.tolist(),
            "n_features": int(X.shape[1]),
        }
        oof_pred = cross_val_predict(clf, X, y, cv=cv)
        labels = sorted(set(y))
        confusion_by_name[name] = {
            "labels": labels,
            "matrix": confusion_matrix(y, oof_pred, labels=labels).tolist(),
        }
        logger.info("%s (%d-dim): %.3f +/- %.3f", name, X.shape[1], scores.mean(), scores.std())

    # Hypothesis check: drawer ↔ cabinet confusion.
    pair_lines: list[str] = []
    labels = confusion_by_name["rgb_only"]["labels"]
    a, b = "bottom drawer", "cabinet"
    if a in labels and b in labels:
        ia, ib = labels.index(a), labels.index(b)
        for name in feature_sets:
            cm = np.array(confusion_by_name[name]["matrix"])
            confused = int(cm[ia, ib] + cm[ib, ia])
            pair_lines.append(f"- {name}: {confused} confusions (out-of-fold)")
    else:
        pair_lines.append("(pair not found in this run's categories)")

    report = {
        "n_instances": int(len(y)),
        "n_splits": n_splits,
        "results": results,
        "confusion_matrices": confusion_by_name,
    }
    (args.out / "probe_v3_results.json").write_text(json.dumps(report, indent=2))

    base = results["rgb_only"]["mean"]
    md_lines = [
        "# Representation Probe v3: hand-crafted features vs frozen SigLIP encoder",
        "",
        f"Same {len(y)} mask instances as v1/v2. RGB features and metadata from "
        "`probe_features.npz`. SigLIP-SO400M features (pooler_output, 1152-dim) from "
        "`vla_features.npz` (extracted by `run_vla_feature_extract.py` on the server). "
        "Shape-aware depth features (15-dim histogram + gradient + planarity) re-extracted "
        "locally from depth PNGs. All feature sets pass through StandardScaler before "
        "logistic regression.",
        "",
        f"{n_splits}-fold stratified cross-validation, {len(y)} instances, "
        f"{len(set(y))} categories.",
        "",
        "| Feature set | Dims | Accuracy (mean ± std) | Δ vs rgb_only |",
        "|---|---|---|---|",
    ]
    for name, r in results.items():
        delta = r["mean"] - base
        md_lines.append(
            f"| {name} | {r['n_features']} | {r['mean']:.3f} ± {r['std']:.3f} | {delta:+.3f} |"
        )
    md_lines += [
        "",
        "## Hypothesis check: `bottom drawer` vs `cabinet`",
        "",
        "Both are wood/grey furniture — visually similar but geometrically different "
        "(drawer protrudes; cabinet face is flat). Out-of-fold confusion counts:",
        "",
        *pair_lines,
        "",
        "## Interpretation",
        "",
        "- If **siglip_only > rgb_plus_depth_rich**: SigLIP's pre-trained representations "
        "already capture the geometric cues that our hand-crafted depth features encode — "
        "the encoder compresses the relevant shape signal implicitly.",
        "- If **siglip_only ≈ rgb_only**: SigLIP's features, despite 400M parameters, don't "
        "provide the *geometric* signal that explicit metric depth does — supporting the "
        "pipeline's investment in DA3.",
        "- If **siglip_plus_depth_rich > siglip_only**: depth adds signal even on top of a "
        "large pre-trained encoder — the strongest possible argument for metric depth.",
        "",
        "## Caveats",
        "",
        f"- Small N ({len(y)} instances), single dataset — an internal-consistency result.",
        "- SigLIP features extracted from per-instance bbox crops (384×384), not full frames — "
        "closer in spirit to the per-instance hand-crafted features but different from how "
        "OpenVLA ingests the full scene.",
    ]
    (args.out / "probe_v3_report.md").write_text("\n".join(md_lines))
    logger.info("Wrote %s", args.out / "probe_v3_report.md")


if __name__ == "__main__":
    main()
