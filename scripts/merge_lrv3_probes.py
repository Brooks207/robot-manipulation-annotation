"""
Merge all_features.npz from multiple lerobotv3.0 sub-datasets into a single
pooled probe, then run episode-level GroupKFold classification.

Usage (run locally after bringing back server outputs):
    python scripts/merge_lrv3_probes.py \
        --inputs outputs/lrv3_book_probe/all_features.npz \
                 outputs/lrv3_cup_probe/all_features.npz \
                 outputs/lrv3_coke_probe/all_features.npz \
        --out outputs/lrv3_merged_probe

Each all_features.npz is expected to have arrays:
    X_rgb       (N, 9)
    X_depth     (N, 27)  — 27-dim depth_rich features (DEPTH_RICH_DIM=27)
    y           (N,)     — string category labels
    episode_idx (N,)     — episode index within that sub-dataset
"""

import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EPISODE_OFFSET = 10_000  # gap between sub-datasets


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def merge(input_paths):
    all_rgb, all_depth, all_y, all_groups = [], [], [], []
    for i, p in enumerate(input_paths):
        d = load_npz(p)
        n = len(d["y"])
        all_rgb.append(d["X_rgb"])
        all_depth.append(d["X_depth"])
        all_y.append(d["y"])
        # offset episode_idx so sub-datasets never share a group id
        all_groups.append(d["episode_idx"] + i * EPISODE_OFFSET)
        print(f"  [{i}] {Path(p).parent.parent.name}: {n} instances, "
              f"{np.unique(d['y']).tolist()}")
    return (
        np.concatenate(all_rgb),
        np.concatenate(all_depth),
        np.concatenate(all_y),
        np.concatenate(all_groups),
    )


def run_probe(X, y, groups, label):
    n_eps = len(np.unique(groups))
    cv = LeaveOneGroupOut() if n_eps <= 10 else GroupKFold(n_splits=min(10, n_eps))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    scores = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="balanced_accuracy")
    print(f"  {label:30s}: {scores.mean():.3f} ± {scores.std():.3f}  "
          f"(n_folds={len(scores)}, n_eps={n_eps})")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/lrv3_merged_probe")
    args = ap.parse_args()

    print("=== lerobotv3.0 cross-embodiment merged probe ===")
    print(f"Loading {len(args.inputs)} sub-dataset probes...")
    X_rgb, X_depth, y, groups = merge(args.inputs)

    X_depth_rgb = np.concatenate([X_rgb, X_depth], axis=1)

    print(f"\nPooled: {len(y)} instances, {len(np.unique(y))} categories, "
          f"{len(np.unique(groups))} unique episode groups")
    print(f"Categories: {sorted(np.unique(y).tolist())}\n")

    print("Running episode-level GroupKFold probes (balanced accuracy)...")
    print("(depth = RELATIVE — intrinsics unknown for real-robot camera)\n")
    s_rgb   = run_probe(X_rgb,       y, groups, f"rgb_only ({X_rgb.shape[1]}-dim)")
    s_depth = run_probe(X_depth,     y, groups, f"depth_only ({X_depth.shape[1]}-dim)")
    s_both  = run_probe(X_depth_rgb, y, groups, f"rgb + depth_rich ({X_depth_rgb.shape[1]}-dim)")

    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes
    delta_h1 = s_both.mean() - s_rgb.mean()
    delta_h2 = s_depth.mean() - chance
    print(f"\n  Random-chance baseline (1/{n_classes}): {chance:.3f}")
    print(f"  H2 (depth_only vs chance):  {delta_h2:+.3f} balanced acc above chance")
    print(f"  H1 (rgb+depth vs rgb_only): {delta_h1:+.3f} balanced acc")
    print("  Note: depth is RELATIVE (no metric scale). Signal reflects")
    print("  geometric shape differences, not absolute depth values.\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "merged_features.npz",
             X_rgb=X_rgb, X_depth=X_depth, y=y, episode_groups=groups)
    print(f"Saved merged features → {out}/merged_features.npz")


if __name__ == "__main__":
    main()
