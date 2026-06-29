"""Sample 150 episodes from BridgeData V2 for Sub-experiment C (H3 test).

Stratified by object geometry category (flat / cylindrical / irregular),
50 episodes per stratum, random seed = 42. Outputs episode indices to
configs/bridge_v2_episodes.txt for use in the annotation config.

Usage (server-side, after downloading bridge_v2 metadata):
    python scripts/bridge_sample.py \
        --meta-dir ./data/bridge_v2/meta \
        --out configs/bridge_v2_episodes.txt

Then paste the output list into configs/lerobot_bridge_v2.yaml under
episode_indices.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Keyword sets for geometry category assignment.
# Applied to lowercased task string; first matching category wins.
GEOMETRY_CATEGORIES: dict[str, list[str]] = {
    "flat": [
        "plate", "book", "cloth", "towel", "sponge", "lid", "cutting board",
        "pan", "tray", "napkin", "mat", "sheet",
    ],
    "cylindrical": [
        "bottle", "can", "cup", "mug", "jar", "container", "bowl", "pot",
        "tube", "cylinder", "glass",
    ],
    "irregular": [
        "toy", "bear", "stuffed", "fruit", "vegetable", "apple", "orange",
        "corn", "carrot", "banana", "lemon", "block", "box", "drawer",
        "knife", "spoon", "fork",
    ],
}

N_PER_STRATUM = 50
SEED = 42


def assign_geometry(task: str) -> str | None:
    t = task.lower()
    for cat, keywords in GEOMETRY_CATEGORIES.items():
        if any(kw in t for kw in keywords):
            return cat
    return None


def load_episodes(meta_dir: Path) -> pd.DataFrame:
    ep_dir = meta_dir / "episodes"
    if not ep_dir.exists():
        # Some LeRobot v3 datasets store episodes directly in meta/
        parquet_files = sorted(meta_dir.glob("*.parquet"))
    else:
        parquet_files = sorted(ep_dir.glob("chunk-*/file-*.parquet"))
        if not parquet_files:
            parquet_files = sorted(ep_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {meta_dir}")
    return pd.concat([pq.read_table(f).to_pandas() for f in parquet_files])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", type=Path, required=True,
                    help="Path to bridge_v2/meta directory")
    ap.add_argument("--out", type=Path, default=Path("configs/bridge_v2_episodes.txt"))
    ap.add_argument("--n-per-stratum", type=int, default=N_PER_STRATUM)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    print(f"Loading episode metadata from {args.meta_dir} ...")
    eps = load_episodes(args.meta_dir)
    print(f"Total episodes: {len(eps)}")

    # Identify task string column
    task_col = None
    for col in ["tasks", "expand_task", "task", "language_instruction"]:
        if col in eps.columns:
            task_col = col
            break
    if task_col is None:
        raise ValueError(f"No task column found. Columns: {eps.columns.tolist()}")
    print(f"Using task column: {task_col!r}")

    # Coerce task string
    def coerce(t):
        if isinstance(t, list):
            return " ".join(str(x) for x in t)
        return str(t) if t is not None else ""

    eps["_task_str"] = eps[task_col].apply(coerce)
    eps["_geometry"] = eps["_task_str"].apply(assign_geometry)

    for cat, count in eps["_geometry"].value_counts().items():
        print(f"  {cat}: {count} episodes")
    print(f"  unclassified: {eps['_geometry'].isna().sum()} episodes")

    rng = np.random.default_rng(args.seed)
    selected = []
    for cat in ["flat", "cylindrical", "irregular"]:
        pool = eps[eps["_geometry"] == cat]["episode_index"].values
        if len(pool) < args.n_per_stratum:
            print(f"WARNING: only {len(pool)} episodes for '{cat}', using all.")
            chosen = pool
        else:
            chosen = rng.choice(pool, args.n_per_stratum, replace=False)
        selected.extend(sorted(chosen.tolist()))
        print(f"  sampled {len(chosen)} '{cat}' episodes")

    selected = sorted(set(selected))
    print(f"\nTotal selected: {len(selected)} episodes")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(i) for i in selected) + "\n")
    print(f"Written to {args.out}")

    # Also print as YAML list for copy-paste into config
    yaml_list = "[" + ", ".join(str(i) for i in selected) + "]"
    print(f"\nPaste into configs/lerobot_bridge_v2.yaml under episode_indices:")
    print(f"episode_indices: {yaml_list}")


if __name__ == "__main__":
    main()
