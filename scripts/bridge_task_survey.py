"""Survey BridgeData V2 task strings to identify candidate object categories.

Prints a frequency table of task strings and object-keyword clusters,
to help manually select episode groups analogous to LIBERO's task groups.

Usage (server-side):
    python scripts/bridge_task_survey.py --meta-dir ./data/bridge_v2/meta
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


# Object keywords to cluster around — edit freely before running.
OBJECT_KEYWORDS: list[str] = [
    "bowl", "bottle", "cup", "mug", "plate", "can",
    "box", "drawer", "cloth", "towel", "sponge",
    "toy", "block", "book", "pot", "pan", "jar",
    "knife", "spoon", "fork", "carrot", "corn", "apple",
    "orange", "banana", "lemon", "bear", "stuffed",
]


def load_episodes(meta_dir: Path) -> pd.DataFrame:
    ep_dir = meta_dir / "episodes"
    if ep_dir.exists():
        files = sorted(ep_dir.glob("chunk-*/file-*.parquet"))
        if not files:
            files = sorted(ep_dir.glob("*.parquet"))
    else:
        files = sorted(meta_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {meta_dir}")
    return pd.concat([pq.read_table(f).to_pandas() for f in files])


def coerce_task(t) -> str:
    if isinstance(t, list):
        return " ".join(str(x) for x in t)
    return str(t) if t is not None else ""


def find_task_col(eps: pd.DataFrame) -> str:
    for col in ["tasks", "expand_task", "task", "language_instruction"]:
        if col in eps.columns:
            return col
    raise ValueError(f"No task column found. Columns: {eps.columns.tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--top-tasks", type=int, default=60,
                    help="How many most-frequent task strings to print")
    ap.add_argument("--min-episodes", type=int, default=20,
                    help="Min episodes for a keyword cluster to be shown")
    args = ap.parse_args()

    print(f"Loading metadata from {args.meta_dir} ...")
    eps = load_episodes(args.meta_dir)
    print(f"Total episodes: {len(eps)}\n")

    task_col = find_task_col(eps)
    print(f"Task column: {task_col!r}\n")
    eps["_task"] = eps[task_col].apply(coerce_task)

    # ── 1. Top raw task strings ──────────────────────────────────────────────
    task_counts = Counter(eps["_task"].str.lower().str.strip())
    print(f"{'='*60}")
    print(f"TOP {args.top_tasks} TASK STRINGS")
    print(f"{'='*60}")
    for task, cnt in task_counts.most_common(args.top_tasks):
        print(f"  {cnt:5d}  {task}")

    # ── 2. Object-keyword clusters ───────────────────────────────────────────
    keyword_to_episodes: dict[str, list[int]] = defaultdict(list)
    for _, row in eps.iterrows():
        t = row["_task"].lower()
        ep_idx = int(row.get("episode_index", row.name))
        for kw in OBJECT_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', t):
                keyword_to_episodes[kw].append(ep_idx)

    print(f"\n{'='*60}")
    print(f"OBJECT KEYWORD CLUSTERS (≥{args.min_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  {'keyword':<20} {'episodes':>8}  {'unique_tasks':>12}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*12}")
    rows = []
    for kw in OBJECT_KEYWORDS:
        idxs = keyword_to_episodes[kw]
        if len(idxs) < args.min_episodes:
            continue
        # count distinct task strings that contain this keyword
        n_tasks = len(set(
            eps.loc[eps["episode_index"].isin(idxs), "_task"]
            .str.lower().str.strip()
        )) if "episode_index" in eps.columns else "?"
        rows.append((kw, len(idxs), n_tasks))

    for kw, n_ep, n_task in sorted(rows, key=lambda x: -x[1]):
        print(f"  {kw:<20} {n_ep:>8}  {n_task:>12}")

    # ── 3. Candidate group suggestion ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("CANDIDATE GROUPS FOR PROBE (manually verify geometric diversity)")
    print(f"{'='*60}")
    print("  Pick 4–6 keywords with:")
    print("  • distinct geometric profiles (flat vs cylindrical vs irregular)")
    print("  • ≥30 episodes each (for GroupKFold-10)")
    print("  • task strings that are unambiguous (one main object)")
    print()
    print("  Suggested workflow:")
    print("  1. Pick keywords from the cluster table above")
    print("  2. Check a few task strings per keyword for cleanliness")
    print("  3. Pass chosen keywords to bridge_filter.py (next script)")


if __name__ == "__main__":
    main()
