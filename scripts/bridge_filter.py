"""Filter BridgeData V2 episodes into manually chosen object categories,
mirroring the LIBERO sub-experiment structure (Sec 4.1 of PROBE_EXPERIMENT.md):
a handful of hand-picked categories with clean, unambiguous task strings and
enough episodes per category for GroupKFold(10).

This replaces bridge_sample.py's automatic 3-stratum keyword clustering
(flat/cylindrical/irregular), which was too coarse and produced ~92 Qwen-parsed
categories in the pilot run. Categories here are chosen by hand after reading
scripts/bridge_task_survey.py output, exactly as the 5 LIBERO task groups were
chosen by hand rather than discovered automatically.

Usage (server-side, after running bridge_task_survey.py and picking keywords):
    python scripts/bridge_filter.py \
        --meta-dir ./data/bridge_v2/meta \
        --categories bowl:bowl bottle:bottle cup:cup,mug plate:plate \
        --max-per-category 50 \
        --out configs/bridge_v2_episodes.txt

Each --categories entry is `label:kw1,kw2,...`. An episode's task string is
matched against each category's keywords in the given order; the first match
wins (mirrors LIBERO's one-target-object-per-episode assumption). Episodes
matching zero or multiple categories' keyword sets are handled per
--ambiguous {drop,first} (default: drop, to keep labels clean).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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


def parse_categories(specs: list[str]) -> dict[str, list[str]]:
    """Parse `label:kw1,kw2` strings into {label: [kw1, kw2]}."""
    out: dict[str, list[str]] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Bad --categories entry {spec!r}; expected label:kw1,kw2")
        label, kws = spec.split(":", 1)
        out[label.strip()] = [k.strip().lower() for k in kws.split(",") if k.strip()]
    return out


def assign_category(task: str, categories: dict[str, list[str]]) -> list[str]:
    """Return every category label whose keywords match this task string."""
    t = task.lower()
    matches = []
    for label, kws in categories.items():
        if any(re.search(r"\b" + re.escape(kw) + r"\b", t) for kw in kws):
            matches.append(label)
    return matches


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--meta-dir", type=Path, required=True,
                    help="Path to bridge_v2/meta directory")
    ap.add_argument("--categories", nargs="+", required=True,
                    help="label:kw1,kw2 entries, e.g. bowl:bowl cup:cup,mug")
    ap.add_argument("--max-per-category", type=int, default=50,
                    help="Cap episodes per category (random subsample if exceeded)")
    ap.add_argument("--min-per-category", type=int, default=20,
                    help="Warn if a category has fewer episodes than this")
    ap.add_argument("--ambiguous", choices=["drop", "first"], default="drop",
                    help="How to handle episodes matching >1 category")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("configs/bridge_v2_episodes.txt"))
    args = ap.parse_args()

    categories = parse_categories(args.categories)
    print(f"Categories ({len(categories)}):")
    for label, kws in categories.items():
        print(f"  {label}: {kws}")
    print()

    print(f"Loading episode metadata from {args.meta_dir} ...")
    eps = load_episodes(args.meta_dir)
    print(f"Total episodes in dataset: {len(eps)}\n")

    task_col = find_task_col(eps)
    eps["_task"] = eps[task_col].apply(coerce_task)
    if "episode_index" not in eps.columns:
        eps["episode_index"] = eps.index

    eps["_matches"] = eps["_task"].apply(lambda t: assign_category(t, categories))
    eps["_n_matches"] = eps["_matches"].apply(len)

    n_zero = int((eps["_n_matches"] == 0).sum())
    n_multi = int((eps["_n_matches"] > 1).sum())
    print(f"Episodes matching 0 categories: {n_zero}")
    print(f"Episodes matching >1 category (ambiguous): {n_multi}")
    if n_multi:
        example_rows = eps[eps["_n_matches"] > 1][["_task", "_matches"]].head(5)
        for _, row in example_rows.iterrows():
            print(f"    e.g. {row['_task']!r} -> {row['_matches']}")
    print()

    if args.ambiguous == "drop":
        eps["_label"] = eps["_matches"].apply(lambda m: m[0] if len(m) == 1 else None)
    else:  # "first" - keep first matching category even if ambiguous
        eps["_label"] = eps["_matches"].apply(lambda m: m[0] if m else None)

    rng = np.random.default_rng(args.seed)
    selected_rows: list[tuple[str, int]] = []  # (label, episode_index)
    summary = []

    for label in categories:
        pool = eps.loc[eps["_label"] == label, "episode_index"].unique()
        n_pool = len(pool)
        if n_pool < args.min_per_category:
            print(f"WARNING: '{label}' has only {n_pool} episodes "
                  f"(< --min-per-category={args.min_per_category}). "
                  "Consider broadening its keywords or dropping it.")
        if n_pool > args.max_per_category:
            chosen = rng.choice(pool, args.max_per_category, replace=False)
        else:
            chosen = pool
        for idx in sorted(int(i) for i in chosen):
            selected_rows.append((label, idx))
        summary.append((label, n_pool, len(chosen)))

    print(f"{'='*60}")
    print("CATEGORY SUMMARY (mirrors LIBERO Sec 4.1 task-group table)")
    print(f"{'='*60}")
    print(f"  {'category':<15} {'available':>10} {'selected':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10}")
    for label, n_pool, n_chosen in summary:
        print(f"  {label:<15} {n_pool:>10} {n_chosen:>10}")

    total = len(selected_rows)
    print(f"\nTotal selected: {total} episodes across {len(categories)} categories")

    # Random-chance baseline, matching PROBE_EXPERIMENT.md Sec 4.1 convention.
    k = len(categories)
    print(f"Random-chance baseline: 1/{k} = {100.0 / k:.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_indices = sorted(set(idx for _, idx in selected_rows))
    args.out.write_text("\n".join(str(i) for i in all_indices) + "\n")
    print(f"\nWritten {len(all_indices)} unique episode indices -> {args.out}")

    # Also write a label map, so run_annotate.py / probe scripts can use
    # hand-picked category labels instead of re-deriving them via Qwen.
    label_map_path = args.out.with_name(args.out.stem + "_labels.csv")
    pd.DataFrame(selected_rows, columns=["category", "episode_index"]).sort_values(
        "episode_index"
    ).to_csv(label_map_path, index=False)
    print(f"Written category labels -> {label_map_path}")

    yaml_list = "[" + ", ".join(str(i) for i in all_indices) + "]"
    print(f"\nPaste into configs/lerobot_bridge_v2.yaml under episode_indices:")
    print(f"episode_indices: {yaml_list}")


if __name__ == "__main__":
    main()
