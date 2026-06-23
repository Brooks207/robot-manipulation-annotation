# Project Context — Robot Manipulation Visual Annotation Pipeline

**Goal**: add visual annotations (instance masks + depth) to LeRobot v3.0 robot manipulation
datasets and store them losslessly for downstream robot-learning / world-model pre-training.

**Storage philosophy**: the storage layer is not bound to any training representation — it
losslessly preserves raw annotations keyed by `(episode_idx, frame_idx)` so any downstream
consumer can ingest them.

> Public fork: reproducible on open data (`lerobot/svla_so100_sorting`). Internal dataset
> paths, server details, and model codenames removed. See `QUICKSTART_PUBLIC.md`.

---

## Architecture (four decoupled, config-driven layers)

1. **Discovery** (`annotation/discovery/`) — extract object queries from the task instruction;
   auto-append `robot hand` / `gripper`. Extractors: `rule` | `manual` | `qwen` | `mock`.
   Instruction source: `episode_field` (string `expand_task` or list-valued LeRobot v3 `tasks`),
   `join_file`, or `none` (fallback to default/vocab). Interface:
   `discover_objects(instruction, config) -> list[str]`.
2. **Segmentation** (`annotation/segmentation/sam3.py`) — SAM3 text-prompted masks; vision
   features reused across queries on the same frame.
3. **Depth** (`annotation/depth/depth_anything3.py`) — Depth-Anything-3 monocular metric depth,
   independent of segmentation.
4. **Storage** (`annotation/storage/`) — Parquet COCO-RLE masks; 16-bit PNG depth + JSON.

## Sampling

`sampling.mode`: `uniform` (linspace over the episode) or `subtask_aware` (group by
`subtask_index`, stride within each segment via `stride_frames`/`stride_seconds`, keep segment
boundary frames). Each run writes `sampling_manifest.parquet` for downstream alignment.

## Stage decoupling

`--stage segmentation|depth|both` (overrides `config.stage`). Each stage uses an independent
output dir + checkpoint and resumes independently; idempotency checks only the active stage's
products. Same `dataset_path` + `episode_indices` + `sampling` ⇒ identical sampled frames ⇒
masks and depth join by `(episode_idx, frame_idx)`.

## Engineering guarantees

Config-driven (zero hardcoding) · checkpoint resume / idempotent · per-frame failure isolation ·
dry-run (discovery only) · mock paths (no-model end-to-end) · QC triptych visualization.

---

## Core commands

```bash
# Discovery-only dry-run (no GPU/models)
python run_dryrun.py configs/lerobot_so100_dryrun.yaml

# Segmentation smoke on public data (GPU + ffmpeg(av1) + SAM3 access)
python run_annotate.py configs/lerobot_so100_smoke.yaml --stage segmentation

# Full pipeline with mock models (architecture check)
python run_annotate.py configs/<cfg>.yaml --use-mock
```

## DA3 metric depth (key gotcha)

Load with `DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")` — plain
construction skips the checkpoint and yields constant depth. `result.depth` is raw; convert
`metric_depth_m = focal_px * raw / 300`, `focal_px` scaled from calibration resolution to DA3's
actual output resolution (from `result.depth.shape`, applied once). Provide camera intrinsics
via `depth.fx/fy/calibration_width/calibration_height` (rig-specific, not committed).

## Troubleshooting

- **SAM3**: `transformers >= 5.0`; correct model id/path; HF token if gated; enough GPU memory.
- **DA3**: `depth-anything-3` installed; use `from_pretrained`; confirm first-layer params
  nonzero; `debug_depth_range: true` to inspect raw+metric ranges; convert metric once.
- **Discovery**: `instruction_source`/`instruction_field` must match the dataset; `rule`
  extractor emits long-phrase/generic noise (known limitation) — use `manual` oracle as the
  quality ceiling, deploy `qwen` for the real fix.

## Conventions

Type annotations on new functions · `logging` not `print` · failure isolation · idempotency ·
feature-branch workflow · `feat/fix/docs:` commit prefixes · unit (RLE, depth scaling) +
integration (dry-run) + mock tests.

## External dependencies

- [SAM 3](https://huggingface.co/facebook/sam3)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [LeRobot v3.0](https://github.com/huggingface/lerobot)
