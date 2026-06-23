# Quickstart: Reproduce on a Public LeRobot v3 Dataset

This pipeline was developed against an internal AnyGrasp dataset. To make it
**reproducible by anyone**, it also runs on the public
[`lerobot/svla_so100_sorting`](https://huggingface.co/datasets/lerobot/svla_so100_sorting)
dataset (LeRobot v3.0, 52 episodes, Apache-2.0) with **config-only changes**.

## 0. Why this works without code changes

The loader (`annotation/lerobot_v3_dataset.py`) reads the standard LeRobot v3.0
layout directly: `meta/episodes/chunk-*/file-*.parquet`, `data/chunk-*/file-*.parquet`,
`videos/<camera>/chunk-*/file-*.mp4`. The only adaptation needed for arbitrary Hub
datasets was instruction extraction: internal AnyGrasp stores a string `expand_task`
field, while standard LeRobot v3 stores a **list-valued `tasks`** column. The loader
now coerces both (`_coerce_instruction`), so `instruction_field: tasks` just works.

## 1. Download the dataset

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download lerobot/svla_so100_sorting \
  --repo-type dataset --local-dir ./data/svla_so100_sorting
```

## 2. Dry-run (Discovery only — no GPU / models / ffmpeg)

Proves the loader + instruction extraction work on public data:

```bash
python run_dryrun.py configs/lerobot_so100_dryrun.yaml
cat outputs/so100_dryrun/discovery_queries.jsonl
```

Expected: 3 episodes loaded, instruction
`"Put the red cube in the right box and the blue cube in the left box."`
extracted from the `tasks` field, and queries written per episode.
(Note: the `rule` extractor emits one dirty long-phrase query here — this is the
documented limitation that motivates the real `qwen` extractor.)

## 3. Segmentation smoke run (needs GPU + ffmpeg(av1) + SAM3 access)

`svla` videos are AV1-encoded, so ffmpeg must have an AV1 decoder. SAM3
(`facebook/sam3`) is gated — request access and set `HF_TOKEN`.

```bash
export HF_TOKEN=...           # SAM3 gated access
python run_annotate.py configs/lerobot_so100_smoke.yaml --stage segmentation
# outputs/so100_smoke/  -> masks.parquet + qc/*.png (RGB | masks | depth)
```

Depth (`--stage depth`) is intentionally skipped in the public smoke run: DA3
metric conversion needs per-camera intrinsics (`fx/fy`, calibration resolution)
that ship with the internal rig but not with this public dataset. Run depth in
relative mode or supply intrinsics to enable it.

## Files added for the public path

- `configs/lerobot_so100_dryrun.yaml` — Discovery-only dry-run
- `configs/lerobot_so100_smoke.yaml` — SAM3 segmentation smoke run
- `annotation/lerobot_v3_dataset.py` — `_coerce_instruction` handles list-valued `tasks`
