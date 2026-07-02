# Server Handoff — Representation Probe Experiment
# Date: 2026-06-30
# Paste this as the server agent's first message. Self-contained.

---

## Who / what

**Binghao Ye** (GitHub: Brooks207), Georgia Tech sophomore (Math & Computing, 4.0).
Applying to US robotics / world-model labs (GT Danfei Xu, Animesh Garg, Zsolt Kira).

This repo is a portfolio project: a config-driven annotation pipeline
(Qwen object discovery → SAM3 segmentation → Depth-Anything-3 metric/relative depth →
Parquet/PNG16 storage → QC) for LeRobot v3 manipulation datasets.

The current task is a **representation probe experiment**: run the pipeline on
two datasets (LIBERO + BridgeData V2), extract features, and bring back outputs
for local analysis.

---

## ⚠️ Hard constraints (do not violate)

- Binghao owns the **data pipeline**, not the world model. Never claim he trained
  a model. Never mention "Marmalade" (internal model codename) anywhere.
- Do **not** commit internal paths (`/mnt/oss/...`), API keys, or output files
  (npz, parquet, depth PNGs) to the public repo.
- `outputs/` and `data/` are gitignored — safe to write there, never commit.
- Do not commit local working files like `configs/bridge_v2_episodes.txt`.

---

## Repo

**https://github.com/Brooks207/robot-manipulation-annotation**

Working directory on server: `/mnt/workspace/binghao/robot-manipulation-annotation`

Key scripts (all in repo after `git pull`):

| Script | Purpose |
|---|---|
| `run_annotate.py` | Pipeline entry point (Qwen → SAM3 → DA3) |
| `run_representation_probe_rigorous.py` | Episode-level probe (GroupKFold, balanced accuracy) |
| `run_representation_probe_richer_depth.py` | 27-dim depth feature library |
| `run_representation_probe.py` | Utilities (decode_mask, rgb_features, load_depth_meters) |
| `run_vla_feature_extract.py` | SigLIP-SO400M (1152-dim) feature extraction |
| `scripts/bridge_sample.py` | Stratified episode sampler for BridgeData V2 |

---

## Experiment overview

Three sub-experiments test whether hand-crafted depth statistics add classifiable
signal beyond RGB, progressively from simple to harder:

| Sub-exp | Dataset | Episodes | Depth type | Hypotheses tested |
|---|---|---|---|---|
| A | LIBERO (simulated Franka) | 202 | **Metric** (fx=fy=618.1px) | H1 + H2 (RGB-controlled) |
| B | BridgeData V2 (real Franka) | 150 | **Relative** (no intrinsics) | **H3** (depth vs SigLIP) |

**H2**: `depth_only` balanced accuracy > random chance (1/K) — Sub-exp A
**H1**: `rgb+depth` balanced accuracy > `rgb_only` by Δ ≥ 0.05 — Sub-exp A
**I-2**: metric vs forced-relative on Sub-exp A data (Ablation, zero extra compute)
**H3**: `siglip+depth` balanced accuracy > `siglip_only` by Δ ≥ 0.03 — Sub-exp B

**Metric**: ALL probes use **balanced accuracy** (sklearn `scoring="balanced_accuracy"` =
mean per-class recall). This removes majority-class bias in Sub-exp A where
`cabinet_top` has ~92 episodes vs 33–42 per drawer task.

Feature sets per run:

| Feature set | Dim | H tested |
|---|---|---|
| `rgb_only` | 9 | — (baseline) |
| `depth_only` | 27 | H2 |
| `rgb_plus_depth_rich` | 36 | H1 |
| `siglip_only` | 1152 | — (learned baseline, Sub-exp B only) |
| `siglip_plus_depth_rich` | 1179 | H3, Sub-exp B only |

**27-dim depth features** (DEPTH_RICH_DIM=27):
- Group 1 (5): mean, std, min, max, median
- Group 2 (2): RANSAC planarity residual, inlier ratio (threshold = 0.005 m for metric / 0.02 × depth_range for relative)
- Group 3 (8): depth-HOG (gradient orientation histogram, 8 bins, magnitude-weighted)
- Group 4 (6): IQR, pct10, pct90, skewness, kurtosis, entropy
- Group 5 (4): quadrant means (TL, TR, BL, BR split at centroid)
- Group 6 (2): Laplacian mean, Laplacian std

**Episode-level CV**: GroupKFold(n_splits=min(10, min_category_episodes), groups=episode_idx) —
test sets contain complete unseen episodes, preventing within-scene memorization.
LeaveOneGroupOut when ≤10 episodes per category.
Sub-exp A: n_splits=10 (all categories ≥33 episodes). Sub-exp B: n_splits=10 (§4.2).

---

## Step 0 — Pull latest code

```bash
cd /mnt/workspace/binghao/robot-manipulation-annotation
git pull origin main
```

Verify all required files exist:

```bash
ls run_annotate.py \
   run_representation_probe.py \
   run_representation_probe_richer_depth.py \
   run_representation_probe_rigorous.py \
   run_vla_feature_extract.py \
   configs/lerobot_libero_large.yaml \
   configs/lerobot_bridge_v2.yaml \
   scripts/bridge_sample.py
```

All 8 must exist. If any missing, `git pull` failed.

---

## Step 1 — Start Qwen vLLM (needed for annotation discovery stage)

```bash
curl -s http://localhost:8000/v1/models \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print([m['id'] for m in d['data']])" \
  2>/dev/null || echo "NOT RUNNING"
```

If not running:

```bash
tmux new -s qwen
vllm serve Qwen/Qwen3-8B --port 8000
# Ctrl+B D to detach; wait ~60s then re-check with curl above
```

---

## Step 2 — Verify datasets

Sub-exp A (LIBERO) and Sub-exp B (BridgeData V2) datasets are handled in their respective sections.

---

## Part A — LIBERO (202 episodes, metric depth)

### Background

202 episodes across 5 task groups, all manipulating the same cabinet object.
Geometric hypothesis: drawer tasks (protruding surface) vs cabinet-top tasks
(flat surface) are separated by depth planarity + quadrant features.

| Task | Episodes | Geometry |
|---|---|---|
| put black bowl in bottom drawer + close | ep 43–77 (35) | bottom drawer |
| open middle drawer | ep 388–429 (42) | middle drawer |
| open top drawer + put bowl inside | ep 430–462 (33) | top drawer |
| put bowl on top of cabinet | ep 602–648 (47) | cabinet top (flat) |
| put wine bottle on top of cabinet | ep 726–770 (45) | cabinet top (flat) |

- Camera: `observation.images.image` (512×512, agentview)
- Depth: **metric** — fx=fy=618.1px exact (robosuite fovy=45°, 512px, tan(22.5°))
- 8 frames/episode → ~1616 frames total

### A1 — Annotate (~30–90 min)

```bash
tmux new -s libero
python run_annotate.py configs/lerobot_libero_large.yaml --stage both \
  2>&1 | tee /tmp/libero_large.log
# Ctrl+B D to detach
```

Monitor: `tail -f /tmp/libero_large.log | grep -E "episode [0-9]+|ERROR|Done"`

Verify when done:
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('outputs/libero_large/segmentation/masks.parquet')
print(len(df), 'instances,', df['category'].nunique(), 'categories')
print(df['category'].value_counts().to_string())
"
```

### A2 — Probe: RGB + depth features

```bash
python run_representation_probe_rigorous.py \
    --dataset-path ./data/libero \
    --masks outputs/libero_large/segmentation/masks.parquet \
    --depth-dir outputs/libero_large/depth_stage/depth \
    --camera-name observation.images.image \
    --out outputs/libero_large_probe \
    --min-episodes-per-class 5
```

**Save the terminal output** — lines like:
```
rgb_only (9-dim):             0.XXX +/- 0.XXX
depth_only (27-dim):          0.XXX +/- 0.XXX
rgb_plus_depth_rich (36-dim): 0.XXX +/- 0.XXX
```

This writes `outputs/libero_large_probe/all_features.npz`
containing X_rgb (N,9), X_depth (N,27), y, episode_idx, frame_idx, instance_id.

---

## Part B — BridgeData V2 (150 episodes, H3 test)

### Background

Tests whether depth adds signal **on top of SigLIP-SO400M** on real-world data.
SigLIP (google/siglip-so400m-patch14-384, 1152-dim pooler_output) is trained on
real photographs (LAION/WebLI), so LIBERO simulation frames create a domain gap.
BridgeData V2 is real Franka arm footage — in-distribution for SigLIP.
H3 positive (Δ ≥ 0.03 balanced accuracy) would justify including hand-crafted
depth features in a SigLIP-based VLA backbone.

### B0 — Install HF CLI, download metadata only

```bash
pip install "huggingface_hub[cli]"

HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download lerobot/bridge_v2 \
  --repo-type dataset \
  --include "meta/**" \
  --local-dir ./data/bridge_v2
```

### B1 — Confirm camera name

```bash
python3 -c "
import json
info = json.load(open('./data/bridge_v2/meta/info.json'))
print('camera_names:', info.get('camera_names', info.get('video_keys', 'not found')))
print('total_episodes:', info.get('total_episodes', '?'))
"
```

**Report the camera name back.** Likely `observation.images.image_0`. Used in B5/B6.

### B2 — Filter episodes into 5 hand-picked categories

`scripts/bridge_sample.py`'s automatic 3-stratum keyword clustering (flat/
cylindrical/irregular) is deprecated — it let Qwen infer categories from the
full unfiltered task-string space and produced ~92 near-random categories in
the pilot run (see `PROBE_EXPERIMENT.md` §4.2 pilot note). Use
`scripts/bridge_filter.py` instead, which mirrors how LIBERO's 5 task groups
were *hand-picked*, not discovered.

Categories were chosen by surveying `scripts/bridge_task_survey.py` output
against BridgeData V2's actual task-string distribution (50,418 episodes) and
picking 5 that (a) span distinct geometric profiles, (b) have a low
unique-task/episode ratio (clean, unambiguous single-object phrasing), and
(c) avoid deformable objects (`cloth`) and multi-context containers
(`pot`/`pan`, which appear in dozens of unrelated task types) — see
`PROBE_EXPERIMENT.md` §4.2 for the full rationale:

| Category | Geometry | Raw match count |
|---|---|---|
| plate | flat | 1978 |
| bowl | concave/curved | 1478 |
| cup | cylindrical (short) | 1104 |
| bottle | cylindrical (elongated) | 467 |
| carrot | irregular/elongated | 1133 |

```bash
python scripts/bridge_filter.py \
    --meta-dir ./data/bridge_v2/meta \
    --categories plate:plate bowl:bowl cup:cup bottle:bottle carrot:carrot \
    --max-per-category 50 \
    --out configs/bridge_v2_episodes.txt

wc -l configs/bridge_v2_episodes.txt   # should be <=250 (5 categories x <=50)
```

This writes two files:
- `configs/bridge_v2_episodes.txt` — episode indices; paste the printed
  `episode_indices: [...]` list into `configs/lerobot_bridge_v2.yaml`
  replacing `episode_indices: []`.
- `configs/bridge_v2_episodes_labels.csv` — `episode_index -> category`
  ground-truth map. **Required in B5** via `--category-override`; do not
  discard this file.

Episodes whose task string matches more than one of the 5 keywords (e.g.
"put carrot on plate" matches both `carrot` and `plate`) are dropped by
default (`--ambiguous drop`) — this is intentional, not a bug: the probe
assumes one target object per episode, and an ambiguous task string has no
single correct label.

### B3 — Download only the sampled episodes' video chunks

```bash
python3 -c "
indices = [int(l) for l in open('configs/bridge_v2_episodes.txt')]
chunks = sorted(set(i // 1000 for i in indices))
print('Chunks needed:', chunks)
patterns = []
for c in chunks:
    patterns += [f'data/chunk-{c:03d}/**', f'videos/chunk-{c:03d}/**']
print(' '.join(f'--include \"{p}\"' for p in patterns))
" > /tmp/bridge_flags.txt

cat /tmp/bridge_flags.txt   # verify before downloading

HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download lerobot/bridge_v2 \
  --repo-type dataset \
  $(cat /tmp/bridge_flags.txt) \
  --local-dir ./data/bridge_v2
```

> If chunks are not 1000-episode blocks, check `ls ./data/bridge_v2/data/`
> and adjust `i // 1000` accordingly.

### B4 — Annotate (~20–60 min)

Confirm `configs/lerobot_bridge_v2.yaml` has:
- `episode_indices:` filled from B2
- `camera_names:` matching B1 result

```bash
tmux new -s bridge
python run_annotate.py configs/lerobot_bridge_v2.yaml --stage both \
  2>&1 | tee /tmp/bridge_v2.log
# Ctrl+B D to detach
```

Verify (note: `category` here is still the Qwen/discovery label used for the
SAM3/Grounded-SAM2 segmentation query — it gets overridden by the hand-picked
label in B5, so don't worry if this shows more than 5 categories or noisy
names):
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('outputs/bridge_v2/segmentation/masks.parquet')
print(len(df), 'instances,', df['category'].nunique(), 'categories')
print(df['category'].value_counts().head(10).to_string())
"
```

### B5 — Probe: RGB + depth features

`--category-override` replaces the discovery-pipeline's per-instance category
with the B2 hand-picked label map, so the probe's `y` matches LIBERO's
task-group-defined ground truth instead of noisy Qwen inference.

```bash
BRIDGE_CAM="observation.images.image_0"   # ← update from B1 if different

python run_representation_probe_rigorous.py \
    --dataset-path ./data/bridge_v2 \
    --masks outputs/bridge_v2/segmentation/masks.parquet \
    --depth-dir outputs/bridge_v2/depth_stage/depth \
    --camera-name "$BRIDGE_CAM" \
    --category-override configs/bridge_v2_episodes_labels.csv \
    --out outputs/bridge_v2_probe \
    --min-episodes-per-class 5
```

**Save accuracy output** (3 lines: rgb_only, depth_only, rgb_plus_depth_rich).
Also save the logged line `Applied category override ... N hand-picked
categories: [...]` — confirms the override actually took effect (should show
5 categories, not ~92).

### B6 — Extract SigLIP-SO400M features (critical for H3)

```bash
BRIDGE_CAM="observation.images.image_0"   # ← same as B5

HF_ENDPOINT=https://hf-mirror.com python run_vla_feature_extract.py \
    --dataset-path ./data/bridge_v2 \
    --masks outputs/bridge_v2/segmentation/masks.parquet \
    --camera-name "$BRIDGE_CAM" \
    --probe-cache outputs/bridge_v2_probe/probe_features.npz \
    --out outputs/bridge_v2/vla_features.npz
```

Verify:
```bash
python3 -c "
import numpy as np; d = np.load('outputs/bridge_v2/vla_features.npz', allow_pickle=True)
X = d['siglip_X']; print('shape:', X.shape, ' NaN rows:', int(np.isnan(X).any(axis=1).sum()))
"
# Expected: shape (N, 1152), NaN rows: 0
```

---

## Files to bring back (6 total)

`all_features.npz` (produced by the rigorous probe script) contains
`X_rgb (N,9)`, `X_depth (N,27)`, `y`, `episode_idx`, `frame_idx`, `instance_id`.
This enables all local ablations without re-running the server.

| File | Source path | Used for |
|---|---|---|
| `libero_all_features.npz` | `outputs/libero_large_probe/all_features.npz` | A local ablations R-1, I-2, A-1, T-1 |
| `libero_masks.parquet` | `outputs/libero_large/segmentation/masks.parquet` | Ablation A-1 (bbox re-extraction) |
| `libero_depth.tar.gz` | `outputs/libero_large/depth_stage/depth/` (tarred) | Ablations I-2, A-1 (need raw depth PNGs) |
| `bridge_v2_all_features.npz` | `outputs/bridge_v2_probe/all_features.npz` | H3 probe with SigLIP (X_rgb + X_depth) |
| `bridge_v2_vla_features.npz` | `outputs/bridge_v2/vla_features.npz` | H3: siglip_only vs siglip+depth |
| `bridge_v2_masks.parquet` | `outputs/bridge_v2/segmentation/masks.parquet` | Local category analysis |

### Package commands

```bash
cp outputs/libero_large_probe/all_features.npz    /tmp/libero_all_features.npz
cp outputs/libero_large/segmentation/masks.parquet /tmp/libero_masks.parquet
tar -czf /tmp/libero_depth.tar.gz outputs/libero_large/depth_stage/depth/

cp outputs/bridge_v2_probe/all_features.npz   /tmp/bridge_v2_all_features.npz
cp outputs/bridge_v2/vla_features.npz          /tmp/bridge_v2_vla_features.npz
cp outputs/bridge_v2/segmentation/masks.parquet /tmp/bridge_v2_masks.parquet

ls -lh /tmp/libero_all_features.npz \
        /tmp/libero_masks.parquet \
        /tmp/libero_depth.tar.gz \
        /tmp/bridge_v2_all_features.npz \
        /tmp/bridge_v2_vla_features.npz \
        /tmp/bridge_v2_masks.parquet
```

---

## What's done locally after files return

The server only needs the 6 files above and the accuracy numbers.
All further analysis and ablations run locally.

### L1 — Sub-exp B: H3 probe with SigLIP

```bash
python run_representation_probe_rigorous.py \
    --masks bridge_v2_masks.parquet \
    --camera-name observation.images.image_0 \
    --probe-cache bridge_v2_all_features.npz \
    --vla-features bridge_v2_vla_features.npz \
    --out outputs/bridge_v2_h3_probe \
    --min-episodes-per-class 5
```

Reports `siglip_only (1152-dim)` vs `siglip_plus_depth_rich (1179-dim)` for H3.

### L2 — Ablation R-1: Feature group drop-one-out (Sub-exp A, no extra files needed)

Six probe variants, each with one feature group zeroed in X_depth:

| Variant | Zeroed dims | Expected signal |
|---|---|---|
| `depth_no_basic` | 0–4 | Are mean/std/min/max/median discriminative? |
| `depth_no_planarity` | 5–6 | Does removing RANSAC planarity increase drawer/cabinet confusion? |
| `depth_no_HOG` | 7–14 | Does cylinder accuracy drop without orientation histogram? |
| `depth_no_shape` | 15–20 | Contribution of skewness, kurtosis, entropy, IQR |
| `depth_no_spatial` | 21–24 | Quadrant mean contribution |
| `depth_no_curvature` | 25–26 | Laplacian second-order signal |

Implementation: `X_depth_ablated = X_depth.copy(); X_depth_ablated[:, dims] = 0`
then re-run StandardScaler → LogisticRegression on same GroupKFold splits.
Output: bar chart of Δ balanced accuracy vs full 27-dim for each group removal.

### L3 — Ablation I-1: GT depth vs DA3 (Sub-exp A only) — **NOT FEASIBLE**

Requires re-running robosuite simulator with `env.render(depth=True)` to get
ground-truth depth buffers. The LIBERO LeRobot v3 dataset does not include
the raw simulator depth channel. Skip unless original LIBERO simulation
environment is set up separately.

### L4 — Ablation I-2: Metric vs forced-relative depth (Sub-exp A, requires `libero_depth/`)

Normalize each depth map by mask-region mean before re-extracting features.
Tests whether absolute metric scale (not just relative ordering) carries
discriminative signal. If balanced accuracy drops Δ ≥ 0.05, metric scale is
load-bearing and Sub-exp B's relative results cannot be directly compared to A.

### L5 — Ablation A-1: Instance mask vs bounding box crop (requires `libero_masks.parquet` + `libero_depth/`)

Re-extract 27 features using rectangular bbox (+12px padding) instead of
SAM3 instance mask. Gap = SAM3 segmentation contribution to feature quality.
- Gap ≈ 0: a YOLO detector would suffice; SAM3 not needed for depth features
- Gap ≥ 0.05: precise instance boundaries matter; SAM3 earns its compute cost

### L6 — Ablation T-1: Per-frame vs episode-level aggregation (uses `all_features.npz`)

Aggregate per-frame feature vectors to episode level (mean across instances),
run episode-level CV on the aggregated features. Compare to per-frame baseline.

### L7 — Permutation test (§5.2)

500 label permutations (episode-level, preserving GroupKFold structure).
If observed `rgb+depth` balanced accuracy > 95th percentile of permutation null
distribution → report p < 0.05 (permutation-based). Supplements but does not
replace the Δ-based decision.

---

## Accuracy numbers to paste back

For each probe run, copy the lines printed to stdout:

**A2 (LIBERO)**:
```
rgb_only (9-dim):             0.XXX +/- 0.XXX
depth_only (27-dim):          0.XXX +/- 0.XXX
rgb_plus_depth_rich (36-dim): 0.XXX +/- 0.XXX
```

**B5 (BridgeData V2)**: same 3 lines

Also report:
- BridgeData V2 camera name from B1
- Total episode counts that passed `--min-episodes-per-class 5` filter in each run

---

## Error handling

**OOM during annotation** — run stages separately:
```bash
python run_annotate.py configs/lerobot_libero_large.yaml --stage segmentation
python run_annotate.py configs/lerobot_libero_large.yaml --stage depth
```

**HuggingFace unreachable** — all commands already use `HF_ENDPOINT=https://hf-mirror.com`.
If still failing, try `https://hf-mirror.com` or ask for an alternative.

**sentencepiece / protobuf missing** (SigLIP model load):
```bash
pip install sentencepiece protobuf
```

**Checkpoint resume** — all annotation stages are idempotent. Re-run same command; it
resumes from last completed episode.

**Qwen timeout** — ensure vLLM is fully loaded before annotation, or increase
`qwen_timeout` in the config.

---

## Completion checklist

Before transferring files:

- [ ] A1: `outputs/libero_large/segmentation/masks.parquet` exists, >500 instances
- [ ] A2: `outputs/libero_large_probe/all_features.npz` exists; 3 accuracy lines recorded
- [ ] B1: camera name confirmed and noted
- [ ] B2: `configs/bridge_v2_episodes.txt` has 150 lines; config `episode_indices` filled
- [ ] B3: bridge_v2 video chunks downloaded
- [ ] B4: `outputs/bridge_v2/segmentation/masks.parquet` exists
- [ ] B5: `outputs/bridge_v2_probe/all_features.npz` exists; 3 accuracy lines recorded
- [ ] B6: `outputs/bridge_v2/vla_features.npz` shape=(N,1152), NaN=0

---

## Hard limits (repeat)

- Do NOT commit `outputs/`, `data/`, `.npz`, `.parquet`, depth files, or API keys
- Do NOT put `/mnt/oss/...` paths or "Marmalade"/"fastwam"/"anygrasp" in any commit
- The repo is public — every commit is visible to the target professors
