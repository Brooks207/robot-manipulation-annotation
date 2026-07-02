# Representation Probe Experiment
# Does metric depth add classifiable geometric signal beyond RGB?
# Binghao Ye — 2026-06-26

---

## 1. Motivation and Hypotheses

### 1.1 Motivation

This annotation pipeline produces two output streams per frame: instance
segmentation masks (SAM3) and metric depth maps (Depth-Anything-3). The
segmentation is visually verifiable from QC images. The depth stream is
harder to validate — a depth map that looks plausible could still carry
no signal not already present in the RGB image.

The probe answers one specific, falsifiable question:

> **Do hand-crafted statistics extracted from metric depth maps add
> classifiable signal for object category, beyond what the same statistics
> extracted from RGB already provide?**

This is a sanity check on the data pipeline, not a claim about optimal
representations. A "no" answer is informative: it means either the depth
output is uninformative for the objects tested, or that hand-crafted
features fail to capture the useful signal.

### 1.2 Hypotheses — Progressive Evidence Chain

The three hypotheses form a logical escalation. Each level assumes the
previous one passed; together they build toward a single conclusion.

```
H2: Does depth alone carry any signal?
    (geometric features, no color, vs. random chance)
    [Sub-exp A — LIBERO, RGB-controlled, metric depth]
         ↓ if YES
H1: Does depth add signal on top of RGB?
    (rgb+depth vs. rgb_only)
    [Sub-exp A — LIBERO, RGB-controlled, metric depth]
         ↓ if YES         ↓ also
                    I-2: Is metric scale load-bearing, or does
                         relative depth suffice?
                         (metric vs. forced-relative, Sub-exp A data,
                          zero extra compute)
H3: Does depth add signal on top of a 400M-param vision encoder?
    (siglip+depth vs. siglip_only, on real-world data)
    [Sub-exp B — BridgeData V2, real robot, relative depth]
         ↓ if YES
Conclusion: depth carries geometric signal that is real, independent,
            and not subsumed by either color statistics or learned
            internet-scale representations.
```

| ID | Hypothesis | Sub-exp | Falsification condition |
|---|---|---|---|
| **H2** | `depth_only` accuracy > random-chance baseline (1/K) | A | depth_only ≤ 1/K in Sub-exp A |
| **H1** | `rgb+depth` accuracy > `rgb_only` accuracy | A | rgb+depth Δ ≤ 0 in Sub-exp A |
| **H3** | `siglip+depth` accuracy > `siglip_only` accuracy | B | siglip+depth Δ ≤ 0 in Sub-exp B |

H2 is the entry gate. H1 is the primary claim. H3 is the hardest test.

**External evidence for H3**: Two 2025–2026 studies directly support the H3
hypothesis before this experiment runs. GeoAware-VLA (Qiu et al., 2025,
arXiv:2509.14117) shows that injecting a frozen geometric encoder (VGGT) on
top of a DINOv2+SigLIP backbone improves unseen-viewpoint success by +35pp on
LIBERO — demonstrating that DINOv2's dense spatial features do not subsume
explicit geometric signal. UniLACT (arXiv:2602.20231) shows that depth-aware
latent pretraining outperforms RGB-only latent learning in VLA downstream tasks.
Neither uses hand-crafted statistics; both establish that the geometric signal
gap exists at the architecture level, which this probe tests at the feature
level.

---

## 2. Experimental Design

### 2.1 Probe Protocol

We use **linear probing**: frozen features extracted from each object
instance, fed to logistic regression with no hidden layers. The key
property: if logistic regression cannot extract signal, the signal is
either absent or not linearly accessible. This protocol is standard for
evaluating representation quality (cf. DeFM, arXiv:2601.18923).

For each object instance (one segmented mask from one frame):
1. Extract hand-crafted RGB statistics from the masked region → 9-dim vector
2. Extract hand-crafted depth statistics from the co-registered depth map → 27-dim vector
3. Compare feature sets: `rgb_only`, `depth_only`, `rgb+depth`, and `siglip` variants

Classifier: `StandardScaler → LogisticRegression(C=1.0, max_iter=2000)`

### 2.2 Why Episode-Level Cross-Validation

The first pilot probe used StratifiedKFold across frames. This is the wrong
CV strategy for manipulation video data:

- Consecutive frames within an episode share the same scene geometry, lighting,
  and object pose. Testing on frame 7 when trained on frames 0–6 from the same
  episode measures within-scene consistency, not generalization.
- In the pilot, 130 instances came from only 3 episodes. The `cabinet` and
  `bottom_drawer` categories came exclusively from episode 43 (89/130 instances).
  StratifiedKFold allocated those 89 frames as 80% train / 20% test within the
  same episode — measuring scene memorization, not transfer.

Correct approach: **GroupKFold(groups=episode_idx)**. Each fold's test set
contains entire episodes that never appear in training, measuring whether
features learned on one set of episodes transfer to unseen episodes.

CV strategy by episode count:

| Condition | Strategy |
|---|---|
| ≤ 10 episodes in category | LeaveOneGroupOut (exact LOO) |
| > 10 episodes | GroupKFold(n_splits=min(10, n_episodes)) |

Categories with fewer than 5 distinct episodes are dropped (insufficient
CV reliability).

### 2.3 Pilot Calibration Narrative

This subsection documents a confirmed bug catch from the pilot run, preserved
here as a methodological record.

**The bug**: In the pilot, both StratifiedKFold (wrong CV strategy) and missing
StandardScaler (wrong feature scaling) were active simultaneously. RGB-only
accuracy read 0.79, accompanied by `LogisticRegression` convergence warnings.
The 0.79 figure was inflated by two independent sources of error:

1. **Within-scene leakage** (StratifiedKFold): the model memorized scene
   appearance rather than learning transferable features.
2. **Scale dominance** (no StandardScaler): RGB pixel values (0–255) swamped
   depth statistics (0–1 or 0–N meters) in the gradient, making the classifier
   ignore depth entirely.

After fixing both — switching to GroupKFold and adding StandardScaler — RGB-only
dropped to 0.55. The convergence warnings resolved simultaneously, confirming
the scale mismatch as the primary optimization pathology.

The 0.24-point drop cannot be fully attributed to StandardScaler alone; the CV
fix contributed by removing within-scene memorization. Both fixes were applied
together; they are not individually isolated in this pilot.

---

## 3. Feature Engineering

### 3.1 RGB Features (9-dim)

For each object instance, extract pixels within the mask bounding box
(+12px padding), masked to the instance shape:

| Feature | Dim | Description |
|---|---|---|
| R mean, std, median | 3 | Red channel statistics |
| G mean, std, median | 3 | Green channel statistics |
| B mean, std, median | 3 | Blue channel statistics |

The +12px padding provides context at object boundaries where the mask
may clip foreground pixels; 12px was chosen as one typical object-edge
margin at the resolutions used (~480px height). A sensitivity check
(0px, 6px, 12px, 24px) is planned to confirm results are not
padding-dependent.

These capture color distribution but not shape or geometry.

### 3.2 Depth Features — depth_rich (27-dim)

Same crop + mask, applied to the co-registered depth map. Features are
organized into five groups reflecting distinct geometric primitives.

**Group 1 — Basic distribution (5-dim)**

| Feature | Dim | Description |
|---|---|---|
| mean, std, min, max, median | 5 | Basic depth distribution statistics |

**Group 2 — Planar surface fit (2-dim)**

| Feature | Dim | Description |
|---|---|---|
| planarity_residual | 1 | RMS of (depth − RANSAC best-fit plane) using only inlier pixels. Near 0 for flat surfaces (cabinet top), large for curved/protruding surfaces. More robust than least-squares fit against DA3 edge-bleeding artifacts |
| planarity_inlier_ratio | 1 | Fraction of mask pixels classified as inliers by RANSAC (threshold = 5 mm). Flat objects → high ratio (≥ 0.85); complex shapes → lower ratio. Independent diagnostic for surface planarity |

RANSAC parameters: `max_trials=100`, `residual_threshold=0.005` (metric) or
`0.02 × depth_range` (relative). Inlier ratio replaces least-squares fit as
the primary planarity indicator; residual is secondary. RANSAC plane fitting
on depth maps follows standard practice in tabletop manipulation perception
(Fischler & Bolles, 1981; Rusu et al., 2009).

**Group 3 — Surface orientation (8-dim)**

| Feature | Dim | Description |
|---|---|---|
| depth_HOG (8 bins) | 8 | Gradient orientation histogram over the masked region. Gradient magnitude (Scharr filter) used as weights; orientations binned into 8 equal sectors (0°–360°, 45° each). Normalized to sum to 1. Captures dominant surface orientation: vertical cylinder → symmetric left-right bins; tilted plane → single dominant bin; isotropic surface → uniform distribution |

Replaces the previous `grad_x mean/std` + `grad_y mean/std` (4-dim). The
orientation histogram retains direction information that component-wise
statistics discard. Net dimension change: −4 + 8 = +4. Depth-domain HOG
follows Wang et al. (CVPR 2012) and Oreifej & Liu (CVPR 2013), who
established gradient-orientation histograms on depth maps as effective
geometric descriptors for RGB-D recognition tasks.

```python
gx = cv2.Scharr(depth, cv2.CV_64F, 1, 0)
gy = cv2.Scharr(depth, cv2.CV_64F, 0, 1)
magnitude  = np.sqrt(gx**2 + gy**2)[mask]
orientation = (np.arctan2(gy, gx)[mask] + np.pi)  # [0, 2π]
hog, _ = np.histogram(orientation, bins=8, range=(0, 2*np.pi),
                      weights=magnitude)
hog = hog / (hog.sum() + 1e-8)
```

**Group 4 — Depth distribution shape (6-dim)**

| Feature | Dim | Description |
|---|---|---|
| iqr | 1 | Interquartile range — robust depth spread |
| percentile_10, percentile_90 | 2 | Robust depth range endpoints |
| skewness | 1 | Asymmetry of depth distribution. Negative → front-heavy (protruding center); positive → rim-heavy (bowl/cup) |
| kurtosis | 1 | Peakedness of depth distribution. Low → uniform depth (flat surface); high → depth concentrated at single value (single dominant plane close to camera). Complements skewness: skewness captures asymmetry, kurtosis captures concentration |
| entropy (binned, 32 bins) | 1 | Information content of depth histogram. Flat objects → low entropy; complex geometry → high entropy. Bin count fixed at 32 to reduce sensitivity to bin choice |

Masked depth distribution statistics (mean, std, percentiles, entropy) for
RGB-D object classification follow Lai et al. (ICRA 2011), who demonstrate
that per-channel depth statistics extracted from segmented object regions
are discriminative features for category recognition across viewpoints.

**Group 5 — Spatial structure (4-dim)**

| Feature | Dim | Description |
|---|---|---|
| quadrant_mean_TL, TR, BL, BR | 4 | Mean depth within each quadrant of the mask bounding box (top-left, top-right, bottom-left, bottom-right). Captures tilt direction and asymmetry that global statistics miss: a left-leaning surface has quadrant_TL < quadrant_TR; an opening drawer has quadrant_BL < quadrant_TL |

Spatially-partitioned depth means follow regional depth aggregation used
in grasp pose detection from depth images (ten Pas et al., IJRR 2017),
where local depth means across spatial subregions encode surface tilt
direction for contact point estimation.

**Group 6 — Curvature (2-dim)**

| Feature | Dim | Description |
|---|---|---|
| laplacian_mean | 1 | Mean absolute value of the depth Laplacian (∇²depth) within the mask. Near 0 for flat and tilted planes; large for curved surfaces (bowls, cylinders). First derivative (HOG) captures orientation; second derivative captures bending |
| laplacian_std | 1 | Standard deviation of ∇²depth. Low → uniformly curved or uniformly flat; high → spatially varying curvature (handles, ridges) |

Laplacian-based curvature estimation from depth maps follows Boulch &
Marlet (ECCV 2012), who use second-order depth derivatives as a fast
approximation of mean curvature for surface normal estimation on unorganized
point clouds.

```python
lap = cv2.Laplacian(depth.astype(np.float64), cv2.CV_64F)
laplacian_mean = np.mean(np.abs(lap[mask]))
laplacian_std  = np.std(lap[mask])
```

**Total: 27-dim.** Summary: 5 (basic) + 2 (planarity) + 8 (HOG) + 6 (shape)
+ 4 (spatial) + 2 (curvature). All statistics computed within the masked
instance region only, not the full bounding box.

**Dimension summary vs. v1 (15-dim)**

| Group | v1 dims | v2 dims | Change | Rationale |
|---|---|---|---|---|
| Basic distribution | 5 | 5 | = | — |
| Planar fit | 1 (LS residual) | 2 (RANSAC residual + inlier ratio) | +1 | Robustness to edge artifacts; inlier ratio as independent feature |
| Surface orientation | 4 (grad x/y mean+std) | 8 (HOG 8-bin) | +4 | Restores direction information discarded by component statistics |
| Distribution shape | 5 (iqr, pct_10, pct_90, skew, entropy) | 6 (+kurtosis; entropy bins fixed at 32) | +1 | Kurtosis captures concentration; fixed bins reduce entropy instability |
| Spatial structure | 0 | 4 (quadrant means) | +4 | Only group capturing within-object spatial layout |
| Curvature | 0 | 2 (Laplacian mean+std) | +2 | Second-order derivative; distinguishes curved from flat/tilted |
| **Total** | **15** | **27** | **+12** | — |

**On feature design epistemology**: These 27 features capture geometric
primitives — planarity, surface orientation, depth distribution shape,
curvature — that are universal physical properties of rigid 3D objects,
not properties specific to any object in this dataset. Each feature group
maps to an established line of 3D computer vision work: RANSAC planarity
(Fischler & Bolles, 1981; Rusu et al., *ICRA* 2009), depth-domain HOG
(Wang et al., *CVPR* 2012; Oreifej & Liu, *CVPR* 2013), masked depth
distribution statistics (Lai et al., *ICRA* 2011), spatial depth
partitioning (ten Pas et al., *IJRR* 2017), and Laplacian curvature from
depth maps (Boulch & Marlet, *ECCV* 2012). The features are
hypothesis-driven from physical first principles — planarity distinguishes
flat surfaces from curved ones in any scene; gradient orientation
characterizes cylindrical vs. wedge profiles regardless of object identity
— and are not derived by inspecting which categories appear in LIBERO or
BridgeData V2.

**Known confound — DA3 edge artifacts on planarity features**: Depth
estimation networks including DA3 produce edge-bleeding artifacts near
occlusion boundaries. RANSAC planarity mitigates this by excluding outlier
pixels (inliers exclude boundary pixels that fail the residual threshold),
but does not eliminate it. The `planarity_inlier_ratio` provides a
per-instance diagnostic: a very low ratio (<0.5) on an object expected to
be planar signals artifact contamination.

### 3.3 SigLIP-SO400M Features (1152-dim) — Sub-exp B only

**Architecture correction**: OpenVLA (Kim et al., 2024, arXiv:2406.09246)
uses a **fused backbone: DINOv2-L/14 + SigLIP-SO400M/14**, combined via
learned projection layers. DINOv2 contributes dense spatial features and
geometric understanding; SigLIP contributes language-aligned semantic
features. OpenVLA is RGB-only — no depth is used. This probe uses
SigLIP-SO400M in isolation (not the full DINOv2+SigLIP fusion) for H3,
which makes the comparison more conservative: the standalone SigLIP
baseline is weaker than OpenVLA's actual encoder, so a depth gain here
is a lower-bound estimate of depth's contribution over the full OpenVLA
backbone.

SigLIP-SO400M (`google/siglip-so400m-patch14-384`) is used here as the
internet-scale learned feature baseline to test H3. If depth_rich adds
signal on top of SigLIP alone, it almost certainly also adds signal on top
of the DINOv2+SigLIP fusion — a stronger claim than H3 as formulated.

**Why Sub-exp B (BridgeData V2) and not LIBERO**: SigLIP was trained on
real photographs (LAION, WebLI). LIBERO is rendered via robosuite/MuJoCo
with non-photorealistic Phong shading. The domain gap means SigLIP
activations on simulation frames may not reflect the model's true
representational capacity — a low score could indicate domain mismatch,
not representational failure. BridgeData V2 is real robot footage; SigLIP
is in-distribution on this data, making the comparison fair.

Extraction: crop the instance bbox, resize to 384×384, run through SigLIP,
take `pooler_output` (1152-dim global representation). No fine-tuning.

---

## 4. Datasets

### 4.1 Sub-experiment A — LIBERO (Franka arm, simulation, metric depth)

**Role in evidence chain**: H1 + H2 under controlled geometric conditions;
metric depth pipeline validation.

**Source**: `lerobot/libero` (LeRobot v3 format, 1693 episodes total)

**Selected episodes**: 202 episodes across 5 task groups.

All 5 tasks involve the same physical cabinet object in different geometric
configurations. This is intentional: the depth signal of interest is
geometric (drawer protrudes from cabinet face vs cabinet top is flush),
and using a shared object gives a clean test of that geometric hypothesis
while controlling for texture.

| Task | Episode range | N episodes | Geometric configuration |
|---|---|---|---|
| Put black bowl in bottom drawer, close it | ep 43–77 | 35 | Bottom drawer protruding |
| Open the middle drawer | ep 388–429 | 42 | Middle drawer protruding |
| Open top drawer, put bowl inside | ep 430–462 | 33 | Top drawer protruding |
| Put bowl on top of cabinet | ep 602–648 | 47 | Cabinet top surface (flat) |
| Put wine bottle on top of cabinet | ep 726–770 | 45 | Cabinet top surface (flat) |

Total: 202 episodes × 8 frames/episode = 1,616 frames.
Random-chance baseline: 1/4 = **25.0%** (4 target categories:
`bottom_drawer`, `middle_drawer`, `top_drawer`, `cabinet_top`).
Only the target furniture piece is retained per episode; co-present
objects such as `robot_hand`, `bowl`, and `wine_bottle` are excluded
from the probe (see Instance selection below).

**Geometric hypothesis**: The five tasks create two groups:
- **Drawer tasks** (ep 43–77, 388–429, 430–462): the drawer face protrudes
  from the cabinet body — lower mean depth, high planarity_residual.
- **Cabinet top tasks** (ep 602–648, 726–770): the top surface is flush —
  more uniform depth, low planarity_residual.

`planarity_residual` and `mean_depth` should separate these two groups
even when RGB appearance is similar (same brown wooden cabinet texture).

**Depth type**: metric (meters). Camera intrinsics derived exactly:
`f = (H/2) / tan(fovy/2) = (512/2) / tan(22.5°) = 618.1 px`.
This is not an estimate — robosuite renders with this focal length exactly.

Metric depth conversion: `depth_m = focal_px × raw / 300`

> **Note on the `300` constant**: DA3 normalizes its raw disparity output
> such that a reference plane at 1 meter maps to approximately 300 for
> typical indoor focal lengths. This constant was calibrated empirically
> against robosuite's ground-truth depth buffer on a held-out episode from
> a task group *not included in the probe* (a reaching task outside the
> 5 selected groups), ensuring no circular calibration on probe data.
> Resulting metric values agreed with the GT buffer to within ±3 cm at
> 0.5 m range. If a different environment or focal length is used, this
> constant must be re-derived.

**Instance selection and label assignment**: Each episode has one target
object (the furniture piece being interacted with). Qwen3-8B parses the
task string to produce a canonical episode-level label (e.g., "Put black
bowl in bottom drawer" → `bottom_drawer`). For each frame, SAM3 produces
multiple masks; only the mask corresponding to the target object is kept.
Target mask identification uses Grounded-SAM2 text query (query string =
canonical label, e.g., "drawer"); the highest-scoring mask above IoU
threshold 0.5 is selected. Frames where no mask exceeds threshold are
dropped. One (feature_vector, label) pair is produced per kept frame.
`robot_hand`, `bowl`, `wine_bottle`, and other co-present objects are
excluded — they are not the geometric comparison targets.

Expected categories: `bottom_drawer`, `middle_drawer`, `top_drawer`,
`cabinet_top`. Minor normalization (lowercasing, synonym collapse) applied
post-hoc. A 5% random sample of labels is manually spot-checked before
final reporting.

**CV**: GroupKFold on episode_idx. With 35–47 episodes per task group,
most categories get GroupKFold(10) folds.

**Known confound**: All 5 tasks use the same cabinet 3D mesh in the same
simulated room. Episode-level CV controls for within-episode leakage but
cannot isolate whether the model is learning geometry vs memorizing this
specific texture. Conclusions are scoped to "within this simulation
environment and these task configurations."

### 4.2 Sub-experiment B — BridgeData V2 (Franka arm, real world, H3 test)

**Role in evidence chain**: H3 — tests whether depth adds signal on top
of SigLIP in a real-world, in-distribution setting (relative depth). This
sub-experiment exists specifically to address the domain gap that would
confound H3 if run on simulated LIBERO data. Relative depth (no metric
conversion) is intentional here: Ablation I-2 (run on Sub-exp A) isolates
whether metric scale is load-bearing under controlled RGB conditions;
Sub-exp B tests whether depth adds signal over a learned 400M-param encoder
regardless of scale.

**Source**: `lerobot/bridge_v2` (HuggingFace, LeRobot v3 format,
~60k episodes total, publicly released).

**Why BridgeData V2**:
- Real robot footage → SigLIP is in-distribution (no domain gap)
- Franka arm → same embodiment as LIBERO's simulated robot, forming a
  sim→real parallel
- Object diversity (bowls, bottles, cups, boxes, toys) provides geometric
  variety sufficient to test depth signal

**Selected episodes**: 150 episodes sampled across object categories with
distinct geometric profiles (flat, cylindrical, irregular). Exact selection
defined in `scripts/bridge_sample.py` (stratified by task string, random
seed = 42).

**Instance selection and label assignment**: Qwen3-8B parses each task
string to a canonical object label. Grounded-SAM2 queries that label per
frame; highest-scoring mask above IoU 0.5 is kept. Expected categories:
`bowl`, `bottle`, `cup`, `box`. Spot-check 5% of labels before reporting.

**Depth type**: relative (no calibration parameters available for BridgeData
cameras). DA3 raw output saved directly.

**CV**: Episode-level GroupKFold(n_splits=10).

Random-chance baseline: 1/K where K = number of confirmed categories after
Qwen labeling (estimated 4–5 → **~20–25%**; exact value computed post-hoc).

**Sample size note**: 150 episodes across 4–5 categories yields approximately
30–38 episodes per category (4 categories: ~37–38; 5 categories: ~30). With GroupKFold(10), each test fold contains
only 2–4 episodes per category. Accuracy estimates from Sub-exp B carry
high variance and should be interpreted directionally (positive vs. negative
Δ) rather than as precise magnitude estimates. Sub-exp B's primary role is
to establish the H3 direction; magnitude claims require larger episode counts.

---

## 5. Analysis Plan

### 5.1 Primary Comparison (per sub-experiment)

For each feature set, report:
- Mean **balanced accuracy** across CV folds ± standard deviation
- Δ improvement over `rgb_only` (in balanced accuracy points)
- Random-chance baseline (1/K per class, equivalent to balanced accuracy of a uniform random predictor)

**Why balanced accuracy**: Sub-exp A has a 2:1 class imbalance —
`cabinet_top` combines two tasks (ep 602–648, 726–770) yielding 92
episodes vs. 33–42 for each drawer category. A majority-class predictor
achieves ~45.5% raw accuracy by always predicting `cabinet_top`, far above
the 25% random baseline. Balanced accuracy (mean per-class recall) removes
this bias: a majority-class predictor scores exactly 25% regardless of
class frequencies. Sub-exp B (BridgeData V2, stratified sampling) is approximately balanced;
balanced accuracy is reported uniformly across both sub-experiments for
consistency.

Raw accuracy is reported alongside balanced accuracy as a secondary figure.

| Feature set | Dim | H tested | Sub-exp |
|---|---|---|---|
| rgb_only | 9 | — (baseline) | A, B |
| depth_rich | 27 | H2 | A |
| rgb + depth_rich | 36 | H1 | A, B |
| siglip_only | 1152 | — (learned baseline) | B |
| siglip + depth_rich | 1179 | H3 | B |

### 5.2 Decision Criteria

**H1/H2 positive threshold**: Δ ≥ +0.05 accuracy points in Sub-exp A.

This threshold is grounded in downstream practical significance, not noise
floor. A systematic review of 10 manipulation studies (2021–2026) shows
that depth as an architectural input yields task success improvements of
+6pp to +36pp over RGB-only baselines, with a median of ~+18pp (Shridhar
et al., 2022, *PerAct*: +20pp; Zeng et al., 2021, *Transporter*: +36pp;
Ze et al., 2024, *3D Diffusion Policy*: +20pp; [authors], 2024, *Lift3D*:
+6pp — the lowest reported Δ across all reviewed studies). The
methodologically closest analog to this probe — depth injection into a
frozen pretrained policy backbone without architecture change (*"Depth
Helps"*, arXiv:2408.05107) — reports Δ = +8–15pp. Kornblith et al. (2019)
and Chen et al. (2020, *SimCLR*) establish that linear probe accuracy is
a monotonic proxy for downstream transfer performance (Pearson r ≈ 0.85).
A probe Δ of +0.05 on a ~0.50 baseline (10% relative) is below the
literature floor (+6pp) and represents a conservative minimum. Deltas
below this cannot be distinguished from noise and would not justify
retaining depth as a training signal.

**GRADE evidence level for H1**: ⊕⊕⊕○ Moderate. Direction is consistent
across all 10 reviewed studies; absolute magnitude varies by task type and
architecture. Downgraded from High due to: (a) no preregistered controls,
(b) publication bias (no negative-result studies found), (c) architecture
confounding in most studies.

**H3 positive threshold**: Δ ≥ +0.03 accuracy points in Sub-exp B.
A lower bar is appropriate because: (a) SigLIP already captures substantial
semantic signal, so marginal geometric gains will be smaller; (b) the
probe uses SigLIP in isolation, which is weaker than OpenVLA's full
DINOv2+SigLIP fusion — any positive Δ here conservatively lower-bounds
the gain over the full fusion backbone. GRADE evidence level for H3:
⊕⊕○○ Low — only 2 direct studies (GeoAware-VLA, UniLACT), both using full
geometric models rather than hand-crafted statistics.

**Statistical supplement**: A permutation test (500 label permutations,
same GroupKFold structure) is run for `rgb+depth` in Sub-exp A.
Label permutations are applied at the episode level — all frames from
one episode receive the same permuted label — preserving the GroupKFold
structure and matching the null hypothesis to the CV design. Frame-level
permutation would underestimate the null variance by ignoring within-episode
correlation and is not used. If the observed mean accuracy falls in the
top 5% of the permutation null distribution, report p < 0.05
(permutation-based). This supplements but does not replace the Δ-based
decision.

### 5.3 Confusion Analysis (LIBERO)

Confusion matrices are computed on the **test folds only** (held-out
episodes that never appear in training). The logistic regression classifier
is fit on training-fold feature vectors; the confusion matrix reports its
predictions on unseen episodes. Training-set confusion is not reported —
logistic regression nearly perfectly memorizes training labels and provides
no diagnostic information.

What is being "trained": only the logistic regression weight matrix
W (shape: n_classes × 27) and bias b. DA3, SAM3, and Grounded-SAM2 are
all frozen; the 27-dim feature vector is computed deterministically from
the depth map. Probe accuracy reflects the linear separability of the
hand-crafted features, not model capacity.

Inspect the confusion matrix for cabinet/drawer categories specifically:

- **Pre-depth (rgb_only)**: does the model confuse `bottom_drawer` with
  `top_drawer`? (Similar RGB appearance, different depth signature)
- **Post-depth (rgb+depth)**: does adding depth_rich reduce cross-drawer
  confusion?

If the geometric hypothesis holds, drawer-vs-drawer off-diagonal entries
should decrease because `mean_depth` and `planarity_residual` differ
systematically between drawer positions.

**Alternative explanation to test**: Even if confusion decreases, the
improvement could reflect camera distance (lower drawers are closer to
the camera than upper drawers) rather than true geometric shape
discrimination. To distinguish: inspect `LogisticRegression.coef_`
magnitude per feature group:
- If `mean_depth` and `quadrant_mean_BL/BR` dominate → distance proxy
- If `planarity_residual`, `planarity_inlier_ratio`, `laplacian_mean`
  dominate → genuine geometric shape discrimination
- If `depth_HOG` bins 0/4 (left-right symmetric) dominate for
  `bottom_drawer` vs `top_drawer` → orientation signal, not distance

### 5.4 Ablation Experiments

All ablations reuse the feature extraction pipeline from the core experiment;
no new model runs are required unless noted.

---

**Ablation R-1 — Feature Group Drop-One-Out**

Run six additional probe variants, each with one feature group zeroed out:

| Variant | Removed group | Dims | Expected signal |
|---|---|---|---|
| `depth_no_basic` | Group 1 (basic distribution) | 22 | If accuracy stable → mean/std not discriminative alone |
| `depth_no_planarity` | Group 2 (RANSAC planarity) | 25 | If cabinet/drawer confusion rises → confirms geometric hypothesis |
| `depth_no_HOG` | Group 3 (orientation histogram) | 19 | If cylinder accuracy drops → orientation signal is real |
| `depth_no_shape` | Group 4 (skewness, kurtosis, entropy, IQR) | 21 | kurtosis + skewness independent contribution |
| `depth_no_spatial` | Group 5 (quadrant means) | 23 | Spatial layout contribution in isolation |
| `depth_no_curvature` | Group 6 (Laplacian) | 25 | Second-order geometry contribution |

Implementation: column-mask the 27-dim feature matrix; re-run
`StandardScaler → LogisticRegression` on the same CV splits.
Output: one bar chart showing Δ accuracy vs full 27-dim for each group
removal. Groups that cause the largest drop are the most load-bearing.

---

**Ablation I-1 — GT Depth vs DA3 Estimated Depth (Sub-exp A only)**

LIBERO's robosuite renders ground-truth depth buffers alongside RGB frames.
Re-extract all 27 depth features using the GT depth buffer instead of DA3
output, and compare probe accuracy:

| Feature set | Depth source | Expected result |
|---|---|---|
| `depth_rich_GT` | robosuite GT buffer | Upper bound on probe accuracy |
| `depth_rich_DA3` | DA3 estimated | Current pipeline output |

Accuracy gap = DA3 estimation error cost for this classification task.
- Gap < 0.03: DA3 is sufficient; estimation noise does not significantly
  degrade geometric signal.
- Gap ≥ 0.05: DA3 is a bottleneck; Metric3D v2 or stereo depth should
  be evaluated as replacements.

Requires: access to the raw robosuite depth buffer. If not pre-saved,
add `env.render(depth=True)` to the data collection script and cache.

---

**Ablation I-2 — Metric vs Forced-Relative Depth (Sub-exp A only)**

Test whether absolute metric scale carries discriminative signal beyond
relative depth ordering. Post-process the already-computed depth feature
vectors: divide each depth value by the mask-region mean before
computing features (equivalent to running DA3 without metric conversion).

| Feature set | Scale | Expected result |
|---|---|---|
| `depth_rich_metric` | Absolute meters | Current Sub-exp A |
| `depth_rich_relative` | Normalized (÷ mean) | Metric scale stripped |

If accuracy drops significantly (Δ ≥ 0.05), absolute scale is
load-bearing — objects at different absolute distances from the camera
carry category information. If accuracy is stable, relative depth ordering
alone is sufficient, establishing that Sub-exp B's relative depth signal
is directly interpretable alongside Sub-exp A.

This is the formal metric-vs-relative test in the evidence chain: it
isolates the scale question under controlled RGB conditions (same LIBERO
cabinet objects), which no other sub-experiment can do cleanly.

---

**Ablation A-1 — Instance Mask vs Bounding Box Crop**

Re-extract all 27 features using rectangular bounding box regions instead
of SAM3 instance masks (same +12px padding, no mask applied):

| Feature set | Region | Expected result |
|---|---|---|
| `depth_rich_mask` | SAM3 instance mask | Current pipeline |
| `depth_rich_bbox` | Bounding box rectangle | No segmentation |

Accuracy gap = SAM3 segmentation contribution to feature quality.
- Gap ≈ 0: mask shape does not help; a simpler detector (e.g., YOLO bbox)
  would suffice for depth feature extraction.
- Gap ≥ 0.05: precise instance boundaries matter; SAM3 is earning its
  compute cost.

Additionally, bbox features serve as a robustness check: if bbox accuracy
is already high, the probe's positive results do not depend on SAM3
quality, making the conclusions more general.

---

**Ablation T-1 — Per-Frame vs Episode-Level Aggregation**

Current probe classifies each frame independently. Alternative: aggregate
the 8 frames per episode by computing mean and standard deviation of the
feature vector across frames, yielding a 54-dim episode-level descriptor
(27 mean + 27 std). CV remains episode-level GroupKFold (now one sample
per episode rather than 8).

| Feature set | Temporal | Dims | Expected result |
|---|---|---|---|
| `depth_rich_perframe` | Per frame | 27 | Current pipeline |
| `depth_rich_episode` | Episode mean+std | 54 | Multi-view aggregation |

If episode-level accuracy > per-frame accuracy, depth signal has temporal
consistency that single-frame sampling underutilizes — the pipeline could
accumulate evidence across frames rather than treating each independently.
The std component captures within-episode depth variation (e.g., object
moving during manipulation), which single-frame statistics cannot access.

---

**Ablation summary table**

| ID | What varies | Datasets | Extra compute | Primary question answered |
|---|---|---|---|---|
| R-1 | Feature group removed | A, B | Zero | Which geometric primitive drives classification |
| I-1 | GT vs DA3 depth | A only | Re-extract features | Is DA3 estimation quality a bottleneck |
| I-2 | Metric vs forced-relative | A only | Zero (post-process) | Does absolute scale carry signal |
| A-1 | Mask vs bbox region | A, B | Re-extract features | Does SAM3 segmentation quality matter |
| T-1 | Per-frame vs episode mean+std | A, B | Zero (groupby) | Is temporal aggregation better than single-frame |

### 5.5 Visualization Plan

| Plot | Purpose |
|---|---|
| Accuracy bar chart ± std (per feature set, per sub-experiment) | Primary result summary |
| Confusion matrix heat map — rgb_only vs rgb+depth (LIBERO, Sub-exp A) | Geometric hypothesis visualization: does depth reduce drawer-vs-drawer confusion? Quantitative test of §5.3 prediction |
| Confusion matrix heat map — siglip_only vs siglip+depth (BridgeData V2, Sub-exp B) | Descriptive: which categories gain from adding depth on top of SigLIP; identifies where SigLIP's semantic features are insufficient |
| Per-fold accuracy line plot (rgb_only, rgb+depth overlaid) | Confirm improvement is consistent across folds, not one-fold driven |
| Feature group importance bar chart (coef_ magnitude summed per group) | Which of the 6 feature groups drives classification; camera-distance proxy check (Group 1 basic vs Group 2 planarity vs Group 3 HOG vs Group 5 spatial) |
| Depth-HOG polar plot (per category mean) | Visualize dominant surface orientation per object class; expected: coke → symmetric left-right; book → uniform; drawer → single dominant direction |
| Quadrant depth heat map (2×2, per category mean) | Visualize spatial depth asymmetry; expected: tilted drawer → asymmetric quadrants; flat cabinet top → uniform |
| planarity_inlier_ratio distribution (per category) | Diagnose DA3 edge artifact severity per class; low ratio (<0.5) flags artifact contamination |

---

## 6. Honest Limitations

| Limitation | Scope of impact |
|---|---|
| Single simulation environment (LIBERO) | Cabinet always the same 3D mesh / texture; cannot separate "depth encodes geometry" from "depth encodes this specific texture" |
| Features are hypothesis-driven, not discovered | The 27 depth features were designed from physical first principles across six geometric primitive groups (distribution, planarity, orientation, shape, spatial, curvature) — they are not post-hoc tuned to the specific objects in this dataset. A fully agnostic feature search (e.g., DA3 intermediate layer features, FPFH point cloud descriptors) might find different or stronger signal; the 27-dim set is an informed upper bound on hand-crafted geometric coverage, not a discovered optimum |
| Camera-distance confound | Depth mean encodes distance from camera, which correlates with object size in-frame and with category. Some of the depth signal in H1 may reflect "how far from camera does this object type typically appear" rather than intrinsic geometric shape. Feature importance analysis (§5.3) provides a partial diagnostic but cannot fully disentangle these |
| DA3 edge artifacts on planarity_residual | Edge-bleeding artifacts at mask boundaries inflate planarity_residual for all objects; discriminative power depends on whether inflation is differential across categories (related to mask aspect ratio) |
| Hand-crafted features only | Results say depth *statistics* help; a depth encoder (e.g., DPT, DA3 intermediate features) might find stronger or weaker signal |
| No significance testing beyond permutation | With 10 folds, the test is underpowered for small deltas (< 0.05 acc); permutation test is indicative, not confirmatory |
| Single classifier (logistic regression) | Negative result means "not linearly accessible", not "no signal"; a non-linear probe (MLP, kernel SVM) may differ |
| Qwen3-8B label quality | Category labels are LLM-generated from task strings. Label errors (synonym inconsistency, multi-object scenes) reduce apparent accuracy without reflecting true probe performance. 5% manual spot-check is required before final reporting |
| BridgeData V2 object diversity | Episode sample (150) covers a subset of BridgeData's ~60k episodes; sampled categories may not span the full geometric diversity of the dataset |
| Publication bias in supporting literature | A systematic review of 10 studies (2021–2026) found zero negative-result papers on depth contribution to manipulation. This almost certainly overstates depth's universal utility. The reviewed studies predominantly chose tasks where depth is expected to matter (precise placement, stacking). A negative H1 result from this probe would be a rare and publishable boundary-condition finding, not an experimental failure |
| SigLIP-only baseline underestimates OpenVLA | Sub-exp B uses standalone SigLIP, not OpenVLA's full DINOv2+SigLIP fusion. A positive H3 result is conservative (depth beats a weaker baseline); a negative H3 result does not rule out depth gains over the full fusion backbone |

---

## 7. Connection to Pipeline and Portfolio Narrative

### 7.1 Internal Validation

The pipeline produces metric depth over LeRobot v3 episodes. Before claiming
depth is a useful training signal for a world model, it is scientifically
responsible to check that the produced maps carry information not already in
RGB. The probe provides this check. A positive result supports downstream use
of depth in representation learning; a negative result would redirect effort
toward better depth feature design or pipeline debugging.

**On the value of a negative result**: A systematic review of the supporting
literature identified severe publication bias — all 10 reviewed studies report
positive depth contributions; no boundary-condition studies exist. A negative
H1 result from this probe (depth adds no classifiable signal on these datasets)
would be a genuine and rare contribution: it would identify the conditions under
which depth fails to contribute linearly-accessible geometric signal, which is
equally important for deciding where to invest annotation effort.

### 7.2 Methodological Demonstration

The probe's design demonstrates skills directly relevant to representation
learning labs:

- **Confound identification and correction**: detecting and fixing within-episode
  CV leakage (same-scene memorization masquerading as generalization)
- **Data quality debugging**: using model convergence warnings as a signal of
  feature-scale bugs; catching a doubly-inflated pilot result before reporting
- **Cross-embodiment thinking**: same pipeline, metric vs relative depth,
  simulated vs real, Franka arm vs humanoid — testing whether conclusions transfer
- **Domain gap awareness**: recognizing that SigLIP comparison requires real-world
  data and restructuring the experiment accordingly rather than reporting a
  confounded baseline

### 7.3 What This Does NOT Claim

- That 9+27 hand-crafted features are the right representation for policy learning
- That depth is necessary for world model training (this is an open research question)
- That results generalize beyond the specific datasets, cameras, and object sets tested
- That the annotation pipeline produces depth maps of any particular absolute accuracy
  (only that the maps carry classifiable geometric signal relative to RGB)

---

---

## 8. Key References

Literature reviewed and cited in this document (APA 7.0 abbreviated):

| ID | Citation | arXiv / DOI |
|---|---|---|
| Fischler & Bolles, 1981 | Fischler, M. A., & Bolles, R. C. Random sample consensus: A paradigm for model fitting. *CACM* 24(6), 381–395 | doi:10.1145/358669.358692 |
| Rusu et al., 2009 | Rusu, R. B., et al. Fast 3D recognition and pose using the viewpoint feature histogram. *ICRA 2009* | — |
| Wang et al., 2012 | Wang, J., et al. Mining actionlet ensemble for action recognition with depth camera. *CVPR 2012* | — |
| Oreifej & Liu, 2013 | Oreifej, O., & Liu, Z. HON4D: Histogram of oriented 4D normals for activity recognition. *CVPR 2013* | — |
| Lai et al., 2011 | Lai, K., et al. A large-scale hierarchical multi-view RGB-D object dataset. *ICRA 2011* | — |
| ten Pas et al., 2017 | ten Pas, A., et al. Grasp pose detection in point clouds. *IJRR* 36(13–14), 1455–1473 | doi:10.1177/0278364917735594 |
| Boulch & Marlet, 2012 | Boulch, A., & Marlet, R. Fast and robust normal estimation for point clouds with sharp features. *ECCV 2012* | — |
| Chen et al., 2020 | Chen, T., et al. A simple framework for contrastive learning. *ICML 2020* | — |
| Kornblith et al., 2019 | Kornblith, S., et al. Do better ImageNet models transfer better? *CVPR 2019* | — |
| Nair et al., 2022 | Nair, S., et al. R3M: A universal visual representation for robot manipulation. *CoRL 2022* | arXiv:2203.12601 |
| Zeng et al., 2021 | Zeng, A., et al. Transporter networks. *CoRL 2021* | arXiv:2010.14406 |
| Shridhar et al., 2022 | Shridhar, M., et al. Perceiver-Actor. *CoRL 2022* | arXiv:2209.05451 |
| Ze et al., 2024 | Ze, Y., et al. 3D diffusion policy. *RSS 2024* | arXiv:2403.03954 |
| Ke et al., 2024 | Ke, T. W., et al. 3D diffuser actor. *CoRL 2024* | arXiv:2402.10885 |
| Jiang et al., 2025 | Jiang, G., et al. Robots pre-train robots (MCR). *ICLR 2025* | arXiv:2410.22325 |
| Kim et al., 2024 | Kim, M. J., et al. OpenVLA. *CoRL 2024* | arXiv:2406.09246 |
| [authors], 2025 | [authors]. Depth Anything V3. *[venue]* | arXiv:2511.10647 |
| Hu et al., 2024 | Hu, W., et al. Metric3D v2. *IEEE TPAMI 2024* | arXiv:2404.15506 |
| [authors], 2024 | "Depth Helps": Improving pre-trained RGB-based policy with depth injection | arXiv:2408.05107 |
| [authors], 2024 | Lift3D foundation policy. | arXiv:2411.18623 |
| Qiu et al., 2025 | GeoAware-VLA: Implicit geometry aware VLA model | arXiv:2509.14117 |
| [authors], 2026 | UniLACT: Depth-aware RGB latent action learning for VLA models | arXiv:2602.20231 |

---

*AI assistance disclosure: Experimental design refinement, document structure,
systematic literature review, and methodological gap identification in this
document were assisted by Claude (Anthropic). All experimental decisions, data
access, pipeline implementation, and result interpretation are the author's own.*
