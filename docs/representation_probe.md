# Representation Probe: Does Depth Help Object Classification?

A linear-probing experiment (frozen features + logistic regression, the standard
"is this representation linearly separable" protocol from representation-learning
papers, e.g. DeFM arXiv:2601.18923) on the 130 SAM3 mask instances from the LIBERO
smoke run, asking a concrete question motivated by this pipeline's metric-depth
stage: **does adding depth to a simple RGB appearance feature improve object
classification, and specifically does it disambiguate categories that look similar
in color but differ in 3D shape?**

Script: `run_representation_probe.py` (extracts RGB + basic depth stats from the
original frames + depth PNGs) and `run_representation_probe_richer_depth.py`
(re-extracts shape-aware depth features locally from the saved `probe_features.npz`
-- no need to re-fetch video frames once the first run has cached RGB features).

## A methodological bug caught mid-experiment

The first pass reported a weak, fold-unstable RGB-vs-RGB+depth gap (+1.5pp) and did
**not** support the hypothesis that depth disambiguates visually-similar categories.
`LogisticRegression` was also throwing `ConvergenceWarning`s, which turned out to be
the real problem: RGB pixel statistics (0-255 scale) and depth statistics (0-3
meter scale) were fed into the same unscaled linear model, which both fails to
converge reliably and lets the larger-magnitude feature dominate the L2-regularized
fit for reasons unrelated to actual information content. Adding `StandardScaler`
(via an sklearn `Pipeline`) before the classifier fixed convergence and **reversed
the conclusion**:

| Feature set | Accuracy (mean +/- std) |
|---|---|
| RGB only (9-dim color stats) | 0.547 +/- 0.040 |
| RGB + basic depth (4-dim mean/std/min/max) | 0.778 +/- 0.056 |
| RGB + shape-aware depth (15-dim: + histogram, gradient, planarity) | 0.824 +/- 0.057 |

Depth is not a marginal addition here -- it is the difference between a near-chance
classifier and a reasonably strong one, and richer (shape-preserving) depth features
add a further +4.6pp over simple aggregate statistics.

## Hypothesis check: `bottom drawer` vs `cabinet`

Both are wood/grey furniture in this dataset -- visually similar in color, but a
drawer protrudes (non-planar) while a cabinet face is flush (planar). Out-of-fold
confusion between this specific pair:

| Feature set | `drawer` <-> `cabinet` confusions |
|---|---|
| RGB only | 21 |
| RGB + basic depth | 10 |
| RGB + shape-aware depth (histogram + gradient + planar-fit residual) | 6 |

Monotonic and substantial: confusion drops by more than half with depth, and again
with richer, shape-preserving depth features -- directly supporting the motivating
hypothesis for why this pipeline invests in metric depth (and not just RGB) as a
training signal.

## Caveats

- Small N (130 instances, single dataset/run) -- a strong internal-consistency
  result, not a statistically powered claim.
- Features are hand-engineered pixel/depth statistics, not learned encoder
  embeddings (e.g. a SAM3 vision-encoder feature) -- a transparent, reproducible
  starting point; swapping in real encoder features is the natural next iteration.
- Classifier is deliberately linear: the goal is to test whether the *features*
  are separable, not whether a stronger classifier can compensate for weak ones.
