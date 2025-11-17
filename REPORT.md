# Path Engineering: Causal Manipulation of the "Road Not Taken"

## 1. Executive Summary
- **Research question**: Does constraining or expanding a transformer sentiment model's hidden-state path space causally change predictive uncertainty when the task stays fixed?
- **Key finding**: Low-rank projection of DistilBERT's `[CLS]` representations did **not** reduce uncertainty, while adding orthogonal components produced a statistically significant but very small increase in entropy without affecting accuracy.
- **Practical implication**: Simple PCA-level interventions at the classifier input are insufficient to materially control model uncertainty; stronger or layer-wise interventions are required for meaningful path engineering.

## 2. Goal
- **Hypothesis**: Shrinking critical hidden-state dimensions will lower uncertainty; expanding them will raise uncertainty while leaving task performance intact.
- **Importance**: Establishing causal levers over uncertainty could enable confidence calibration or safety filters without touching inputs.
- **Problem addressed**: Lack of concrete evidence that "path multiplicity" drives uncertainty rather than being a correlated artifact.
- **Expected impact**: Identify whether small architectural interventions can act as dials on model confidence and motivate deeper interpretability work if unsuccessful.

## 3. Data Construction

### Dataset Description
- **Source**: GLUE SST-2 via HuggingFace (`nyu-mll/glue`).
- **Version**: Downloaded January 2025 snapshot (last modified 2024-01-30 per API metadata).
- **Size**: 67,349 training sentences; 872 validation sentences used for evaluation.
- **Characteristics**: Short movie review snippets with binary sentiment labels (0 = negative, 1 = positive). Validation labels are roughly balanced (positive: 444, negative: 428).
- **Collection**: Curated from movie reviews with manual labels; inherits GLUE licensing/limitations.
- **Biases**: Movie-review domain, English only, may encode stylistic and demographic biases from source websites.

### Example Samples
| Row ID | Sentence | Label |
|--------|----------|-------|
| 0 | `hide new secretions from the parental units` | 0 |
| 195 | `my thoughts were focused on the characters .` | 1 |
| 734 | `this isn't even madonna's swept away .` | 0 |

### Data Quality
- Missing values: 0% (GLUE provides complete entries).
- Outliers: None detected; max length < 60 tokens after tokenization.
- Class distribution: Val set ~51% positive, ~49% negative.
- Validation checks: ensured `row_id` uniqueness, token lengths capped at 128, no duplicate rows in evaluation subset.

### Preprocessing Steps
1. **ID assignment**: Added `row_id` to train/validation splits for deterministic merges.
2. **Tokenization**: DistilBERT tokenizer with `max_length=128`, padding to fixed length, truncation for longer sentences (rare).
3. **Format conversion**: Set dataset format to PyTorch tensors for `input_ids`, `attention_mask`, `label`, `row_id`.
4. **PCA subset selection**: Randomly traversed the training loader to collect 2,000 `[CLS]` embeddings for PCA fitting.

### Train/Val/Test Splits
- Train: 67,349 examples (used only for PCA statistics, not for accuracy evaluation).
- Validation: 872 examples (used for all reported metrics and comparisons).
- Test: Not used because SST-2 test labels are hidden.
- Stratification: Not required; validation already near-balanced.

## 4. Experiment Description

### Methodology
#### High-Level Approach
1. Load DistilBERT SST-2 checkpoint and datasets on CPU.
2. Extract `[CLS]` vectors, learn PCA basis, and define interventions:
   - **Constrained**: Project onto top-128 components and reconstruct (low-rank path space).
   - **Expanded**: Add amplified bottom-64 components plus orthogonal noise to open new paths.
3. Run inference on the same validation set under baseline + interventions.
4. Measure uncertainty (entropy), calibration (Brier), and accuracy; test paired differences.

#### Why This Method?
- DistilBERT SST-2 provides strong baseline accuracy without fine-tuning, ensuring observed effects stem from interventions rather than training noise.
- PCA captures dominant representational directions, letting us systematically remove or introduce subspace components with minimal code changes.
- Paired evaluation on identical inputs isolates causal impacts of hidden-state manipulation.

### Implementation Details
#### Tools and Libraries
- Python 3.12.2, PyTorch 2.9.1, Transformers 4.57.1, Datasets 4.4.1, scikit-learn 1.7.2, SciPy 1.16.3, pandas 2.3.3, seaborn 0.13.2, nbconvert 7.16.6. Full list: `requirements.txt`.

#### Algorithms/Models
- `distilbert-base-uncased-finetuned-sst-2-english` (67M parameters).
- PCA (`svd_solver="randomized"`, components equal to min(2,000, 768)).
- Interventions implemented as deterministic functions applied to `[CLS]` embeddings prior to DistilBERT's `pre_classifier` layer.

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| `max_length` | 128 | Standard SST-2 recipe |
| `batch_size` | 32 | CPU-friendly throughput |
| `PCA_SAMPLE_SIZE` | 2000 | Balance between coverage and runtime |
| `TOP_K` | 128 | Aggressive 6× reduction of 768-d hidden state |
| `BOTTOM_K` | 64 | Enough orthogonal dims without destabilizing logits |
| `NOISE_SCALE` | 0.08 | Small perturbation to avoid large accuracy shifts |
| `BOOST_FACTOR` | 1.8 | Amplifies latent bottom components to mimic extra paths |

#### Training / Analysis Pipeline
1. **Baseline inference**: run model with `output_hidden_states=True`, capture `[CLS]`, and compute logits via classification head (ensures parity with intervention path).
2. **PCA computation**: gather `2,000 × 768` matrix, fit PCA, store explained variance.
3. **Intervention**: apply `PCAHelper.constrain` or `.expand` to `[CLS]` before feeding through classifier layers.
4. **Metric logging**: For each example, store probabilities, entropy, Brier score, logit margin, predictions, and text in `results/per_sample_metrics.csv`.
5. **Aggregation & statistics**: compute means, paired t-tests, bootstrap CIs, and McNemar-style accuracy checks (`results/statistics.json`).
6. **Visualization**: Generate entropy bars, scatter plots, KDEs, and PCA curves (`results/plots/`).

### Experimental Protocol
#### Reproducibility
- Runs averaged over a single deterministic pass (dropout disabled during `model.eval()`).
- Random seeds: 42 for Python, NumPy, Torch, PCA randomness, and bootstrap sampling.
- Hardware: CPU-only Linux container (no GPU), 8 vCPUs, 16GB RAM (peak usage <5GB).
- Execution time: ~2 minutes for PCA fitting + inference; ~1 minute for stats/plots.
- Notebook: `notebooks/path_engineering.ipynb` (executed via `jupyter nbconvert`).

#### Evaluation Metrics
- **Accuracy**: proportion of correct predictions; verifies reasoning unchanged.
- **Predictive entropy**: `-∑ p log p` (nats) over sentiment classes; proxy for uncertainty.
- **Brier score**: squared error between predicted probabilities and one-hot labels; measures calibration.
- **Logit margin**: difference between positive and negative logits; indicates confidence gap.

#### Raw Results
| Condition | Accuracy | Mean Entropy | Mean Brier | Mean Logit Margin |
|-----------|----------|--------------|------------|-------------------|
| Baseline | 0.9106 | 0.04711 | 0.16673 | 0.77610 |
| Constrained (top-128) | 0.9106 | 0.04714 | 0.16669 | 0.77630 |
| Expanded (bottom-64) | 0.9106 | 0.04714 | 0.16672 | 0.77581 |

Detailed metrics stored in `results/metrics.json`; per-example values in `results/per_sample_metrics.csv`.

#### Visualizations
- `results/plots/entropy_bar.png`: bar chart of mean entropies (differences barely visible).
- `results/plots/pca_variance.png`: cumulative variance, showing top-128 components cover >98% variance.
- `results/plots/entropy_scatter_constrained.png` and `entropy_scatter_expanded.png`: scatter of baseline vs intervention entropies (points concentrate near diagonal).
- `results/plots/entropy_density.png`: KDE overlays showing subtle right-shift for expansion.

#### Output Locations
- Metrics: `results/metrics.json`
- Statistics: `results/statistics.json`
- Config/parameters: `results/config.json`
- PCA info: `results/pca_info.json`
- Plots: `results/plots/*.png`
- Notebook: `notebooks/path_engineering.ipynb`

## 5. Result Analysis

### Key Findings
1. **Accuracy invariant**: All conditions achieved 91.1% accuracy on SST-2 validation (McNemar p=1.0), confirming the task itself remained unchanged.
2. **Constrained projection slightly increased entropy**: Mean entropy rose by `3.5e-05` nats (Cohen's d ≈ 0.016, p=0.63). Effect not significant and opposite of hypothesis.
3. **Expansion induced minute but significant entropy increase**: Adding bottom-64 components raised entropy by `3.0e-05` nats with p≈0.011 and d≈0.086, indicating a causal—but tiny—uncertainty increase.
4. **Calibration barely moved**: Brier scores improved by ~0.00005 in constrained case and worsened by ~0.00001 in expanded case—well within noise.

### Hypothesis Testing Results
- **H1 (constraint lowers uncertainty)**: Not supported. Entropy difference CI includes zero and leans positive.
- **H2 (expansion raises uncertainty)**: Supported directionally but effect size is very small (≈0.086). Despite statistical significance, the absolute change is negligible for practical use.
- **H3 (accuracy stability)**: Supported; accuracy differences exactly zero (no errors flipped).

### Comparison to Baselines
- Improvements/degradations over baseline are within `±3e-05` nats, far below one standard deviation of entropy distribution (~0.26). Thus, baseline essentially matches interventions.
- Since accuracy/Brier remain constant, baseline remains preferred due to simplicity; interventions offer no meaningful gain.

### Visualizations
- Entropy bar chart visually confirms minimal differences.
- Scatter plots show near-identity line slopes; expansions only slightly tilt above diagonal for higher-entropy samples.
- PCA curve indicates top-128 components cover >98% variance, explaining why projecting onto them preserves behavior.

### Surprises and Insights
- Removing low-variance components failed to reduce uncertainty because `[CLS]` activations already live in a highly concentrated subspace; interventions removed information rather than competing reasoning branches.
- Expansion effect reached significance despite small magnitude, suggesting that adding orthogonal noise can slightly destabilize logits even when accuracy stays constant.
- Sentences with long, neutral phrasing (e.g., row 195 "my thoughts were focused on the characters") experienced the largest entropy increases, implying these interventions primarily affect borderline cases.

### Error Analysis
- Reviewed top ±5 entropy shifts for each condition (see `per_sample_metrics.csv`).
- Positive shifts concentrated on nuanced reviews referencing narrative structure or sarcasm, where baseline entropy already moderate (0.2–0.5).
- Negative shifts occurred on confident statements (e.g., row 697 "not since tom cruise …"), indicating projection occasionally sharpened certainty on very positive statements but not consistently.
- No examples flipped predictions, highlighting stability of the classifier head despite interventions.

### Limitations
- Only a single dataset/model pair evaluated; results may not generalize to reasoning-heavy tasks where uncertainty is higher.
- Interventions limited to last-layer `[CLS]`; ignoring earlier path bifurcations may miss the mechanisms hypothesized in the prompt.
- PCA assumes linear structure and may not capture "critical directions" tied to reasoning; alternative bases (e.g., ICA, Hessian eigenvectors) could behave differently.
- CPU-only constraint limited exploration of larger PCA samples or deeper layer hooks.

## 6. Conclusions
### Summary
Constricting DistilBERT's `[CLS]` embeddings to a top-128 PCA subspace did **not** lower uncertainty and may slightly raise it. Expanding the path space via bottom-component amplification significantly—but minutely—increased entropy. Accuracy and calibration stayed constant across conditions.

### Implications
- Path engineering at the final hidden layer is insufficient to control uncertainty; richer interventions (layer-wise gating, learned projections) are needed.
- However, the slight entropy increase under expansion shows that adding orthogonal components can act as a controllable noise source for uncertainty tuning.

### Confidence in Findings
- High confidence that current intervention strength is too weak: metrics stable, tests show microscopic differences with well-characterized CI.
- Moderate confidence in directionality (expansion increases entropy) but low confidence in effect size utility; more evidence required on other tasks and models.

## 7. Next Steps
### Immediate Follow-ups
1. **Layer-wise interventions**: Apply PCA-based constraints at earlier transformer layers or multiple stages to determine if cumulative effects are stronger.
2. **Vary k and noise**: Run grid search over projection rank and noise amplitude to map effect curve and locate thresholds causing meaningful entropy shifts.
3. **Evaluate on reasoning benchmarks**: Repeat experiments on BoolQ or QNLI where baseline entropy is higher, testing whether path manipulation has larger impact.

### Alternative Approaches
1. Use learned low-rank adapters (LoRA-style) that explicitly constrain/explore hidden directions rather than PCA.
2. Perform causal tracing by ablating attention heads or MLP neurons associated with alternative reasoning paths instead of linear projections.

### Broader Extensions
- Integrate interventions into decoding for generative LLMs to examine uncertainty modulation in open-ended outputs.
- Explore calibration effects when combining path engineering with temperature scaling or focal loss.

### Open Questions
- Which internal layers actually encode divergent reasoning paths, and how can we target them precisely?
- Can we design interventions that affect uncertainty without adding observable noise to embeddings?
- How does path engineering interact with human-calibrated confidence assessments or interpretability probes?

