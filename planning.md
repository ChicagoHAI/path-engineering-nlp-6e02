# Planning Document (Phase 1)

## Research Question
Can constraining or expanding the effective path space within a transformer sentiment classifier—by projecting `[CLS]` embeddings onto reduced or expanded principal subspaces—causally decrease or increase predictive uncertainty while holding the input task (GLUE/SST2) constant?

## Background and Motivation
Prior causal-tracing studies show that linear interventions on hidden representations can steer model behavior. However, most analyses coincide with semantic edits rather than explicit manipulation of the "path space" (dimensional routes a computation can traverse). Demonstrating that deliberately shrinking the dimensional span of hidden states reduces uncertainty, while expanding it increases uncertainty, would provide causal evidence that path multiplicity itself modulates confidence. This links representation geometry to calibration and could guide safety interventions that modulate uncertainty without changing task inputs or labels.

## Hypothesis Decomposition
1. **H1 (Constraint effect)**: Projecting `[CLS]` representations onto a low-rank PCA subspace (k < d) before the classifier will reduce predictive entropy relative to baseline.
2. **H2 (Expansion effect)**: Injecting additional orthogonal components (via bottom PCA directions + noise) before the classifier will increase predictive entropy relative to baseline.
3. **H3 (Causality on accuracy)**: Entropy shifts will occur without significantly altering accuracy beyond ±2%, ensuring the reasoning task remains effectively unchanged.

Independent variables: representation intervention (baseline, constrained, expanded), PCA dimensionality k, noise scale for expansion. Dependent variables: mean predictive entropy, Brier score, accuracy.

## Proposed Methodology

### Approach
- Use the `distilbert-base-uncased-finetuned-sst-2-english` checkpoint to ensure reliable predictions on SST-2.
- Extract `[CLS]` vectors with `output_hidden_states=True` and learn PCA bases on a subset (~2k examples).
- Implement intervention functions applied just before the classifier head to simulate path restriction/expansion at inference only.
- Evaluate on held-out SST-2 validation set; compute uncertainty metrics and compare via paired tests.

### Experimental Steps
1. **Data prep**: Load SST-2 train split for PCA fitting and validation split for evaluation. Tokenize with DistilBERT tokenizer (max length 128). Rationale: consistent text processing.
2. **Baseline run**: Record logits, probabilities, entropy, Brier score, and accuracy on validation data without interventions; ensures pipeline correctness.
3. **PCA fitting**: Collect `[CLS]` embeddings from ~2000 training samples, compute PCA (scikit-learn) to obtain ordered components and explained variance.
4. **Constrained intervention**: Project embeddings to top-k (k choices: 64, 128). Reconstruct to original dimension before feeding classifier. Measure metrics.
5. **Expanded intervention**: Add scaled components from bottom PCA directions (<= bottom 64) plus Gaussian noise to mimic additional degrees of freedom. Measure metrics for different noise scales.
6. **Statistical analysis**: Use paired t-tests comparing entropy distributions between baseline vs interventions; also test accuracy difference within ±2% tolerance via McNemar or simple proportion difference (since paired). Compute effect sizes (Cohen's d) for entropy shift.
7. **Visualization**: Plot entropy histograms and scatter (baseline vs interventions) plus explained-variance curves to show component importance.

### Baselines
- **Baseline**: Original model inference with no interventions; establishes reference metrics.
- **Simple random noise control**: Optional check where we add isotropic Gaussian noise without PCA guidance to ensure targeted expansion differs from naive noise (time permitting).

### Evaluation Metrics
- **Accuracy**: ensures label predictions stay consistent (task constant).
- **Predictive entropy**: measures uncertainty; mean change indicates hypothesized effect.
- **Brier score**: captures calibration differences.
- **Logit margin**: optional to see confidence gap changes.

### Statistical Analysis Plan
- For each metric, compute per-example differences relative to baseline.
- Use paired t-tests (α = 0.05) on entropy and Brier differences due to approximate normality for large n (~872).
- Report Cohen's d and 95% CI via bootstrapping (1k samples) for entropy differences.
- For accuracy, compute paired contingency table and apply McNemar's test; also report simple difference with CI.

## Expected Outcomes
- Support hypothesis if constrained condition lowers mean entropy (negative difference) with significant p < 0.05 and expanded condition increases entropy with p < 0.05, while accuracy remains within ±2% of baseline.
- Refute if entropy changes are insignificant or move opposite directions, or if accuracy drastically shifts (indicating task change rather than uncertainty modulation).

## Timeline and Milestones
1. **Setup & PCA data extraction (20 min)**: Install deps, load data, gather embeddings.
2. **Baseline + interventions implementation (25 min)**: Build notebook pipeline, verify hooks.
3. **Experiment runs & metrics (25 min)**: Evaluate baseline, constrained, expanded; store outputs.
4. **Analysis & visuals (25 min)**: Statistical tests, plots, interpret.
5. **Documentation (20 min)**: REPORT.md, README.md, finalize artifacts.

## Potential Challenges & Mitigation
- **Hooking complexity**: DistilBERT forward hooks may be tricky; fallback to manual forward pass capturing hidden states via `output_hidden_states`. Then pass modified `[CLS]` through classifier by reusing model head modules directly.
- **PCA fit size**: Extracting all training embeddings may be slow; limit to random subset (2k) to keep CPU manageable.
- **Intervention intensity**: Constraining too aggressively may hurt accuracy; tune k and noise scales quickly via small validation subset before full evaluation.
- **Statistical reliability**: If entropy shifts small, rely on bootstrap CI for robustness.

## Success Criteria
- Completed experiments with metrics + plots saved under `results/`.
- planning assumptions documented in REPORT.md; reproducible pipeline with random seeds set.
- Evidence showing directional entropy changes aligned with hypothesis alongside discussion of limitations.
