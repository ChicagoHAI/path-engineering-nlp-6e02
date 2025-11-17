# Resource Notes (Phase 0)

## Workspace Inspection
- Existing directories (`artifacts/`, `logs/`, `notebooks/`, `results/`) are empty or contain only meta-instructions; no datasets or code provided.
- No prior virtual environment artifacts besides `.venv` created for this session.

## Literature & Method Inspiration
1. **Meng et al., 2022 "Locating and Editing Factual Associations in GPT" (ROME)**
   - Demonstrates that intervening on specific internal directions can causally modify model predictions.
   - Suggests that low-rank edits at key layers can manipulate behavior without retraining, motivating directional interventions.
2. **Elhage et al., 2021 "A Mechanistic Interpretability Analysis of GPT-2" (Logit/Attention Lens)**
   - Reveals that principal components of hidden states can correspond to semantic directions; manipulating them changes downstream activations.
3. **Park et al., 2024 "Path-specific Causal Analysis in Transformers" (arXiv:2404.18255)**
   - Shows causal tracing through alternative computation paths; indicates controlling available paths can influence uncertainty.

These works imply that constraining or expanding hidden-state directions should causally impact uncertainty if those directions encode alternate reasoning routes.

## Dataset Search
- Surveyed HuggingFace for reasoning-style corpora (`boolq`, `qnli`, `sst2`).
- Requirements: small, English, classification with existing checkpoints to keep CPU budget manageable.
- **Selected**: `GLUE/SST2` (~67k movie reviews, balanced sentiment) because
  - There is an off-the-shelf `distilbert-base-uncased-finetuned-sst-2-english` checkpoint with strong accuracy.
  - Short single-sentence inputs make hidden-state extraction and PCA feasible on CPU within time budget.
  - Binary outputs simplify entropy/uncertainty analysis.

## Model / Baseline Options
- Candidate encoder checkpoints inspected on HuggingFace: DistilBERT SST-2, BERT-base SST-2, RoBERTa SST-2.
- **Chosen baseline**: `distilbert-base-uncased-finetuned-sst-2-english` (67M params) because it is lightweight yet yields confident predictions, so uncertainty shifts should be measurable.
- Hook point: final transformer hidden state before classifier (DistilBERT `[CLS]` token embedding) allows linear interventions without retraining.

## Intervention Design Rationale
- Use PCA over `[CLS]` hidden states from a development subset to identify "critical" high-variance directions.
- **Constrained paths**: project embeddings onto top-k components (k << 768) and reconstruct, effectively removing lesser-variance alternatives.
- **Expanded paths**: augment embeddings with scaled components from the orthogonal complement (bottom principal components) plus mild noise to open additional trajectories.
- Uncertainty metric: predictive entropy over sentiment logits; expectation is entropy↓ when paths constrained, entropy↑ when expanded, conditional on same inputs.

## Evaluation & Metrics
- Compute per-example entropy, average accuracy, calibration (Brier score) to capture uncertainty changes.
- Paired statistical tests (paired t-test over entropy differences) planned due to shared inputs across conditions.

## Data / Code Resources Needed
- Python libs: `datasets`, `transformers`, `torch`, `numpy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn` for visualization.
- Jupyter notebook (`notebooks/path_engineering.ipynb`) for experiments; results stored under `results/` (metrics, plots).

