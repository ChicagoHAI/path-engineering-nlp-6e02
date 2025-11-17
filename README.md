# Path Engineering Workspace

Brief exploration of whether PCA-based interventions on DistilBERT's `[CLS]` embeddings can causally control predictive uncertainty on SST-2 sentiment classification.

## Key Findings
- Low-rank projection onto the top-128 PCA components **did not** reduce entropy (mean change +3.5e-05 nats, p=0.63).
- Expanding the path space with bottom-64 components + noise increased entropy slightly (+3.0e-05 nats, p≈0.011) while accuracy stayed at 91.1%.
- Calibration (Brier score) remained ~0.167 in all conditions, so interventions failed to produce meaningful uncertainty control.

## Reproduction Steps
1. **Environment**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
2. **Run notebook** (executed headlessly in this study):
   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/path_engineering.ipynb
   ```
   Outputs will land in `notebooks/results/`; copy to `results/` as done in this session if you prefer repo-level storage.
3. **Inspect artifacts**: metrics/statistics JSON, per-sample CSV, and plots saved under `results/`.

## Repository Structure
- `resources.md` – phase-0 literature/data notes.
- `planning.md` – experimental plan and hypothesis decomposition.
- `notebooks/path_engineering.ipynb` – full experiment code (auto-executable).
- `results/` – metrics (`metrics.json`), stats (`statistics.json`), per-sample outputs, PCA info, and plots directory.
- `REPORT.md` – comprehensive study report.
- `README.md` – this overview.
- `requirements.txt` – frozen environment (installed via `uv pip`).

See `REPORT.md` for detailed methodology, statistical analysis, and future work proposals.
