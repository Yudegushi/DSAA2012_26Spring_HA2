# HA2: CLIP Zero-Shot Classification on Caltech101

This project implements HA2 using OpenAI CLIP for zero-shot image classification.
The final pipeline is notebook-based and includes setup, full experiments, and analysis.

## What We Did

- Dataset: Caltech101 (`8677` images, `101` classes in torchvision split).
- Models: `ViT-B/32` and `ViT-B/16` from CLIP.
- Prompt settings:
  - `B1_simple`: `a photo of a {CLASS}.`
  - `M1_ensemble_all`: 34 templates from CLIP `prompts.md`.
- Ablations:
  - `A1`: template count (`k = 1, 2, 4, 8, 16, 34`)
  - `A2`: aggregation method (`feature_mean` vs `logit_mean`)
  - `A3`: normalization (`norm_on` vs `norm_off`)
- Statistics and visualization:
  - bootstrap confidence intervals
  - McNemar test (paired significance)
  - confusion pairs and qualitative case grids

## Key Full-Run Results (Top-1 Accuracy)

- `ViT-B/16`: simple `0.8834` -> ensemble `0.8871` (`+0.0037`)
- `ViT-B/32`: simple `0.8759` -> ensemble `0.8729` (`-0.0030`)
- McNemar (simple vs ensemble): not significant for both models (`p > 0.05`)

## Project Structure

- `notebooks/00_setup_data.ipynb`: data setup and metadata artifacts
- `notebooks/01_run_experiments.ipynb`: baselines + ablations + saved predictions
- `notebooks/02_analysis_viz.ipynb`: statistics, figures, tables
- `src/ha2_common.py`: setup utilities
- `src/ha2_experiments.py`: experiment and inference utilities
- `src/ha2_analysis.py`: statistical analysis utilities
- `artifacts/`: cached outputs, full results, tables, and figures
- `report/`: LaTeX report source and compiled PDF

## How To Reproduce

1. Run setup (only needed once):
   - `notebooks/00_setup_data.ipynb`
2. Run full experiments:
   - set `RUN_MODE = "full"` in `notebooks/01_run_experiments.ipynb`
   - execute the notebook
3. Run full analysis:
   - set `RUN_MODE = "full"` in `notebooks/02_analysis_viz.ipynb`
   - execute the notebook
4. Generate report PDF:
   - see `report/README.md`
