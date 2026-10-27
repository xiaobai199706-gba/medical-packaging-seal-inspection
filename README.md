# Seal breach detection and localization

Core research code and numerical results for **Seal Breach Detection and Localization in Sterile Medical Packaging under Sparse Defect Supervision**.

This repository compares real-defect and synthetic-anomaly supervision using a fixed SuperSimpleNet baseline. It contains the training/evaluation implementation, reported experiment configurations and compact result tables.

## Main results

The official P01 source comparison and a separately trained content-group comparison both show higher pixel AP and lower image AUROC/AP for joint training than for real-only training in all paired seeds. Thresholded overlap depends on the seed and operating point.

### Official P01

| Training | Image AUROC | Pixel AP | Dice |
|---|---:|---:|---:|
| Mixed | 0.7446 | 0.8235 | 0.7310 |
| Real-only | 0.9582 | 0.7819 | 0.7453 |
| Synthetic-only | 0.5527 | 0.0188 | 0.0362 |

### Content-group split

| Training | Image AUROC | Pixel AP | Dice |
|---|---:|---:|---:|
| Mixed | 0.8058 | 0.7616 | 0.7084 |
| Real-only | 0.9684 | 0.7277 | 0.7190 |

Values above are means across seeds 42, 7 and 2024. Full-precision values, sample standard deviations and error counts are provided in [results](results/README.md). The grouped table uses 20 defective validation images; the fixed-one-defect calibration analysis is reported separately in the CSV files.

## Getting started

- [Setup and experiment commands](docs/REPRODUCE.md)
- [Result tables and metric definitions](results/README.md)
- [Pinned baseline and third-party materials](THIRD_PARTY_NOTICES.md)
- [File integrity manifest](MANIFEST.sha256)

```text
src/           model integration, data handling, training and metrics
scripts/       baseline/grouped training, evaluation, export and summaries
configs/       reported official and grouped experiment settings
tools/         streaming original-resolution evaluator
tests/         regression tests for the shipped components
results/       compact CSV/JSON tables for reading and analysis
environment/   pinned SuperSimpleNet source identity
```

The experiments use 256 by 256 inputs and restore pixel outputs to the original 4,110 by 3,008 grid for evaluation. Image thresholds maximize validation Youden's J; pixel thresholds maximize pooled validation Dice. Formal thresholds are fixed before testing. Both official and grouped comparisons receive 1,020 optimizer updates per run.

## Dataset and additional materials

SSI images and annotations are available from https://zenodo.org/records/7834077. The official Partial protocols and the retrospective content-group split are different evaluation designs and are reported separately. Their score differences do not quantify the effect of content dependence.

Raw images, model weights, prediction tensors, full logs and full audit/split manifests are not bundled. Exact grouped-list reconstruction requires the original frozen manifest and audit inputs; their hashes are published, and the materials can be requested from the corresponding author. The public CSV files can be inspected and summarized without these large artifacts.

## Result summary

```bash
python scripts/summarize_results.py
```

This command reads the supplied CSVs and prints descriptive summaries; it does not train a model or recompute prediction-level metrics. Run model-dependent tests in the recorded PyTorch environment as described in the reproduction guide.
