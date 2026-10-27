# Setup and reproduction

## Environment

The recorded training stack uses Python 3.10.16, PyTorch 2.1.0+cu118, torchvision 0.16.0+cu118, anomalib 0.7.0 and NumPy 1.26.0. See `environment/upstream.json`.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
.venv/bin/python scripts/fetch_supersimplenet.py --destination ../upstream
```

On Windows, use `.venv/Scripts/python.exe`. The public requirements retain the recorded package pins but omit two machine-specific pip/packaging build-path entries. A fresh environment installation and the full GPU suite are not rerun as part of packaging this release.

The fetch script verifies the exact upstream archive and prints its extracted path. Keep the recorded ImageNet initialization. The grouped driver requires CUDA. Obtain SSI and the matching official Partial split files from https://zenodo.org/records/7834077.

## Official protocols

Example official P01 Mixed command (replace paths):

```bash
.venv/bin/python scripts/run_baseline.py   --config configs/ds1_partial01_seed42_updates1020.json   --upstream-root /absolute/path/to/pinned/SuperSimpleNet   --data-root /absolute/path/to/DS1   --split-archive /absolute/path/to/Partial-01.zip   --output /absolute/path/to/new/run
```

Use the corresponding seed/source configuration for each condition. `evaluate_baseline.py`, `diagnose_baseline.py` and `export_baseline_predictions.py` provide the existing evaluation/diagnostic/export entry points. Legacy summaries may use a lower-resolution evaluation path; the paper's original-scale reported values are the CSVs in `results/`. Do not substitute one for the other.

## Content-group experiment

The original frozen manifest and four full audit CSVs are additional author-supplied artifacts. The public compact results/hashes alone cannot reconstruct these lists. The freeze command verifies the four input hashes in `docs/evidence_audit_20261023/local_artifact_manifest.csv` and refuses substitutes.

```bash
.venv/bin/python scripts/freeze_content_group_protocol.py   --audit /absolute/path/to/original-audit   --data-root /absolute/path/to/DS1   --output /absolute/path/to/new-experiment/protocol
```

All six configurations require manifest SHA256 `f3295d6d545d1c2b9e59b6025ff9abdce991ee43c8ec611a309d0a214b136dc5`.

```bash
.venv/bin/python scripts/run_content_group_experiment.py --phase preflight   --config-directory configs/ssi_contentgroup_sparse3_v1   --data-root /absolute/path/to/DS1   --upstream-root /absolute/path/to/pinned/SuperSimpleNet   --split-manifest /absolute/path/to/new-experiment/protocol/frozen_manifest.json   --output /absolute/path/to/new-experiment/gates
```

Next run `--phase smoke` for Mixed and Real-only. Complete the six `--phase train` runs (seeds 42, 7 and 2024, both modes) before the corresponding `--phase evaluate` runs. Train/evaluate/smoke commands use `--config`, the same roots/manifest, a fresh `--output` run directory and `--gate-directory` pointing to the passed checks. Resume uses only that run's own compatible checkpoint. The driver selects final-update weights and fixes the main20 and fixed1 validation thresholds before test evaluation.

Use `scripts/verify_content_group_exposures.py --experiment-directory ... --data-root ...` to check the recorded paired exposures. Full outputs stay outside Git. The detailed internal provenance-report generator is not part of this minimal public release; the compact published CSVs are already supplied.

## Tests

From the repository root, in the model environment:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests
```

On Windows, set `PYTHONPATH` to `src` first. PyTorch-dependent tests require the pinned model stack. No training is launched by pytest or by the CSV summary command.
