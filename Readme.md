# MONAI Architecture Benchmark for 3D MSD Segmentation Tasks

This repository contains the code for the study

> **Architecture–Task Interaction Analysis in 3D Medical Image Segmentation  
> using CNN and Transformer Architectures Across Multiple MSD Tasks**

submitted to **Neurocomputing** as an Open Source Project (OSP).

The code provides a **unified MONAI-based pipeline** to train and evaluate a set of 3D architectures
(CNNs and transformers) on multiple **Medical Segmentation Decathlon (MSD)** tasks under a
compute-aware 64³ patch regime.

Current experiments in the paper cover **five MSD tasks**:

- Task09 Spleen
- Task02 Heart
- Task04 Hippocampus
- Task08 Hepatic Vessels
- Task01 Brain Tumour (BraTS-like)

The implementation is **task agnostic**. Each task is configured via a small YAML file.

---

## 1. Repository Structure

```text
monai-arch-benchmark/
├─ .github/
│  └─ workflows/
│     └─ tests.yml              # CI: install + pytest
│
├─ configs/
│  └─ spleen.yaml               # Example task config (MSD Spleen)
│     # You can add: heart.yaml, hippocampus.yaml, hepatic_vessel.yaml, brats.yaml
│
├─ experiments/
│  └─ run_spleen_all_models.py  # Example runner script using spleen.yaml
│     # You can add: run_heart_all_models.py, run_brats_all_models.py, ...
│
├─ monai_arch_benchmark/
│  ├─ __init__.py               # Package exports
│  ├─ config.py                 # TaskConfig and YAML loading
│  ├─ data.py                   # Dataset + transforms (train / val / viz)
│  ├─ engine.py                 # Training and evaluation loops
│  ├─ env_logging.py            # Environment logging helper
│  ├─ metrics.py                # Per-class metric aggregation
│  ├─ models.py                 # build_model_zoo() and complexity helpers
│  └─ viz.py                    # Whole-volume overlay rendering
│
├─ tests/
│  ├─ conftest.py               # Test configuration / fixtures
│  └─ test_models.py            # Unit tests for model zoo and forward pass
│
├─ requirements.txt             # Python dependencies
├─ CONTRIBUTING.md              # Contribution guidelines
├─ LICENSE                      # MIT license
└─ README.md                    # This file
