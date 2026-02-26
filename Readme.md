# MONAI Architecture Benchmark for 3D MSD Segmentation Tasks


⚠️ This repository is maintained under a personal GitHub account. It is purely academic work developed at the Laboratory of Computer Sciences, Faculty of Sciences, Ibn Tofail University, Kénitra, Morocco.

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
```
The Spleen configuration and script are provided as a complete example.
The manuscript uses the same pattern for the other four MSD tasks.

2. Installation
2.1. Clone the repository
```text
git clone https://github.com/<your-username>/monai-arch-benchmark.git
cd monai-arch-benchmark
```
2.2. Create a virtual environment (recommended)

Example with Conda:
```text
conda create -n monai-arch-bench python=3.10 -y
conda activate monai-arch-bench
```
2.3. Install dependencies
```text
pip install -r requirements.txt
```
2.4. Install the package in editable mode
```text
pip install -e .
```

This makes monai_arch_benchmark importable from anywhere in the environment.

3. Data Preparation

The framework expects the standard MSD folder layout for each task.

Below are typical locations. Adjust the paths to match your system.

3.1. Spleen (Task09)
```text
/your/data/root/
└─ Task09_Spleen/
   ├─ imagesTr/
   └─ labelsTr/
```

Example configs/spleen.yaml:
```text
name: spleen_msd
data_root: /your/data/root/Task09_Spleen
images_dir: imagesTr
labels_dir: labelsTr

in_channels: 1
num_classes: 2
label_names:
  0: background
  1: spleen

patch_size: [64, 64, 64]
target_spacing: [1.0, 1.0, 1.0]
val_fraction: 0.2
seed: 42
```
3.2. Other MSD tasks

You can create similar YAML files for the remaining tasks, for example:

configs/heart.yaml pointing to Task02_Heart

configs/hippocampus.yaml pointing to Task04_Hippocampus

configs/hepatic_vessel.yaml pointing to Task08_HepaticVessel

configs/brats.yaml pointing to Task01_BrainTumour (or your BraTS folder)

Each config only needs to specify:

data_root, images_dir, labels_dir

in_channels

num_classes

label_names per dataset

patch_size, target_spacing

val_fraction, seed

The rest of the pipeline is shared.

The MSD datasets themselves are not included in this repository.
Please download them from the official MSD source and follow the respective licenses.

4. Running Experiments
4.1. Spleen example (fully wired)

From the project root:
```text
python experiments/run_spleen_all_models.py \
    --config configs/spleen.yaml \
    --out_dir ./ablation_results_spleen
```

This script will:

Load the YAML config into a TaskConfig instance.

Log the software and hardware environment to out_dir/environment.json.

Build MONAI datasets and dataloaders for training, validation and visualisation.

Build the model zoo via build_model_zoo:

CNNs: UNet, BasicUNet, BasicUNetPlusPlus, SegResNet,
DynUNet, AttentionUnet, VNet, HighResNet

Transformers: UNETR, SwinUNETR

Train each model with Dice+CE loss, AdamW, early stopping.

Save the best checkpoint per model as best_<ModelName>.pth.

Evaluate per-class metrics on the validation set and aggregate results.

Compute model footprints (GFLOPs at 64³ and parameter counts).

Render qualitative whole-volume overlays for a subset of validation cases.

Key CSV outputs in out_dir:

per_class_summary.csv

summary_models.csv

model_complexity.csv

Overlay figures:

overlay_full_<case>_axis-<axis>_slice-XXX.png

4.2. Extending to other tasks

To run another MSD task you can:

Add a task config, for example configs/heart.yaml.

Either:

Reuse the same script with a different config, if it is written generically, or

Create experiments/run_heart_all_models.py which is a copy of the spleen script
with task-specific defaults.

Example generic call if your runner supports arbitrary configs:
```text
python experiments/run_spleen_all_models.py \
    --config configs/heart.yaml \
    --out_dir ./ablation_results_heart
```

The core library code in monai_arch_benchmark/ does not assume a specific task.

5. Metrics and Outputs

For each model and class, the evaluation reports:

Macro Dice (average over cases where the class is present)

Micro Dice (voxel-pooled)

Micro precision, recall and F1

Hausdorff distance (HD95)

Mean surface distance

These are aggregated by the PerClassAggregator in metrics.py and written to CSV files.

Model complexity metrics are computed by params_gflops in models.py using thop when available.

6. Reproducibility

Reproducibility is supported through:

A single YAML configuration per task (configs/*.yaml)

A fixed random seed for random, numpy, torch and torch.cuda

monai.utils.set_determinism(seed=...)

Environment logging:

Python, PyTorch, MONAI versions

CUDA availability and GPU name

Patch size, target spacing, speed mode

To repeat an experiment, use the same:

Dataset version and preprocessing

YAML config

Software environment (see environment.json)

7. Testing

From the project root:
```text
pytest
```
Expected output:

tests/test_models.py ..                             [100%]


The tests currently verify:

The model zoo can be constructed with a minimal TaskConfig

BasicUNet runs a 3D forward pass and returns a tensor with the expected shape

The test suite is intentionally lightweight so that CI runs quickly.

8. Continuous Integration

GitHub Actions is configured under:

.github/workflows/tests.yml


The workflow:

Sets up Python

Installs dependencies from requirements.txt

Installs monai_arch_benchmark in editable mode

Runs pytest

This provides a minimal software quality check for every push and pull request.

9. License

This project is released under the MIT License.
See LICENSE for details.

10. Contributing

Contributions are welcome. Typical extensions include:

Adding new task configs in configs/

Adding new experiment scripts in experiments/

Extending the model zoo or metrics

Improving tests and documentation

Please see CONTRIBUTING.md before opening an issue or pull request.

11. Contact

For questions about the code or the paper:

Hasnae Briouya
hasnae.briouya@uit.ac.ma
