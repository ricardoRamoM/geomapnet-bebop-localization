# GeoMapNet-Bebop: Visual Localization for Indoor Drones

**An adaptation of NVIDIA's GeoMapNet / MapNet for monocular visual localization using a Parrot Bebop 2 drone.**

## Overview

This repository contains a Python 3 and PyTorch 2 adaptation of the original [GeoMapNet](https://github.com/NVlabs/geomapnet) implementation, developed to investigate deep learning-based visual localization for indoor drone applications.

The project focuses on adapting the original codebase to a modern software environment, integrating a custom Parrot Bebop 2 dataset, and evaluating the localization performance of MapNet using monocular image sequences.

The work is organized into two main stages:

1. **Implementation and validation:** Porting the original codebase from Python 2.7 and PyTorch 0.4.1 to Python 3.11 and PyTorch 2.4, followed by training and evaluation on the 7-Scenes benchmark.
2. **Drone adaptation:** Integrating the Parrot Bebop 2 dataset and investigating the application of sequence-based visual localization to indoor drone environments.

The repository includes the adapted implementation, experiment configurations, dataset preparation utilities, evaluation scripts, and selected experimental results.

## Main Features

- Python 3.11 and PyTorch 2.4 compatibility.
- CUDA 12.1 support.
- Compatibility with NVIDIA Ada architecture GPUs.
- Adapted dataset loader for the Parrot Bebop 2.
- Training and evaluation on the 7-Scenes dataset.
- Custom experiment configurations for indoor drone localization.
- Evaluation tools for trajectory visualization and quantitative pose estimation metrics.

## Results

### 7-Scenes benchmark

The adapted implementation was evaluated using the public 7-Scenes dataset.

The following results were obtained using MapNet trained from scratch on three scenes.

| Scene | Epochs | Median translation error | Mean translation error | Median rotation error | Mean rotation error |
|---|---:|---:|---:|---:|---:|
| Heads | 300 | **0.20 m** | 0.22 m | **14.23°** | 13.31° |
| Chess | 100 | **0.18 m** | 0.21 m | **5.33°** | 6.61° |
| Fire | 100 | **0.40 m** | 0.43 m | **12.15°** | 16.69° |

The experiments used the official 7-Scenes training and test splits.

### Comparison with published results

For the `heads` scene, the original paper reports median errors of 0.18 m in translation and 13.33° in rotation.

The adapted implementation achieved:

| Metric | Original paper | Adapted implementation |
|---|---:|---:|
| Median translation error | 0.18 m | **0.20 m** |
| Median rotation error | 13.33° | **14.23°** |

These results indicate that the adapted implementation achieves performance close to the published MapNet results.

The original public checkpoint was also evaluated independently, obtaining 0.24 m median translation error and 21.66° median rotation error. Training from scratch improved the results on this benchmark.

**Note:** The `fire` experiment was limited to 100 epochs and had not fully converged. Its results should therefore be interpreted in the context of the training duration.

### Experimental visualizations

Selected trajectory comparisons and evaluation results are available in:

- `eval_results/`
- `figures/`

## Hardware and Software

The implementation was tested with the following configuration:

| Component | Configuration |
|---|---|
| GPU | NVIDIA RTX 3500 Ada Generation Laptop GPU |
| GPU memory | 12 GB |
| Compute capability | 8.9 |
| Operating system | Ubuntu 24.04 |
| Python | 3.11 |
| PyTorch | 2.4.0 |
| torchvision | 0.19.0 |
| CUDA runtime | 12.1 |
| cuDNN | 9.1 |

The project uses Conda for environment management.

Other hardware configurations may require adjustments to the CUDA and PyTorch installation.

## Repository Structure

```text
geomapnet-bebop-localization/
│
├── geomapnet/
│   ├── common/                 # Training utilities and pose operations
│   ├── dataset_loaders/        # Dataset loading implementations
│   ├── models/                 # PoseNet, MapNet and VidLoc models
│   ├── scripts/                # Training and evaluation scripts
│   ├── data/                   # Dataset statistics and metadata
│   ├── figures/                # Original implementation figures
│   └── README.md               # Original project documentation
│
├── configs/
│   ├── 7scenes/                # 7-Scenes experiment configurations
│   └── Bebop/                  # Bebop experiment configurations
│
├── src/
│   ├── data/                   # Dataset preparation
│   ├── diagnostics/            # Dataset diagnostics
│   ├── evaluation/             # Evaluation and analysis tools
│   └── setup/                  # Compatibility and setup utilities
│
├── docs/                       # Development documentation
├── eval_results/               # Selected experimental results
├── figures/                    # Selected visualizations
│
├── environment_mapnet.yml      # Conda environment
├── .gitignore
└── README.md
```

Datasets, training checkpoints, temporary logs, and intermediate outputs are not included in this repository.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/geomapnet-bebop-localization.git
cd geomapnet-bebop-localization
```

### 2. Create the Conda environment

```bash
conda env create -f environment_mapnet.yml
conda activate mapnet-py311
```

### 3. Install additional dependencies

Visdom requires a compatible setuptools version.

```bash
pip install "setuptools<80"
pip install --no-build-isolation visdom
```

### 4. Verify the installation

```bash
python - <<'PY'
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
PY
```

The tested configuration uses PyTorch 2.4.0, torchvision 0.19.0, and CUDA 12.1.

## Datasets

### 7-Scenes

The public [7-Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) is used to validate the implementation.

Dataset files must be downloaded separately.

The dataset directory should be connected to the expected path:

```text
geomapnet/data/deepslam_data/7Scenes/
```

The scene-specific metadata required by the original implementation is maintained separately under:

```text
geomapnet/data/7Scenes/
```

In particular, `pose_stats.txt` and `stats.txt` must match the experimental setup.

### Parrot Bebop 2

A custom indoor dataset collected with a Parrot Bebop 2 drone is used for the drone localization experiments.

The dataset contains image sequences and associated camera pose annotations.

The dataset files are not distributed in this repository. Users must prepare the data locally and configure the corresponding paths and experiment settings.

See `docs/` and `configs/Bebop/` for the available dataset preparation notes and experiment configurations.

## Training and Evaluation

The original training and evaluation entry points are maintained in:

```text
geomapnet/scripts/
```

Experiment-specific configurations are maintained in:

```text
configs/
```

For 7-Scenes experiments, the original training and evaluation scripts can be used with the corresponding scene and configuration.

The general training workflow is:

```text
Dataset preparation
        ↓
Dataset loading
        ↓
Model training
        ↓
Checkpoint generation
        ↓
Model evaluation
        ↓
Trajectory visualization and metrics
```

Refer to the original implementation documentation and the experiment notes in `docs/` for detailed commands and configuration parameters.

## Documentation

Additional documentation is available in `docs/`, including:

- Python 3 and PyTorch compatibility modifications.
- Dataset adaptation and preparation.
- Coordinate and pose handling.
- 7-Scenes training experiments.
- Evaluation procedures and experimental analysis.

## Acknowledgments

This project builds upon the original GeoMapNet implementation developed by NVIDIA.

The original method is described in:

Brahmbhatt, S., Gu, J., Kim, K., Hays, J., and Kautz, J. (2018). *Geometry-Aware Learning of Maps for Camera Localization*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Original repository: https://github.com/NVlabs/geomapnet

All original authorship, attribution, and licensing requirements are retained.

## License

The original GeoMapNet code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

This repository retains the original licensing terms for the adapted code. Refer to `geomapnet/LICENSE.md` and the accompanying license information for details.

## Author

**Ricardo Ramos Morales**

Research project developed at the Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE).
