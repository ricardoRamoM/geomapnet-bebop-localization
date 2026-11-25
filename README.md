# GeoMapNet-Bebop: Visual Localization for Indoor Drones

**An adaptation of NVIDIA's GeoMapNet / MapNet for monocular visual localization using a Parrot Bebop 2 drone.**

## Overview

This repository contains a Python 3.11 and PyTorch 2.4 adaptation of the original [GeoMapNet](https://github.com/NVlabs/geomapnet) implementation, developed to investigate deep learning-based visual localization for indoor drone applications.

The project focuses on adapting sequence-based camera localization methods to a custom indoor dataset collected using a Parrot Bebop 2 drone.

The work covers three main areas:

1. **Codebase adaptation:** Porting the original implementation from Python 2.7 and PyTorch 0.4.1 to Python 3.11 and PyTorch 2.4, including compatibility with modern NVIDIA GPUs.
2. **Benchmark validation:** Training and evaluating the adapted implementation on the public 7-Scenes benchmark.
3. **Drone localization:** Preparing a custom Bebop dataset, investigating different camera pose representations, training MapNet models, and evaluating their 6-DoF localization performance.

The repository includes the adapted implementation, experiment configurations, dataset preparation utilities, evaluation scripts, analysis tools, documentation, and selected experimental results.

## Main Features

- Python 3.11 and PyTorch 2.4 compatibility.
- CUDA 12.1 runtime support.
- Compatibility with NVIDIA Ada architecture GPUs.
- Adapted dataset loader for the Parrot Bebop 2.
- Training and evaluation on the 7-Scenes benchmark.
- Three Bebop dataset versions: v2_0, v2_2, and v2_3.
- Support for World-to-Camera (W2C) and Camera-to-World (C2W) pose representations.
- Configurable MapNet training experiments.
- Learnable loss weighting parameters for absolute and relative pose losses.
- Quantitative 6-DoF pose evaluation.
- Trajectory visualization and per-frame error analysis.
- Experiment comparison and automated result summarization.

## Results

### 1. Parrot Bebop 2 Indoor Localization

MapNet was evaluated on a custom indoor dataset collected using a Parrot Bebop 2 drone.

Three dataset versions were investigated:

| Version | Pose representation | Description |
|---|---|---|
| v2_0 | W2C training / C2W evaluation | Initial experimental configuration |
| v2_2 | W2C | World-to-Camera pose representation |
| v2_3 | C2W | Camera-to-World pose representation |

The experiments investigate the effects of dataset preparation, pose representation, training duration, learning rate, weight decay, and other optimization parameters.

#### Best experimental results

The following table summarizes selected results from the training experiments.

| Dataset | Selected experiment | Position mean (m) | Position RMSE (m) | Orientation mean (°) |
|---|---|---:|---:|---:|
| v2_0 | baseline | 0.6616 | 0.8476 | 17.4570 |
| v2_2 (W2C) | wd1e-4_2000 | 0.3477 | 0.4151 | **8.3944** |
| v2_3 (C2W) | 500 | **0.3322** | **0.4018** | 10.9258 |

For v2_0, the reported values correspond to the common-image evaluation of the baseline experiment. For v2_2 and v2_3, the results correspond to their respective evaluations using 508 common validation images.

**Best results identified:**

- **Position:** v2_3, `bebop_mapnet_500`, with 0.3322 m mean error and 0.4018 m RMSE.
- **Orientation:** v2_2, `bebop_mapnet_wd1e-4_2000`, with 8.3944° mean angular error.

These results indicate substantial improvements over the initial v2_0 experiments.

However, the comparison between W2C and C2W should be interpreted carefully, since dataset versions and other pipeline configurations also changed. Additional controlled experiments are necessary to isolate the effect of pose representation.

#### Evaluation metrics

The evaluation includes:

- Mean and median translation error.
- Translation RMSE.
- 95th percentile translation error.
- Mean and median orientation error.
- 95th percentile orientation error.

Translation errors are expressed in meters and orientation errors in degrees.

Detailed experimental comparisons, including individual training configurations and common-image evaluations, are documented in:

`docs/MAPNET_TRAINING_SUMMARY.md`

### 2. 7-Scenes Benchmark

The adapted implementation was also evaluated on the public [7-Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).

The following results were obtained using MapNet trained from scratch:

| Scene | Epochs | Median translation error | Mean translation error | Median rotation error | Mean rotation error |
|---|---:|---:|---:|---:|---:|
| Heads | 300 | **0.20 m** | 0.22 m | 14.23° | 13.31° |
| Chess | 100 | **0.18 m** | 0.21 m | **5.33°** | 6.61° |
| Fire | 100 | **0.40 m** | 0.43 m | 12.15° | 16.69° |

The experiments used the official 7-Scenes training and test splits.

For the Heads scene, the original MapNet paper reports median errors of 0.18 m in translation and 13.33° in rotation.

The adapted implementation achieved 0.20 m and 14.23°, respectively.

The Fire experiment was limited to 100 epochs and had not fully converged.

## Experimental Visualizations

Selected trajectory comparisons, quantitative evaluations, and experimental visualizations are available under:

- `eval_results/`
- `figures/`

The Bebop evaluation results are organized by dataset version and experiment.

Training checkpoints, datasets, and intermediate outputs are not distributed in this repository.

## Hardware and Software

The implementation was tested with the following configuration:

| Component | Configuration |
|---|---|
| GPU | NVIDIA RTX 3500 Ada Generation Laptop GPU |
| GPU memory | 12 GB |
| Compute capability | 8.9 |
| NVIDIA driver | 595.71.05 |
| Operating system | Ubuntu 24.04 |
| Python | 3.11 |
| PyTorch | 2.4.0 |
| torchvision | 0.19.0 |
| NumPy | 1.26.4 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.2 |
| Pillow | 13.0.0 |
| CUDA runtime (PyTorch) | 12.1 |
| cuDNN | 9.1.0 |

The project uses Conda for environment management.

The tested configuration uses the PyTorch CUDA 12.1 runtime. The installed NVIDIA driver supports CUDA versions up to 13.2.

A compatible NVIDIA GPU is recommended for training.

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
│   └── README.md               # Original implementation documentation
│
├── configs/
│   ├── 7scenes/                # 7-Scenes configurations
│   └── Bebop/
│       ├── v2_0/
│       ├── v2_2/
│       └── v2_3/
│
├── src/
│   ├── data/                   # Dataset preparation
│   ├── analysis/               # Training experiment analysis
│   ├── evaluation/             # Evaluation and visualization tools
│   ├── diagnostics/            # Dataset diagnostics
│   ├── setup/                  # Compatibility and setup utilities
│   └── tests/                  # Validation and testing scripts
│
├── scripts/                    # Auxiliary utilities
│
├── docs/                       # Technical and experimental documentation
│   └── MAPNET_TRAINING_SUMMARY.md
│
├── eval_results/                # Selected evaluation results
├── figures/                     # Selected visualizations
│
├── environment_mapnet.yml
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ricardoRamoM/geomapnet-bebop-localization.git
cd geomapnet-bebop-localization
```

### 2. Create the Conda Environment

```bash
conda env create -f environment_mapnet.yml
conda activate mapnet-py311
```

### 3. Install Additional Dependencies

Visdom requires a compatible setuptools version and is installed separately:

```bash
pip install "setuptools<80"
pip install --no-build-isolation visdom
```

### 4. Verify the Installation

```bash
python -c "import torch; import torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
```

## Datasets

### 7-Scenes

The public [7-Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) is used to validate the adapted implementation.

Dataset files must be downloaded separately.

The expected dataset directory is:

```text
geomapnet/data/deepslam_data/7Scenes/
```

Scene-specific metadata is maintained under:

```text
geomapnet/data/7Scenes/
```

In particular, `pose_stats.txt` and `stats.txt` must correspond to the experimental setup.

### Parrot Bebop 2

A custom indoor dataset collected using a Parrot Bebop 2 drone is used for the localization experiments.

The dataset contains image sequences and associated camera pose annotations.

Three dataset versions are currently investigated:

- `v2_0`
- `v2_2`
- `v2_3`

The dataset files are not distributed in this repository. Users must prepare the data locally and configure the corresponding dataset paths and experiment settings.

Refer to `docs/` and `configs/Bebop/` for additional information.

## Training and Evaluation

The main training and evaluation entry points are located in:

```text
geomapnet/scripts/
```

Experiment-specific configurations are maintained under:

```text
configs/
```

### Training

A Bebop experiment can be launched using:

```bash
python geomapnet/scripts/train.py \
    --dataset Bebop \
    --scene v2_3 \
    --config_file configs/Bebop/v2_3/bebop_mapnet_500.ini \
    --model mapnet \
    --device 0 \
    --learn_beta \
    --learn_gamma
```

The configuration filename must correspond to an existing experiment configuration.

The `--learn_beta` and `--learn_gamma` options enable learnable weighting parameters for the absolute pose and relative motion losses, respectively.

### Evaluation

An example evaluation command is:

```bash
python geomapnet/scripts/eval.py \
    --dataset Bebop \
    --scene v2_3 \
    --model mapnet \
    --weights experiments/Bebop/v2_3/bebop_mapnet_500/best_model.pth.tar \
    --config_file configs/Bebop/v2_3/bebop_mapnet_500.ini \
    --val \
    --eval_name bebop_mapnet_500
```

### Experiment Analysis

The repository includes an analysis utility to compare multiple training runs automatically.

For v2_0:

```bash
python src/analysis/analyze_mapnet_runs.py \
    --dataset Bebop \
    --scene v2_0
```

For v2_2:

```bash
python src/analysis/analyze_mapnet_runs.py \
    --dataset Bebop \
    --scene v2_2
```

For v2_3:

```bash
python src/analysis/analyze_mapnet_runs.py \
    --dataset Bebop \
    --scene v2_3
```

The analysis tool collects available training and evaluation artifacts, compares experimental configurations, and generates quantitative summaries.

## Documentation

Additional documentation is available under `docs/`, including:

- Python 3 and PyTorch compatibility modifications.
- Dataset adaptation and preparation.
- Coordinate systems and pose representations.
- W2C and C2W transformations.
- 7-Scenes training experiments.
- Bebop dataset preparation.
- Training configurations and experimental comparisons.
- Evaluation procedures and results.

The detailed comparison of Bebop training experiments is available in:

`docs/MAPNET_TRAINING_SUMMARY.md`

## Limitations

- The custom Parrot Bebop 2 dataset is not publicly distributed with this repository.
- Training checkpoints and intermediate experiment outputs are not included.
- Reproducing the reported results requires the corresponding datasets, configurations, and compatible software environment.
- The reported results correspond to specific training and evaluation configurations.
- Comparisons between dataset versions must account for changes in dataset preparation and pose representation.
- The reported performance does not represent a fully optimized or deployed real-time localization system.
- Performance may vary depending on hardware, dataset preparation, and training configuration.

## Acknowledgments

This project builds upon the original GeoMapNet implementation developed by NVIDIA.

The original method is described in:

Brahmbhatt, S., Gu, J., Kim, K., Hays, J., and Kautz, J. (2018). *Geometry-Aware Learning of Maps for Camera Localization*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Original repository: https://github.com/NVlabs/geomapnet

All original authorship, attribution, and applicable licensing requirements are retained.

## License

The original GeoMapNet code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

Refer to `geomapnet/LICENSE.md` for the original license and its terms.

The licensing conditions applicable to adapted portions must be considered in accordance with the original license. Other contributions and materials should be interpreted according to their respective authorship and licensing terms.

## Author

**Ricardo Ramos Morales**

Research project developed at the Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE).