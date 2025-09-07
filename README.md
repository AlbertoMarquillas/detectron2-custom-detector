# detectron2-custom-detector

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23fe5196.svg)](https://www.conventionalcommits.org/en/v1.0.0/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Detectron2](https://img.shields.io/badge/Framework-Detectron2-lightgrey)
![Status](https://img.shields.io/badge/Release-v0.1.0-informational)

Custom object detector built on **Detectron2** (PyTorch). Includes **training**, **inference**, **dataset registration**, and **loss visualization**, with a clean, portfolio‑ready layout and PowerShell‑first commands for Windows.

---

## Table of Contents

* [Overview](#overview)
* [Repository Structure](#repository-structure)
* [Getting Started](#getting-started)

  * [1) Environment (PowerShell)](#1-environment-powershell)
  * [2) Install PyTorch & Detectron2](#2-install-pytorch--detectron2)
  * [3) Project Dependencies](#3-project-dependencies)
* [Datasets](#datasets)
* [Models / Weights](#models--weights)
* [Training](#training)
* [Inference / Demo](#inference--demo)
* [Loss Tracking & Plots](#loss-tracking--plots)
* [Configuration](#configuration)
* [Troubleshooting](#troubleshooting)
* [Features](#features)
* [What I Learned](#what-i-learned)
* [Roadmap](#roadmap)
* [License](#license)

---

## Overview

This repository demonstrates a **custom object detection** workflow using Detectron2, from dataset registration to training and visual inference. It is designed as a compact, **portfolio‑ready** project with a standard structure and clear entrypoints:

* **`src/train.py`** — Training launcher (based on Detectron2 trainer).
* **`src/infer.py`** — Inference/visualization on images or folders.
* **`src/utils/loss.py`** — Loss helpers.
* **`src/analysis/plot_loss.py`** — Produces neat loss curves from training logs.
* **`configs/base.yaml`** — Minimal config you can adapt to your dataset.

> **Note**: Large datasets and pretrained weights are intentionally **excluded** from git. See [Datasets](#datasets) and [Models / Weights](#models--weights).

---

## Repository Structure

```
├─ src/
│  ├─ train.py                # training entrypoint
│  ├─ infer.py                # inference/visualization entrypoint
│  ├─ utils/
│  │  ├─ loss.py
│  │  └─ __init__.py
│  ├─ analysis/
│  │  ├─ plot_loss.py
│  │  └─ __init__.py
│  └─ __init__.py
├─ configs/
│  └─ base.yaml               # Detectron2 config (edit NUM_CLASSES, datasets, etc.)
├─ data/                      # (ignored) put datasets here; see data/README.md
├─ models/                    # (ignored) put weights here; see models/README.md
├─ docs/
│  └─ assets/                 # small images/gifs for README
├─ notebooks/                 # optional experiments
├─ test/                      # unit/functional tests (optional)
├─ build/                     # outputs (checkpoints, eval, logs)
├─ README.md
└─ LICENSE
```

---

## Getting Started

### 1) Environment (PowerShell)

```powershell
# From repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install PyTorch & Detectron2

Detectron2 requires a **matching** PyTorch/CUDA build. Install PyTorch first (CPU or CUDA build that matches your system), then install Detectron2 built against that PyTorch version.

> Tip: Check the official install matrix for PyTorch ↔ CUDA ↔ Detectron2 compatibility. Pick **fixed versions** rather than unconstrained `latest`.

```powershell
# Example (adjust versions to your setup)
# PyTorch (CPU example)
pip install torch==2.3.1 torchvision==0.18.1

# Detectron2 matching your torch build
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 3) Project Dependencies

```powershell
pip install -r requirements.txt
```

> If you don’t have a `requirements.txt` yet, typical extras include: `opencv-python`, `pycocotools`, `tqdm`, `numpy`, `Pillow`, `matplotlib`, `yacs`, `termcolor`.

---

## Datasets

Place datasets under `data/` (ignored by git). For COCO‑style datasets, your layout could look like:

```
data/
└─ my_dataset/
   ├─ annotations/
   │  ├─ instances_train.json
   │  └─ instances_val.json
   ├─ train/
   │  ├─ img_0001.jpg
   │  └─ ...
   └─ val/
      ├─ img_1001.jpg
      └─ ...
```

Register them in your training script (or a small helper) using Detectron2’s `DatasetCatalog`/`MetadataCatalog`.

> A more advanced setup can centralize registration into `src/datasets/register.py` and switch datasets by name via CLI flags.

**Do not commit data.** Instead, add instructions in `data/README.md` (created in Step 5) on how to obtain/prepare the dataset.

---

## Models / Weights

Store checkpoints under `models/` (ignored by git). For reproducible demos, the release includes a compressed weight file as a **GitHub Release asset** (download and unpack into `models/`).

* Expected path (as referenced by `configs/base.yaml`): `models/model_final.pth`
* You can also change the path in `MODEL.WEIGHTS` within the config.

**Do not commit weights.** Use release assets and `models/README.md` (Step 5) for instructions.

---

## Training

Edit `configs/base.yaml` first:

* `ROI_HEADS.NUM_CLASSES`: set to your number of classes.
* `DATASETS.TRAIN` / `DATASETS.TEST`: update names if needed.
* `MODEL.WEIGHTS`: optional init from a backbone or a previous checkpoint.

Then run:

```powershell
# Basic training run\ npython .\src\train.py --config .\configs\base.yaml --output .\build\output
```

Common options your script may support (adjust to your CLI):

* `--dataset-name custom_train` (and corresponding val/test)
* `--resume` to continue training from the last checkpoint
* `--visualize` to periodically visualize predictions in `build/output/vis`

**Outputs** (typical):

* `build/output/metrics.json` (loss/acc/…)
* `build/output/events.out.tfevents*` (TensorBoard)
* `build/output/model_final.pth` (final weights)

---

## Inference / Demo

Point to a single image or a directory:

```powershell
# Single image
python .\src\infer.py --config .\configs\base.yaml --weights .\models\model_final.pth --input .\docs\assets\demo.jpg --output .\build\demo_out

# Directory (images)
python .\src\infer.py --config .\configs\base.yaml --weights .\models\model_final.pth --input .\data\my_dataset\val --output .\build\demo_out
```

The script will save visualizations (boxes/masks) to `--output`.

---

## Loss Tracking & Plots

Training typically generates a metrics file or TensorBoard logs. Use:

```powershell
python .\src\analysis\plot_loss.py --logs .\build\output --out .\build\loss
```

This produces PNG/SVG plots in `build/loss/`.

---

## Configuration

Key fields in `configs/base.yaml` you may want to tune:

* **`MODEL`**: `WEIGHTS`, backbone depth, `MASK_ON`, device (`cuda`/`cpu`).
* **`INPUT`**: input size and format.
* **`DATASETS`**: dataset names for train/test.
* **`SOLVER`**: learning rate, schedule (`STEPS`), `MAX_ITER`, `IMS_PER_BATCH`.
* **`OUTPUT_DIR`**: where checkpoints and logs are stored.

Keep configurations **version‑controlled** and prefer **fixed seeds** when comparing experiments.

---

## Troubleshooting

* **CUDA/PyTorch mismatch**: Ensure your PyTorch build matches your CUDA driver/runtime. If in doubt, try the CPU build first to validate the pipeline.
* **Detectron2 build errors**: Use a Detectron2 wheel/source that matches your exact PyTorch version. Avoid mixing “latest” on one side and pinned versions on the other.
* **`pycocotools` on Windows**: Prefer a prebuilt wheel if available or use WSL for smoother builds.
* **Out of Memory (GPU)**: Reduce `IMS_PER_BATCH`, image size, or switch to gradient accumulation.
* **Dataset not found / empty**: Double‑check registration names and paths; print a sample from `DatasetCatalog` to verify.

---

## Features

* Clean, portfolio‑ready layout with **powerful defaults**.
* **Detectron2** trainer setup with configurable YAML.
* **CLI** for training and inference (PowerShell examples).
* **Loss visualization** and clean output folders.
* Git‑ignored **data/** and **models/** with clear instructions.

---

## What I Learned

* Structuring a production‑leaning repo for Computer Vision.
* Navigating **PyTorch/Detectron2** version compatibilities on Windows.
* Dataset registration patterns for **COCO‑style** data.
* Practical tips for visualization, debugging, and reproducibility.

---

## Roadmap

* [ ] Add `src/datasets/register.py` utility with unified CLI flags.
* [ ] Provide a small synthetic sample dataset for quick smoke tests.
* [ ] Add unit tests for dataset registration and inference.
* [ ] Publish a prebuilt demo weight as a GitHub Release asset.
* [ ] Optional: Dockerfile and `devcontainer.json`.

---

## License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.
