# DocForge — Document Forgery Detection and Localisation

An MSc Data Science project focused on detecting and localising tampering in financial documents using deep learning and image analysis.

## 1. Project Overview

DocForge aims to investigate methods for identifying manipulated areas in document images, such as bank statements, receipts, vouchers and other financial documents.

The project explores two main approaches:

1. **ELA + CNN:** Error Level Analysis (ELA) combined with a Convolutional Neural Network (CNN) as a document-level forgery-detection baseline.
2. **Image Segmentation:** A U-Net-based approach to predict and localise potentially manipulated pixels.

A pretrained TruFor model is also planned as a potential comparison method.

The longer-term objective is to develop a reproducible detection workflow that can be served through an inference API and monitored using Prometheus and Grafana.

## 2. Project Objectives

- Explore and prepare document-forgery datasets.
- Investigate ELA as an image-forensics preprocessing technique.
- Develop a CNN baseline for forgery classification.
- Develop a segmentation model for pixel-level forgery localisation.
- Evaluate models using appropriate classification and segmentation metrics.
- Compare the proposed approach with TruFor, if feasible.
- Explore model serving, containerisation and monitoring.

## 3. Dataset

The project currently contains two DocTamper LMDB datasets.

| Dataset | Samples | Image size | Format |
|---|---:|---|---|
| Small | 2,000 | 512 × 512 | RGB JPEG |
| Large | 120,000 | 512 × 512 | RGB JPEG |

Each sample has a corresponding grayscale mask.

### Current verification status

Both datasets have been inspected using the project dataset utilities.

- Small dataset: 2,000 image/mask pairs verified.
- Large dataset: 120,000 image/mask pairs verified.
- No missing or decoding errors were reported.
- No empty masks were reported.
- Large-dataset masks have been observed with pixel values `0`, `1` and `255`.

**Important:** Dataset integrity verification does not establish the official dataset split, annotation semantics or the availability of genuine examples. These must be confirmed before final model training and evaluation.

### Dataset locations

```text
data/
├── archives/
├── raw/
│   ├── small/
│   └── large/
├── processed/
└── splits/
    ├── small_dataset_split.csv
    └── large_dataset_split.csv
```

The raw datasets and archive files are stored locally and should not be committed to Git.

## 4. Technology Stack

| Area | Technology |
|---|---|
| Programming language | Python 3.12 |
| Dependency management | Poetry |
| Deep learning | PyTorch |
| Image processing | Pillow, NumPy |
| Dataset storage | LMDB |
| Experimentation | Jupyter Notebook |
| Development environment | VS Code |
| Local hardware | Apple Silicon MacBook Pro (M3) |
| Containerisation | Docker / Docker Compose |
| Monitoring | Prometheus / Grafana |

PyTorch 2.3.1 and Apple MPS availability were confirmed in the previous local environment checks. The active environment should be checked again when running new experiments.

## 5. Repository Structure

```text
doc-forgery-detection/
├── README.md
├── pyproject.toml
├── poetry.lock
│
├── data/
│   ├── archives/
│   ├── raw/
│   │   ├── small/
│   │   └── large/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── src/
│   └── docforge/
│       ├── data/
│       │   ├── dataset_tools.py
│       │   ├── lmdb_reader.py
│       │   ├── split_dataset.py
│       │   ├── split_large_dataset.py
│       │   ├── verify_dataset.py
│       │   └── visualize_lmdb.py
│       │
│       ├── models/
│       │   ├── cnn.py
│       │   └── unet.py
│       │
│       ├── training/
│       │   ├── dataset.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── metrics.py
│       │
│       ├── serving/
│       └── monitoring/
│
├── docker/
│   ├── docker-compose.yml
│   ├── exporter/
│   └── prometheus/
│
└── tests/
```

## 6. Setup

Clone the repository and enter the project directory:

```bash
git clone https://github.com/odette-eleanor/PROM02_240291316.git
cd PROM02_240291316/doc-forgery-detection
```

Install project dependencies:

```bash
poetry install
```

Activate the Poetry environment:

```bash
poetry shell
```

Alternatively, execute commands directly using:

```bash
poetry run python --version
```

### Jupyter setup

Install Jupyter and the IPython kernel if they are not already included in the project dependencies:

```bash
poetry add --group dev jupyterlab ipykernel
```

Register the Poetry environment:

```bash
poetry run python -m ipykernel install --user \
  --name doc-forgery-detection \
  --display-name "Python (DocForgery)"
```

Launch JupyterLab:

```bash
poetry run jupyter lab
```

Select the `Python (DocForgery)` kernel when opening project notebooks.

## 7. Dataset Inspection and Verification

Run commands from the `doc-forgery-detection` project root.

Inspect the small dataset:

```bash
poetry run python src/docforge/data/dataset_tools.py inspect --dataset small
```

Inspect the large dataset:

```bash
poetry run python src/docforge/data/dataset_tools.py inspect --dataset large
```

Verify the small dataset:

```bash
poetry run python src/docforge/data/dataset_tools.py verify --dataset small
```

Verify the large dataset:

```bash
poetry run python src/docforge/data/dataset_tools.py verify --dataset large
```

Visualise a sample:

```bash
poetry run python src/docforge/data/dataset_tools.py visualize --dataset large --index 100
```

These utilities support dataset inspection, integrity checking and visualisation. Final split creation and training preparation require additional validation.

## 8. Model Development Plan

### Approach 1 — ELA + CNN

1. Explore ELA and JPEG recompression.
2. Establish the classification labels and dataset split.
3. Prepare ELA inputs.
4. Develop and train the CNN baseline.
5. Evaluate classification performance.

Planned classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

### Approach 2 — Segmentation

1. Prepare image/mask pairs.
2. Develop the U-Net training pipeline.
3. Train the segmentation model.
4. Evaluate pixel-level localisation.
5. Compare with TruFor, if feasible.

Planned segmentation metrics:

- Intersection over Union (IoU)
- Pixel-level F1-score

### Deployment and Monitoring

The repository contains initial Docker, exporter and Prometheus configuration files.

The intended future workflow is:

- Model inference API.
- Docker-based deployment.
- Prometheus metrics collection.
- Grafana dashboards.
- Inference performance monitoring.

These components are still subject to implementation and testing.

## 9. Experiment Reproducibility

Experiments should record:

- Dataset version and split.
- Random seed.
- Python and PyTorch versions.
- Model architecture and configuration.
- Optimiser, learning rate and batch size.
- Number of epochs.
- Training and validation loss.
- Evaluation metrics.
- Git commit associated with the experiment.

Save checkpoints, logs, plots and predictions outside the source-code directory. Do not commit large datasets or model weights to Git unless there is a deliberate reason and an appropriate storage method.

## 10. Current Development Status

| Component | Status |
|---|---|
| Project structure | In place |
| Small dataset inspection and verification | Completed |
| Large dataset inspection and verification | Completed |
| Dataset split validation | Pending |
| PyTorch data preparation | In development |
| ELA exploration | Planned |
| CNN training and evaluation | Pending |
| U-Net training and evaluation | Pending |
| TruFor comparison | Planned |
| Inference API | Pending |
| Monitoring and deployment | In development |

## 11. Important Notes

- Keep raw datasets outside Git.
- Do not assume an empty mask represents a genuine document without confirming the dataset documentation.
- Normalise mask values consistently before segmentation training.
- Avoid data leakage between training, validation and test sets.
- Prefer source-document-aware splitting if samples share original documents.
- Keep experiment configurations and results reproducible.

---

**Project:** DocForge — Document Forgery Detection and Localisation  
**Repository:** [PROM02_240291316](https://github.com/odette-eleanor/PROM02_240291316.git)  
**Academic context:** MSc Data Science dissertation project
