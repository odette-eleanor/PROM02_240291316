# Document Forgery Detection for Loan Fraud Prevention

MSc Computing project (PROM02). Detects and localises forged/tampered
financial documents (receipts, invoices, bank statements) submitted as
supporting evidence in loan applications, using deep learning. The trained
model is served behind an API with a live Prometheus/Grafana monitoring
layer tracking inference latency, confidence and drift.

## Project structure

```
doc-forgery-detection/
├── src/docforge/
│   ├── data/          # dataset loading & preprocessing (LMDB readers etc.)
│   ├── models/        # model architectures
│   ├── training/       # training loops, evaluation scripts
│   ├── serving/         # inference API (FastAPI)
│   └── monitoring/     # Prometheus exporter for the served model
├── notebooks/           # exploratory notebooks (not for pipeline code)
├── docker/               # docker-compose stack: exporter + Prometheus + Grafana
├── tests/                 # unit tests
└── data/                   # local dataset storage (gitignored - never commit data)
```

## Setup

1. Install Poetry if you don't have it:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
2. Install dependencies:
   ```bash
   poetry install
   ```
3. Activate the environment:
   ```bash
   poetry shell
   ```
4. Download the dataset (not committed to git - see below) into `data/`.

## Dataset

This project uses the **DocTamper** dataset (Qu et al., CVPR 2023) and the
**Find It Again** receipt forgery dataset (Martinez Tornes et al., ICDAR
2023) as a real-world evaluation set. Download instructions:

- DocTamper (Kaggle mirror): https://www.kaggle.com/datasets/dinmkeljiame/doctamper/data
- Find It Again: https://l3i-share.univ-lr.fr/2023Finditagain/index.html

Place extracted data under `data/` (this folder is gitignored - the raw
dataset should never be committed to the repository).

## Monitoring stack

```bash
cd docker
docker compose up --build
```

- Exporter metrics: http://localhost:9100/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Status

Early setup stage - dataset exploration in progress.
