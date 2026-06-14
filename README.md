# Healthcare Provider Fraud Detection

A transformer-based pipeline that flags potentially fraudulent Medicare
providers from inpatient/outpatient claims and beneficiary data.

Everything is driven by a single config file: [`configs/train_config.yaml`](configs/train_config.yaml).

## Pipeline

```
healthcare_fraud_data/*.csv
        │  src/aggregate_data.py   (merge claims+beneficiary+labels, engineer features, split by provider)
        ▼
data/train.csv, data/test.csv, data/feature_metadata.json
        │  src/dataset.py          (FraudClaimsDataset + DataLoaders, train-stat normalization)
        ▼
src/model.py                       (FT-Transformer: per-feature tokens + [CLS] + encoder head)
        │  src/train.py            (training loop, periodic test eval, logging, checkpointing)
        ▼
runs/<exp>/  (logs + TensorBoard)   checkpoints/<exp>/  (rolling + best.pt)
```

The fraud label (`PotentialFraud`) is provider-level, so the train/test split is
done **by provider** to prevent a provider's claims leaking across both sets.

## Setup

Dependencies are installed in the conda env `hpfda`:

```bash
conda activate hpfda
pip install -r requirements.txt
```

## Run

```bash
# 1. Build aggregated train/test CSVs + feature metadata
python src/aggregate_data.py --config configs/train_config.yaml

# 2. Train (logs to runs/, checkpoints to checkpoints/)
python src/train.py --config configs/train_config.yaml

# 3. (optional) Inspect training curves
tensorboard --logdir runs/
```

## Configuration

All paths and hyperparameters live in `configs/train_config.yaml`:

- `data.*` — raw file names, output CSV paths, split ratio/strategy
- `features.*` — exact numeric & categorical feature lists + label column
- `model.*` — transformer width/depth/heads
- `training.*` — batch size, epochs, lr, device (`auto`/`cpu`/`cuda`/`mps`), class weights, eval cadence
- `logging.*` — `log_dir`, `log_file`, TensorBoard toggle, step log interval
- `checkpoint.*` — `dir`, save cadence, rolling `keep_last`, best-model metric/mode
