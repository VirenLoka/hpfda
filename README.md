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
        │  src/inference.py        (load best.pt, score raw feature inputs)
        ▼
samples/predictions.csv
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

# 4. Predict on new claims (defaults to checkpoints/exp1/best.pt + samples/sample_claims.csv)
python src/inference.py
python src/inference.py --checkpoint checkpoints/exp1/best.pt \
    --input samples/single_claim.csv --output samples/single_prediction.csv
```

The aggregation and training scripts show `tqdm` progress bars and print
summary blocks (dataset size, feature counts, model parameter count / footprint,
per-epoch gradient-norm statistics).

## Inference inputs

`src/inference.py` loads the best checkpoint and scores a CSV of **raw** feature
values (the 15 numeric + 6 categorical feature columns — no label needed). It
applies the exact training-time preprocessing (numeric standardization +
categorical vocab encoding from `feature_metadata.json`) and writes a copy of the
input with `FraudProbability` and `PredictedFraud` columns appended. Two ready
samples are provided: [`samples/sample_claims.csv`](samples/sample_claims.csv)
(8 rows) and [`samples/single_claim.csv`](samples/single_claim.csv) (1 row).

## Configuration

All paths and hyperparameters live in `configs/train_config.yaml`:

- `data.*` — raw file names, output CSV paths, split ratio/strategy
- `features.*` — exact numeric & categorical feature lists + label column
- `model.*` — transformer width/depth/heads
- `training.*` — batch size, epochs, lr, device (`auto`/`cpu`/`cuda`/`mps`), class weights, eval cadence
- `logging.*` — `log_dir`, `log_file`, TensorBoard toggle, step log interval
- `checkpoint.*` — `dir`, save cadence, rolling `keep_last`, best-model metric/mode
