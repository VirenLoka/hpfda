"""Aggregate the raw healthcare-fraud CSVs into model-ready train/test CSVs.

The raw data ships as separate inpatient, outpatient, beneficiary and label
files. The fraud label (``PotentialFraud``) is defined at the *provider* level,
while claims are per-row. This script:

  1. concatenates inpatient + outpatient claims (tagging which is which),
  2. joins beneficiary demographics/health history onto each claim,
  3. joins the provider-level fraud label onto each claim,
  4. engineers a fixed set of numeric + categorical features,
  5. splits into train/test *by provider* (so a provider never appears in both),
  6. writes train.csv, test.csv and a feature_metadata.json describing the
     numeric columns and categorical vocabularies for the model/dataset.

All inputs, outputs and feature lists come from configs/train_config.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import get_logger, load_config, resolve_path, set_seed

# Raw column groups used during feature engineering.
DIAG_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
PHYS_COLS = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
CHRONIC_COLS = [
    "ChronicCond_Alzheimer",
    "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression",
    "ChronicCond_Diabetes",
    "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=True, na_values=["", "NA"])


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def build_claims(cfg: dict, logger) -> pd.DataFrame:
    """Load raw files and produce one engineered row per claim."""
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    files = cfg["data"]["files"]

    logger.info("Reading raw CSVs from %s", raw_dir)
    read_targets = {
        "inpatient": files["inpatient"],
        "outpatient": files["outpatient"],
        "beneficiary": files["beneficiary"],
        "train_labels": files["train_labels"],
    }
    frames = {}
    for key, fname in tqdm(read_targets.items(), desc="Reading CSVs", unit="file"):
        frames[key] = _read_csv(raw_dir / fname)
    inp, out = frames["inpatient"], frames["outpatient"]
    bene, labels = frames["beneficiary"], frames["train_labels"]
    logger.info(
        "inpatient=%d outpatient=%d beneficiaries=%d providers=%d",
        len(inp), len(out), len(bene), len(labels),
    )

    inp["IsInpatient"] = 1
    out["IsInpatient"] = 0
    # Columns only present on inpatient records; fill on outpatient so concat aligns.
    for col in ["AdmissionDt", "DischargeDt", "DiagnosisGroupCode"]:
        if col not in out.columns:
            out[col] = np.nan

    claims = pd.concat([inp, out], ignore_index=True, sort=False)
    logger.info("Combined claims: %d", len(claims))

    # ---- join beneficiary + provider label --------------------------------
    pbar = tqdm(total=6, desc="Engineering features", unit="step")
    claims = claims.merge(bene, on="BeneID", how="left")
    claims = claims.merge(labels, on="Provider", how="left")
    claims = claims[claims["PotentialFraud"].notna()].copy()
    logger.info("Claims with a provider label: %d", len(claims))
    pbar.update(1)  # joins

    # ---- engineered features ---------------------------------------------
    feat = pd.DataFrame(index=claims.index)
    feat["Provider"] = claims["Provider"]

    # Monetary
    feat["InscClaimAmtReimbursed"] = _to_num(claims["InscClaimAmtReimbursed"]).fillna(0.0)
    feat["DeductibleAmtPaid"] = _to_num(claims["DeductibleAmtPaid"]).fillna(0.0)
    pbar.update(1)  # monetary

    # Durations (days)
    start = _to_date(claims["ClaimStartDt"])
    end = _to_date(claims["ClaimEndDt"])
    feat["ClaimDurationDays"] = (end - start).dt.days.fillna(0).clip(lower=0)
    adm = _to_date(claims["AdmissionDt"])
    dis = _to_date(claims["DischargeDt"])
    feat["AdmitDurationDays"] = (dis - adm).dt.days.fillna(0).clip(lower=0)
    pbar.update(1)  # durations

    # Counts
    feat["NumDiagnosisCodes"] = claims[DIAG_COLS].notna().sum(axis=1)
    feat["NumProcedureCodes"] = claims[PROC_COLS].notna().sum(axis=1)
    feat["NumPhysicians"] = claims[PHYS_COLS].notna().sum(axis=1)
    pbar.update(1)  # counts

    # Patient age at claim start; death flag
    dob = _to_date(claims["DOB"])
    dod = _to_date(claims["DOD"])
    age = (start - dob).dt.days / 365.25
    feat["Age"] = age.fillna(age.median()).clip(lower=0, upper=120)
    feat["IsDead"] = dod.notna().astype(int)

    # Coverage months
    feat["NoOfMonths_PartACov"] = _to_num(claims["NoOfMonths_PartACov"]).fillna(0)
    feat["NoOfMonths_PartBCov"] = _to_num(claims["NoOfMonths_PartBCov"]).fillna(0)

    # Annual amounts
    for col in [
        "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
        "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt",
    ]:
        feat[col] = _to_num(claims[col]).fillna(0.0)

    # Chronic conditions are encoded 1=yes, 2=no -> count the "yes"es.
    chronic = claims[CHRONIC_COLS].apply(_to_num)
    feat["NumChronicConditions"] = (chronic == 1).sum(axis=1)
    pbar.update(1)  # patient/coverage/chronic

    # Categorical (kept raw here; encoded to integer codes below)
    feat["Gender"] = claims["Gender"]
    feat["Race"] = claims["Race"]
    feat["RenalDiseaseIndicator"] = claims["RenalDiseaseIndicator"]
    feat["State"] = claims["State"]
    feat["IsInpatient"] = claims["IsInpatient"].astype(int)
    # IsDead already set above.

    feat["PotentialFraud"] = (claims["PotentialFraud"] == "Yes").astype(int)
    pbar.update(1)  # categoricals + label
    pbar.close()
    return feat


def split_by_provider(df: pd.DataFrame, cfg: dict, logger):
    """Group-aware train/test split so providers don't leak across partitions."""
    group_col = cfg["data"]["split_by"]
    test_size = cfg["data"]["test_size"]
    seed = cfg["seed"]

    prov = df.groupby(group_col)["PotentialFraud"].max().reset_index()
    stratify = prov["PotentialFraud"] if cfg["data"].get("stratify", True) else None
    train_prov, test_prov = train_test_split(
        prov[group_col], test_size=test_size, random_state=seed, stratify=stratify
    )
    train_set, test_set = set(train_prov), set(test_prov)
    train_df = df[df[group_col].isin(train_set)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_set)].reset_index(drop=True)
    logger.info(
        "Split by %s -> train providers=%d (%d claims), test providers=%d (%d claims)",
        group_col, len(train_set), len(train_df), len(test_set), len(test_df),
    )
    return train_df, test_df


def encode_categoricals(train_df, test_df, cat_cols: List[str]) -> Dict[str, dict]:
    """Map categorical values to integer codes (0 reserved for unknown/missing).

    Vocabularies are built on the *train* split only; unseen test values map to 0.
    Returns {col: {"vocab": {value: code}, "cardinality": N}}.
    """
    meta: Dict[str, dict] = {}
    for col in tqdm(cat_cols, desc="Encoding categoricals", unit="col"):
        values = train_df[col].astype("string").fillna("__nan__").unique().tolist()
        vocab = {str(v): i + 1 for i, v in enumerate(sorted(values))}  # 0 = unknown
        for frame in (train_df, test_df):
            frame[col] = (
                frame[col].astype("string").fillna("__nan__").map(vocab).fillna(0).astype(int)
            )
        meta[col] = {"vocab": vocab, "cardinality": len(vocab) + 1}
    return meta


def main():
    parser = argparse.ArgumentParser(description="Aggregate raw CSVs into train/test sets.")
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = get_logger("aggregate", cfg["logging"].get("log_file"))

    df = build_claims(cfg, logger)
    train_df, test_df = split_by_provider(df, cfg, logger)

    num_cols = cfg["features"]["numeric"]
    cat_cols = cfg["features"]["categorical"]
    label = cfg["features"]["label"]

    cat_meta = encode_categoricals(train_df, test_df, cat_cols)

    # Normalization stats (mean/std) from the *train* split only.
    norm = {
        c: {"mean": float(train_df[c].mean()), "std": float(train_df[c].std() or 1.0)}
        for c in num_cols
    }

    keep = num_cols + cat_cols + [label]
    out_dir = resolve_path(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = resolve_path(cfg["data"]["train_csv"])
    test_path = resolve_path(cfg["data"]["test_csv"])
    train_df[keep].to_csv(train_path, index=False)
    test_df[keep].to_csv(test_path, index=False)

    metadata = {
        "numeric": num_cols,
        "categorical": cat_cols,
        "label": label,
        "normalization": norm,
        "categorical_meta": cat_meta,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "train_fraud_rate": float(train_df[label].mean()),
        "test_fraud_rate": float(test_df[label].mean()),
    }
    meta_path = resolve_path(cfg["data"]["metadata_json"])
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Wrote %s (%d rows)", train_path, len(train_df))
    logger.info("Wrote %s (%d rows)", test_path, len(test_df))
    logger.info("Wrote %s", meta_path)

    total = len(train_df) + len(test_df)
    summary = [
        "",
        "==================== Aggregation summary ====================",
        f"  Total claims         : {total:,}",
        f"  Train / Test claims  : {len(train_df):,} / {len(test_df):,}",
        f"  Numeric features     : {len(num_cols)}",
        f"  Categorical features : {len(cat_cols)}  "
        f"(cardinalities: {[cat_meta[c]['cardinality'] for c in cat_cols]})",
        f"  Fraud rate train/test: {metadata['train_fraud_rate']:.3f} / "
        f"{metadata['test_fraud_rate']:.3f}",
        "=============================================================",
    ]
    logger.info("\n".join(summary))


if __name__ == "__main__":
    main()
