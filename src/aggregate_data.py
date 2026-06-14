"""Build a heterogeneous graph from the raw healthcare-fraud CSVs.

The raw data ships as separate inpatient, outpatient, beneficiary and label
files. Rather than flattening them into a tabular matrix (which destroys the
relational structure organized fraud lives in), this script constructs a single
heterogeneous graph:

  node types
    provider     - the labeled targets (PotentialFraud), with aggregate features
    beneficiary  - patients, with demographic / chronic-condition features
    physician    - doctors (attending/operating/other), with aggregate features
    claim        - one node per claim, carrying financial features + the raw ICD
                   diagnosis/procedure codes (embedded end-to-end by the model)

  edges (both directions)
    beneficiary <-> claim       (patient filed the claim)
    claim       <-> provider    (claim billed to the provider)
    physician   <-> claim       (doctor worked the claim)

Provider nodes are split into train/test *masks* (stratified by fraud label) for
transductive node classification. The graph is saved to ``data/graph.pt`` and a
companion ``graph_metadata.json`` records vocab sizes, feature dims and the
provider-id ordering needed for inference.

Everything is driven by configs/train_config.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from tqdm import tqdm

from utils import get_logger, load_config, resolve_path, set_seed

DIAG_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
PHYS_COLS = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
CHRONIC_COLS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=True, na_values=["", "NA"])


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _zscore(arr: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Standardize columns; return (scaled, {mean, std}) with std floored at 1."""
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    scaled = np.nan_to_num((arr - mean) / std)
    return scaled.astype(np.float32), {"mean": mean.tolist(), "std": std.tolist()}


# ---------------------------------------------------------------------------
# Raw load + claim assembly
# ---------------------------------------------------------------------------
def load_claims(cfg: dict, logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    files = cfg["data"]["files"]

    targets = {
        "inpatient": files["inpatient"], "outpatient": files["outpatient"],
        "beneficiary": files["beneficiary"], "train_labels": files["train_labels"],
    }
    frames = {}
    for key, fname in tqdm(targets.items(), desc="Reading CSVs", unit="file"):
        frames[key] = _read_csv(raw_dir / fname)
    inp, out = frames["inpatient"], frames["outpatient"]
    bene, labels = frames["beneficiary"], frames["train_labels"]

    inp["IsInpatient"] = 1
    out["IsInpatient"] = 0
    for col in ["AdmissionDt", "DischargeDt", "DiagnosisGroupCode"]:
        if col not in out.columns:
            out[col] = np.nan
    claims = pd.concat([inp, out], ignore_index=True, sort=False)

    # Keep only claims whose provider has a label.
    labeled = set(labels["Provider"])
    claims = claims[claims["Provider"].isin(labeled)].reset_index(drop=True)
    logger.info(
        "claims=%d providers=%d beneficiaries=%d", len(claims), len(labels), len(bene)
    )
    return claims, bene, labels


# ---------------------------------------------------------------------------
# Per-node feature engineering
# ---------------------------------------------------------------------------
def build_claim_tensors(claims: pd.DataFrame, code_vocab: Dict[str, int], cfg: dict):
    """Numeric claim features + padded diagnosis/procedure code-index matrices."""
    max_d = cfg["features"]["icd"]["max_diagnosis"]
    max_p = cfg["features"]["icd"]["max_procedure"]

    start, end = _date(claims["ClaimStartDt"]), _date(claims["ClaimEndDt"])
    adm, dis = _date(claims["AdmissionDt"]), _date(claims["DischargeDt"])

    num = pd.DataFrame(index=claims.index)
    num["InscClaimAmtReimbursed"] = _num(claims["InscClaimAmtReimbursed"]).fillna(0.0)
    num["DeductibleAmtPaid"] = _num(claims["DeductibleAmtPaid"]).fillna(0.0)
    num["ClaimDurationDays"] = (end - start).dt.days.fillna(0).clip(lower=0)
    num["AdmitDurationDays"] = (dis - adm).dt.days.fillna(0).clip(lower=0)
    num["IsInpatient"] = claims["IsInpatient"].astype(int)
    num["NumDiagnosisCodes"] = claims[DIAG_COLS].notna().sum(axis=1)
    num["NumProcedureCodes"] = claims[PROC_COLS].notna().sum(axis=1)
    num["NumPhysicians"] = claims[PHYS_COLS].notna().sum(axis=1)
    claim_num_cols = list(num.columns)

    def code_matrix(cols: List[str], width: int) -> np.ndarray:
        mat = np.zeros((len(claims), width), dtype=np.int64)
        for j, col in enumerate(cols[:width]):
            vals = claims[col].astype("string")
            mat[:, j] = vals.map(lambda v: code_vocab.get(str(v), 0) if pd.notna(v) else 0)
        return mat

    diag_idx = code_matrix(DIAG_COLS, max_d)
    proc_idx = code_matrix(PROC_COLS, max_p)
    return num.values.astype(np.float32), claim_num_cols, diag_idx, proc_idx


def build_code_vocab(claims: pd.DataFrame) -> Dict[str, int]:
    """Single shared vocabulary over all diagnosis + procedure codes (0 = pad)."""
    codes = set()
    for col in DIAG_COLS + PROC_COLS:
        codes.update(claims[col].dropna().astype(str).unique().tolist())
    return {c: i + 1 for i, c in enumerate(sorted(codes))}


def build_beneficiary_tensors(bene: pd.DataFrame, cfg: dict):
    """Numeric + categorical feature tensors for beneficiary nodes."""
    cat_cols = cfg["features"]["beneficiary_categorical"]

    num = pd.DataFrame(index=bene.index)
    dob, dod = _date(bene["DOB"]), _date(bene["DOD"])
    ref = pd.Timestamp("2009-12-31")  # claims are from 2009
    age = (ref - dob).dt.days / 365.25
    num["Age"] = age.clip(lower=0, upper=120)
    num["IsDead"] = dod.notna().astype(int)
    num["NoOfMonths_PartACov"] = _num(bene["NoOfMonths_PartACov"]).fillna(0)
    num["NoOfMonths_PartBCov"] = _num(bene["NoOfMonths_PartBCov"]).fillna(0)
    for col in ["IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
                "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt"]:
        num[col] = _num(bene[col]).fillna(0.0)
    chronic = bene[CHRONIC_COLS].apply(_num)
    num["NumChronicConditions"] = (chronic == 1).sum(axis=1)
    bene_num_cols = list(num.columns)

    # Categorical vocabs (0 = unknown/missing).
    cat_meta: Dict[str, dict] = {}
    cat = np.zeros((len(bene), len(cat_cols)), dtype=np.int64)
    for j, col in enumerate(cat_cols):
        values = sorted(bene[col].astype("string").fillna("__nan__").unique().tolist())
        vocab = {v: i + 1 for i, v in enumerate(values)}
        cat[:, j] = bene[col].astype("string").fillna("__nan__").map(vocab).fillna(0)
        cat_meta[col] = {"vocab": vocab, "cardinality": len(vocab) + 1}
    return num.values.astype(np.float32), bene_num_cols, cat, cat_meta


def aggregate_provider_features(claims: pd.DataFrame, prov_idx: Dict[str, int]):
    """Provider-level statistics used as initial provider node features."""
    df = pd.DataFrame({
        "Provider": claims["Provider"],
        "reimb": _num(claims["InscClaimAmtReimbursed"]).fillna(0.0),
        "deduct": _num(claims["DeductibleAmtPaid"]).fillna(0.0),
        "ip": claims["IsInpatient"].astype(int),
        "dur": (_date(claims["ClaimEndDt"]) - _date(claims["ClaimStartDt"])).dt.days.fillna(0).clip(lower=0),
        "bene": claims["BeneID"],
        "phys": claims["AttendingPhysician"],
        "ndiag": claims[DIAG_COLS].notna().sum(axis=1),
        "nproc": claims[PROC_COLS].notna().sum(axis=1),
    })
    g = df.groupby("Provider")
    feats = pd.DataFrame({
        "n_claims": g.size(),
        "n_unique_bene": g["bene"].nunique(),
        "n_unique_phys": g["phys"].nunique(),
        "inpatient_ratio": g["ip"].mean(),
        "sum_reimb": g["reimb"].sum(),
        "mean_reimb": g["reimb"].mean(),
        "std_reimb": g["reimb"].std().fillna(0.0),
        "mean_deduct": g["deduct"].mean(),
        "mean_duration": g["dur"].mean(),
        "mean_ndiag": g["ndiag"].mean(),
        "mean_nproc": g["nproc"].mean(),
    })
    cols = list(feats.columns)
    arr = np.zeros((len(prov_idx), len(cols)), dtype=np.float64)
    for pid, row in feats.iterrows():
        arr[prov_idx[pid]] = row.values
    return arr, cols


def aggregate_physician_features(claims: pd.DataFrame, phys_idx: Dict[str, int]):
    """Physician-level statistics across all roles."""
    records = []
    for col in PHYS_COLS:
        sub = pd.DataFrame({
            "phys": claims[col],
            "provider": claims["Provider"],
            "bene": claims["BeneID"],
            "reimb": _num(claims["InscClaimAmtReimbursed"]).fillna(0.0),
            "ip": claims["IsInpatient"].astype(int),
        })
        records.append(sub[sub["phys"].notna()])
    df = pd.concat(records, ignore_index=True)
    g = df.groupby("phys")
    feats = pd.DataFrame({
        "n_claims": g.size(),
        "n_unique_provider": g["provider"].nunique(),
        "n_unique_bene": g["bene"].nunique(),
        "mean_reimb": g["reimb"].mean(),
        "inpatient_ratio": g["ip"].mean(),
    })
    cols = list(feats.columns)
    arr = np.zeros((len(phys_idx), len(cols)), dtype=np.float64)
    for pid, row in feats.iterrows():
        if pid in phys_idx:
            arr[phys_idx[pid]] = row.values
    return arr, cols


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph(cfg: dict, logger) -> Tuple[HeteroData, dict]:
    claims, bene, labels = load_claims(cfg, logger)

    pbar = tqdm(total=7, desc="Building graph", unit="step")

    # --- node index maps -----------------------------------------------------
    provider_ids = sorted(labels["Provider"].unique().tolist())
    prov_idx = {p: i for i, p in enumerate(provider_ids)}

    phys_series = pd.unique(claims[PHYS_COLS].values.ravel("K"))
    physician_ids = sorted([p for p in phys_series if pd.notna(p)])
    phys_idx = {p: i for i, p in enumerate(physician_ids)}

    # Only keep beneficiaries that actually appear in claims.
    used_bene = set(claims["BeneID"].dropna().unique())
    bene = bene[bene["BeneID"].isin(used_bene)].reset_index(drop=True)
    bene_ids = bene["BeneID"].tolist()
    bene_idx = {b: i for i, b in enumerate(bene_ids)}
    pbar.update(1)

    # --- claim node features -------------------------------------------------
    code_vocab = build_code_vocab(claims)
    claim_num, claim_num_cols, diag_idx, proc_idx = build_claim_tensors(claims, code_vocab, cfg)
    claim_num_scaled, claim_norm = _zscore(claim_num)
    pbar.update(1)

    # --- beneficiary node features ------------------------------------------
    bene_num, bene_num_cols, bene_cat, cat_meta = build_beneficiary_tensors(bene, cfg)
    bene_num_scaled, bene_norm = _zscore(bene_num)
    pbar.update(1)

    # --- provider / physician aggregate features ----------------------------
    prov_feat, prov_cols = aggregate_provider_features(claims, prov_idx)
    prov_scaled, prov_norm = _zscore(prov_feat)
    phys_feat, phys_cols = aggregate_physician_features(claims, phys_idx)
    phys_scaled, phys_norm = _zscore(phys_feat)
    pbar.update(1)

    # --- edges ---------------------------------------------------------------
    claim_node = np.arange(len(claims), dtype=np.int64)
    c_bene = claims["BeneID"].map(bene_idx).to_numpy()
    c_prov = claims["Provider"].map(prov_idx).to_numpy()

    bene_claim = np.vstack([c_bene, claim_node])               # bene -> claim
    claim_prov = np.vstack([claim_node, c_prov])               # claim -> provider

    phys_src, claim_dst = [], []
    for col in PHYS_COLS:
        mapped = claims[col].map(phys_idx)
        mask = mapped.notna().to_numpy()
        phys_src.append(mapped[mask].to_numpy())
        claim_dst.append(claim_node[mask])
    phys_claim = np.vstack([np.concatenate(phys_src), np.concatenate(claim_dst)])
    pbar.update(1)

    # --- labels + masks ------------------------------------------------------
    y = (labels.set_index("Provider").loc[provider_ids, "PotentialFraud"] == "Yes").astype(int).to_numpy()
    idx = np.arange(len(provider_ids))
    train_i, test_i = train_test_split(
        idx, test_size=cfg["data"]["test_size"], random_state=cfg["seed"],
        stratify=y if cfg["data"].get("stratify", True) else None,
    )
    train_mask = np.zeros(len(provider_ids), dtype=bool); train_mask[train_i] = True
    test_mask = np.zeros(len(provider_ids), dtype=bool); test_mask[test_i] = True
    pbar.update(1)

    # --- assemble HeteroData -------------------------------------------------
    data = HeteroData()
    data["provider"].x = torch.tensor(prov_scaled, dtype=torch.float32)
    data["provider"].y = torch.tensor(y, dtype=torch.long)
    data["provider"].train_mask = torch.tensor(train_mask)
    data["provider"].test_mask = torch.tensor(test_mask)

    data["physician"].x = torch.tensor(phys_scaled, dtype=torch.float32)

    data["beneficiary"].x_num = torch.tensor(bene_num_scaled, dtype=torch.float32)
    data["beneficiary"].x_cat = torch.tensor(bene_cat, dtype=torch.long)

    data["claim"].x_num = torch.tensor(claim_num_scaled, dtype=torch.float32)
    data["claim"].diag_idx = torch.tensor(diag_idx, dtype=torch.long)
    data["claim"].proc_idx = torch.tensor(proc_idx, dtype=torch.long)

    data["beneficiary", "files", "claim"].edge_index = torch.tensor(bene_claim, dtype=torch.long)
    data["claim", "billed_to", "provider"].edge_index = torch.tensor(claim_prov, dtype=torch.long)
    data["physician", "treats", "claim"].edge_index = torch.tensor(phys_claim, dtype=torch.long)
    # reverse relations so messages flow both ways
    data["claim", "rev_files", "beneficiary"].edge_index = torch.tensor(bene_claim[[1, 0]], dtype=torch.long)
    data["provider", "rev_billed_to", "claim"].edge_index = torch.tensor(claim_prov[[1, 0]], dtype=torch.long)
    data["claim", "rev_treats", "physician"].edge_index = torch.tensor(phys_claim[[1, 0]], dtype=torch.long)
    pbar.update(1)
    pbar.close()

    metadata = {
        "provider_ids": provider_ids,
        "node_counts": {
            "provider": len(provider_ids), "physician": len(physician_ids),
            "beneficiary": len(bene_ids), "claim": len(claims),
        },
        "feature_dims": {
            "provider": prov_scaled.shape[1], "physician": phys_scaled.shape[1],
            "beneficiary_num": bene_num_scaled.shape[1], "claim_num": claim_num_scaled.shape[1],
        },
        "feature_cols": {
            "provider": prov_cols, "physician": phys_cols,
            "beneficiary_num": bene_num_cols, "claim_num": claim_num_cols,
        },
        "icd": {
            "code_vocab_size": len(code_vocab) + 1,  # +1 for padding index 0
            "embedding_dim": cfg["features"]["icd"]["embedding_dim"],
            "max_diagnosis": cfg["features"]["icd"]["max_diagnosis"],
            "max_procedure": cfg["features"]["icd"]["max_procedure"],
        },
        "beneficiary_categorical": cfg["features"]["beneficiary_categorical"],
        "beneficiary_cat_meta": cat_meta,
        "cat_embedding_dim": cfg["features"]["cat_embedding_dim"],
        "normalization": {
            "provider": prov_norm, "physician": phys_norm,
            "beneficiary_num": bene_norm, "claim_num": claim_norm,
        },
        "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        "train_fraud_rate": float(y[train_mask].mean()),
        "test_fraud_rate": float(y[test_mask].mean()),
        "code_vocab": code_vocab,
    }
    return data, metadata


def main():
    parser = argparse.ArgumentParser(description="Build the heterogeneous fraud graph.")
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    logger = get_logger("graph", cfg["logging"].get("log_file"))

    data, metadata = build_graph(cfg, logger)

    out_dir = resolve_path(cfg["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = resolve_path(cfg["data"]["graph_path"])
    meta_path = resolve_path(cfg["data"]["metadata_json"])
    torch.save(data, graph_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    # Write a few held-out providers for the inference demo.
    test_ids = [metadata["provider_ids"][i]
                for i in np.where(data["provider"].test_mask.numpy())[0]]
    n_sample = min(cfg["data"]["n_sample_providers"], len(test_ids))
    sample_path = resolve_path(cfg["data"]["sample_providers_csv"])
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Provider": test_ids[:n_sample]}).to_csv(sample_path, index=False)

    logger.info("Saved graph -> %s", graph_path)
    logger.info("Saved metadata -> %s", meta_path)
    logger.info("Wrote %d sample providers -> %s", n_sample, sample_path)
    nc = metadata["node_counts"]
    summary = [
        "",
        "==================== Graph summary ==========================",
        f"  Nodes  : provider={nc['provider']:,}  beneficiary={nc['beneficiary']:,}  "
        f"physician={nc['physician']:,}  claim={nc['claim']:,}",
        f"  Edges  : bene-claim={data['beneficiary','files','claim'].edge_index.shape[1]:,}  "
        f"claim-provider={data['claim','billed_to','provider'].edge_index.shape[1]:,}  "
        f"physician-claim={data['physician','treats','claim'].edge_index.shape[1]:,}",
        f"  ICD codes (vocab): {metadata['icd']['code_vocab_size']:,}",
        f"  Provider labels  : train={metadata['n_train']:,}  test={metadata['n_test']:,}",
        f"  Fraud rate       : train={metadata['train_fraud_rate']:.3f}  "
        f"test={metadata['test_fraud_rate']:.3f}",
        "============================================================",
    ]
    logger.info("\n".join(summary))


if __name__ == "__main__":
    main()
