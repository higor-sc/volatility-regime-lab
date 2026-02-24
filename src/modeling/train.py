from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import json

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import mlflow

DATA_DIR = Path("/opt/airflow/data")
GOLD_DIR = DATA_DIR / "gold"
MODEL_DIR = GOLD_DIR / "models"

def _ensure_list_of_str(x) -> List[str]:
    # Airflow params may return list of mixed types; normalize to list[str]
    return [str(v) for v in x]

def _select_risk_feature(df: pd.DataFrame, features: List[str]) -> str:
    # Prefer a volatility-based feature to rank clusters into regimes
    for candidate in ["vol_20", "vol_60"]:
        if candidate in features and candidate in df.columns:
            return candidate
    # Fallback to the first feature
    return features[0]

def train_regime_model(
    asset_name: str,
    model_type: str = "kmeans",          # "kmeans" or "gmm"
    n_regimes: int = 3,
    features: Optional[List[str]] = None,
    experiment_name: str = "volatility_regime_lab",
    tracking_uri: Optional[str] = None,  # e.g. "http://mlflow:5000"
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train an unsupervised regime model (KMeans/GMM) from Gold features.

    Inputs:
      - Gold parquet: /opt/airflow/data/gold/{asset}_features.parquet
      - features: list of columns to use for clustering

    Outputs:
      - Model artifacts saved locally and logged to MLflow
      - Regime parquet: /opt/airflow/data/gold/{asset}_regimes.parquet

    Returns:
      - dict with output paths and metadata
    """

    if features is None:
        features = ["vol_20", "vol_60"]

    features = _ensure_list_of_str(features)
    n_regimes = int(n_regimes)
    if n_regimes <= 0:
        raise ValueError("n_regimes must be a positive integer.")

    model_type = str(model_type).strip().lower()
    if model_type not in {"kmeans", "gmm"}:
        raise ValueError("model_type must be 'kmeans' or 'gmm'.")

    in_path = GOLD_DIR / f"{asset_name}_features.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Gold features not found: {in_path}")

    df = pd.read_parquet(in_path).copy()
    if "Date" not in df.columns:
        raise ValueError("Gold dataset must contain a 'Date' column.")

    # Validate feature columns exist
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in Gold: {missing}")

    # Prepare training matrix
    X = df[features].astype(float).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Configure MLflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Prepare artifact paths
    asset_model_dir = MODEL_DIR / asset_name
    asset_model_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = asset_model_dir / "scaler.joblib"
    model_path = asset_model_dir / f"{model_type}_k{n_regimes}.joblib"
    out_regimes_path = GOLD_DIR / f"{asset_name}_regimes.parquet"

    with mlflow.start_run(run_name=f"{asset_name}_{model_type}_k{n_regimes}"):

        mlflow.log_param("asset_name", asset_name)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_regimes", n_regimes)
        mlflow.log_param("features", ",".join(features))
        mlflow.log_param("random_state", random_state)

        # Train
        if model_type == "kmeans":
            model = KMeans(n_clusters=n_regimes, random_state=random_state, n_init="auto")
            cluster_id = model.fit_predict(Xs)
            # A simple metric to track across runs
            mlflow.log_metric("inertia", float(model.inertia_))
        else:
            model = GaussianMixture(n_components=n_regimes, random_state=random_state)
            cluster_id = model.fit_predict(Xs)
            mlflow.log_metric("gmm_lower_bound", float(model.lower_bound_))

        # Attach cluster ids
        df_out = df[["Date"] + features].copy()
        df_out["cluster_id"] = cluster_id.astype(int)

        # Convert clusters into ordered regimes (0=low risk .. k-1=high risk)
        risk_feature = _select_risk_feature(df_out, features)
        cluster_risk = (
            df_out.groupby("cluster_id")[risk_feature]
            .mean()
            .sort_values()
        )
        # Example: {cluster_with_low_risk:0, ...}
        regime_map = {int(cid): int(rank) for rank, cid in enumerate(cluster_risk.index)}
        df_out["regime"] = df_out["cluster_id"].map(regime_map).astype(int)

        regime_map_path = asset_model_dir / f"{model_type}_k{n_regimes}_regime_map.json"
        with open(regime_map_path, "w", encoding="utf-8") as f:
            json.dump(regime_map, f, ensure_ascii=False, indent=2)

        # Save outputs
        df_out.to_parquet(out_regimes_path, index=False)

        joblib.dump(scaler, scaler_path)
        joblib.dump(model, model_path)

        # Log artifacts to MLflow
        mlflow.log_artifact(str(scaler_path), artifact_path="model")
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(out_regimes_path), artifact_path="outputs")
        mlflow.log_artifact(str(regime_map_path), artifact_path="model")

        # Log a quick summary metric (risk ordering)
        mlflow.log_param("risk_feature_for_ordering", risk_feature)

    return {
        "regimes_path": str(out_regimes_path),
        "scaler_path": str(scaler_path),
        "model_path": str(model_path),
        "used_features": features,
        "model_type": model_type,
        "n_regimes": n_regimes,
        "regime_map_path": str(regime_map_path),
    }