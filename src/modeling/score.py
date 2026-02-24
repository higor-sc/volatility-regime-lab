from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
import json

import pandas as pd
import numpy as np
import joblib

DATA_DIR = Path("/opt/airflow/data")
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MODEL_DIR = GOLD_DIR / "models"

def score_regime_model(
    asset_name: str,
    model_type: str = "kmeans",
    n_regimes: int = 3,
    features: Optional[List[str]] = None,
    include_gold_features: bool = True,
    output_name: str = "scored",  # file suffix
) -> Dict[str, Any]:
    """
    Score a trained regime model on the latest Gold features and merge with Silver
    to produce a front-end friendly dataset.

    Reads:
      - Gold features:  /opt/airflow/data/gold/{asset}_features.parquet
      - Silver data:    /opt/airflow/data/silver/{asset}_returns.parquet
      - Model artifacts:
          /opt/airflow/data/gold/models/{asset}/scaler.joblib
          /opt/airflow/data/gold/models/{asset}/{model_type}_k{n_regimes}.joblib
          /opt/airflow/data/gold/models/{asset}/{model_type}_k{n_regimes}_regime_map.json

    Writes:
      - /opt/airflow/data/gold/{asset}_{output_name}.parquet
    """

    if features is None:
        features = ["vol_20", "vol_60"]

    model_type = str(model_type).strip().lower()
    n_regimes = int(n_regimes)

    gold_path = GOLD_DIR / f"{asset_name}_features.parquet"
    silver_path = SILVER_DIR / f"{asset_name}_returns.parquet"

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold features not found: {gold_path}")
    if not silver_path.exists():
        raise FileNotFoundError(f"Silver data not found: {silver_path}")

    # Load datasets
    df_gold = pd.read_parquet(gold_path).copy()
    df_silver = pd.read_parquet(silver_path).copy()

    df_gold["Date"] = pd.to_datetime(df_gold["Date"])
    df_silver["Date"] = pd.to_datetime(df_silver["Date"])

    # Validate feature columns
    missing = [c for c in features if c not in df_gold.columns]
    if missing:
        raise ValueError(f"Missing feature columns in Gold: {missing}")

    # Load artifacts
    asset_model_dir = MODEL_DIR / asset_name
    scaler_path = asset_model_dir / "scaler.joblib"
    model_path = asset_model_dir / f"{model_type}_k{n_regimes}.joblib"
    map_path = asset_model_dir / f"{model_type}_k{n_regimes}_regime_map.json"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not map_path.exists():
        raise FileNotFoundError(
            f"Regime map not found: {map_path}. "
            "Re-run training after adding regime_map persistence."
        )

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    with open(map_path, "r", encoding="utf-8") as f:
        regime_map = json.load(f)

    # Ensure JSON keys are ints
    regime_map = {int(k): int(v) for k, v in regime_map.items()}

    # Score
    X = df_gold[features].astype(float).values
    Xs = scaler.transform(X)
    cluster_id = model.predict(Xs).astype(int)

    df_pred = df_gold[["Date"]].copy()
    df_pred["cluster_id"] = cluster_id
    df_pred["regime"] = df_pred["cluster_id"].map(regime_map).astype(int)

    # Optionally include Gold features in the output for plotting/analysis
    if include_gold_features:
        # columns already present in Silver that we do NOT want to duplicate
        silver_cols = set(df_silver.columns)

        # keep only Gold columns that are not in Silver (plus Date for join)
        gold_extra_cols = [c for c in df_gold.columns if c not in silver_cols and c != "Date"]

        df_pred = df_pred.merge(
            df_gold[["Date"] + gold_extra_cols],
            on="Date",
            how="left",
        )

    # Merge with Silver to get OHLCV/returns for candlesticks + labels
    df_out = df_silver.merge(df_pred, on="Date", how="inner").sort_values("Date")

    out_path = GOLD_DIR / f"{asset_name}_{output_name}.parquet"
    df_out.to_parquet(out_path, index=False)

    return {"scored_path": str(out_path), "rows": int(len(df_out))}