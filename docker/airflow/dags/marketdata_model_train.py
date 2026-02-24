from datetime import datetime
import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

sys.path.append("/opt/airflow/src")

from modeling.train import train_regime_model

def train_from_params(**context):
    p = context["params"]

    # Tracking URI: prefer env var, fallback to docker service name
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

    return train_regime_model(
        asset_name=str(p["asset_name"]),
        model_type=str(p["model_type"]),
        n_regimes=int(p["n_regimes"]),
        features=list(p["features"]),
        experiment_name=str(p["experiment_name"]),
        tracking_uri=tracking_uri,
        random_state=int(p["random_state"]),
    )


with DAG(
    dag_id="marketdata_model_train",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # keep manual for now (training is usually not daily)
    catchup=False,
    tags=["modeling", "train", "mlflow"],
    params={
        "asset_name": Param("ibov", type="string", title="Asset name"),
        "model_type": Param("kmeans", type="string", title="Model: kmeans or gmm"),
        "n_regimes": Param(3, type="integer", title="Number of regimes"),
        "features": Param(["vol_20", "vol_60"], type="array", title="Feature columns"),
        "experiment_name": Param("volatility_regime_lab", type="string", title="MLflow experiment"),
        "random_state": Param(42, type="integer", title="Random state"),
    },
) as dag:

    train_task = PythonOperator(
        task_id="train_regime_model",
        python_callable=train_from_params,
    )