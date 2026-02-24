from datetime import datetime
import sys
import json

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

sys.path.append("/opt/airflow/src")

from modeling.score import score_regime_model

def score_from_params(**context):
    #p = context["params"]
    conf = (context.get("dag_run").conf or {})
    p = {**context["params"], **conf}

    features = json.loads(p["features"])
    n_regimes = int(p["n_regimes"])
    include_gold_features = str(p["include_gold_features"]).lower() == "true"

    return score_regime_model(
        asset_name=str(p["asset_name"]),
        model_type=str(p["model_type"]),
        n_regimes=n_regimes,
        features=features,
        include_gold_features=include_gold_features,
        output_name=str(p["output_name"]),
    )

with DAG(
    dag_id="marketdata_model_score",
    start_date=datetime(2024, 1, 1),
    #schedule="@daily",   # scoring can be daily
    schedule=None,  
    catchup=False,
    tags=["modeling", "score", "frontend"],
    params={
        "asset_name": Param("ibov", type="string"),
        "model_type": Param("kmeans", type="string"),
        "n_regimes": Param("3", type="string"),
        "features": Param('["vol_20", "vol_60"]', type="string"),
        "include_gold_features": Param("true", type="string"),
        "output_name": Param("scored", type="string"),
    },
) as dag:

    score_task = PythonOperator(
        task_id="score_regime_model",
        python_callable=score_from_params,
    )