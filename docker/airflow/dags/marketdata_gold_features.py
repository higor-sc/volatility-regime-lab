from datetime import datetime
import sys
import json

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

sys.path.append("/opt/airflow/src")

from features.gold import make_gold_features

def gold_from_params(**context):
    #p = context["params"]
    conf = (context.get("dag_run").conf or {})
    p = {**context["params"], **conf}

    vol_windows = json.loads(p["vol_windows"]) if isinstance(p["vol_windows"], str) else p["vol_windows"]
    ret_windows = json.loads(p["ret_windows"]) if isinstance(p["ret_windows"], str) else p["ret_windows"]
    max_dd_window = int(p["max_dd_window"])

    return make_gold_features(
        asset_name=p["asset_name"],
        price_col=p["price_col"],
        vol_windows=tuple(int(x) for x in vol_windows),
        ret_windows=tuple(int(x) for x in ret_windows),
        max_dd_window=max_dd_window,
    )

with DAG(
    dag_id="marketdata_gold_features",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["gold", "features"],
    params={
        "asset_name": Param("ibov", type="string", title="Asset name"),
        "price_col": Param("Adj Close", type="string", title="(Adj Close or Close)"),
        "vol_windows": Param("[20, 60]", type="string", title="Volatility windows"),
        "ret_windows": Param("[20, 60]", type="string", title="Accumulated return windows"),
        "max_dd_window": Param("60", type="string", title="Max Drawdown window"),
    },
) as dag:

    gold_task = PythonOperator(
        task_id="silver_to_gold_features",
        python_callable=gold_from_params,
    )