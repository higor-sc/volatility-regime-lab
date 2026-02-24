from datetime import datetime
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

sys.path.append("/opt/airflow/src")

from transform.silver import bronze_to_silver

def silver_from_params(**context):
    #p = context["params"]
    conf = (context.get("dag_run").conf or {})
    p = {**context["params"], **conf}

    return bronze_to_silver(asset_name=p["asset_name"])

with DAG(
    dag_id="marketdata_silver_transform",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["silver", "transform"],
    params={
        "asset_name": Param("ibov", type="string", title="Asset name"),
    },
) as dag:

    silver_task = PythonOperator(
        task_id="bronze_to_silver",
        python_callable=silver_from_params,
    )