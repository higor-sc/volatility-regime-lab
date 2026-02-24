from datetime import datetime
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models.param import Param

# permitir importar seu /src
sys.path.append("/opt/airflow/src")

from ingest.yahoo import ingest_yahoo_to_bronze

def ingest_from_params(**context):
    """
    Wrapper para puxar params do Airflow e chamar a ingestão genérica.
    """
    #p = context["params"]
    conf = (context.get("dag_run").conf or {})
    p = {**context["params"], **conf}

    return ingest_yahoo_to_bronze(
        ticker=p["ticker"],
        asset_name=p["asset_name"],
        period=p["period"],
    )


with DAG(
    dag_id="marketdata_bronze_ingest",
    start_date=datetime(2024, 1, 1),
    schedule=None,   # manual por enquanto (didático)
    catchup=False,
    tags=["bronze", "ingest", "yfinance"],
    params={
        "ticker": Param("^BVSP", type="string", title="Yahoo ticker"),
        "asset_name": Param("ibov", type="string", title="Asset Name (bronze folder)"),
        "period": Param("max", type="string", title="period (ex: 1y, 5y, max)"),
    },
) as dag:

    ingest_task = PythonOperator(
        task_id="download_to_bronze",
        python_callable=ingest_from_params,
    )