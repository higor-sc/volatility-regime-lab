from datetime import datetime

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.models.param import Param

with DAG(
    dag_id="marketdata_orchestrator",
    start_date=datetime(2024, 1, 1),
    schedule=None,     # manual por enquanto
    catchup=False,
    tags=["orchestrator"],
    params={
        "asset_name": Param("ibov", type="string"),
        "ticker": Param("^BVSP", type="string"),
        "period": Param("max", type="string"),

        # Gold params
        "price_col": Param("Adj Close", type="string"),
        "vol_windows": Param([20, 60], type="array"),
        "ret_windows": Param([20, 60], type="array"),
        "max_dd_window": Param(60, type="integer"),

        # Score params (must match trained model)
        "model_type": Param("kmeans", type="string"),
        "n_regimes": Param(3, type="integer"),
        "features": Param(["vol_20", "vol_60"], type="array"),
        "include_gold_features": Param(True, type="boolean"),
        "output_name": Param("scored", type="string"),
    },
) as dag:

    run_bronze = TriggerDagRunOperator(
        task_id="trigger_bronze_ingest",
        trigger_dag_id="marketdata_bronze_ingest",
        conf={
            "ticker": "{{ params.ticker }}",
            "asset_name": "{{ params.asset_name }}",
            "period": "{{ params.period }}",
        },
        wait_for_completion=True,
        reset_dag_run=True,
    )

    run_silver = TriggerDagRunOperator(
        task_id="trigger_silver_transform",
        trigger_dag_id="marketdata_silver_transform",
        conf={
            "asset_name": "{{ params.asset_name }}",
        },
        wait_for_completion=True,
        reset_dag_run=True,
    )

    run_gold = TriggerDagRunOperator(
        task_id="trigger_gold_features",
        trigger_dag_id="marketdata_gold_features",
        conf={
            "asset_name": "{{ params.asset_name }}",
            "price_col": "{{ params.price_col }}",
            "vol_windows": "{{ params.vol_windows | tojson }}",
            "ret_windows": "{{ params.ret_windows | tojson }}",
            "max_dd_window": "{{ params.max_dd_window }}",
        },
        wait_for_completion=True,
        reset_dag_run=True,
    )

    run_score = TriggerDagRunOperator(
        task_id="trigger_model_score",
        trigger_dag_id="marketdata_model_score",
        conf={
            "asset_name": "{{ params.asset_name }}",
            "model_type": "{{ params.model_type }}",
            "n_regimes": "{{ params.n_regimes }}",
            "features": "{{ params.features | tojson }}",
            "include_gold_features": "{{ params.include_gold_features }}",
            "output_name": "{{ params.output_name }}",
        },
        wait_for_completion=True,
        reset_dag_run=True,
    )

    run_bronze >> run_silver >> run_gold >> run_score