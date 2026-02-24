from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

def hello():
    print("Hello from Airflow! 🚀")

with DAG(
    dag_id="hello_world",
    start_date=datetime(2024, 1, 1),
    schedule=None,   # roda só quando você clicar (manual)
    catchup=False,
    tags=["tutorial"],
) as dag:

    hello_task = PythonOperator(
        task_id="say_hello",
        python_callable=hello,
    )