from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("/opt/airflow/data")
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"

def bronze_to_silver(
    asset_name: str,
) -> str:
    """
    Transforma dados da camada Bronze em Silver:
    - ordena por data
    - calcula retorno simples (ret)
    - calcula log retorno (log_ret)
    """

    # caminhos de entrada/saída
    in_path = BRONZE_DIR / f"{asset_name}_ohlcv.parquet"
    out_path = SILVER_DIR / f"{asset_name}_returns.parquet"

    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    # leitura
    df = pd.read_parquet(in_path)

    # garantir tipo datetime e ordenação
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # cálculo de retornos
    df["ret"] = df["Adj Close"].pct_change()
    df["log_ret"] = np.log(df["Adj Close"]).diff()

    # remove a primeira linha (retornos NaN)
    df = df.dropna(subset=["ret", "log_ret"])

    # salva Silver
    df.to_parquet(out_path, index=False)

    return str(out_path)