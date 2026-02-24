from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path("/opt/airflow/data")
BRONZE_DIR = DATA_DIR / "bronze"

def ingest_yahoo_to_bronze(
    ticker: str,
    asset_name: str,
    period: str = "max",
) -> str:
    """
    Baixa OHLCV via Yahoo Finance e salva em Parquet na camada Bronze.
    """
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)

    # se vier com MultiIndex nas colunas, achatar
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    out_path = BRONZE_DIR / f"{asset_name}_ohlcv.parquet"
    df.to_parquet(out_path, index=False)

    return str(out_path)