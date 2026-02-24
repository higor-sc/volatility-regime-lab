from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("/opt/airflow/data")
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

def make_gold_features(
    asset_name: str,
    price_col: str = "Adj Close",
    vol_windows: tuple[int, int] = (20, 60),
    ret_windows: tuple[int, int] = (20, 60),
    max_dd_window: int = 60,
) -> str:
    """
    Gera a camada Gold (features) a partir da Silver.

    - Usa log_ret como base estatística
    - Cria vol rolling (vol_20, vol_60)
    - Cria retorno acumulado (ret_20, ret_60)
    - Cria drawdown e max drawdown rolling (max_dd_60)

    Retorna o caminho do parquet gerado.
    """
    in_path = SILVER_DIR / f"{asset_name}_returns.parquet"
    out_path = GOLD_DIR / f"{asset_name}_features.parquet"
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path).copy()

    # Normalizar janelas (Params do Airflow podem vir como str/float)
    vol_windows = tuple(int(w) for w in vol_windows)
    ret_windows = tuple(int(w) for w in ret_windows)
    max_dd_window = int(max_dd_window)

    # Garantir ordenação temporal (fundamental para rolling)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Escolha de coluna de preço (preferimos Adj Close)
    if price_col not in df.columns:
        # fallback comum
        if "Close" in df.columns:
            price_col = "Close"
        else:
            raise ValueError(f"Without '{price_col}' or 'Close' in Silver.")

    # ----------------------------
    # 1) Volatilidade rolling (std do log_ret)
    # ----------------------------
    for w in vol_windows:
        df[f"vol_{w}"] = df["log_ret"].rolling(w).std()

    # ----------------------------
    # 2) Retorno acumulado em janela (base log_ret)
    #    sum(log_ret) -> exp(sum)-1
    # ----------------------------
    for w in ret_windows:
        df[f"ret_{w}"] = np.exp(df["log_ret"].rolling(w).sum()) - 1.0

    # ----------------------------
    # 3) Drawdown (do pico histórico até a data)
    # ----------------------------
    price = df[price_col].astype(float)
    peak = price.cummax()
    df["drawdown"] = (price / peak) - 1.0  # <= 0

    # ----------------------------
    # 4) Max drawdown rolling (pior drawdown na janela)
    # ----------------------------
    df[f"max_dd_{max_dd_window}"] = df["drawdown"].rolling(max_dd_window).min()

    # ----------------------------
    # Limpeza: remover linhas iniciais com NaN (rolling)
    # (Entregar Gold pronta para ML)
    # ----------------------------
    needed_cols = ["ret", "log_ret"] \
        + [f"vol_{w}" for w in vol_windows] \
        + [f"ret_{w}" for w in ret_windows] \
        + ["drawdown", f"max_dd_{max_dd_window}"]

    df_gold = df[["Date"] + needed_cols].dropna().reset_index(drop=True)

    df_gold.to_parquet(out_path, index=False)
    return str(out_path)