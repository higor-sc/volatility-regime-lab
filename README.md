# Volatility Regime Lab (IBOV)

End-to-end study/portfolio project for volatility regime detection on financial time series, starting with the Brazilian Ibovespa index (IBOV / ^BVSP).

## Goals
- Automated market data ingestion
- Local Medallion architecture (Bronze / Silver / Gold)
- Unsupervised regime detection (KMeans/GMM first, HMM later)
- Daily regime signal for risk monitoring
- Hands-on Data Engineering + MLOps + DevOps (100% local and free)

## Stack
- Python 3.11
- yfinance (market data)
- Parquet (storage)
- Docker + Docker Compose
- Apache Airflow (orchestration)
- MLflow (experiment tracking)
- scikit-learn (KMeans / GMM)
- Streamlit (simple dashboard)

## Repository Structure

volatility-regime-lab/
docker/
airflow/
dags/
mlflow/
Dockerfile
data/
bronze/
silver/
gold/
mlruns/ # MLflow artifacts (ignored by git)
src/
ingest/
transform/
features/
modeling/
frontend/
app.py
tests/
docker-compose.yml
README.md
.gitignore

## Data Layers (Medallion)
- **Bronze**: raw OHLCV downloaded from Yahoo Finance
- **Silver**: cleaned data + `ret` and `log_ret`
- **Gold**: engineered features (rolling vol, rolling returns, drawdown, etc.)
- **Modeling output**: regimes (`cluster_id`, `regime`) and a scored dataset for visualization

## Airflow DAGs
- `marketdata_bronze_ingest`: download OHLCV → `data/bronze`
- `marketdata_silver_transform`: compute returns → `data/silver`
- `marketdata_gold_features`: compute features → `data/gold`
- `marketdata_model_train`: train KMeans/GMM and log to MLflow
- `marketdata_model_score`: score latest data and generate `*_scored.parquet`
- `marketdata_orchestrator`: triggers Bronze → Silver → Gold → Score

## Quickstart
### 1) Start services
docker compose up -d --build

Airflow UI: http://localhost:8080
MLflow UI: http://localhost:5000

### 2) Run the end-to-end pipeline
Trigger DAG: marketdata_orchestrator

### 3) Run Dashboard
pip install streamlit plotly pandas pyarrow
streamlit run frontend/app.py

## Notes
Training is not meant to run daily in production-style workflows. The expected pattern is:
- Train occasionally (manual/weekly)
- Score daily using a fixed model
This project is educational: infrastructure was introduced gradually to avoid overengineering.

## Roadmap
- Add GMM runs and compare against KMeans
- Champion model selection in MLflow
- HMM-based regime detection
- CI (lint + tests) on GitHub Actions