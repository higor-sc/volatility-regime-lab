#O que isso faz?
#cria um container Python
#instala MLflow
#inicia o servidor MLflow
#guarda o “banco” do MLflow em mlflow.db (sqlite)
#guarda os artefatos em /mlruns

# Esta é uma imagem base oficial
FROM python:3.11-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instala as dependências 
RUN pip install --no-cache-dir mlflow

# Expõe a porta que a aplicação utiliza
EXPOSE 5000

# Comando para rodar a aplicação
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlruns"]