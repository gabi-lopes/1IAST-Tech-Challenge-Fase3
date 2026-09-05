# Imagem para rodar a pipeline da Fase 3 de forma reproduzível.
#   docker build -t tc-fase3 .
#   docker run --rm --env-file .env -v "$PWD/data:/app/data" \
#              -v "$PWD/reports:/app/reports" -v "$PWD/images:/app/images" tc-fase3
FROM python:3.12-slim

# build-essential só é necessário se algum wheel não estiver disponível;
# mantido enxuto — descomente se o pip install falhar ao compilar.
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/raw/ ./data/raw/

# A Gold (data/gold/) NÃO é copiada — vem por volume (cache local) ou é baixada
# no runtime com `--refresh` (requer credenciais AWS via --env-file .env).
CMD ["python", "-m", "src.run_all"]
