# Atalhos do projeto. No Windows sem `make`, use os comandos `python -m ...` direto
# (ver README) ou rode via Docker (`make docker-run`).

PYTHON ?= python

.PHONY: help install gold train predict plots all notebooks clean docker-build docker-run

help:
	@echo "install       instala as dependencias (requirements.txt)"
	@echo "gold          baixa/atualiza a camada Gold do S3 (requer credenciais AWS)"
	@echo "train         treina o modelo + gera model card"
	@echo "predict       gera o ranking de risco (reports/previsao_risco_*.csv)"
	@echo "plots         gera as figuras (images/*.png)"
	@echo "all           pipeline completa: train -> predict -> plots"
	@echo "notebooks     executa todos os notebooks in-place"
	@echo "clean         remove artefatos gerados (modelo, figuras, previsao)"
	@echo "docker-build  constroi a imagem"
	@echo "docker-run    roda a pipeline completa no container"

install:
	$(PYTHON) -m pip install -r requirements.txt

gold:
	$(PYTHON) -m src.preprocessing.gold_consumer --refresh

train:
	$(PYTHON) -m src.modeling.pipeline_modelo_d

predict:
	$(PYTHON) -m src.modeling.prever_proximo_ano

plots:
	$(PYTHON) -m src.visualization.plots

all:
	$(PYTHON) -m src.run_all

notebooks:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=300 notebooks/*.ipynb

clean:
	-rm -f data/model/*.pkl reports/model_card_*.json reports/previsao_risco_*.csv images/*.png

docker-build:
	docker build -t tc-fase3 .

docker-run:
	docker compose run --rm pipeline
