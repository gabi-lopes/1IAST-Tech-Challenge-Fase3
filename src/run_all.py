"""
Pipeline completa da Fase 3, de ponta a ponta.

    Gold (S3/cache)  →  treino + avaliação  →  ranking de risco  →  figuras

Saídas:
    data/model/modelo_d_defasagem_temporal.pkl
    reports/model_card_modelo_d.json
    reports/previsao_risco_<ano>.csv
    images/*.png

Uso:
    python -m src.run_all              # usa o cache local da Gold
    python -m src.run_all --refresh    # re-baixa a Gold do S3 (requer credenciais AWS)
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")  # headless — só grava PNG, não abre janela

from src.modeling import prever_proximo_ano
from src.modeling.pipeline_modelo_d import run_pipeline
from src.visualization import plots

log = logging.getLogger("run_all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Roda a pipeline completa da Fase 3.")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-baixa a Gold do S3 antes de treinar (ignora o cache local).")
    args = parser.parse_args()

    log.info("=" * 64)
    log.info("[1/3] Treino + avaliação")
    log.info("=" * 64)
    resultado = run_pipeline(force_refresh=args.refresh, save=True)

    log.info("=" * 64)
    log.info("[2/3] Ranking de risco (previsão do próximo ano)")
    log.info("=" * 64)
    prever_proximo_ano.main()

    log.info("=" * 64)
    log.info("[3/3] Figuras")
    log.info("=" * 64)
    for caminho in plots.gerar_todas(resultado):
        log.info("      OK: %s", caminho)

    log.info("Pipeline concluída.")


if __name__ == "__main__":
    main()
