"""
Demonstração — Previsão de risco para o próximo ano
---------------------------------------------------
Pega o último ano disponível na Gold, aplica o modelo já treinado
(data/model/modelo_d_defasagem_temporal.pkl) e estima, pra cada município, a
probabilidade de ficar EM RISCO de não bater a meta no ano seguinte.

⚠️ É extrapolação: o modelo foi validado só na transição 2023 -> 2024. A saída
serve de demonstração, não de previsão validada.

Uso:
    python -m src.modeling.prever_proximo_ano
"""

from __future__ import annotations

import joblib

from src import config
from src.modeling.pipeline_modelo_d import predict_ano_seguinte
from src.preprocessing.gold_consumer import load_gold


def main() -> None:
    print("Carregando Gold...")
    gold = load_gold(config.GOLD_DATASET)

    ano_base = int(gold["ano"].max())
    ano_alvo = ano_base + 1
    print(f"Último ano na Gold: {ano_base} -> prevendo risco em {ano_alvo}")

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {config.MODEL_PATH}. Rode antes:\n"
            f"    python -m src.modeling.pipeline_modelo_d"
        )
    pipeline = joblib.load(config.MODEL_PATH)

    print("Prevendo...")
    pred = predict_ano_seguinte(pipeline, gold, ano_base).sort_values("prob_risco", ascending=False)

    n = len(pred)
    n_risco = int(pred["em_risco"].sum())
    print(f"\n{n} municípios | {n_risco} previstos EM RISCO em {ano_alvo} ({n_risco / n * 100:.1f}%)")
    print(f"\nTop 10 municípios com maior risco previsto pra {ano_alvo}:")
    print(pred.head(10).to_string(index=False))

    saida = config.REPORTS_DIR / f"previsao_risco_{ano_alvo}.csv"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_csv(saida, index=False)
    print(f"\nRanking completo salvo em: {saida.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
