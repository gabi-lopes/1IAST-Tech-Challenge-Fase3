"""
Demonstração — Previsão para o próximo ano com o Modelo D
------------------------------------------------------------
Pega o último ano disponível no Gold, usa o modelo já treinado
(data/model/modelo_d_defasagem_temporal.pkl) e prevê o ano seguinte
pra cada município. É a mesma lógica da pipeline, só que aplicada
em dado real (sem saber a resposta), pra mostrar a previsão de verdade.

Uso:
    python -m src.modeling.prever_proximo_ano
"""

from __future__ import annotations

import joblib

from src.modeling.pipeline_modelo_d import MODEL_PATH, _preparar_um_por_ano, predict
from src.preprocessing.gold_consumer import load_gold


def main():
    print("Carregando Gold...")
    df = load_gold("indicador_municipio")
    um_por_ano = _preparar_um_por_ano(df)

    ultimo_ano = int(um_por_ano["ano"].max())
    proximo_ano = ultimo_ano + 1
    print(f"Último ano disponível: {ultimo_ano} -> prevendo {proximo_ano}")

    dados_ultimo_ano = um_por_ano[um_por_ano["ano"] == ultimo_ano].copy()
    print(f"{len(dados_ultimo_ano)} municípios com dado em {ultimo_ano}")

    print("Carregando modelo treinado...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. Rode antes:\n"
            f"    python -m src.modeling.pipeline_modelo_d"
        )
    pipeline = joblib.load(MODEL_PATH)

    print("Prevendo...")
    previsoes = predict(pipeline, dados_ultimo_ano)

    resultado = dados_ultimo_ano[["id_municipio"]].copy()
    resultado["taxa_alfabetizacao_" + str(ultimo_ano)] = dados_ultimo_ano["taxa_alfabetizacao"]
    resultado[f"previsao_bate_meta_{proximo_ano}"] = previsoes

    print(f"\nExemplo (10 primeiros municípios) — previsão pra {proximo_ano}:")
    print(resultado.head(10).to_string(index=False))

    total = len(resultado)
    positivos = int(previsoes.sum())
    print(
        f"\nResumo: {positivos}/{total} municípios previstos para bater "
        f"a meta de alfabetização em {proximo_ano} ({positivos / total * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
