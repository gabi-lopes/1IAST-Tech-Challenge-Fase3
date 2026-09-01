"""
Pipeline do Modelo D — Alfabetização (defasagem temporal)
-----------------------------------------------------------
Treina e salva o Modelo D: usa só dado do ano anterior (2023) pra prever
o resultado do ano seguinte (2024). Corrige o vazamento de dados do
Modelo A (notebook 03), que usava features do mesmo ano do resultado.

Reproduz, em formato de script/pipeline formal, a lógica já validada
no notebook 03b_modelo_defasagem_temporal.ipynb.

Uso como script (treina e salva o modelo em data/model/):
    python -m src.modeling.pipeline_modelo_d
    python -m src.modeling.pipeline_modelo_d --refresh   # força re-download do Gold no S3

Uso como módulo:
    from src.modeling.pipeline_modelo_d import run_pipeline
    resultado = run_pipeline()
    resultado["pipeline"]      # sklearn Pipeline treinado, pronto pra .predict()
    resultado["metricas"]      # dict com accuracy/precision/recall/f1/auc
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing.gold_consumer import load_gold

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "model" / "modelo_d_defasagem_temporal.pkl"

FEATURE_COLS = [
    "media_portugues_ant", "taxa_alfabetizacao_ant", "percentual_participacao_ant",
    "meta_2024", "meta_2025", "meta_2026", "meta_2027", "meta_2028", "meta_2029", "meta_2030",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Etapas da pipeline ──────────────────────────────────────────────────────

def _preparar_um_por_ano(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 linha por (município, ano).

    O Gold oficial do S3 (pipeline da Fase 2) já vem assim — não separa por
    rede. A reconstrução local (notebook 01) tem duplicidade Estadual/Privada;
    quando a coluna `rede_label` existir, agregamos as duas antes de seguir
    (evita o LAG parear ano errado). Nos dois casos, o resultado é 1 linha
    por município-ano.
    """
    filtro_rede = "WHERE rede_label IN ('Estadual', 'Privada')" if "rede_label" in df.columns else ""
    return duckdb.sql(f"""
        SELECT id_municipio, ano,
               AVG(taxa_alfabetizacao) AS taxa_alfabetizacao,
               AVG(media_portugues)    AS media_portugues,
               AVG(percentual_participacao) AS percentual_participacao,
               AVG(meta_2024) AS meta_2024, AVG(meta_2025) AS meta_2025, AVG(meta_2026) AS meta_2026,
               AVG(meta_2027) AS meta_2027, AVG(meta_2028) AS meta_2028, AVG(meta_2029) AS meta_2029,
               AVG(meta_2030) AS meta_2030
        FROM df
        {filtro_rede}
        GROUP BY id_municipio, ano
    """).df()


def _construir_features_defasadas(um_por_ano: pd.DataFrame) -> pd.DataFrame:
    """Traz os valores do ano anterior pra mesma linha do ano seguinte (LAG)."""
    return duckdb.sql("""
        SELECT
            id_municipio,
            ano,
            taxa_alfabetizacao AS taxa_alfabetizacao_atual,
            (taxa_alfabetizacao >= 50)::INT AS target,
            LAG(media_portugues)          OVER w AS media_portugues_ant,
            LAG(taxa_alfabetizacao)       OVER w AS taxa_alfabetizacao_ant,
            LAG(percentual_participacao)  OVER w AS percentual_participacao_ant,
            meta_2024, meta_2025, meta_2026, meta_2027, meta_2028, meta_2029, meta_2030
            -- as metas ficam sem defasagem: são valores definidos ANTES do resultado
            -- do ano vigente (meta oficial do programa, não consequência do resultado)
        FROM um_por_ano
        WINDOW w AS (PARTITION BY id_municipio ORDER BY ano)
        QUALIFY taxa_alfabetizacao_ant IS NOT NULL
        ORDER BY id_municipio
    """).df()


def _montar_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), FEATURE_COLS),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1,
        )),
    ])


def _avaliar(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": auc(fpr, tpr),
        "n_teste": len(X_test),
        "baseline": max(y_test.mean(), 1 - y_test.mean()),
    }


# ── API pública ──────────────────────────────────────────────────────────────

def run_pipeline(force_refresh: bool = False, save: bool = True) -> dict:
    """
    Roda a pipeline completa do Modelo D: carrega o Gold, prepara as features
    defasadas, treina, avalia e (por padrão) salva o modelo em data/model/.
    """
    log.info("1/5 — Carregando Gold (indicador_municipio)...")
    df = load_gold("indicador_municipio", force_refresh=force_refresh)
    log.info(f"      {len(df)} registros carregados")

    log.info("2/5 — Agregando Estadual/Privada (1 linha por município-ano)...")
    um_por_ano = _preparar_um_por_ano(df)

    log.info("3/5 — Construindo features defasadas (LAG do ano anterior)...")
    lagged = _construir_features_defasadas(um_por_ano)
    log.info(f"      {len(lagged)} pares (ano anterior -> ano atual) disponíveis")

    if lagged.empty:
        raise ValueError(
            "Nenhum par (ano anterior -> ano atual) encontrado — verifique se o "
            "Gold tem pelo menos 2 anos de dado por município."
        )

    X = lagged[FEATURE_COLS].copy()
    y = lagged["target"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    log.info("4/5 — Treinando o Modelo D (RandomForest, features defasadas)...")
    pipeline = _montar_pipeline()
    pipeline.fit(X_train, y_train)

    metricas = _avaliar(pipeline, X_test, y_test)
    log.info(
        "      Accuracy: %.1f%% | Precision: %.1f%% | Recall: %.1f%% | "
        "F1: %.1f%% | AUC-ROC: %.3f | baseline: %.1f%%",
        metricas["accuracy"] * 100, metricas["precision"] * 100,
        metricas["recall"] * 100, metricas["f1"] * 100,
        metricas["auc"], metricas["baseline"] * 100,
    )

    if save:
        log.info("5/5 — Salvando modelo treinado...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        log.info(f"      ✔ Salvo em: {MODEL_PATH.relative_to(ROOT)}")

    return {
        "pipeline": pipeline,
        "metricas": metricas,
        "n_treino": len(X_train),
        "n_teste": len(X_test),
        "feature_cols": FEATURE_COLS,
    }


def predict(pipeline: Pipeline, df_ano_anterior: pd.DataFrame) -> pd.Series:
    """
    Aplica o Modelo D já treinado em dado de um ano anterior real, pra gerar
    a previsão do ano seguinte (ex.: usar 2024 pra prever 2025).

    df_ano_anterior precisa ter as colunas: media_portugues, taxa_alfabetizacao,
    percentual_participacao (do ano anterior) + meta_2024..meta_2030.
    """
    entrada = pd.DataFrame({
        "media_portugues_ant": df_ano_anterior["media_portugues"],
        "taxa_alfabetizacao_ant": df_ano_anterior["taxa_alfabetizacao"],
        "percentual_participacao_ant": df_ano_anterior["percentual_participacao"],
        **{c: df_ano_anterior[c] for c in
           ["meta_2024", "meta_2025", "meta_2026", "meta_2027", "meta_2028", "meta_2029", "meta_2030"]},
    })
    return pipeline.predict(entrada[FEATURE_COLS])


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Treina e salva o Modelo D (defasagem temporal) de alfabetização."
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Força re-download do Gold no S3, ignorando o cache local.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Pipeline — Modelo D (defasagem temporal)")
    log.info("=" * 60)
    run_pipeline(force_refresh=args.refresh)


if __name__ == "__main__":
    main()
