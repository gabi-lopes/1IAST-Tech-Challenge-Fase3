"""
Pipeline do Modelo D — Alfabetização (defasagem temporal)
-----------------------------------------------------------
Treina e salva o modelo final da Fase 3: usa só dado do ano anterior (2023)
pra prever se o município fica EM RISCO EDUCACIONAL no ano seguinte (2024).
Nenhuma feature do mesmo ano do target — sem data leakage
(ver notebooks/03a_diagnostico_vazamento_dados.ipynb).

Target: em_risco = (taxa_alfabetizacao < taxa_media_nacional[ano]).
Classe positiva = município abaixo do nível médio do país.

Uso como script (treina, avalia, salva o modelo + model card):
    python -m src.modeling.pipeline_modelo_d
    python -m src.modeling.pipeline_modelo_d --refresh   # re-baixa a Gold do S3

Uso como módulo:
    from src.modeling.pipeline_modelo_d import run_pipeline, predict
    r = run_pipeline()
    r["pipeline"]   # sklearn Pipeline treinado
    r["metricas"]   # holdout + validação cruzada por UF
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.evaluation.metrics import cv_metrics, holdout_metrics
from src.preprocessing.features import (
    build_inference_frame, build_lagged_frame, ref_nacional_por_ano,
)
from src.preprocessing.gold_consumer import load_gold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_PATH = config.MODEL_PATH  # retrocompat p/ quem importa daqui


# ── Pipeline ─────────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """Pré-processamento (imputação + scaling) + regressão logística, num Pipeline só."""
    num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(transformers=[("num", num, config.FEATURE_COLS)])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(**config.LOGREG_PARAMS)),
    ])


# ── API pública ──────────────────────────────────────────────────────────────

def run_pipeline(force_refresh: bool = False, save: bool = True) -> dict:
    """Carrega a Gold, monta as features defasadas, treina, avalia e salva."""
    log.info("1/5 — Carregando Gold (%s + %s)...", config.GOLD_DATASET, config.GOLD_REF_NACIONAL)
    gold = load_gold(config.GOLD_DATASET, force_refresh=force_refresh)
    ref = ref_nacional_por_ano(load_gold(config.GOLD_REF_NACIONAL, force_refresh=force_refresh))
    log.info("      %d registros | referência nacional: %s", len(gold), ref)

    log.info("2/5 — Montando features defasadas (LAG do ano anterior) + target...")
    frame = build_lagged_frame(gold, ref)
    if frame.empty:
        raise ValueError(
            "Nenhum par (ano anterior -> ano atual) com target definido. "
            "A Gold precisa de >= 2 anos por município e de meta_ano_vigente."
        )
    X = frame[config.FEATURE_COLS]
    y = frame[config.TARGET_COL]
    groups = frame[config.GROUP_COL]
    log.info("      %d municípios | %.1f%% em risco | %d UFs",
             len(frame), 100 * y.mean(), groups.nunique())

    log.info("3/5 — Holdout estratificado (20%%, municípios não vistos)...")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_STATE)
    tr, te = next(splitter.split(X, y))
    X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    log.info("4/5 — Treinando (%s, class_weight=balanced)...", config.FINAL_MODEL)
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    holdout = holdout_metrics(pipe, X_test, y_test)
    cv = cv_metrics(build_pipeline(), X, y, groups)
    log.info(
        "      HOLDOUT  recall_risco: %.1f%% | precision_risco: %.1f%% | "
        "PR-AUC: %.3f | ROC-AUC: %.3f | acc: %.1f%% (baseline %.1f%%)",
        100 * holdout["recall_risco"], 100 * holdout["precision_risco"],
        holdout["pr_auc"], holdout["roc_auc"],
        100 * holdout["accuracy"], 100 * holdout["baseline_acuracia"],
    )
    log.info(
        "      CV/UF    recall_risco: %.1f%%±%.1f | PR-AUC: %.3f±%.3f | ROC-AUC: %.3f±%.3f",
        100 * cv["recall"]["media"], 100 * cv["recall"]["desvio"],
        cv["average_precision"]["media"], cv["average_precision"]["desvio"],
        cv["roc_auc"]["media"], cv["roc_auc"]["desvio"],
    )

    # modelo final: treina em tudo antes de salvar
    pipe_final = build_pipeline().fit(X, y)
    metricas = {"holdout_municipios": holdout, "cv_por_uf": cv}

    if save:
        log.info("5/5 — Salvando modelo + model card...")
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe_final, config.MODEL_PATH)
        _salvar_model_card(gold, frame, metricas, ref)
        log.info("      OK: %s", config.MODEL_PATH.relative_to(config.ROOT))
        log.info("      OK: %s", config.MODEL_CARD_PATH.relative_to(config.ROOT))

    return {
        "pipeline": pipe_final,
        "metricas": metricas,
        "n_treino": len(X_train),
        "n_teste": len(X_test),
        "n_total": len(frame),
        "feature_cols": config.FEATURE_COLS,
        "frame": frame,
        # arrays do holdout — pra visualização / inspeção fora daqui
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
    }


def predict(pipeline: Pipeline, frame_features: pd.DataFrame) -> pd.Series:
    """Aplica o modelo treinado num frame que já tenha config.FEATURE_COLS."""
    return pd.Series(pipeline.predict(frame_features[config.FEATURE_COLS]), name=config.TARGET_COL)


def predict_ano_seguinte(pipeline: Pipeline, gold: pd.DataFrame, ano_base: int) -> pd.DataFrame:
    """Previsão de risco pro ano seguinte a `ano_base` (extrapolação, sem validação)."""
    feats = build_inference_frame(gold, ano_base)
    proba = pipeline.predict_proba(feats[config.FEATURE_COLS])[:, 1]
    return feats[["id_municipio", config.GROUP_COL]].assign(
        prob_risco=proba,
        em_risco=(proba >= 0.5).astype(int),
    )


# ── Model card ───────────────────────────────────────────────────────────────

def _salvar_model_card(gold: pd.DataFrame, frame: pd.DataFrame, metricas: dict,
                       ref_nacional: dict) -> None:
    card = {
        "modelo": f"Modelo D — defasagem temporal ({config.FINAL_MODEL})",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gold": {
            "dataset": config.GOLD_DATASET,
            "n_linhas": int(len(gold)),
            "anos": sorted(int(a) for a in gold["ano"].dropna().unique()),
            "colunas": sorted(gold.columns),
        },
        "target": {
            "coluna": config.TARGET_COL,
            "definicao": "taxa_alfabetizacao < taxa_media_nacional[ano]",
            "referencia_nacional_por_ano": {int(a): float(t) for a, t in ref_nacional.items()},
            "n_amostras": int(len(frame)),
            "prevalencia_risco": float(frame[config.TARGET_COL].mean()),
        },
        "features": config.FEATURE_COLS,
        "hiperparametros": config.LOGREG_PARAMS,
        "metricas": metricas,
        "limitacoes": [
            "Gold só tem 2023 e 2024 — uma única transição temporal.",
            "Holdout é cross-municípios do mesmo ano, não validação temporal.",
            "Previsão de anos futuros é extrapolação não validada.",
        ],
    }
    config.MODEL_CARD_PATH.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina e salva o Modelo D (defasagem temporal) de alfabetização."
    )
    parser.add_argument("--refresh", action="store_true",
                        help="Força re-download da Gold no S3, ignorando o cache local.")
    args = parser.parse_args()

    log.info("=" * 64)
    log.info("Pipeline — Modelo D (defasagem temporal)")
    log.info("=" * 64)
    run_pipeline(force_refresh=args.refresh)


if __name__ == "__main__":
    main()
