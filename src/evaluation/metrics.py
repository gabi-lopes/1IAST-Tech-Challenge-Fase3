"""
Métricas de avaliação da Fase 3.

Classe positiva (1) = município EM RISCO educacional. Por isso `precision`,
`recall` e `f1` reportados são **da classe de risco** (a que importa para
política pública), e `accuracy` sempre vem ao lado do baseline.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import config

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]

_LABELS = ["fora de risco", "EM RISCO"]


def holdout_metrics(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Métricas no conjunto de teste isolado."""
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_risco": precision_score(y_test, y_pred, zero_division=0),
        "recall_risco": recall_score(y_test, y_pred, zero_division=0),
        "f1_risco": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "baseline_acuracia": max(y_test.mean(), 1 - y_test.mean()),
        "prevalencia_risco": float(y_test.mean()),
        "confusao": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_teste": int(len(X_test)),
    }


def cv_metrics(estimator, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
               n_splits: int | None = None) -> dict:
    """Validação cruzada por UF: cada fold testa em estados fora do treino."""
    n_splits = min(n_splits or config.CV_FOLDS, groups.nunique())
    res = cross_validate(
        estimator, X, y, groups=groups, cv=GroupKFold(n_splits=n_splits),
        scoring=SCORING, n_jobs=-1,
    )
    return {
        "esquema": f"GroupKFold por UF (n_splits={n_splits})",
        **{m: {"media": float(res[f"test_{m}"].mean()), "desvio": float(res[f"test_{m}"].std())}
           for m in SCORING},
    }


def _pipe(estimator, scale: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    prep = ColumnTransformer([("num", Pipeline(steps), config.FEATURE_COLS)])
    return Pipeline([("preprocessor", prep), ("model", estimator)])


def comparar_modelos(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                     n_splits: int | None = None) -> pd.DataFrame:
    """Baseline + 4 modelos na mesma pipeline/CV. Índice = modelo, colunas = métricas."""
    seed = config.RANDOM_STATE
    modelos = {
        "Baseline (classe majoritária)": (DummyClassifier(strategy="prior"), False),
        "Regressão Logística": (LogisticRegression(**config.LOGREG_PARAMS), True),
        "Árvore de Decisão": (DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed), False),
        "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                                 class_weight="balanced", random_state=seed, n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=seed), False),
    }
    n_splits = min(n_splits or config.CV_FOLDS, groups.nunique())
    linhas = []
    for nome, (est, scale) in modelos.items():
        r = cross_validate(_pipe(est, scale), X, y, groups=groups,
                           cv=GroupKFold(n_splits=n_splits), scoring=SCORING, n_jobs=-1)
        linhas.append({
            "modelo": nome,
            "ROC-AUC": r["test_roc_auc"].mean(),
            "PR-AUC": r["test_average_precision"].mean(),
            "recall_risco": r["test_recall"].mean(),
            "precision_risco": r["test_precision"].mean(),
            "accuracy": r["test_accuracy"].mean(),
        })
    return pd.DataFrame(linhas).set_index("modelo").round(3)


def relatorio_classificacao(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=_LABELS)
