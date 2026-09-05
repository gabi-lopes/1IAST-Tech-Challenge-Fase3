"""
Interpretabilidade do modelo final (regressão logística sobre features defasadas).

Tudo aqui opera sobre o `Pipeline` treinado (`build_pipeline` de
`src.modeling.pipeline_modelo_d`), sempre sobre o modelo HONESTO — nunca o
contraexemplo com data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src import config


def tabela_coeficientes(pipe: Pipeline) -> pd.DataFrame:
    """
    Coeficientes da regressão logística (features padronizadas) + odds ratio.

    coef < 0  -> a feature REDUZ o log-odds de risco (bom desempenho anterior).
    odds_ratio = exp(coef): fator multiplicativo na chance de risco por +1 desvio.
    """
    model = pipe.named_steps["model"]
    coef = pd.Series(model.coef_[0], index=config.FEATURE_COLS, name="coef")
    return (
        pd.DataFrame({"coef": coef, "odds_ratio": np.exp(coef)})
        .sort_values("coef")
        .round(3)
    )


def tabela_permutation_importance(
    pipe: Pipeline, X: pd.DataFrame, y: pd.Series,
    scoring: str = "average_precision", n_repeats: int = 20,
) -> pd.DataFrame:
    """Queda média da métrica ao embaralhar cada feature (± desvio)."""
    perm = permutation_importance(
        pipe, X, y, scoring=scoring, n_repeats=n_repeats, random_state=config.RANDOM_STATE,
    )
    return (
        pd.DataFrame(
            {"importancia": perm.importances_mean, "desvio": perm.importances_std},
            index=config.FEATURE_COLS,
        )
        .sort_values("importancia", ascending=False)
        .round(4)
    )


def shap_explanation(pipe: Pipeline, X: pd.DataFrame):
    """
    Retorna um shap.Explanation (LinearExplainer) sobre X já pré-processado.
    Requer `shap` instalado (está no requirements.txt).
    """
    import shap

    X_prep = pipe.named_steps["preprocessor"].transform(X)
    explainer = shap.LinearExplainer(pipe.named_steps["model"], X_prep)
    sv = explainer(X_prep)
    sv.feature_names = list(config.FEATURE_COLS)
    return sv
