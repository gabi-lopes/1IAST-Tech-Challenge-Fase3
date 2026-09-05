"""
Geração das figuras do projeto → `images/*.png`.

Uso como script (gera todas as figuras da entrega):
    python -m src.visualization.plots

Uso como módulo:
    from src.visualization import plots
    fig = plots.volatilidade_taxa(gold)          # devolve a Figure
    plots.salvar(fig, "eda_volatilidade")        # grava em images/
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from src import config

sns.set_theme(style="whitegrid", palette="muted")

IMAGES_DIR = config.ROOT / "images"

_UF_REGIAO = {1: "Norte", 2: "Nordeste", 3: "Sudeste", 4: "Sul", 5: "Centro-Oeste"}


def salvar(fig: plt.Figure, nome: str) -> str:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    caminho = IMAGES_DIR / f"{nome}.png"
    fig.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return str(caminho.relative_to(config.ROOT))


# ── EDA ──────────────────────────────────────────────────────────────────────

def volatilidade_taxa(gold: pd.DataFrame) -> plt.Figure:
    w = gold.assign(id_municipio=gold.id_municipio.astype("int64")).pivot_table(
        index="id_municipio", columns="ano", values="taxa_alfabetizacao")
    anos = sorted(w.columns)
    delta = (w[anos[-1]] - w[anos[0]]).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(delta, bins=40, ax=axes[0])
    axes[0].axvline(0, color="k", lw=1)
    axes[0].set_title(f"Variação da taxa por município ({anos[0]}→{anos[-1]}, σ = {delta.std():.1f} p.p.)")
    axes[0].set_xlabel("Δ taxa_alfabetizacao")
    axes[1].scatter(w[anos[0]], w[anos[-1]], s=6, alpha=.3)
    axes[1].plot([0, 100], [0, 100], "r--")
    axes[1].set_xlabel(f"taxa {anos[0]}"); axes[1].set_ylabel(f"taxa {anos[-1]}")
    axes[1].set_title("Dispersão em torno da diagonal")
    fig.tight_layout()
    return fig


def taxa_por_regiao(gold: pd.DataFrame, ref_nacional: dict[int, float]) -> plt.Figure:
    g = gold.copy()
    g["cod_uf"] = g.id_municipio.astype(str).str.zfill(7).str[:2].astype(int)
    g["regiao"] = g.cod_uf.astype(str).str[0].astype(int).map(_UF_REGIAO)
    ano = int(g.ano.max())
    g_ano = g[g.ano == ano]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    g_ano.groupby("regiao").taxa_alfabetizacao.mean().sort_values().plot.barh(ax=axes[0], color="#4c72b0")
    axes[0].axvline(ref_nacional[ano], color="r", ls="--", label="média nacional")
    axes[0].set_title(f"Taxa média por região ({ano})"); axes[0].legend()
    g_ano.groupby("cod_uf").taxa_alfabetizacao.mean().sort_values().plot.barh(ax=axes[1], color="#55a868")
    axes[1].set_title(f"Taxa média por UF (cód. IBGE, {ano})")
    fig.tight_layout()
    return fig


def correlacoes_alvo(gold: pd.DataFrame) -> plt.Figure:
    num = ["taxa_alfabetizacao", "media_portugues", "taxa_meta_base",
           "percentual_participacao", "meta_2024", "meta_2027"]
    num = [c for c in num if c in gold.columns]
    corr = gold[num].corr()["taxa_alfabetizacao"].drop("taxa_alfabetizacao").sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    cores = ["#c44e52" if abs(v) > 0.6 else "#4c72b0" for v in corr]
    corr.plot.barh(ax=ax, color=cores)
    ax.set_title("Correlação com taxa_alfabetizacao (vermelho = quase-cópia do alvo → leakage)")
    fig.tight_layout()
    return fig


# ── Modelagem ────────────────────────────────────────────────────────────────

def comparacao_modelos(df_comparacao: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    df_comparacao[["ROC-AUC", "PR-AUC", "recall_risco"]].plot.barh(ax=ax)
    ax.set_title("Comparação de modelos — CV GroupKFold por UF")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def matriz_confusao(y_true, y_pred) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["fora", "risco"], yticklabels=["fora", "risco"], ax=ax)
    ax.set_xlabel("previsto"); ax.set_ylabel("real")
    ax.set_title("Matriz de confusão — holdout")
    fig.tight_layout()
    return fig


def curvas_roc_pr(y_true, y_proba) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "--", color="gray"); axes[0].set_title("ROC")
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=axes[1])
    axes[1].set_title("Precisão × Recall")
    fig.tight_layout()
    return fig


def shap_summary(shap_values) -> plt.Figure:
    import shap
    shap.plots.beeswarm(shap_values, show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 3.5)
    fig.suptitle("SHAP — impacto das features (modelo honesto)", y=1.02)
    fig.tight_layout()
    return fig


def risco_por_uf(pred: pd.DataFrame, ano_alvo: int) -> plt.Figure:
    por_uf = pred.groupby("cod_uf").prob_risco.mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 4))
    por_uf.plot.bar(ax=ax, color="#c44e52")
    ax.set_title(f"Risco médio previsto por UF (cód. IBGE, {ano_alvo})")
    ax.set_ylabel("prob. média de risco")
    fig.tight_layout()
    return fig


# ── Orquestração ─────────────────────────────────────────────────────────────

def gerar_todas() -> list[str]:
    """Roda a pipeline e gera todas as figuras da entrega em images/."""
    from src.evaluation import interpretability, metrics
    from src.modeling.pipeline_modelo_d import predict_ano_seguinte, run_pipeline
    from src.preprocessing.features import build_lagged_frame, ref_nacional_por_ano
    from src.preprocessing.gold_consumer import load_gold

    gold = load_gold(config.GOLD_DATASET)
    ref = ref_nacional_por_ano(load_gold(config.GOLD_REF_NACIONAL))
    frame = build_lagged_frame(gold, ref)
    X = frame[config.FEATURE_COLS]
    y = frame[config.TARGET_COL]
    groups = frame[config.GROUP_COL]

    r = run_pipeline(save=False)
    pipe = r["pipeline"]
    comp = metrics.comparar_modelos(X, y, groups)
    sv = interpretability.shap_explanation(pipe, X)
    ano_base = int(gold.ano.max())
    pred = predict_ano_seguinte(pipe, gold, ano_base)

    saved = [
        salvar(volatilidade_taxa(gold), "eda_volatilidade"),
        salvar(taxa_por_regiao(gold, ref), "eda_taxa_por_regiao"),
        salvar(correlacoes_alvo(gold), "eda_correlacoes_alvo"),
        salvar(comparacao_modelos(comp), "comparacao_modelos"),
        salvar(matriz_confusao(r["y_test"], r["y_pred"]), "matriz_confusao"),
        salvar(curvas_roc_pr(r["y_test"], r["y_proba"]), "curvas_roc_pr"),
        salvar(shap_summary(sv), "shap_summary"),
        salvar(risco_por_uf(pred, ano_base + 1), "ranking_risco_uf"),
    ]
    return saved


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    for caminho in gerar_todas():
        print("OK:", caminho)


if __name__ == "__main__":
    main()
