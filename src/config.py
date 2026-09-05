"""
Configuração central da Fase 3.

Fonte única de verdade pra paths, seed, schema esperado da Gold, definição do
target e listas de features. Antes disso, essas constantes viviam duplicadas
(e divergentes) entre notebooks e scripts.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GOLD_DIR = DATA_DIR / "gold"
MODEL_DIR = DATA_DIR / "model"
REPORTS_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "modelo_d_defasagem_temporal.pkl"
MODEL_CARD_PATH = REPORTS_DIR / "model_card_modelo_d.json"

# ── Reprodutibilidade ────────────────────────────────────────────────────────

RANDOM_STATE = 42

# ── Gold (camada oficial da Fase 2, granularidade município-ano) ──────────────

GOLD_DATASET = "indicador_municipio"
GOLD_REF_NACIONAL = "painel_nacional"   # traz taxa_media_nacional por ano

# Colunas mínimas que a modelagem exige. validate_gold_schema() falha com
# mensagem clara se a Fase 2 mudar o schema e remover alguma.
GOLD_SCHEMA_MINIMO = frozenset({
    "id_municipio", "ano",
    "taxa_alfabetizacao", "media_portugues", "percentual_participacao",
    "meta_ano_vigente",
})

# ── Target ───────────────────────────────────────────────────────────────────
# PRINCIPAL — classe positiva (1) = município EM RISCO EDUCACIONAL:
#   em_risco = (taxa_alfabetizacao < taxa_media_nacional[ano])
# "Está abaixo do nível médio do país." Responde "quais municípios priorizar".
# O nível de alfabetização é fortemente autocorrelacionado ano a ano, então
# esse alvo é genuinamente previsível a partir do ano anterior (ROC-AUC ~0.83).
TARGET_COL = "em_risco"

# SECUNDÁRIO — só pra a seção de achado do relatório, NÃO é o modelo entregue:
#   nao_bate_meta = (taxa_alfabetizacao < meta_ano_vigente)
# A meta anual é fixada ~1 p.p. acima da taxa do ano anterior, e a taxa oscila
# ±16 p.p./ano — então "bater a meta anual" é dominado por ruído (ROC-AUC ~0.66).
TARGET_META = "nao_bate_meta"

# ── Features do modelo (defasagem temporal) ──────────────────────────────────
# Tudo aqui é do próprio município no ano ANTERIOR (LAG) — nada do ano do target.
FEATURE_COLS = [
    "taxa_alfabetizacao_ant",
    "media_portugues_ant",
    "percentual_participacao_ant",
]

# Coluna de agrupamento pra validação cruzada por UF (holdout de estados inteiros).
GROUP_COL = "cod_uf"

# ── Modelo final ─────────────────────────────────────────────────────────────
# Escolhido por comparação (ver notebooks/03b_modelo_defasagem_temporal.ipynb):
# com 3 features e sinal quase linear, a regressão logística supera árvores /
# boosting em ROC-AUC, PR-AUC e recall da classe de risco — e é interpretável
# (coeficientes). class_weight='balanced' porque a classe de risco é minoritária.
FINAL_MODEL = "logistic_regression"
LOGREG_PARAMS = dict(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)

CV_FOLDS = 5
