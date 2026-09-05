"""
Preparação de features pra modelagem (Fase 3).

Fluxo de treino:
    Gold (município-ano) + referência nacional por ano
      -> validate_gold_schema      (contrato com a Fase 2)
      -> add_cod_uf                (grupo pra CV por UF)
      -> collapse_municipio_ano    (defensivo: a Gold oficial já vem 1 linha/mun-ano)
      -> build_lagged_frame        (LAG do ano anterior + targets)

Targets no frame:
    em_risco       = taxa_alfabetizacao < taxa_media_nacional[ano]   (principal)
    nao_bate_meta  = taxa_alfabetizacao < meta_ano_vigente           (secundário/achado)

Uso:
    from src.preprocessing.features import build_lagged_frame, ref_nacional_por_ano
    ref = ref_nacional_por_ano(load_gold(config.GOLD_REF_NACIONAL))
    frame = build_lagged_frame(load_gold(config.GOLD_DATASET), ref)
"""

from __future__ import annotations

import duckdb
import pandas as pd

from src import config


def validate_gold_schema(df: pd.DataFrame) -> None:
    """Falha cedo e com mensagem clara se a Gold não tiver o schema esperado."""
    faltando = set(config.GOLD_SCHEMA_MINIMO) - set(df.columns)
    if faltando:
        raise ValueError(
            f"Gold sem colunas exigidas pela modelagem: {sorted(faltando)}.\n"
            f"Colunas recebidas: {sorted(df.columns)}\n"
            f"A Fase 2 pode ter mudado o schema. Se for intencional, ajuste "
            f"src/config.GOLD_SCHEMA_MINIMO e as features."
        )


def ref_nacional_por_ano(df_painel: pd.DataFrame) -> dict[int, float]:
    """{ano: taxa_media_nacional} a partir do dataset painel_nacional da Gold."""
    if not {"ano", "taxa_media_nacional"} <= set(df_painel.columns):
        raise ValueError(
            "painel_nacional sem 'ano'/'taxa_media_nacional' — schema da Fase 2 mudou."
        )
    return {int(a): float(t) for a, t in zip(df_painel["ano"], df_painel["taxa_media_nacional"])}


def add_cod_uf(df: pd.DataFrame) -> pd.DataFrame:
    """Código da UF = 2 primeiros dígitos do código IBGE de 7 dígitos do município."""
    out = df.copy()
    cod = out["id_municipio"].astype("int64").astype(str).str.zfill(7)
    out[config.GROUP_COL] = cod.str[:2].astype(int)
    return out


def collapse_municipio_ano(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante 1 linha por (município, ano).

    A Gold oficial da Fase 2 já vem assim — nesse caso é passthrough. Mantido
    como rede de segurança: se o schema regredir e trouxer duplicidade por rede,
    agrega as numéricas por média (e avisa).
    """
    tamanho_grupos = df.groupby(["id_municipio", "ano"]).size()
    if (tamanho_grupos == 1).all():
        return df.reset_index(drop=True)

    import warnings

    n_dup = int((tamanho_grupos > 1).sum())
    warnings.warn(
        f"{n_dup} pares (município, ano) com mais de 1 linha na Gold — agregando "
        f"as colunas numéricas por média. Verifique o schema da Fase 2.",
        RuntimeWarning,
        stacklevel=2,
    )
    cols_num = [
        c for c in df.columns
        if c not in ("id_municipio", "ano") and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df.groupby(["id_municipio", "ano"], as_index=False)[cols_num].mean()


def build_lagged_frame(df_gold: pd.DataFrame, ref_nacional: dict[int, float]) -> pd.DataFrame:
    """
    Frame de treino: features do ano anterior -> targets do ano seguinte.

    Colunas: id_municipio, ano, cod_uf, <config.FEATURE_COLS>,
             em_risco, nao_bate_meta, taxa_alfabetizacao_atual, ref_nacional.
    """
    validate_gold_schema(df_gold)
    df = collapse_municipio_ano(add_cod_uf(df_gold))
    df = df.copy()
    df["ref_nacional"] = df["ano"].map(ref_nacional)
    if df["ref_nacional"].isna().any():
        anos_sem_ref = sorted(df.loc[df["ref_nacional"].isna(), "ano"].unique())
        raise ValueError(f"Sem taxa_media_nacional pra os anos: {anos_sem_ref}")
    tem_meta = "meta_ano_vigente" in df.columns

    frame = duckdb.sql(f"""
        SELECT
            id_municipio,
            ano,
            {config.GROUP_COL},
            taxa_alfabetizacao                             AS taxa_alfabetizacao_atual,
            ref_nacional,
            (taxa_alfabetizacao < ref_nacional)::INT        AS {config.TARGET_COL},
            {"(taxa_alfabetizacao < meta_ano_vigente)::INT" if tem_meta else "NULL"}
                                                           AS {config.TARGET_META},
            LAG(taxa_alfabetizacao)      OVER w             AS taxa_alfabetizacao_ant,
            LAG(media_portugues)         OVER w             AS media_portugues_ant,
            LAG(percentual_participacao) OVER w             AS percentual_participacao_ant
        FROM df
        WINDOW w AS (PARTITION BY id_municipio ORDER BY ano)
        QUALIFY taxa_alfabetizacao_ant IS NOT NULL
        ORDER BY id_municipio, ano
    """).df()

    frame = frame.dropna(subset=[config.TARGET_COL]).reset_index(drop=True)
    frame[config.TARGET_COL] = frame[config.TARGET_COL].astype(int)
    return frame


def build_inference_frame(df_gold: pd.DataFrame, ano_base: int) -> pd.DataFrame:
    """
    Frame pra prever o risco do ano seguinte a `ano_base`, sem conhecer a resposta.

    Usa os dados de `ano_base` como "ano anterior". Retorna: id_municipio,
    cod_uf, <config.FEATURE_COLS>.
    """
    validate_gold_schema(df_gold)
    df = collapse_municipio_ano(add_cod_uf(df_gold))

    base = df[df["ano"] == ano_base]
    if base.empty:
        raise ValueError(f"Gold não tem dado de {ano_base}.")

    return pd.DataFrame({
        "id_municipio": base["id_municipio"].values,
        config.GROUP_COL: base[config.GROUP_COL].values,
        "taxa_alfabetizacao_ant": base["taxa_alfabetizacao"].values,
        "media_portugues_ant": base["media_portugues"].values,
        "percentual_participacao_ant": base["percentual_participacao"].values,
    })
