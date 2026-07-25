"""
Gold Consumer — Tarefa 1 (Fase 3)
---------------------------------
Consome os datasets da camada Gold gerados pela pipeline da Fase 2
(armazenados no S3) e salva uma cópia local em `data/gold/`, para que
o time não precise rodar a pipeline (nem ter credenciais AWS) toda vez.

Ordem de resolução ao carregar um dataset:
    1. Cache local (data/gold/<nomeDoRoleDagold>.parquet)         →  offline
    2. S3 da pipeline da Fase 2 (se credenciais AWS)  → baixa e salva no cache
    3. Fallback: gold local reconstruída pelo notebook 01 (se existir em algum lugar do repositório) → offline

Uso como script (baixa tudo do S3 e salva localmente):
    python -m src.preprocessing.gold_consumer            # usa cache se existir
    python -m src.preprocessing.gold_consumer --refresh  # força re-download do S3

Uso como módulo (em notebooks / outros scripts):
    from src.preprocessing.gold_consumer import load_gold
    df = load_gold("indicador_municipio")
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv não instalado — segue sem .env (usa aws configure)
    def load_dotenv(*args, **kwargs):
        return None

# ── Configuração pra fazer a aws e local

ROOT = Path(__file__).resolve().parents[2]          # raiz do repositório
load_dotenv(ROOT / ".env")                          # carrega credenciais AWS, se existirem
LOCAL_GOLD_DIR = ROOT / "data" / "gold"             # cache local (gitignored)

# Bucket da pipeline da Fase 2 (gold_builder.py grava aqui)
S3_GOLD_DIR = "s3://tech-challenge-fase2-fiap-vitor/layers/gold"

# Datasets gerados pelo gold_builder.py da Fase 2.
# `partitioned=True` → gravado como diretório particionado por `ano`
# (o pandas lê diretórios particionados normalmente, basta apontar pro path).
GOLD_DATASETS = {
    "indicador_municipio":  {"partitioned": True},
    "ranking_uf":           {"partitioned": False},
    "meta_vs_realizado_uf": {"partitioned": True},
    "evolucao_uf":          {"partitioned": False},
    "painel_nacional":      {"partitioned": False},
}

# Fallback: gold reconstruída localmente pelo notebook 01_gold_layer_build_
NOTEBOOK_GOLD_FALLBACK = {
    "indicador_municipio": LOCAL_GOLD_DIR / "gold_indicador_municipio.parquet",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Funções internas ─────────────────────────────────────────────────────────

def _local_path(name: str) -> Path:
    return LOCAL_GOLD_DIR / f"{name}.parquet"


def _s3_path(name: str) -> str:
    partitioned = GOLD_DATASETS[name]["partitioned"]
    return f"{S3_GOLD_DIR}/{name}" if partitioned else f"{S3_GOLD_DIR}/{name}.parquet"


def _read_from_s3(name: str) -> pd.DataFrame:
    """Lê um dataset Gold direto do S3 (requer credenciais AWS + s3fs)."""
    path = _s3_path(name)
    log.info(f"Lendo do S3: {path}")
    df = pd.read_parquet(path)

    # Em datasets particionados por `ano`, a coluna de partição volta como
    # category/string — normalizamos para inteiro pra facilitar as análises.
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")

    return df


def _save_local(df: pd.DataFrame, name: str) -> Path:
    """Salva o dataset no cache local em um único parquet."""
    LOCAL_GOLD_DIR.mkdir(parents=True, exist_ok=True)
    path = _local_path(name)
    df.to_parquet(path, index=False)
    log.info(f"  ✔ Cache local salvo: {path.relative_to(ROOT)} — {len(df)} registros")
    return path


# ── API pública ──────────────────────────────────────────────────────────────

def load_gold(name: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Carrega um dataset da camada Gold.

    paramentros
    ----------
    name : str
        Nome do dataset (ver GOLD_DATASETS).
    force_refresh : bool
        Se True, ignora o cache local e baixa novamente do S3.
    """
    if name not in GOLD_DATASETS:
        raise ValueError(
            f"Dataset desconhecido: '{name}'. "
            f"Disponíveis: {sorted(GOLD_DATASETS)}"
        )

    local = _local_path(name)

    # 1) Cache local
    if local.exists() and not force_refresh:
        log.info(f"Usando cache local: {local.relative_to(ROOT)}")
        return pd.read_parquet(local)

    # 2) S3 (pipeline da Fase 2)
    try:
        df = _read_from_s3(name)
        _save_local(df, name)
        return df
    except ImportError as e:
        log.warning(f"Dependência ausente para ler do S3 ({e}). Instale: pip install s3fs")
    except Exception as e:
        log.warning(f"Não foi possível ler '{name}' do S3: {type(e).__name__}: {e}")

    # 3) Fallback: gold reconstruída pelo notebook 01
    fallback = NOTEBOOK_GOLD_FALLBACK.get(name)
    if fallback and fallback.exists():
        log.info(f"Usando fallback do notebook 01: {fallback.relative_to(ROOT)}")
        return pd.read_parquet(fallback)

    raise FileNotFoundError(
        f"Dataset '{name}' indisponível: sem cache local, sem acesso ao S3 "
        f"e sem fallback local.\n"
        f"Opções:\n"
        f"  a) configure as credenciais AWS (.env / aws configure) e rode:\n"
        f"     python -m src.preprocessing.gold_consumer\n"
        f"  b) peça a alguém do time o arquivo data/gold/{name}.parquet\n"
        f"  c) rode o notebook notebooks/01_gold_layer_build_.ipynb para "
        f"reconstruir a gold a partir de data/raw/"
    )


def download_all(force_refresh: bool = False) -> dict:
    """Baixa/atualiza todos os datasets Gold para o cache local."""
    resultados = {}
    for name in GOLD_DATASETS:
        try:
            df = load_gold(name, force_refresh=force_refresh)
            resultados[name] = {"status": "ok", "registros": len(df)}
        except Exception as e:
            resultados[name] = {"status": f"erro: {e.__class__.__name__}"}
            log.error(f"Falha em '{name}': {e}")
    return resultados


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Consome a camada Gold da pipeline (Fase 2) e salva cache local."
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Força re-download do S3, ignorando o cache local.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Consumo da camada Gold — pipeline Fase 2 → cache local")
    log.info("=" * 60)

    resultados = download_all(force_refresh=args.refresh)

    log.info("")
    log.info("RESUMO")
    log.info("-" * 60)
    for name, r in resultados.items():
        if r["status"] == "ok":
            log.info(f"  ✔ {name:<25} {r['registros']:>6} registros")
        else:
            log.warning(f"  ✘ {name:<25} {r['status']}")
    log.info("-" * 60)


if __name__ == "__main__":
    main()
