# 🎓 Predição e Inteligência Analítica para Alfabetização no Brasil

> **Tech Challenge – Fase 3 | PosTech FIAP**

## 👥 Equipe
Projeto desenvolvido para o Tech Challenge – Fase 3 da PosTech FIAP

Gabriela de Lima Lopes (RM372467) ➔ LinkedIn
Vitor Lopes Rodrigues (RM372427) ➔ LinkedIn
Lucas Oliveira dos Santos Lima (RM372651) ➔ LinkedIn
Mateus Quintino Vieira dos Santos (RM371795) ➔ LinkedIn

---

## 📌 Contexto do Problema

_(preencher)_ Contextualizar o Compromisso Nacional Criança Alfabetizada, o
ponto de corte de 743 pontos no SAEB e por que antecipar risco educacional
importa para gestores públicos.

## 🎯 Objetivo Analítico

Desenvolver um modelo supervisionado de classificação binária capaz de indicar
se **um município, em determinado ano, tende a atingir (ou não) o patamar de
alfabetização esperado**, apoiando a priorização de políticas públicas.

> **Nota metodológica importante:** os dados públicos do INEP/SAEB utilizados
> neste projeto são agregados por **município/UF/ano** — não existe microdado
> público por aluno. Por isso, a unidade observacional do modelo é o par
> (**município, ano**), e o target original do desafio ("aluno
> alfabetizado/não alfabetizado") foi reformulado como "a taxa de
> alfabetização do município naquele ano atingiu o referencial esperado".
> Essa é uma adaptação necessária e documentada, mantendo o espírito da
> pergunta de negócio (identificar quem está em risco).

## 📊 Descrição da Base Utilizada

| Fonte | Granularidade | Principais colunas |
|---|---|---|
| `indicador_municipio.csv` | Município-ano | `taxa_alfabetizacao`, `media_portugues`, `proporcao_aluno_nivel_0..8` |
| `indicador_uf.csv` | UF-ano | idem, agregado por UF |
| `meta_alfabetizacao_municipio.csv` | Município-ano | `meta_alfabetizacao_2024..2030`, `percentual_participacao` |
| `meta_alfabetizacao_uf.csv` | UF-ano | idem |
| `meta_alfabetizacao_brasil.csv` | Brasil-ano | idem, nacional |

Fontes externas de enriquecimento (opcional): _(preencher se utilizado —
IBGE, Censo Escolar, FUNDEB, Atlas do Desenvolvimento Humano, PNAD)_.

## 🔄 Etapas de Modelagem

1. **Camada Gold** (`notebooks/01`): join das fontes brutas em nível município-ano.
2. **EDA** (`notebooks/02`): distribuições, séries temporais, correlações, hipóteses.
3. **Target e Features** (`notebooks/03`): definição do corte binário e engenharia de atributos, com prevenção explícita de data leakage.
4. **Pipeline de ML** (`notebooks/04`): `ColumnTransformer` (imputação + encoding/scaling) + modelo, dentro de um `sklearn.Pipeline` único; split temporal; validação cruzada.
5. **Interpretabilidade** (`notebooks/05`): Feature Importance + SHAP.
6. **Perguntas de negócio** (`notebooks/06`): tradução dos resultados em insights acionáveis.

## 🤖 Escolha do Algoritmo

_(preencher)_

## 📈 Métricas de Avaliação

_(preencher)_

## 🔍 Interpretação dos Resultados

_(preencher)_

## 💡 Insights Encontrados

_(preencher)_

## ⚠️ Limitações do Projeto

- Dados agregados por município (não por aluno) — target adaptado.
- Possível desbalanceamento de classes.
- Séries temporais curtas (poucos anos disponíveis) limitam a robustez da validação temporal.
- _(adicionar outras limitações identificadas durante o projeto)_

## 🏛️ Aplicação Prática para Políticas Públicas

_(preencher)_ Como o ranking de risco por município pode orientar alocação de
recursos do FUNDEB, priorização de formação de professores, PDDE, etc.

## 🚀 Possíveis Evoluções Futuras

- Incorporar microdados de avaliações municipais quando disponíveis.
- Enriquecer com dados socioeconômicos (IDH, Cadastro Único, PNAD).
- Modelo de série temporal (ex.: prever trajetória multi-ano por município).
- Dashboard interativo para gestores (Streamlit/PowerBI).

---

## 📁 Estrutura do Repositório

```
tech-challenge-fase3/
│
├── data/
│   ├── raw/                 # CSVs originais (herdados da Fase 2)
│   └── gold/                # Dataset analítico consolidado (parquet)
├── notebooks/
│   ├── 01_gold_layer_build.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_target_and_feature_engineering.ipynb --> a construir
│   ├── 04_modeling_pipeline.ipynb --> a construir
│   ├── 05_interpretability_and_shap.ipynb --> a construir
│   └── 06_business_questions_and_insights.ipynb --> a construir
├── src/
│   ├── preprocessing/   # target_engineering.py
│   ├── modeling/        # pipeline.py, train.py
│   ├── evaluation/      # metrics.py, interpretability.py
│   └── visualization/   # plots.py
├── reports/             # modelo treinado, relatórios exportados
├── images/              # gráficos exportados para o README/vídeo
├── requirements.txt
├── README.md
└── .gitignore
```

## ▶️ Como Rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
# Execute os notebooks em ordem: 01 -> 02 -> 03 -> 04 -> 05 -> 06
```

## 🎥 Vídeo Executivo

_(link para o vídeo — até 5 minutos, simulando reunião com gestores públicos)_
