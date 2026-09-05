# Relatório Técnico — Predição de Risco Educacional em Alfabetização

**Tech Challenge – Fase 3 | PosTech FIAP**

Este documento consolida as decisões analíticas, a metodologia e os resultados
do projeto. Para narrativa e instruções de execução, ver o `README.md`; para os
resultados reproduzíveis célula a célula, os notebooks em `notebooks/`.

---

## 1. Problema e objetivo

O **Compromisso Nacional Criança Alfabetizada (CNCA)** estabelece metas anuais de
alfabetização por município, medidas pelo SAEB/INEP, com o alvo de 80% das
crianças alfabetizadas ao fim do 2º ano até 2030. Antecipar quais municípios
estão em situação de risco educacional permite ao gestor público priorizar
recursos (FUNDEB, PDDE, formação de professores) **antes** do próximo ciclo de
avaliação.

**Objetivo analítico:** um modelo supervisionado de classificação binária que,
a partir de informação do ano anterior, indica se um município tende a estar
**em risco educacional** no ano seguinte — apoiando a priorização de políticas
públicas.

## 2. Base de dados e unidade de análise

- **Fonte:** camada **Gold** da pipeline da Fase 2 (dados INEP/SAEB tratados),
  em `s3://tech-challenge-fase2-fiap-vitor/layers/gold`, consumida por
  `src/preprocessing/gold_consumer.py` com cache local em `data/gold/`.
- **Datasets usados:** `indicador_municipio` (município-ano) e `painel_nacional`
  (referência nacional por ano).
- **Unidade observacional:** o par **(município, ano)**.
- **Granularidade temporal:** anual. **Anos disponíveis: 2023 e 2024.**

> **Nota metodológica.** O enunciado fala em prever alfabetização "de alunos",
> mas os dados públicos do INEP/SAEB são **agregados por município/UF/ano** — não
> há microdado por aluno. A unidade do modelo é, portanto, (município, ano), e o
> alvo foi reformulado (ver §4), mantendo o espírito da pergunta: identificar
> quem está em risco.

## 3. Análise exploratória — achados que orientaram a modelagem

Detalhe em `notebooks/02_eda.ipynb`.

| # | Achado | Consequência para a modelagem |
|---|---|---|
| A1 | Distribuição da taxa muito heterogênea entre municípios (≈5%–100%, concentrada em 50–75%) | Teto de performance moderado |
| A2 | **Volatilidade anual altíssima:** a taxa municipal varia com desvio de **±16,4 p.p.** de 2023 para 2024 (ganho médio nacional foi de só +2,6 p.p.) | Define o alvo: prever "nível estrutural" é viável; prever "cruzar a meta anual" não |
| A3 | `taxa_meta_base` (r ≈ 0,98) e `media_portugues` (r ≈ 0,93) são quase cópias da `taxa_alfabetizacao` do mesmo ano | **Data leakage** — excluídas das features |
| A4 | `diferenca_meta`, `nivel_alfabetizacao`, `classificacao`, `bateu_meta_2030` são cálculos diretos da taxa | Excluídas das features |
| A5 | `percentual_participacao` tem correlação fraca (r ≈ 0,28) | Feature legítima; usada defasada |
| A6 | Desigualdade regional forte: Norte 50,4 / Nordeste 56,5 / Sul 64,1 / Sudeste 70,7 / Centro-Oeste 72,0 (2024) | Análise por UF/região; validação cruzada por UF |
| A7 | A meta anual é fixada em média só **+1,4 p.p.** acima da taxa do ano anterior | Combinada com A2 → atingir a meta num ano é quase sorte |

## 4. Definição do target

### 4.1 Target principal (modelo entregue)

```
em_risco = (taxa_alfabetizacao < taxa_media_nacional[ano])
```

Classe positiva (1) = município **abaixo do nível médio do país** naquele ano
("em risco educacional"). A referência nacional vem de `painel_nacional`
(55,3 em 2023; 57,4 em 2024).

- **Prevalência:** 37,8% (quase balanceado).
- **Por que este alvo:** o nível de alfabetização é fortemente autocorrelacionado
  ano a ano, o que o torna **genuinamente previsível** a partir do ano anterior.
- **É data leakage usar a média nacional do ano do alvo?** Não de forma
  relevante: é um agregado macro (~57), não derivado de um município específico
  (1 de ~5.500), e todas as features são do ano anterior. Ainda assim, um
  endurecimento possível é usar a média nacional de *t‑1* (resultado idêntico nos
  testes) — registrado como melhoria futura.

### 4.2 Target secundário — investigado e reportado como achado

```
nao_bate_meta = (taxa_alfabetizacao < meta_ano_vigente)
```

É o alvo literal do enunciado ("municípios que podem não atingir a meta"). Foi
testado com as mesmas features defasadas e a mesma validação:

| Alvo | ROC-AUC (CV/UF) |
|---|---|
| `em_risco` (abaixo da média nacional) | **≈ 0,82** |
| `nao_bate_meta` (abaixo da meta anual) | **≈ 0,60** |

**Conclusão:** por causa de A2 + A7, o atingimento da meta num ano específico é
dominado por ruído e **não é previsível** com os dados disponíveis. Este é um
resultado honesto do projeto (`03b`, seção 7), não uma falha do modelo — a
pergunta de negócio é respondida pelo proxy `em_risco`, que identifica quem está
estruturalmente atrás.

## 5. Features e engenharia de atributos

Módulo: `src/preprocessing/features.py` (`build_lagged_frame`).

- **Colapso para 1 linha por (município, ano)** — a Gold oficial já vem assim;
  a função mantém uma agregação por média como rede de segurança caso o schema
  regrida para a duplicidade Estadual/Privada da base bruta.
- **Defasagem temporal (LAG, via DuckDB):** para cada município, os valores de
  2023 são trazidos para a linha de 2024.
- **Features finais (todas do ano anterior):**

| Feature | Origem |
|---|---|
| `taxa_alfabetizacao_ant` | taxa de alfabetização do município no ano anterior |
| `media_portugues_ant` | nota média de português no ano anterior |
| `percentual_participacao_ant` | participação na avaliação no ano anterior |

Não há variáveis categóricas no modelo final (a agregação removeu rede/série).
`cod_uf` (2 primeiros dígitos do código IBGE) é derivada apenas para agrupar a
validação cruzada.

## 6. Prevenção de data leakage

Diagnóstico completo em `notebooks/03a_diagnostico_vazamento_dados.ipynb`.

| Coluna | Risco | Motivo | Ação |
|---|---|---|---|
| `media_portugues` (mesmo ano) | Crítico | r ≈ 0,93 com o alvo — mesma prova | Excluída |
| `taxa_meta_base` (mesmo ano) | Crítico | r ≈ 0,98 — é a própria taxa | Excluída |
| `nivel_alfabetizacao`, `diferenca_meta`, `classificacao`, `bateu_meta_2030` | Crítico | cálculos diretos da taxa | Excluídas |
| `meta_2024..2030` | Alto | derivadas da taxa-base do município | Não usadas como feature |
| `*_ant` (LAG) | Baixo | anteriores ao resultado previsto | **Usadas** |
| Imputação / scaling | — | dentro do `Pipeline`, `fit` só no treino | OK |

**Prova quantitativa:** um Random Forest com features do **mesmo ano** atinge
ROC-AUC ≈ 0,996 (aparência de modelo perfeito); com as features **defasadas**,
ROC-AUC ≈ 0,82. A diferença é o tamanho do vazamento. O "Modelo A" (mesmo ano)
foi mantido em `notebooks/03_modelo_A_contraexemplo.ipynb` **apenas como
contraexemplo, explicitamente rotulado como inválido**.

## 7. Pipeline de pré-processamento e modelo

`src/modeling/pipeline_modelo_d.build_pipeline`:

```
ColumnTransformer([
    ("num", Pipeline([SimpleImputer(strategy="median"), StandardScaler()]), FEATURE_COLS)
])
→ LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
```

Todo o pré-processamento está **dentro** do `sklearn.Pipeline` e é ajustado
**somente no conjunto de treino**.

## 8. Validação

- **Holdout estratificado (20%)** — como cada município aparece uma única vez no
  frame (só há linhas de 2024), um split aleatório já é um holdout limpo de
  municípios não vistos, sem *group leakage*.
- **Validação cruzada `GroupKFold` por UF (5 folds)** — cada fold testa em
  **estados inteiros** que não estavam no treino (generalização espacial).
- **Limitação:** com apenas 2 anos, há **uma única transição temporal**. O
  holdout é cross-municípios do mesmo ano — **não é validação temporal**. A
  generalização para anos futuros fica **não validada**; a previsão para 2025 é
  extrapolação.

## 9. Comparação de modelos

Mesma pipeline, mesma seed, CV `GroupKFold` por UF (`notebooks/03b`, seção 4):

| Modelo | ROC-AUC | PR-AUC | Recall (risco) | Precisão (risco) | Acurácia |
|---|---|---|---|---|---|
| Baseline (classe majoritária) | 0,500 | 0,378 | 0,00 | 0,00 | 0,62 |
| **Regressão Logística** | **0,836** | **0,755** | **0,755** | 0,63 | 0,73 |
| Árvore de Decisão | 0,806 | 0,707 | 0,723 | 0,66 | 0,75 |
| Random Forest | 0,822 | 0,749 | 0,713 | 0,66 | 0,75 |
| Gradient Boosting | 0,817 | 0,747 | 0,610 | 0,72 | 0,76 |

**Escolha: Regressão Logística.** Com apenas 3 features e um sinal quase linear
(desempenho passado → risco futuro), a LogReg entrega o melhor ROC-AUC, PR-AUC e
o **melhor recall da classe de risco** — a métrica prioritária para política
pública (não deixar município em risco passar despercebido) — e é interpretável.

## 10. Métricas e resultados

`reports/model_card_modelo_d.json` registra os números de cada execução.

| Métrica | Holdout (municípios) | CV `GroupKFold`/UF |
|---|---|---|
| ROC-AUC | 0,82 | 0,84 ± 0,05 |
| PR-AUC | 0,75 | 0,76 ± 0,07 |
| Recall (classe de risco) | 0,76 | 0,76 ± 0,15 |
| Precisão (classe de risco) | 0,62 | 0,66 |
| Acurácia | 0,73 (baseline 0,62) | 0,75 |

Matriz de confusão (holdout, n = 1.096): VN 491 · FP 190 · FN 101 · VP 314.

**Métrica priorizada:** recall da classe de risco + PR-AUC. Accuracy é reportada
sempre ao lado do baseline (0,62) para não induzir a erro num alvo desbalanceado.

## 11. Interpretabilidade

`notebooks/03b`, seção 6 — coeficientes + odds ratio + *permutation importance* +
SHAP, todos sobre o **modelo honesto**.

- **`media_portugues_ant`** é o fator dominante — a nota de português do ano
  anterior praticamente já indica se o município está abaixo da média nacional.
- `percentual_participacao_ant` e `taxa_alfabetizacao_ant` contribuem na margem.
- Todos no sentido esperado: quanto pior o ano anterior, maior o risco.

**Limitação importante:** todos os "fatores" são medidas do **próprio desempenho
passado** do município. A Gold não traz variáveis de contexto (renda,
investimento, formação docente, infraestrutura) que apontem **alavancas de
política pública**. Enriquecer com essas fontes é a principal evolução futura.

## 12. Respostas às perguntas de negócio

| # | Pergunta | Resposta |
|---|---|---|
| 1 | Quais fatores mais impactam a alfabetização? | **Parcial.** O desempenho passado (nota de português, taxa, participação) explica o risco; não há na base fatores acionáveis de política pública (§11). |
| 2 | Quais municípios apresentam maior risco educacional? | **Respondido.** `reports/previsao_risco_2025.csv` — probabilidade de risco por município, ordenável. |
| 3 | Quais regiões têm padrões semelhantes? | **Parcial.** EDA e `03b` mostram o risco concentrado em Norte e Nordeste; falta uma clusterização formal por perfil (evolução futura). |
| 4 | Como prever municípios que podem não atingir metas futuras? | **Parcial + achado.** O alvo literal ("bater a meta anual") é ~ruído (§4.2); o proxy `em_risco` entrega o ranking de risco. |
| 5 | Quais variáveis têm maior influência? | **Respondido** (§11). |

## 13. Aplicação prática

- **Priorização orçamentária:** o ranking de `prob_risco` pode orientar a
  alocação de recursos suplementares (FUNDEB, PDDE) para os municípios de maior
  risco previsto.
- **Foco regional:** Norte e Nordeste concentram o risco — programas de formação
  de professores e busca ativa podem ser direcionados por UF.
- **Uso responsável:** o modelo sinaliza **prioridade de atenção**, não
  "fracasso garantido"; a decisão final é do gestor, com o contexto local.

## 14. Limitações

1. **Apenas 2 anos de dados** (2023–2024) — uma única transição temporal; sem
   validação de generalização para anos futuros.
2. **Alta volatilidade** da taxa municipal (±16 p.p./ano), sobretudo em
   municípios pequenos — limita o teto de performance.
3. **Atingimento de meta anual não é previsível** com os dados atuais (§4.2).
4. **Sem variáveis socioeconômicas / de insumo** — a interpretabilidade não
   aponta alavancas acionáveis.
5. O limiar do target usa a média nacional contemporânea ao ano do alvo (correção
   trivial disponível: usar *t‑1*).

## 15. Evoluções futuras

- Incorporar novos anos da Gold quando disponíveis → validação temporal real
  (treina ≤ *t‑1*, testa *t*) e `TimeSeriesSplit`.
- Enriquecer com IDH, Cadastro Único, PNAD, Censo Escolar, FUNDEB → fatores
  acionáveis.
- Clusterização de municípios por perfil (KMeans/HDBSCAN) para a pergunta 3.
- `GridSearchCV` para ajuste fino da regressão logística.
- Dashboard para gestores (Streamlit / Power BI) sobre o ranking de risco.

## 16. Reprodutibilidade

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python -m src.preprocessing.gold_consumer --refresh # baixa a Gold (requer credenciais AWS 1x)
python -m src.modeling.pipeline_modelo_d            # treina, avalia, salva modelo + model card
python -m src.modeling.prever_proximo_ano           # gera o ranking de risco
# notebooks: 01 -> 02 -> 03a -> 03b  (03 é apêndice/contraexemplo)
```

`src/config.py` centraliza paths, seed (`RANDOM_STATE = 42`), schema esperado da
Gold, definição do target e lista de features.
`src/preprocessing/features.validate_gold_schema` falha com mensagem clara se a
Fase 2 mudar o schema. `reports/model_card_modelo_d.json` registra a proveniência
da Gold usada em cada treino (dataset, nº de linhas, anos, colunas, referência
nacional, data).
