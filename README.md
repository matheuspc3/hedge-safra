# Derivativos Climáticos para Hedge de Produtividade da Soja

Modelagem, precificação e simulação de estratégias de hedge com **opções CDD**
(*Cooling Degree Days*) sobre a produtividade da soja em 3 regiões do Brasil.

**Tese:** CDD options são hedge viável para soja brasileira, com efetividade de
**83–86%** em 2 das 3 principais regiões e prêmio médio de **5–9%** do valor
segurado — comparável a derivativos de commodities e superior ao seguro agrícola
tradicional (10–15% de prêmio, 30%+ de inadimplência no Proagro).

---

## Resultados Principais

| Região | Efetividade | Prêmio | K ótimo | R² prod. | Volatilidade OU |
|--------|-------------|--------|---------|-----------|-----------------|
| **Sorriso/MT** ★ | **83,1%** | 7,3% | 10 °C·dia | 0,81 | 1,02 |
| **Londrina/PR** ★ | **86,1%** | 8,7% | 25 °C·dia | 0,70 | 1,64 |
| **Rio Verde/GO** ⚠ | 65,1% | 4,8% | 25 °C·dia | 0,75 | 1,21 |

★ Recomendado · ⚠ Cautela (hedge complementar necessário)

**Benchmark:** prêmio de 7,3–8,7% vs. seguro agrícola que custa 10–15% do valor
segurado e tem inadimplência de 30%+ no Proagro. CDD options têm liquidação
objetiva (temperatura, não perícia), eliminando risco moral.

---

## Pipeline

```
download_dados.py   → NASA POWER (1980–2025, 3 regiões)
       ↓
modelagem_ou.py     → Ornstein-Uhlenbeck MLE (R² ~65–68%)
       ↓
clima_produtividade.py → CDD → produtividade (R² 0,70–0,81)
       ↓
precificacao.py      → Monte Carlo 50k trajetórias → prêmio justo
       ↓
hedge_simulacao.py   → 50k cenários → efetividade + VaR
       ↓
sensibilidade.py     → grid search (K × nocional) → otimização
```

Cada script em `scripts/` corresponde a um notebook em `notebooks/`.

---

## Estrutura

```
├── scripts/                        # Pipeline executável (Python puro)
│   ├── download_dados.py           # Coleta NASA POWER → parquet
│   ├── modelagem_ou.py             # OU + MLE → parâmetros + resíduos
│   ├── clima_produtividade.py      # Função clima→produtividade
│   ├── precificacao.py             # Monte Carlo → prêmio justo
│   ├── hedge_simulacao.py          # Hedge + métricas de risco
│   ├── sensibilidade.py            # Grid search K × nocional
│   ├── calibrar_produtividade.py   # Calibração γ, K, perda máxima
│   └── regraficos_profissionais.py # Regenera gráficos com estilo Itaú
│
├── notebooks/
│   └── 01_exploracao_dados.ipynb   # EDA inicial
│
├── src/
│   ├── __init__.py
│   └── plot_utils.py               # Estilo profissional (Itaú-inspired)
│
├── data/
│   ├── raw/                        # Cache NASA POWER (parquet)
│   └── processed/                  # Parâmetros calibrados + simulações
│       ├── temperatura_diaria_MT_PR_GO.parquet
│       ├── parametros_ou.json
│       ├── parametros_produtividade.json
│       ├── precificacao_cdd.json
│       ├── hedge_simulacao.json
│       └── sensibilidade_hedge.json
│
├── output/graficos/                # 30+ figuras (300 dpi, formato publicação)
│   ├── 05_ou_*.png                 # Termograma OU (3 regiões)
│   ├── 06_residuos_ou_*.png        # Resíduos + Q-Q plot
│   ├── 07_cdd_historico_v2.png     # Série histórica CDD (DEZ-FEV)
│   ├── 08_produtividade_cdd.png    # Produtividade vs CDD
│   ├── 09_precificacao_*.png       # Distribuição de payoff
│   ├── 10_caminhos_temp_*.png      # Caminhos Monte Carlo
│   ├── 11_hedge_distrib_v2.png     # Receita antes/depois do hedge
│   ├── 12_sensibilidade_hedge_v2.png # Heatmap efetividade × K × nocional
│   ├── 13_curvas_efetividade_*.png # Curvas de efetividade
│   ├── 14_calibracao_produtividade_v2.png # Scatter + curva exponencial
│   └── 15_comparacao_params.png    # Parâmetros por região
│
├── relatorio/                      # Relatório LaTeX (formato publicação)
│   ├── main.tex                    # 10 seções, Palatino/Latin Modern
│   ├── main.pdf                    # PDF compilado
│   └── Makefile                    # make → pdflatex
│
├── pyproject.toml
└── README.md
```

---

## Instalação

```bash
pip install -e .
```

Ou com uv:

```bash
uv sync
```

**Dependências:** Python ≥ 3.11, numpy, pandas, scipy, matplotlib, seaborn,
requests, jupyter, pyarrow, tqdm.

---

## Uso

### Pipeline completo (script a script)

```bash
# 1. Download dos dados climáticos (NASA POWER)
python scripts/download_dados.py

# 2. Modelagem Ornstein-Uhlenbeck
python scripts/modelagem_ou.py

# 3. Função clima → produtividade
python scripts/clima_produtividade.py

# 4. Precificação Monte Carlo
python scripts/precificacao.py

# 5. Simulação de hedge
python scripts/hedge_simulacao.py

# 6. Análise de sensibilidade
python scripts/sensibilidade.py
```

### Notebooks

```bash
cd notebooks
jupyter notebook 01_exploracao_dados.ipynb
```

Ou abrir diretamente no VS Code.

### Regenerar gráficos com estilo profissional

```bash
python scripts/regraficos_profissionais.py
```

Gera 4 gráficos _v2 em `output/graficos/` com paleta Itaú, anotações
estilizadas e 300 dpi.

### Compilar relatório

```bash
cd relatorio && make
```

Requer texlive-core + lmodern (pdflatex).

---

## Metodologia

### Modelo de Temperatura: Ornstein-Uhlenbeck

A temperatura diária é modelada como um processo de reversão à média:

```
dT(t) = θ(μ - T(t))dt + σdW(t)
```

Calibrado via **MLE** com 16.071 observações diárias (1980–2025) por região:

| Região | θ (rev.) | μ (drift) | σ (vol.) | Meia-vida | R² |
|--------|----------|-----------|----------|-----------|----|
| Sorriso/MT | 0,192 | 0,0002 | 1,02 | 3,6 dias | 0,68 |
| Londrina/PR | 0,214 | 0,0005 | 1,64 | 3,2 dias | 0,65 |
| Rio Verde/GO | 0,197 | 0,0001 | 1,21 | 3,5 dias | 0,67 |

A meia-vida de **3,2–3,6 dias** significa que choques térmicos se dissipam
rapidamente — o prêmio de risco temporal é curto.

### CDD e Perda de Produtividade

**CDD** (Cooling Degree Days) acumulado em DEZ-FEV:

```
CDD = Σ max(0, T_média - 25 °C)
```

Perda de produtividade modelada como função piecewise:

```
Perda = min(max(0, γ · (CDD - K)), perda_máxima)
Y = Y_max × (1 - Perda)
```

| Região | γ | K (°C·dia) | perda_máx | R² |
|--------|---|------------|-----------|----|
| Sorriso/MT | 0,0036 | 30,9 | 40,5% | 0,81 |
| Londrina/PR | 0,0045 | 33,6 | 36,1% | 0,70 |
| Rio Verde/GO | 0,0052 | 27,7 | 44,7% | 0,75 |

### Precificação: Monte Carlo

Opção de venda digital com barreira, precificada via Monte Carlo com 50.000
trajetórias e discretização exata do processo OU:

- **Strike:** K = 30,9 °C·dia
- **Nocional:** R$ 30.000 / ponto de CDD
- **Prob. de exercício:** 65–71% (opção bem no dinheiro)

| Região | Prêmio justo | CDD médio sim. | P95 CDD |
|--------|-------------|----------------|---------|
| Sorriso/MT | R$ 418.881 | 41,9 | 82,6 |
| Londrina/PR | R$ 604.451 | 49,0 | 100,6 |
| Rio Verde/GO | R$ 481.212 | 44,3 | 89,1 |

### Simulação de Hedge

Produtor com 1.000 ha, 60 sacas/ha, preço R$ 140/saca. 50.000 cenários de
temperatura → produtividade → receita.

Resultados **otimizados** (grid search K × nocional):

| Região | K ótimo | Nocional | Efetividade | Prêmio | VaR95 (s/ hedge) | VaR95 (c/ hedge) |
|--------|---------|----------|-------------|--------|------------------|------------------|
| Sorriso/MT | 10 | R$ 20k | **83,1%** | 7,3% | R$ 8,0M | R$ 7,5M |
| Londrina/PR | 25 | R$ 30k | **86,1%** | 8,7% | R$ 8,0M | R$ 7,5M |
| Rio Verde/GO | 25 | R$ 20k | 65,1% | 4,8% | R$ 8,0M | R$ 7,8M |

### Tese de Investimento

1. **Sorriso/MT — RECOMENDADO:** R² mais alto (0,81), γ mais baixo (0,0036),
   meia-vida 3,6 dias. Estruturável como produto OTC.

2. **Londrina/PR — RECOMENDADO:** Maior efetividade (86,1%). Prêmio mais alto
   (8,7%) justificado pela maior volatilidade (σ = 1,64). Ideal para produtores
   que já usam hedge de preço.

3. **Rio Verde/GO — CAUTELA:** Efetividade marginal (65,1%). γ alto (0,0052)
   indica perda abrupta por CDD. Necessário hedge complementar.

---

## Dados

- **Fonte:** [NASA POWER](https://power.larc.nasa.gov/) — grade 0,5°×0,5°,
  dados diários de temperatura máxima, mínima e precipitação
- **Período:** 1980–2025 (46 safras)
- **Período crítico:** Dezembro–Fevereiro (DEZ-FEV), enchimento de grãos
- **Regiões:**
  | Região | Latitude | Longitude | Estado |
  |--------|----------|-----------|--------|
  | Sorriso | 12,55°S | 55,71°O | MT |
  | Londrina | 23,31°S | 51,16°O | PR |
  | Rio Verde | 17,80°S | 50,93°O | GO |
- **Cache local:** `data/raw/nasa_power_raw.parquet` para evitar limites de taxa

---

## Gráficos

Os gráficos seguem estilo inspirado em **Bloomberg / Itaú BBA**, gerados com
`src/plot_utils.py`:

- Paleta: azul institucional (#005CA9), verde retorno (#00995D),
  vermelho (#CC3333), fundo bege (#F7F3EB)
- 300 dpi, formato PNG
- Grid limpo (somente eixo Y), fonte sempre creditada
- Anotações estilizadas com setas e fundo branco

Os arquivos `*_v2.png` são a versão mais recente com layout profissional.

---

## Licença

Projeto acadêmico. Dados NASA POWER de uso livre mediante atribuição.
