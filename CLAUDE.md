# CLAUDE.md — Derivativos Climáticos / Soja

## Objetivo

Modelar, precificar e simular estratégias de hedge com derivativos climáticos (CDD options) sobre a produtividade da soja nas principais regiões produtoras do Brasil.

## Stack

- **Python** ≥ 3.11 — numpy, pandas, scipy, matplotlib, seaborn, requests, jupyter
- **Dados:** NASA POWER API (temperatura diária, precipitação, 1980–2025)
- **Regiões:** Sorriso/MT, Londrina/PR, Rio Verde/GO
- **Ambiente:** pip (`pip install -e .`) ou uv (`uv sync`)

## Estrutura

```
├── notebooks/
│   ├── 01_exploracao_dados.ipynb    # Coleta NASA POWER + EDA
│   ├── 02_modelagem_ou.ipynb        # (planejado) OU + MLE
│   ├── 03_clima_produtividade.ipynb # (planejado) função clima → prod.
│   ├── 04_precificacao.ipynb        # (planejado) Monte Carlo
│   ├── 05_hedge_simulacao.ipynb     # (planejado) hedge + risco
│   └── 06_sensibilidade.ipynb       # (planejado) sensibilidade
├── data/
│   ├── raw/          # Cache da NASA POWER
│   └── processed/    # Dados tratados + parâmetros calibrados
├── output/graficos/  # Figuras (300dpi)
├── pyproject.toml    # Dependências + metadados do projeto
└── README.md
```

## Comandos

```bash
# Instalar
pip install -e .
# ou
uv sync

# Executar
cd notebooks && jupyter notebook 01_exploracao_dados.ipynb
# ou abrir no VS Code diretamente
```

## Padrões

- Notebooks numerados (01–06) — pipeline sequencial: dados → modelo OU → clima→prod → precificação → hedge → sensibilidade.
- Dados brutos em `data/raw/`, processados em `data/processed/`. Ambos no `.gitignore`.
- Figuras em `output/graficos/`, 300dpi, formato publicação.
- Código em português (nomes, comentários, markdown).
- Dados climáticos diários, grade NASA POWER 0.5°×0.5°.

## Restrições

- API da NASA POWER pode ter limite de taxa — caching local em `data/raw/` via parquet.
- Período de dados: 1980–2025 (~46 safras).
- Período crítico da soja: dezembro–fevereiro (DEZ-FEV).
- Apenas o notebook 01 existe — 02–06 são planejados.
