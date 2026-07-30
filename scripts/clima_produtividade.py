"""Modela a relação entre temperatura (CDD) e produtividade da soja.

CDD (Cooling Degree Days) acumulado no período crítico DEZ-FEV:
    CDD = Σ max(0, T_média - T_base)

Para soja brasileira, T_base ≈ 28°C (temperatura acima disso durante
floração/enchimento de grãos reduz produtividade).

A função produtividade:  Y(CDD) = Y_max × (1 - f(CDD))
onde f(CDD) é a fração de perda, modelada como:
    - linear:      f(CDD) = max(0, min(1, γ · CDD))
    - piecewise:   f(CDD) = 0                        se CDD ≤ K
                           = max(0, min(1, γ·(CDD-K))) se CDD > K
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_IN = Path("data/processed/temperatura_diaria_MT_PR_GO.parquet")
CDD_OUT = Path("data/processed/cdd_safras.parquet")
PARAMS_OUT = Path("data/processed/parametros_produtividade.json")
FIGS_OUT = Path("output/graficos")
FIGS_OUT.mkdir(parents=True, exist_ok=True)

DPI = 300
T_BASE = 25.0          # °C — limiar estresse térmico soja
PERDA_MAX = 0.50       # 50% de perda máxima (fallback se sem JSON)
GAMMA = 0.003           # perda linear por CDD (fallback se sem JSON)
STRIKE = 30.0          # CDD mínimo pra começar a perder (fallback se sem JSON)

# Carrega parâmetros calibrados do JSON se existir
if PARAMS_OUT.exists():
    try:
        with open(PARAMS_OUT, encoding="utf-8") as f:
            _calib = json.load(f)
        _calib_gamma = _calib.get("gamma")
        _calib_strike = _calib.get("strike")
        _calib_perda = _calib.get("perda_maxima")
        if _calib_gamma is not None:
            GAMMA = _calib_gamma
        if _calib_strike is not None:
            STRIKE = _calib_strike
        if _calib_perda is not None:
            PERDA_MAX = _calib_perda
    except (json.JSONDecodeError, KeyError):
        pass


def carregar():
    df = pd.read_parquet(DATA_IN)
    df = df.reset_index().set_index("data").sort_index()
    return df


def calcular_cdd(df, t_base=T_BASE):
    """Calcula CDD acumulado por safra × região no período DEZ-FEV."""
    # Período crítico: dezembro a fevereiro
    mask = df.index.get_level_values("data").month.isin([12, 1, 2])
    cd = df.loc[mask, ["t2m", "safra", "regiao"]].copy()
    cd["cdd_diario"] = np.maximum(0, cd["t2m"] - t_base)

    cdd_safra = (
        cd.groupby(["regiao", "safra"])["cdd_diario"]
        .sum()
        .reset_index()
        .rename(columns={"cdd_diario": "cdd"})
    )
    cdd_safra["dias_periodo"] = (
        cd.groupby(["regiao", "safra"]).size().values
    )
    cdd_safra["dias_estresse"] = (
        cd[cd["cdd_diario"] > 0]
        .groupby(["regiao", "safra"]).size()
        .reindex(cdd_safra.set_index(["regiao", "safra"]).index)
        .fillna(0)
        .values
    )
    return cdd_safra


def produtividade_linear(cdd, y_max=100.0, gamma=GAMMA, perda_max=PERDA_MAX):
    """Y(CDD) = Y_max × (1 - min(γ · CDD, perda_max))"""
    fracao_perda = np.minimum(gamma * cdd, perda_max)
    return y_max * (1.0 - fracao_perda)


def produtividade_piecewise(cdd, y_max=100.0, strike=STRIKE, gamma=GAMMA,
                            perda_max=PERDA_MAX):
    """Y(CDD) = Y_max × (1 - min(max(0, γ·(CDD-K)), perda_max))"""
    excesso = np.maximum(0, cdd - strike)
    fracao_perda = np.minimum(gamma * excesso, perda_max)
    return y_max * (1.0 - fracao_perda)


def plotar_cdd_serie(cdd_df):
    """Série histórica de CDD por região."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for regiao in cdd_df["regiao"].unique():
        sub = cdd_df[cdd_df["regiao"] == regiao].sort_values("safra")
        ax.plot(sub["safra"], sub["cdd"], "-o", markersize=3, linewidth=0.8,
                label=regiao)

    ax.axhline(STRIKE, color="gray", linestyle="--", linewidth=0.7,
               label=f"Strike K={STRIKE}°C")
    ax.set_xlabel("Safra")
    ax.set_ylabel("CDD Acumulado DEZ-FEV (°C·dia)")
    ax.set_title("CDD no Período Crítico da Soja — DEZ-FEV")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGS_OUT / "07_cdd_historico.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Figura salva: {path}")


def plotar_produtividade(cdd_df):
    """Relação CDD × produtividade (modelos linear e piecewise)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (nome, fn) in enumerate([
        ("Linear", produtividade_linear),
        ("Piecewise K=50", produtividade_piecewise),
    ]):
        ax = axes[i]
        cdd_range = np.linspace(0, cdd_df["cdd"].max() * 1.1, 200)
        y = fn(cdd_range)

        # Pontos reais
        for regiao in cdd_df["regiao"].unique():
            sub = cdd_df[cdd_df["regiao"] == regiao]
            y_real = fn(sub["cdd"].values)
            ax.scatter(sub["cdd"], y_real, s=15, alpha=0.5, label=regiao)

        ax.plot(cdd_range, y, "k--", linewidth=1, label="Modelo")
        ax.set_xlabel("CDD (°C·dia)")
        ax.set_ylabel("Produtividade (%)")
        ax.set_title(nome)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGS_OUT / "08_produtividade_cdd.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Figura salva: {path}")


def main():
    print("Carregando dados...")
    df = carregar()

    print("Calculando CDD por safra/região (DEZ-FEV)...")
    cdd = calcular_cdd(df)
    print(f"  {len(cdd)} registros, {cdd['regiao'].nunique()} regiões, "
          f"{cdd['safra'].nunique()} safras")

    print("\nEstatísticas CDD por região:")
    for regiao in cdd["regiao"].unique():
        sub = cdd[cdd["regiao"] == regiao]
        print(f"  {regiao}: média={sub['cdd'].mean():.1f}  "
              f"min={sub['cdd'].min():.1f}  max={sub['cdd'].max():.1f}  "
              f"dias_estresse_med={sub['dias_estresse'].mean():.0f}")

    # Salvar CDD
    cdd.to_parquet(CDD_OUT, index=False)
    print(f"\nCDD salvo em {CDD_OUT}")

    # Salvar parâmetros da função produtividade
    # Preserva keys existentes (por_regiao, calibracao) escritas pelo
    # calibrar_produtividade.py — atualiza apenas os campos deste script.
    params = {}
    if PARAMS_OUT.exists():
        try:
            with open(PARAMS_OUT, encoding="utf-8") as f:
                params = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    params.update({
        "t_base": T_BASE,
        "periodo_critico": "DEZ-FEV",
        "modelo": "piecewise",
        "strike": STRIKE,
        "gamma": GAMMA,
        "perda_maxima": PERDA_MAX,
        "y_max": 100.0,
        "descricao": (
            f"CDD = Σ max(0, T - {T_BASE}) em DEZ-FEV. "
            f"Perda = min(max(0, {GAMMA}·(CDD-{STRIKE})), {PERDA_MAX}). "
            f"Y = Y_max × (1 - Perda). "
            f"γ e K calibrados via regressão contra produtividade IBGE/literatura."
        ),
    })
    PARAMS_OUT.write_text(json.dumps(params, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"Parâmetros produtividade salvos em {PARAMS_OUT}")

    # Figuras
    plotar_cdd_serie(cdd)
    plotar_produtividade(cdd)

    # Resumo
    print(f"\n{'='*50}")
    print("  PRODUTIVIDADE — Exemplo (safra 2024, Sorriso/MT)")
    print(f"{'='*50}")
    ex = cdd[(cdd["regiao"] == "Sorriso_MT") & (cdd["safra"] >= 2019)]
    for _, row in ex.iterrows():
        y_lin = produtividade_linear(np.array([row["cdd"]]))[0]
        y_pw = produtividade_piecewise(np.array([row["cdd"]]))[0]
        print(f"  Safra {int(row['safra'])}: CDD={row['cdd']:.0f}  "
              f"Y_linear={y_lin:.0f}%  Y_piecewise={y_pw:.0f}% "
              f"dias_estresse={int(row['dias_estresse'])}")


if __name__ == "__main__":
    main()
