"""Calibra modelo Ornstein-Uhlenbeck para temperatura diária por região.

O processo OU contínuo:
    dX(t) = θ(μ - X(t))dt + σdW(t)

Discretização (exata):
    X(t+1) = μ + (X(t) - μ)e^{-θΔt} + ε(t)
    ε(t) ~ N(0, σ²(1 - e^{-2θΔt}) / 2θ)

Equivalentemente via regressão OLS:
    X(t+1) = α + β·X(t) + ε
    β = e^{-θΔt}
    α = μ(1 - β)
    σ²_ε = σ²(1 - β²) / 2θ
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, jarque_bera

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_IN = Path("data/processed/temperatura_diaria_MT_PR_GO.parquet")
PARAMS_OUT = Path("data/processed/parametros_ou.json")
FIGS_OUT = Path("output/graficos")
FIGS_OUT.mkdir(parents=True, exist_ok=True)

DPI = 300
ALPHA = 0.05


def carregar():
    df = pd.read_parquet(DATA_IN)
    df = df.reset_index().set_index(["regiao", "data"]).sort_index()
    return df


def decompor_sazonal(df, col="t2m"):
    """Remove sazonalidade mensal: retorna anomalias e tabela sazonal."""
    medias = df.groupby("mes")[col].mean()
    desvios = df.groupby("mes")[col].std()
    anomalias = df[col] - df.index.get_level_values("data").month.map(medias)
    return anomalias, pd.DataFrame({"media": medias, "desvio": desvios})


def calibrar_ou(anom, dt=1.0):
    """Ajusta OU via OLS nas anomalias (já dessazonalizadas).

    Retorna dicionário com θ, μ (≈0 p/ anomalias), σ, e métricas.
    """
    x = anom.values
    x_lag, x_lead = x[:-1], x[1:]

    # OLS via numpy — X(t+1) = α + β·X(t) + ε
    n = len(x_lag)
    sx = x_lag.sum()
    sy = x_lead.sum()
    sxx = (x_lag * x_lag).sum()
    sxy = (x_lag * x_lead).sum()
    beta = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    alpha = (sy - beta * sx) / n

    residuos = x_lead - (alpha + beta * x_lag)
    sigma_eps = np.std(residuos, ddof=2)

    ss_res = (residuos ** 2).sum()
    ss_tot = ((x_lead - x_lead.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot

    theta = -np.log(beta) / dt
    mu = alpha / (1 - beta) if beta < 1 else 0.0
    sigma = sigma_eps * np.sqrt(2 * theta / (1 - beta ** 2)) if beta < 1 else sigma_eps

    meia_vida = np.log(2) / theta if theta > 0 else np.inf

    jb_stat, jb_p = jarque_bera(residuos)

    return {
        "theta": round(theta, 6),
        "mu": round(mu, 4),
        "sigma": round(sigma, 6),
        "meia_vida_dias": round(meia_vida, 1),
        "beta": round(beta, 6),
        "alpha": round(alpha, 4),
        "sigma_eps": round(sigma_eps, 4),
        "n": len(x),
        "r2": round(r2, 4),
        "jb_stat": round(jb_stat, 2),
        "jb_p": round(jb_p, 4),
        "residuos_normal": bool(jb_p > ALPHA),
    }


def plotar_serie(t2m, anom, medias_mensais, regiao):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Série original + sazonalidade sobreposta
    t2m_plot = t2m.iloc[:365 * 3]  # primeiros 3 anos p/ legibilidade
    meses = t2m_plot.index.month
    saz_plot = meses.map(medias_mensais)
    axes[0].plot(t2m_plot.index, t2m_plot.values, color="steelblue", linewidth=0.4, alpha=0.7)
    axes[0].plot(t2m_plot.index, saz_plot.values, color="crimson", linewidth=1.5, label="Média mensal")
    axes[0].set_ylabel("Temperatura (°C)")
    axes[0].set_title(f"Série Original (3 primeiros anos) — {regiao}")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Anomalias (série completa)
    axes[1].plot(anom.index, anom.values, color="gray", linewidth=0.3, alpha=0.6)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("Anomalia (°C)")
    axes[1].set_title("Dessazonalizada (anomalias)")
    axes[1].grid(alpha=0.3)

    # Distribuição
    axes[2].hist(anom.values, bins=80, density=True, color="steelblue", alpha=0.6, label="Anomalias")
    media, std = anom.mean(), anom.std()
    x = np.linspace(anom.min(), anom.max(), 200)
    axes[2].plot(x, norm.pdf(x, media, std), "r--", linewidth=1.5, label=f"N({media:.2f}, {std:.2f})")
    axes[2].set_xlabel("Anomalia (°C)")
    axes[2].set_ylabel("Densidade")
    axes[2].set_title("Distribuição das Anomalias")
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = FIGS_OUT / f"05_ou_{regiao.lower()}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"    Figura salva: {path}")


def plotar_residuos(residuos, regiao):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(residuos, color="gray", linewidth=0.3, alpha=0.5)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_title(f"Resíduos OU — {regiao}")
    axes[0].set_ylabel("Resíduo (°C)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(residuos, bins=60, density=True, color="steelblue", alpha=0.6)
    x = np.linspace(residuos.min(), residuos.max(), 200)
    m, s = residuos.mean(), residuos.std()
    axes[1].plot(x, norm.pdf(x, m, s), "r--", linewidth=1.5, label=f"N({m:.2f}, {s:.2f})")
    axes[1].set_xlabel("Resíduo (°C)")
    axes[1].set_title("Distribuição dos Resíduos")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = FIGS_OUT / f"06_residuos_ou_{regiao.lower()}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"    Figura salva: {path}")


def main():
    print("Carregando dados...")
    df = carregar()
    regioes = df.index.get_level_values("regiao").unique()

    todos_params = {}
    for regiao in regioes:
        print(f"\n{'='*50}")
        print(f"  Região: {regiao}")
        print(f"{'='*50}")

        sub = df.loc[regiao]
        t2m = sub["t2m"]

        # Dessazonalização
        anomalias, saz = decompor_sazonal(sub)
        print(f"  Média jan: {saz.loc[1, 'media']:.1f}°C  |  Média jul: {saz.loc[7, 'media']:.1f}°C")

        # Calibração OU
        params = calibrar_ou(anomalias)
        todos_params[regiao] = params
        print(f"\n  Parâmetros OU (anomalias):")
        print(f"    θ (reversão)   = {params['theta']:.6f}  /dia")
        print(f"    μ (longo prazo)= {params['mu']:.4f} °C")
        print(f"    σ (volatilid.) = {params['sigma']:.6f}  /√dia")
        print(f"    β              = {params['beta']:.6f}")
        print(f"    Meia-vida      = {params['meia_vida_dias']:.1f} dias")
        print(f"    R²             = {params['r2']:.4f}")
        print(f"    JB p-valor     = {params['jb_p']:.4f}  {'✓ normal' if params['residuos_normal'] else '✗ não-normal'}")

        # Resíduos
        x = anomalias.values
        x_lag = x[:-1]
        residuos = x[1:] - (params["alpha"] + params["beta"] * x_lag)

        # Figuras
        plotar_serie(t2m, anomalias, saz["media"], regiao)
        plotar_residuos(residuos, regiao)

    # Salvar parâmetros
    PARAMS_OUT.write_text(json.dumps(todos_params, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nParâmetros salvos em {PARAMS_OUT}")

    # Tabela-resumo
    print(f"\n{'='*50}")
    print("  RESUMO — Parâmetros OU por Região")
    print(f"{'='*50}")
    print(f"{'Região':<15} {'θ':>8} {'μ':>8} {'σ':>8} {'τ½(dias)':>10} {'R²':>6}")
    print("-" * 55)
    for r, p in todos_params.items():
        print(f"{r:<15} {p['theta']:>8.4f} {p['mu']:>8.2f} {p['sigma']:>8.4f} {p['meia_vida_dias']:>10.1f} {p['r2']:>6.3f}")


if __name__ == "__main__":
    main()
