"""Análise de sensibilidade — varre K e nocional para ótimo do hedge.

Para cada região, simula Monte Carlo uma única vez, depois varre:
  - Strike K:  0 a 100 (passo 5)
  - Notional:  R$ 10k a R$ 100k (passo 10k)

Métrica alvo: efetividade do hedge = 1 - σ(hedged)/σ(unhedged)
Ótimo = máxima efetividade com prêmio ≤ 10% da receita esperada.
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_IN = Path("data/processed/temperatura_diaria_MT_PR_GO.parquet")
OU_PARAMS = Path("data/processed/parametros_ou.json")
CDD_IN = Path("data/processed/cdd_safras.parquet")
PARAMS_PROD = Path("data/processed/parametros_produtividade.json")
FIGS_OUT = Path("output/graficos")
FIGS_OUT.mkdir(parents=True, exist_ok=True)

DPI = 300
T_BASE = 25.0
N_SIM = 50_000
DIAS_SAFRA = 90
TAXA = 0.10
AREA_HA = 1_000
PROD_BASE = 60
PRECO_SACA = 140.0
RECEITA_ESPERADA = AREA_HA * PROD_BASE * PRECO_SACA  # R$ 8.4M

# Carrega parâmetros calibrados do JSON se existir
GAMMA_CALIB = 0.003
K_CALIB = 30.0
PERDA_MAX_CALIB = 0.50
if PARAMS_PROD.exists():
    try:
        with open(PARAMS_PROD, encoding="utf-8") as f:
            _p = json.load(f)
        if _p.get("gamma") is not None:
            GAMMA_CALIB = _p["gamma"]
        if _p.get("strike") is not None:
            K_CALIB = _p["strike"]
        if _p.get("perda_maxima") is not None:
            PERDA_MAX_CALIB = _p["perda_maxima"]
    except (json.JSONDecodeError, KeyError):
        pass

# Grid de varredura
K_GRID = list(range(0, 105, 5))
NOCIONAL_GRID = list(range(10_000, 105_000, 10_000))


def carregar():
    df = pd.read_parquet(DATA_IN)
    df = df.reset_index().set_index("data").sort_index()
    with open(OU_PARAMS, encoding="utf-8") as f:
        ou = json.load(f)
    cdd = pd.read_parquet(CDD_IN)
    return df, ou, cdd


def medias_sazonais(df, regiao):
    sub = df[df["regiao"] == regiao]
    return sub.groupby("mes")["t2m"].agg("mean")


def simular_cdd(mu_saz, params, n_sims=N_SIM, n_dias=DIAS_SAFRA):
    """Simula CDD acumulado — roda uma vez, reuso pra todo grid."""
    theta = params["theta"]
    sigma = params["sigma"]
    mu_ou = params["mu"]
    dt = 1.0

    exp_neg = np.exp(-theta * dt)
    var_eps = sigma ** 2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
    std_eps = np.sqrt(var_eps)

    anomalias = np.zeros((n_sims, n_dias))
    rng = np.random.default_rng(42)
    for t in range(1, n_dias):
        anomalias[:, t] = (
            mu_ou + (anomalias[:, t - 1] - mu_ou) * exp_neg
            + rng.normal(0, std_eps, n_sims)
        )

    temp = np.zeros_like(anomalias)
    dias_por_mes = [31, 31, 28]
    inicio = 0
    for i, mes in enumerate([12, 1, 2]):
        fim = inicio + dias_por_mes[i]
        temp[:, inicio:fim] = mu_saz.loc[mes] + anomalias[:, inicio:fim]
        inicio = fim

    cdd = np.maximum(0, temp - T_BASE).sum(axis=1)
    return cdd


def efetividade_hedge(cdd_hist, receitas, strike, notional, premio):
    """Efetividade para um (K, notional)."""
    payoffs = np.maximum(0, cdd_hist - strike) * notional
    receita_hedged = receitas + payoffs - premio
    std_fisica = receitas.std(ddof=1)
    std_hedged = receita_hedged.std(ddof=1)
    if std_fisica == 0:
        return 0.0
    return 1.0 - std_hedged / std_fisica


def produtividade(cdd, gamma=None, strike=None, perda_max=None):
    if gamma is None:
        gamma = GAMMA_CALIB
    if strike is None:
        strike = K_CALIB
    if perda_max is None:
        perda_max = PERDA_MAX_CALIB
    excesso = np.maximum(0, cdd - strike)
    fracao = np.minimum(gamma * excesso, perda_max)
    return 100.0 * (1.0 - fracao)


def main():
    print("Carregando dados...")
    df, ou_params, cdd_hist = carregar()
    regioes = df["regiao"].unique()

    todos_resultados = {}

    for regiao in regioes:
        print(f"\n{'='*50}")
        print(f"  SENSIBILIDADE — {regiao}")
        print(f"{'='*50}")

        mu_saz = medias_sazonais(df, regiao)
        params = ou_params[regiao]
        print(f"  θ={params['theta']:.4f}  σ={params['sigma']:.4f}")

        # CDD histórico p/ região
        ch = cdd_hist[cdd_hist["regiao"] == regiao].sort_values("safra")
        cdd_vals = ch["cdd"].values

        # Receitas históricas físicas
        y_pct = produtividade(cdd_vals)
        sacas_ha = PROD_BASE * y_pct / 100.0
        receitas = sacas_ha * AREA_HA * PRECO_SACA

        # Simular CDD uma vez
        print(f"  Simulando {N_SIM:,} caminhos...")
        cdd_sim = simular_cdd(mu_saz, params)

        # Varredura
        resultados = []
        for K in K_GRID:
            payoff_sim = np.maximum(0, cdd_sim - K)
            premio = payoff_sim.mean() * TAXA  # sem desconto (proxy)
            # desconto real
            premio_real = payoff_sim.mean() * np.exp(-TAXA * 180 / 365)

            for N in NOCIONAL_GRID:
                premium_final = premio_real * N
                efet = efetividade_hedge(cdd_vals, receitas, K, N,
                                         premium_final)
                custo_pct = premium_final / RECEITA_ESPERADA * 100

                # VaR95 da receita hedgeada
                payoffs_hist = np.maximum(0, cdd_vals - K) * N
                rh = receitas + payoffs_hist - premium_final
                var95_hedge = np.percentile(rh, 5)

                resultados.append({
                    "K": K,
                    "nocional": N,
                    "premio": round(premium_final, 2),
                    "custo_pct": round(custo_pct, 2),
                    "efetividade": round(efet, 4),
                    "var95_hedge": round(var95_hedge, 2),
                })

        df_res = pd.DataFrame(resultados)

        # Top 5 por efetividade máxima (com custo ≤ 10%)
        viavel = df_res[df_res["custo_pct"] <= 10.0]
        top5 = viavel.nlargest(5, "efetividade") if len(viavel) > 0 else df_res.nlargest(5, "efetividade")

        todos_resultados[regiao] = {
            "top5": top5.to_dict(orient="records"),
            "melhor": top5.iloc[0].to_dict() if len(top5) > 0 else None,
        }

        print(f"\n  TOP 5 — Melhores parâmetros (custo ≤ 10% receita):")
        print(f"  {'K':>4} {'Nocional':>10} {'Prêmio':>10} {'Custo%':>7} "
              f"{'Efetiv.':>8} {'VaR95':>10}")
        print(f"  {'-'*52}")
        for _, r in top5.iterrows():
            print(f"  {r['K']:>4} {r['nocional']:>10,} {r['premio']:>10,.0f} "
                  f"{r['custo_pct']:>6.1f}% {r['efetividade']*100:>7.1f}% "
                  f"{r['var95_hedge']:>10,.0f}")

        # Heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Heatmap efetividade
        ax = axes[0]
        pivot = df_res.pivot_table(index="K", columns="nocional",
                                   values="efetividade")
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       vmin=-1, vmax=1, origin="lower")
        ax.set_xticks(range(len(NOCIONAL_GRID)))
        ax.set_xticklabels([f"{n//1000:.0f}k" for n in NOCIONAL_GRID])
        ax.set_yticks(range(len(K_GRID)))
        ax.set_yticklabels(K_GRID)
        ax.set_xlabel("Nocional (R$/ponto)")
        ax.set_ylabel("Strike K")
        ax.set_title(f"Efetividade do Hedge — {regiao}")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Efetividade")

        # Heatmap custo (%)
        ax = axes[1]
        pivot2 = df_res.pivot_table(index="K", columns="nocional",
                                    values="custo_pct")
        im2 = ax.imshow(pivot2.values, aspect="auto", cmap="YlOrRd",
                        origin="lower")
        ax.set_xticks(range(len(NOCIONAL_GRID)))
        ax.set_xticklabels([f"{n//1000:.0f}k" for n in NOCIONAL_GRID])
        ax.set_yticks(range(len(K_GRID)))
        ax.set_yticklabels(K_GRID)
        ax.set_xlabel("Nocional (R$/ponto)")
        ax.set_ylabel("Strike K")
        ax.set_title(f"Custo do Hedge (% Receita) — {regiao}")
        plt.colorbar(im2, ax=ax, shrink=0.7, label="Custo (%)")

        plt.tight_layout()
        path = FIGS_OUT / f"12_sensibilidade_{regiao.lower()}.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"\n  Figura salva: {path}")

        # Efetividade vs Custo (curva para cada nocional)
        fig, ax = plt.subplots(figsize=(12, 5))
        for N in [20_000, 30_000, 50_000, 80_000, 100_000]:
            sub = df_res[df_res["nocional"] == N].sort_values("K")
            ax.plot(sub["K"], sub["efetividade"] * 100, "-o", markersize=4,
                    linewidth=0.8, label=f"N={N//1000}k")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Efetividade (%)")
        ax.set_title(f"Efetividade vs Strike — {regiao}")
        ax.legend(fontsize=8, title="Nocional")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = FIGS_OUT / f"13_curvas_efetividade_{regiao.lower()}.png"
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        print(f"  Figura salva: {path}")

    # Salvar resultados
    out_path = Path("data/processed/sensibilidade_hedge.json")
    out_path.write_text(
        json.dumps(todos_resultados, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nResultados salvos em {out_path}")

    # Resumo final
    print(f"\n{'='*50}")
    print("  RESUMO — Estrutura Ótima por Região")
    print(f"{'='*50}")
    print(f"{'Região':<15} {'K':>4} {'Nocional':>10} {'Prêmio':>10} "
          f"{'Custo':>7} {'Efetiv.':>8}")
    print("-" * 58)
    for regiao, res in todos_resultados.items():
        if res["melhor"]:
            m = res["melhor"]
            print(f"{regiao:<15} {m['K']:>4} {m['nocional']:>10,} "
                  f"{m['premio']:>10,.0f} {m['custo_pct']:>6.1f}% "
                  f"{m['efetividade']*100:>7.1f}%")


if __name__ == "__main__":
    main()
