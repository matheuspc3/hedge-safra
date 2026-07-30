"""Backtest Walk-Forward — validação histórica da estratégia de hedge com CDD.

Para cada safra de 1990 a 2025, recalibra o modelo OU com janela expansiva,
calcula o prêmio forward-looking via Monte Carlo, e testa o hedge contra o CDD
real daquela safra.

Metodologia:
  1. A cada ano t, recalibra OU com todos os dados de temperatura até t
  2. Simula N_SIM trajetórias de CDD para precificar a opção (prêmio justo)
  3. Calcula produtividade real da safra t com base no CDD observado
  4. Aplica hedge ótimo (K, nocional) do grid search — payoff vs prêmio
  5. Acumula resultado: receita física + payoff - prêmio

Uso:
    python scripts/backtest_hedge.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_IN = Path("data/processed/temperatura_diaria_MT_PR_GO.parquet")
CDD_IN = Path("data/processed/cdd_safras.parquet")
SENS_PATH = Path("data/processed/sensibilidade_hedge.json")
PARAMS_PROD = Path("data/processed/parametros_produtividade.json")
FIGS_OUT = Path("output/graficos")
FIGS_OUT.mkdir(parents=True, exist_ok=True)

OUT_PATH = Path("data/processed/backtest_results.parquet")
OUT_METRICS = Path("data/processed/backtest_metricas.json")

DPI = 300
N_SIM = 20_000  # trajetórias para precificação forward-looking
DIAS_SAFRA = 90
T_BASE = 25.0
TAXA_DESC = 0.10

# Parâmetros do produtor
AREA_HA = 1_000
PROD_BASE_SC = 60  # sacas/ha
PRECO_SC = 140.0  # R$/saca

# Parâmetros ótimos do grid search (K_opcao, nocional)
OTIMO = {
    "Sorriso_MT": {"K": 10, "nocional": 20_000},
    "Londrina_PR": {"K": 25, "nocional": 30_000},
    "Rio Verde_GO": {"K": 25, "nocional": 20_000},
}

# Mapeamento região → nome do arquivo
NORM = {"Sorriso_MT": "Sorriso_MT", "Londrina_PR": "Londrina_PR", "Rio Verde_GO": "Rio Verde_GO"}


def calibrar_ou_ols(anom):
    """Ajusta OU via OLS nas anomalias — retorna (theta, sigma, mu_ou, beta)."""
    x = anom.values
    x_lag, x_lead = x[:-1], x[1:]

    n = len(x_lag)
    sx, sy = x_lag.sum(), x_lead.sum()
    sxx = (x_lag * x_lag).sum()
    sxy = (x_lag * x_lead).sum()

    denom = n * sxx - sx * sx
    beta = (n * sxy - sx * sy) / denom if denom != 0 else 0.0
    alpha = (sy - beta * sx) / n

    residuos = x_lead - (alpha + beta * x_lag)
    sigma_eps = np.std(residuos, ddof=2)
    theta = -np.log(beta)
    sigma = sigma_eps * np.sqrt(2 * theta / (1 - beta ** 2)) if theta > 0 and beta < 1 else sigma_eps
    mu_ou = alpha / (1 - beta) if beta < 1 else 0.0

    return theta, sigma, mu_ou, beta, residuos


def medias_sazonais(sub):
    """Médias mensais de temperatura."""
    return sub.groupby("mes")["t2m"].mean()


def simular_premio(theta, sigma, mu_ou, mu_saz, K, notional, n_sims=N_SIM):
    """Simula CDD via OU e calcula prêmio forward-looking da opção."""
    dt = 1.0
    exp_neg = np.exp(-theta * dt)
    var_eps = sigma ** 2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta) if theta > 0 else sigma ** 2
    std_eps = np.sqrt(var_eps)

    anomalias = np.zeros((n_sims, DIAS_SAFRA))
    rng = np.random.default_rng()
    for t in range(1, DIAS_SAFRA):
        anomalias[:, t] = (
            mu_ou + (anomalias[:, t - 1] - mu_ou) * exp_neg + rng.normal(0, std_eps, n_sims)
        )

    temp = np.zeros_like(anomalias)
    dias_por_mes = [31, 31, 28]
    inicio = 0
    for i, mes in enumerate([12, 1, 2]):
        fim = inicio + dias_por_mes[i]
        temp[:, inicio:fim] = mu_saz.loc[mes] + anomalias[:, inicio:fim]
        inicio = fim

    cdd_sim = np.maximum(0, temp - T_BASE).sum(axis=1)
    payoff_sim = np.maximum(0, cdd_sim - K)
    premio_justo = payoff_sim.mean() * np.exp(-TAXA_DESC * 180 / 365) * notional
    return premio_justo


def produtividade(cdd, gamma, strike, perda_max):
    """Y = Y_max × (1 - min(max(0, γ·(CDD-K)), perda_max))"""
    excesso = np.maximum(0, cdd - strike)
    fracao = np.minimum(gamma * excesso, perda_max)
    return 100.0 * (1.0 - fracao)


def preparar_temperatura(df):
    """Prepara índice e colunas auxiliares."""
    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"])
        df = df.set_index("data").sort_index()
    if "safra" not in df.columns:
        df["ano"] = df.index.year
        df["mes"] = df.index.month
        df["safra"] = np.where(df["mes"] >= 10, df["ano"] + 1, df["ano"])
    return df


def main():
    print("═" * 55)
    print("  BACKTEST WALK-FORWARD — Estratégia de Hedge CDD")
    print("═" * 55)

    # Carrega dados
    print("\nCarregando dados...")
    temp = pd.read_parquet(DATA_IN)
    cdd_hist = pd.read_parquet(CDD_IN)

    with open(PARAMS_PROD, encoding="utf-8") as f:
        params_prod = json.load(f)

    regioes = temp["regiao"].unique()
    print(f"  Regiões: {list(regioes)}")

    todos_resultados = {}
    todas_metricas = {}

    for regiao in regioes:
        print(f"\n{'─' * 50}")
        print(f"  ▸ {regiao.replace('_', ' ')}")
        print(f"  {'─' * 50}")

        # Filtra dados da região
        sub = temp[temp["regiao"] == regiao].copy()
        sub = preparar_temperatura(sub)

        # Parâmetros de produtividade específicos da região
        pp = params_prod.get("por_regiao", {}).get(regiao, params_prod)
        gamma = pp.get("gamma", 0.0045)
        strike_prod = pp.get("strike", 30.9)
        perda_max = pp.get("perda_maxima", 0.40)

        # Hedge ótimo
        otimo = OTIMO.get(regiao, {"K": 25, "nocional": 20_000})
        K_opcao = otimo["K"]
        nocional = otimo["nocional"]

        # Dados de CDD histórico para esta região
        cdd_r = cdd_hist[cdd_hist["regiao"] == regiao].sort_values("safra")

        # Safras testáveis (precisam de dados anteriores para calibração)
        anos = sorted(cdd_r["safra"].unique())
        anos_testaveis = [a for a in anos if a >= 1985]  # 5 anos mínimos de warm-up

        print(f"  K_opção={K_opcao}, Nocional=R${nocional:,}")
        print(f"  Safras: {len(anos_testaveis)} ({int(anos_testaveis[0])}–{int(anos_testaveis[-1])})")

        resultados = []

        for safra in anos_testaveis:
            safra_int = int(safra)

            # ── 1. Dados de temperatura até esta safra ──────
            # Último dia de maio da safra (colheita) — dados até maio
            cutoff = pd.Timestamp(year=safra_int, month=6, day=1)
            dados_ateh = sub[sub.index < cutoff].copy()

            if len(dados_ateh) < 365:
                continue  # dados insuficientes para calibrar

            # ── 2. Recalibra OU com janela expansiva ────────
            anomalias, saz = dados_ateh.groupby("mes")["t2m"].mean(), None
            anomalias_vals = (
                dados_ateh["t2m"]
                - dados_ateh.index.month.map(dados_ateh.groupby("mes")["t2m"].mean())
            )
            theta, sigma, mu_ou, beta, _ = calibrar_ou_ols(anomalias_vals)

            # ── 3. Prêmio forward-looking via Monte Carlo ──
            mu_saz = dados_ateh.groupby("mes")["t2m"].mean()
            premio = simular_premio(theta, sigma, mu_ou, mu_saz, K_opcao, nocional)

            # ── 4. CDD real da safra ────────────────────────
            row = cdd_r[cdd_r["safra"] == safra]
            if len(row) == 0:
                continue
            cdd_real = row["cdd"].values[0]

            # ── 5. Resultado físico ─────────────────────────
            y_pct = produtividade(np.array([cdd_real]), gamma, strike_prod, perda_max)[0]
            sacas_ha = PROD_BASE_SC * y_pct / 100.0
            receita_fisica = sacas_ha * AREA_HA * PRECO_SC

            # ── 6. Resultado do hedge ───────────────────────
            payoff = max(0, cdd_real - K_opcao) * nocional
            receita_hedgeada = receita_fisica + payoff - premio

            resultados.append({
                "regiao": regiao,
                "safra": safra_int,
                "cdd_real": round(cdd_real, 2),
                "produtividade_pct": round(y_pct, 2),
                "receita_fisica": round(receita_fisica, 2),
                "premio_pago": round(premio, 2),
                "payoff": round(payoff, 2),
                "receita_hedgeada": round(receita_hedgeada, 2),
                "exercicio": payoff > 0,
                "theta_calib": round(theta, 4),
                "sigma_calib": round(sigma, 4),
            })

        if not resultados:
            print("  ⚠ Nenhum resultado gerado")
            continue

        df_res = pd.DataFrame(resultados)
        todos_resultados[regiao] = df_res

        # ── Métricas ───────────────────────────────────────
        rf = df_res["receita_fisica"]
        rh = df_res["receita_hedgeada"]
        payoffs = df_res["payoff"]
        premios = df_res["premio_pago"]

        efetividade = 1.0 - rh.std(ddof=1) / rf.std(ddof=1) if rf.std(ddof=1) > 0 else 0.0
        hit_rate = df_res["exercicio"].mean()
        payoff_medio = payoffs[payoffs > 0].mean() if (payoffs > 0).any() else 0.0
        premio_medio = premios.mean()
        payout_ratio = payoffs.sum() / premios.sum() if premios.sum() > 0 else 0.0
        reducao_var = rf.quantile(0.05) - rh.quantile(0.05)

        # Drawdown máximo
        cum_fisica = rf.cumsum()
        cum_hedge = rh.cumsum()
        dd_fisica = (cum_fisica.cummax() - cum_fisica) / cum_fisica.cummax()
        dd_hedge = (cum_hedge.cummax() - cum_hedge) / cum_hedge.cummax()

        metricas = {
            "regiao": regiao,
            "n_safras": len(df_res),
            "periodo": f"{int(df_res['safra'].min())}–{int(df_res['safra'].max())}",
            "k_opcao": K_opcao,
            "nocional": nocional,
            "receita_media_fisica": round(rf.mean(), 2),
            "receita_media_hedgeada": round(rh.mean(), 2),
            "std_fisica": round(rf.std(ddof=1), 2),
            "std_hedgeada": round(rh.std(ddof=1), 2),
            "efetividade_historica": round(efetividade, 4),
            "var95_fisica": round(rf.quantile(0.05), 2),
            "var95_hedgeada": round(rh.quantile(0.05), 2),
            "reducao_var95": round(reducao_var, 2),
            "hit_rate": round(hit_rate, 4),
            "payoff_medio_exercicio": round(payoff_medio, 2),
            "premio_medio": round(premio_medio, 2),
            "payout_ratio_total": round(payout_ratio, 4),
            "drawdown_max_fisico": round(dd_fisica.max(), 4),
            "drawdown_max_hedgeado": round(dd_hedge.max(), 4),
            "melhor_ano_fisico": int(df_res.loc[rf.idxmax(), "safra"]),
            "pior_ano_fisico": int(df_res.loc[rf.idxmin(), "safra"]),
        }
        todas_metricas[regiao] = metricas

        # ── Print ──────────────────────────────────────────
        print(f"\n  Resultados ({len(df_res)} safras, {metricas['periodo']}):")
        print(f"    Efetividade histórica: {efetividade * 100:.1f}%")
        print(f"    Hit rate (exercício): {hit_rate * 100:.1f}%")
        print(f"    Prêmio médio: R$ {premio_medio:,.0f}")
        print(f"    Payoff médio (qdo ex.): R$ {payoff_medio:,.0f}")
        print(f"    Payout ratio (total): {payout_ratio:.2f}x")
        print(f"    σ física: R$ {rf.std()/1e6:.2f}M → σ hedge: R$ {rh.std()/1e6:.2f}M")
        print(f"    VaR95 física: R$ {rf.quantile(0.05)/1e6:.2f}M → hedge: R$ {rh.quantile(0.05)/1e6:.2f}M")
        print(f"    Drawdown máx físico: {dd_fisica.max()*100:.1f}% → hedge: {dd_hedge.max()*100:.1f}%")
        print(f"    Melhor safra: {metricas['melhor_ano_fisico']}  Pior safra: {metricas['pior_ano_fisico']}")

        # 5 anos extremos
        tops = df_res.nlargest(5, "receita_fisica")
        bots = df_res.nsmallest(5, "receita_fisica")
        print(f"    Top 3 safras: {[int(s) for s in tops['safra'].values[:3]]}")
        print(f"    Piores 3 safras: {[int(s) for s in bots['safra'].values[:3]]}")

    # ── Salvar ─────────────────────────────────────────────
    if todos_resultados:
        df_out = pd.concat(todos_resultados.values(), ignore_index=True)
        df_out.to_parquet(OUT_PATH, index=False)
        print(f"\nResultados detalhados salvos em {OUT_PATH}")

    if todas_metricas:
        with open(OUT_METRICS, "w", encoding="utf-8") as f:
            json.dump(todas_metricas, f, indent=2, ensure_ascii=False)
        print(f"Métricas salvas em {OUT_METRICS}")

    # ── Resumo final ───────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("  RESUMO — Backtest vs Monte Carlo")
    print(f"{'=' * 55}")
    print(f"{'Região':<15} {'Efetiv. BT':>11} {'Efetiv. MC':>11} {'Dif':>6} {'Hit Rate':>9}")
    print("-" * 52)
    for regiao, m in sorted(todas_metricas.items()):
        efet_mc = 0.0
        if SENS_PATH.exists():
            with open(SENS_PATH) as f:
                sens = json.load(f)
            efet_mc = sens.get(regiao, {}).get("melhor", {}).get("efetividade", 0) * 100
        dif = m["efetividade_historica"] * 100 - efet_mc
        print(f"{regiao:<15} {m['efetividade_historica']*100:>10.1f}% "
              f"{efet_mc:>10.1f}% {dif:>+5.1f}% "
              f"{m['hit_rate']*100:>7.1f}%")

    print(f"\n{'─' * 55}")
    print("  Conclusão do Backtest:")
    print(f"{'─' * 55}")
    for regiao, m in sorted(todas_metricas.items()):
        print(f"  {regiao.replace('_', ' ')}: {m['n_safras']} safras, "
              f"efet. {m['efetividade_historica']*100:.1f}%, "
              f"hit rate {m['hit_rate']*100:.0f}%, "
              f"payout {m['payout_ratio_total']:.2f}x")

    return todos_resultados, todas_metricas


if __name__ == "__main__":
    main()
