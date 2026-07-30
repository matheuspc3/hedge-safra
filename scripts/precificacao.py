"""Precifica opções climáticas CDD via Monte Carlo.

Processo:
  1. Carrega parâmetros OU calibrados para cada região
  2. Carrega sazonalidade (médias mensais) da temperatura
  3. Simula N caminhos de temperatura diária para a safra futura
  4. Calcula CDD acumulado no período crítico DEZ-FEV
  5. Precifica opção: payoff = max(CDD - K_strike, 0) × notional
  6. Desconta a valor presente (taxa livre de risco)

A opção CDD paga quando o acúmulo de calor no período crítico
ultrapassa o strike — hedge contra perda de produtividade por calor.
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
T_BASE = 25.0        # °C — limiar CDD soja
N_SIMULACOES = 50_000
DIAS_SAFRA = 90      # DEZ-FEV ≈ 90 dias (DJF)
TAXA_LIVRE_RISCO = 0.10   # 10% a.a. (Selic real + prêmio)
STRIKE_PADRAO = 30.0     # CDD strike default (fallback)
NOTIONAL = 30_000         # R$ 30k por ponto de CDD

# Tenta carregar strike calibrado
if PARAMS_PROD.exists():
    try:
        with open(PARAMS_PROD, encoding="utf-8") as f:
            _p = json.load(f)
        _s = _p.get("strike")
        if _s is not None:
            STRIKE_PADRAO = _s
    except (json.JSONDecodeError, KeyError):
        pass

# Safra alvo para precificação
SAFRA_ALVO = 2025


def carregar():
    """Carrega dados históricos + parâmetros calibrados."""
    df = pd.read_parquet(DATA_IN)
    df = df.reset_index().set_index("data").sort_index()

    with open(OU_PARAMS, encoding="utf-8") as f:
        ou_params = json.load(f)

    cdd_hist = pd.read_parquet(CDD_IN)

    return df, ou_params, cdd_hist


def medias_sazonais(df, regiao):
    """Média e desvio mensal da temperatura para uma região."""
    sub = df[df["regiao"] == regiao]
    medias = sub.groupby("mes")["t2m"].agg(["mean", "std"])
    return medias["mean"], medias["std"]


def simular_temperaturas(mu_sazonal, params_ou, n_sims=N_SIMULACOES,
                         n_dias=DIAS_SAFRA):
    """Simula caminhos de temperatura diária via OU.

    O processo começa da média sazonal de novembro (pré-safra) e
    evolui diariamente pelos 90 dias do DEZ-FEV.

    X(t+1) = μ + (X(t) - μ)e^{-θΔt} + ε
    ε ~ N(0, σ²(1 - e^{-2θΔt}) / 2θ)

    Retorna array (n_sims, n_dias) com temperaturas simuladas.
    """
    theta = params_ou["theta"]
    sigma = params_ou["sigma"]
    mu_ou = params_ou["mu"]  # ≈ 0 (anomalias)

    dt = 1.0

    # Parâmetros da discretização exata
    exp_neg_theta = np.exp(-theta * dt)
    var_eps = (sigma ** 2) * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
    std_eps = np.sqrt(var_eps)

    # Temperatura inicial = anomalia zero + sazonalidade de novembro (mês 11)
    # Começamos da média de novembro (pré-safra)
    x0 = 0.0  # anomalia inicial = 0

    # Simular anomalias
    anomalias = np.zeros((n_sims, n_dias))
    anomalias[:, 0] = x0

    rng = np.random.default_rng(42)
    for t in range(1, n_dias):
        anomalias[:, t] = (
            mu_ou
            + (anomalias[:, t - 1] - mu_ou) * exp_neg_theta
            + rng.normal(0, std_eps, n_sims)
        )

    # Adicionar sazonalidade: temperatura = média mensal + anomalia
    # Mês 12 (dez): índice 0-30, mês 1 (jan): 31-61, mês 2 (fev): 62-89
    # (aproximado — ignoramos anos bissextos para simplicidade)
    temp = np.zeros_like(anomalias)

    dias_por_mes = [31, 31, 28]  # dez, jan, fev
    inicio = 0
    for i, mes in enumerate([12, 1, 2]):
        fim = inicio + dias_por_mes[i]
        temp[:, inicio:fim] = mu_sazonal.loc[mes] + anomalias[:, inicio:fim]
        inicio = fim

    return temp


def calcular_cdd_simulado(temp, t_base=T_BASE):
    """Calcula CDD acumulado para cada simulação."""
    cdd_diario = np.maximum(0, temp - t_base)
    return cdd_diario.sum(axis=1)


def precificar_opcao(cdd_simulado, strike=STRIKE_PADRAO,
                     notional=NOTIONAL, taxa=TAXA_LIVRE_RISCO,
                     dias_ate_vencimento=180):
    """Precifica opção CDD via Monte Carlo.

    Payoff por unidade: max(CDD - strike, 0)
    Prêmio justo = E[payoff] × notional × exp(-r·T)
    """
    payoffs = np.maximum(0, cdd_simulado - strike)
    payoff_medio = payoffs.mean()
    payoff_std = payoffs.std(ddof=1)

    # Descontar a valor presente
    fator_desconto = np.exp(-taxa * dias_ate_vencimento / 365)
    premio_justo = payoff_medio * notional * fator_desconto

    # Erro padrão da estimativa Monte Carlo
    erro_padrao = payoff_std / np.sqrt(len(cdd_simulado))
    premio_erro = erro_padrao * notional * fator_desconto

    # Probabilidade de exercício (in-the-money)
    prob_exercicio = (cdd_simulado > strike).mean()

    # Percentis do CDD simulado
    percentis = np.percentile(cdd_simulado, [5, 25, 50, 75, 95])

    return {
        "strike": strike,
        "notional": notional,
        "payoff_medio": round(payoff_medio, 2),
        "premio_justo": round(premio_justo, 2),
        "premio_erro_padrao": round(premio_erro, 2),
        "prob_exercicio": round(prob_exercicio, 4),
        "cdd_medio_simulado": round(cdd_simulado.mean(), 2),
        "cdd_mediano_simulado": round(np.median(cdd_simulado), 2),
        "cdd_std_simulado": round(cdd_simulado.std(ddof=1), 2),
        "cdd_percentis": {
            "p5": round(percentis[0], 1),
            "p25": round(percentis[1], 1),
            "p50": round(percentis[2], 1),
            "p75": round(percentis[3], 1),
            "p95": round(percentis[4], 1),
        },
    }


def plotar_distribuicao_cdd(cdd_simulado, regiao, strike, premio):
    """Histograma do CDD simulado + payoff da opção."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma CDD
    ax = axes[0]
    ax.hist(cdd_simulado, bins=80, density=True, alpha=0.6,
            color="tomato", edgecolor="white", linewidth=0.3)
    ax.axvline(strike, color="black", linestyle="--", linewidth=1.2,
               label=f"Strike K={strike}")
    ax.set_xlabel("CDD Acumulado (°C·dia)")
    ax.set_ylabel("Densidade")
    ax.set_title(f"Distribuição do CDD Simulado — {regiao}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Payoff
    ax = axes[1]
    payoffs = np.maximum(0, cdd_simulado - strike)
    ax.hist(payoffs, bins=60, density=True, alpha=0.6,
            color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Payoff (°C·dia)")
    ax.set_ylabel("Densidade")
    ax.set_title(f"Payoff max(CDD - {strike}, 0) — Prêmio: R$ {premio:.0f}")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGS_OUT / f"09_precificacao_{regiao.lower()}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Figura salva: {path}")


def plotar_caminhos(temp_sim, regiao):
    """Alguns caminhos simulados de temperatura."""
    fig, ax = plt.subplots(figsize=(14, 5))

    datas = pd.date_range("2025-12-01", periods=90, freq="D")

    # Plotar 50 caminhos aleatórios
    amostra = np.random.choice(temp_sim.shape[0], min(50, temp_sim.shape[0]),
                               replace=False)
    for i in amostra:
        ax.plot(datas, temp_sim[i], linewidth=0.3, alpha=0.4, color="steelblue")

    # Média dos caminhos
    media = temp_sim.mean(axis=0)
    ax.plot(datas, media, linewidth=2, color="crimson", label="Média")

    ax.axhline(T_BASE, color="orange", linestyle="--", linewidth=0.8,
               label=f"T_base={T_BASE}°C")
    ax.set_xlabel("Data")
    ax.set_ylabel("Temperatura (°C)")
    ax.set_title(f"Caminhos Simulados — Safra {SAFRA_ALVO} ({regiao})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGS_OUT / f"10_caminhos_temp_{regiao.lower()}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Figura salva: {path}")


def analisar_cenario(precos, cdd_hist):
    """Compara CDD simulado com histórico."""
    print(f"\n{'='*50}")
    print("  ANÁLISE DE CENÁRIOS — CDD Histórico vs Simulado")
    print(f"{'='*50}")
    for regiao, info in precos.items():
        p = info["precificacao"]
        ch = cdd_hist[cdd_hist["regiao"] == regiao]["cdd"]
        print(f"\n  {regiao}:")
        print(f"    CDD histórico: média={ch.mean():.1f}  "
              f"máx={ch.max():.0f}  p95={ch.quantile(0.95):.0f}")
        print(f"    CDD simulado:  média={p['cdd_medio_simulado']:.1f}  "
              f"p95={p['cdd_percentis']['p95']:.0f}")
        print(f"    Prêmio justo: R$ {p['premio_justo']:>10.2f}  "
              f"(±R$ {p['premio_erro_padrao']:.2f})")
        print(f"    Prob. exercício: {p['prob_exercicio']*100:.1f}%")


def main():
    print("Carregando dados históricos e parâmetros...")
    df, ou_params, cdd_hist = carregar()
    regioes = df["regiao"].unique()

    precos = {}

    for regiao in regioes:
        print(f"\n{'='*50}")
        print(f"  PRECIFICAÇÃO — {regiao}")
        print(f"{'='*50}")

        # Médias sazonais
        mu_saz, _ = medias_sazonais(df, regiao)
        params = ou_params[regiao]

        print(f"  θ={params['theta']:.4f}  σ={params['sigma']:.4f}  "
              f"τ½={params['meia_vida_dias']:.1f}d")

        # Simular
        print(f"  Simulando {N_SIMULACOES:,} caminhos × {DIAS_SAFRA} dias...")
        temp = simular_temperaturas(mu_saz, params)

        # CDD simulado
        cdd_sim = calcular_cdd_simulado(temp)

        # Precificar
        premio = precificar_opcao(cdd_sim)
        precos[regiao] = {
            "safra": SAFRA_ALVO,
            "n_simulacoes": N_SIMULACOES,
            "parametros_ou": params,
            "precificacao": premio,
        }
        print(f"  CDD médio simulado: {premio['cdd_medio_simulado']:.1f}")
        print(f"  Prêmio justo: R$ {premio['premio_justo']:>10.2f}")
        print(f"  Prob. exercício: {premio['prob_exercicio']*100:.1f}%")

        # Figuras
        plotar_distribuicao_cdd(cdd_sim, regiao, STRIKE_PADRAO,
                                premio["premio_justo"])
        plotar_caminhos(temp, regiao)

    # Salvar precificações
    out_path = Path("data/processed/precificacao_cdd.json")
    out_path.write_text(
        json.dumps(precos, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nPrecificação salva em {out_path}")

    # Análise de cenários
    analisar_cenario(precos, cdd_hist)


if __name__ == "__main__":
    main()
