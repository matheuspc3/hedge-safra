"""Simula estratégia de hedge para produtor de soja com opção CDD.

Estrutura:
  - Produtor planta soja em outubro, colhe em março
  - No plantio, compra opção de CALL de CDD (vencimento DEZ-FEV)
  - Na colheita: se CDD > strike, opção paga a diferença
  - Payoff compensa a perda de produtividade por estresse térmico

Métricas:
  - Receita física (soja) = produtividade × preço × área
  - Receita hedgeada   = receita física + payoff da opção - prêmio pago
  - Efetividade do hedge = 1 - σ(hedged) / σ(unhedged)
  - Value at Risk (VaR 5%) comparado entre cenários
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CDD_IN = Path("data/processed/cdd_safras.parquet")
PARAMS_PROD = Path("data/processed/parametros_produtividade.json")
PRECIFICACAO_IN = Path("data/processed/precificacao_cdd.json")
FIGS_OUT = Path("output/graficos")
FIGS_OUT.mkdir(parents=True, exist_ok=True)

DPI = 300
AREA_HA = 1_000          # hectares plantados
PROD_BASE = 60           # sacas/ha em safra normal
PRECO_SACA = 140.0       # R$/saca (60kg)
STRIKE = 30.0            # CDD strike da opção (fallback se sem JSON)
NOTIONAL = 30_000         # R$ por ponto de CDD acima do strike

# Carrega parâmetros calibrados do JSON se existir
if PARAMS_PROD.exists():
    try:
        with open(PARAMS_PROD, encoding="utf-8") as f:
            _p = json.load(f)
        _gamma = _p.get("gamma")
        _strike = _p.get("strike")
        _perda = _p.get("perda_maxima")
        if _gamma is not None:
            GAMMA_CALIB = _gamma
        if _strike is not None:
            STRIKE = _strike  # também atualiza strike da opção
        if _perda is not None:
            PERDA_MAX_CALIB = _perda
    except (json.JSONDecodeError, KeyError):
        pass

# Fallback se JSON não tiver os valores
try:
    GAMMA_CALIB
except NameError:
    GAMMA_CALIB = 0.003
try:
    PERDA_MAX_CALIB
except NameError:
    PERDA_MAX_CALIB = 0.50


def carregar():
    cdd = pd.read_parquet(CDD_IN)
    with open(PARAMS_PROD, encoding="utf-8") as f:
        params_prod = json.load(f)
    with open(PRECIFICACAO_IN, encoding="utf-8") as f:
        precos = json.load(f)
    return cdd, params_prod, precos


def produtividade(cdd_arr, gamma=None, strike=None, perda_max=None,
                  y_max=100.0):
    """Função produtividade piecewise: Y = Y_max × (1 - perda)"""
    if gamma is None:
        gamma = GAMMA_CALIB
    if strike is None:
        strike = STRIKE
    if perda_max is None:
        perda_max = PERDA_MAX_CALIB
    excesso = np.maximum(0, cdd_arr - strike)
    fracao_perda = np.minimum(gamma * excesso, perda_max)
    return y_max * (1.0 - fracao_perda)


def prod_real(y_pct):
    """Converte % de produtividade para sacas/ha."""
    return PROD_BASE * y_pct / 100.0


def simular_safra(cdd_regiao, premio_opcao=0):
    """Simula resultado econômico safra a safra para uma região.

    Retorna DataFrame com cada safra: físico, payoff, hedgeado.
    """
    resultados = []
    for _, row in cdd_regiao.iterrows():
        cdd = row["cdd"]
        safra = int(row["safra"])

        # Produtividade física
        y_pct = produtividade(np.array([cdd]))[0]
        sacas_ha = prod_real(y_pct)
        producao = sacas_ha * AREA_HA
        receita_fisica = producao * PRECO_SACA

        # Payoff da opção (se CDD > strike)
        payoff_opcao = max(0, cdd - STRIKE) * NOTIONAL

        # Resultado hedgeado
        receita_hedgeada = receita_fisica + payoff_opcao - premio_opcao

        resultados.append({
            "safra": safra,
            "cdd": cdd,
            "produtividade_pct": y_pct,
            "sacas_ha": sacas_ha,
            "receita_fisica": receita_fisica,
            "payoff": payoff_opcao,
            "premio_pago": premio_opcao,
            "receita_hedgeada": receita_hedgeada,
        })

    return pd.DataFrame(resultados)


def calcular_metricas(df):
    """Calcula métricas de risco do cenário."""
    rf = df["receita_fisica"]
    rh = df["receita_hedgeada"]

    return {
        "receita_media_fisica": rf.mean(),
        "receita_media_hedgeada": rh.mean(),
        "std_fisica": rf.std(ddof=1),
        "std_hedgeada": rh.std(ddof=1),
        "var_95_fisica": rf.quantile(0.05),
        "var_95_hedgeada": rh.quantile(0.05),
        "min_fisica": rf.min(),
        "min_hedgeada": rh.min(),
        "max_fisica": rf.max(),
        "max_hedgeada": rh.max(),
        "efetividade": 1.0 - rh.std(ddof=1) / rf.std(ddof=1),
    }


def plotar_comparacao(resultados, metricas, regiao):
    """Comparação receita física vs hedgeada ao longo das safras."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Série temporal
    ax = axes[0, 0]
    ax.plot(resultados["safra"], resultados["receita_fisica"] / 1e6,
            "-o", markersize=4, linewidth=0.8, label="Física (sem hedge)",
            color="steelblue")
    ax.plot(resultados["safra"], resultados["receita_hedgeada"] / 1e6,
            "-s", markersize=4, linewidth=0.8, label="Hedgeada",
            color="crimson")
    ax.axhline(metricas["receita_media_fisica"] / 1e6, color="steelblue",
               linestyle="--", linewidth=0.6)
    ax.axhline(metricas["receita_media_hedgeada"] / 1e6, color="crimson",
               linestyle="--", linewidth=0.6)
    ax.set_xlabel("Safra")
    ax.set_ylabel("Receita (R$ milhões)")
    ax.set_title(f"Receita por Safra — {regiao}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Payoff da opção
    ax = axes[0, 1]
    payoff_plot = resultados[resultados["payoff"] > 0]
    if len(payoff_plot) > 0:
        ax.bar(payoff_plot["safra"], payoff_plot["payoff"] / 1e6,
               width=0.7, color="forestgreen", alpha=0.7)
    ax.set_xlabel("Safra")
    ax.set_ylabel("Payoff (R$ milhões)")
    ax.set_title("Payoff da Opção CDD (safras com exercício)")
    ax.grid(alpha=0.3)

    # Distribuição: física vs hedgeada
    ax = axes[1, 0]
    ax.hist(resultados["receita_fisica"] / 1e6, bins=15, alpha=0.5,
            color="steelblue", label="Física", edgecolor="white")
    ax.hist(resultados["receita_hedgeada"] / 1e6, bins=15, alpha=0.5,
            color="crimson", label="Hedgeada", edgecolor="white")
    ax.axvline(metricas["var_95_fisica"] / 1e6, color="steelblue",
               linestyle="--", linewidth=1, label=f"VaR físico: R$ {metricas['var_95_fisica']/1e6:.1f}M")
    ax.axvline(metricas["var_95_hedgeada"] / 1e6, color="crimson",
               linestyle="--", linewidth=1, label=f"VaR hedge: R$ {metricas['var_95_hedgeada']/1e6:.1f}M")
    ax.set_xlabel("Receita (R$ milhões)")
    ax.set_ylabel("Frequência")
    ax.set_title("Distribuição da Receita")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Payoff vs CDD
    ax = axes[1, 1]
    ax.scatter(resultados["cdd"], resultados["payoff"] / 1e6, s=30,
               alpha=0.6, color="forestgreen")
    ax.axvline(STRIKE, color="gray", linestyle="--", linewidth=0.8,
               label=f"Strike K={STRIKE}")
    ax.set_xlabel("CDD (°C·dia)")
    ax.set_ylabel("Payoff (R$ milhões)")
    ax.set_title(f"Payoff vs CDD (premium: R$ {resultados['premio_pago'].iloc[0]:.0f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGS_OUT / f"11_hedge_{regiao.lower()}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Figura salva: {path}")


def print_resumo(regiao, m, df):
    """Imprime resumo formatado."""
    print(f"\n  {regiao}:")
    print(f"    Receita física:    média=R$ {m['receita_media_fisica']/1e6:.2f}M  "
          f"std=R$ {m['std_fisica']/1e6:.2f}M  "
          f"VaR95=R$ {m['var_95_fisica']/1e6:.1f}M")
    print(f"    Receita hedgeada:  média=R$ {m['receita_media_hedgeada']/1e6:.2f}M  "
          f"std=R$ {m['std_hedgeada']/1e6:.2f}M  "
          f"VaR95=R$ {m['var_95_hedgeada']/1e6:.1f}M")
    print(f"    Efetividade hedge: {m['efetividade']*100:.1f}%  "
          f"(redução de volatilidade)")
    print(f"    Safras com exercício: {(df['payoff']>0).sum()}/{len(df)}  "
          f"({(df['payoff']>0).mean()*100:.0f}%)")


def main():
    print("Carregando dados...")
    cdd, params_prod, precos = carregar()
    regioes = cdd["regiao"].unique()

    print(f"\n{'='*50}")
    print(f"  SIMULAÇÃO DE HEDGE — Opção CDD K={STRIKE}")
    print(f"{'='*50}")
    print(f"  Área: {AREA_HA:,} ha  |  Prod. base: {PROD_BASE} sacas/ha  |  "
          f"Preço: R$ {PRECO_SACA:.0f}/saca")
    print(f"  Notional opção: R$ {NOTIONAL:,}/ponto CDD")
    print(f"  Nocional opção: R$ {NOTIONAL*100:,}")
    print()

    todos_metrics = {}

    for regiao in regioes:
        premio = precos.get(regiao, {}).get("precificacao", {}).get("premio_justo", 0)

        cdd_regiao = cdd[cdd["regiao"] == regiao].sort_values("safra")
        res = simular_safra(cdd_regiao, premio_opcao=premio)
        metrics = calcular_metricas(res)

        todos_metrics[regiao] = {
            "area_ha": AREA_HA,
            "prod_base_sacas_ha": PROD_BASE,
            "preco_saca": PRECO_SACA,
            "strike_cdd": STRIKE,
            "notional_por_ponto": NOTIONAL,
            "premio_opcao": premio,
            "metrics": {
                "receita_media_fisica": round(metrics["receita_media_fisica"], 2),
                "receita_media_hedgeada": round(metrics["receita_media_hedgeada"], 2),
                "std_fisica": round(metrics["std_fisica"], 2),
                "std_hedgeada": round(metrics["std_hedgeada"], 2),
                "var_95_fisica": round(metrics["var_95_fisica"], 2),
                "var_95_hedgeada": round(metrics["var_95_hedgeada"], 2),
                "min_fisica": round(metrics["min_fisica"], 2),
                "min_hedgeada": round(metrics["min_hedgeada"], 2),
                "efetividade": round(metrics["efetividade"], 4),
            },
        }

        plotar_comparacao(res, metrics, regiao)
        print_resumo(regiao, metrics, res)

    # Salvar resultados
    out_path = Path("data/processed/hedge_simulacao.json")
    out_path.write_text(
        json.dumps(todos_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResultados salvos em {out_path}")

    # Resumo final
    print(f"\n{'='*50}")
    print("  RESUMO — Efetividade do Hedge por Região")
    print(f"{'='*50}")
    print(f"{'Região':<15} {'Prêmio(R$)':>12} {'σ_física(R$)':>14} "
          f"{'σ_hedge(R$)':>14} {'Efetiv.':>8}")
    print("-" * 65)
    for r, rr in sorted(todos_metrics.items()):
        m = rr["metrics"]
        print(f"{r:<15} {rr['premio_opcao']:>12.0f} {m['std_fisica']/1e6:>12.2f}M "
              f"{m['std_hedgeada']/1e6:>10.2f}M {m['efetividade']*100:>7.1f}%")


if __name__ == "__main__":
    main()
