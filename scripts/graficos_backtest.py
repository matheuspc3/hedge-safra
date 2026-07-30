"""
Gera gráficos específicos para o backtest e apresentação 16:9.

Depende de:
  - data/processed/backtest_results.parquet (gerado por backtest_hedge.py)
  - data/processed/temperatura_diaria_MT_PR_GO.parquet

Uso:
    python scripts/graficos_backtest.py
"""
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJETO))
from src.plot_utils import (
    configurar_estilo,
    PALETA_REGIOES,
    COR_AZUL,
    COR_VERDE,
    COR_VERMELHO,
    COR_CINZA,
    COR_BEGE,
    COR_BRANCO,
    COR_PRETA,
    adicionar_fonte,
    salvar_figura,
    formatar_eixo,
)

configurar_estilo("relatorio")

PASTA_GRAFICOS = PROJETO / "output" / "graficos"
PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
PASTA_DADOS = PROJETO / "data" / "processed"

# ── 1. Backtest — Série temporal: receita física vs hedgeada ─
print("\n📊 17_backtest_timeseries (série temporal física vs hedge)...")

bt_path = PASTA_DADOS / "backtest_results.parquet"
if bt_path.exists():
    bt = pd.read_parquet(bt_path)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, facecolor=COR_BEGE)
    fig.suptitle("Backtest Walk-Forward — Receita Física vs Hedgeada",
                 fontsize=14, fontweight="bold", color=COR_PRETA, x=0.10, ha="left")

    for ax, (regiao, df_r) in zip(axes, bt.groupby("regiao")):
        cor = PALETA_REGIOES[regiao]
        ax.set_facecolor(COR_BEGE)
        df_r = df_r.sort_values("safra")

        ax.plot(df_r["safra"], df_r["receita_fisica"] / 1e6,
                color=COR_CINZA, lw=1.5, marker="o", ms=4, label="Física")
        ax.plot(df_r["safra"], df_r["receita_hedgeada"] / 1e6,
                color=cor, lw=2.0, marker="s", ms=4, label="Hedgeada")

        # Destaque para anos de El Niño/La Niña fortes
        eventos = {1998: "El Niño", 2016: "El Niño", 2024: "El Niño",
                   2009: "La Niña", 2011: "La Niña", 2022: "La Niña"}
        for ano, evento in eventos.items():
            if ano in df_r["safra"].values:
                row = df_r[df_r["safra"] == ano].iloc[0]
                ax.axvspan(ano - 0.3, ano + 0.3, color=COR_CINZA, alpha=0.08)
                ax.annotate(evento, xy=(ano, row["receita_fisica"] / 1e6),
                            xytext=(ano, row["receita_fisica"] / 1e6 + 0.6),
                            ha="center", fontsize=6.5, color=COR_CINZA,
                            arrowprops=dict(arrowstyle="->", color=COR_CINZA, lw=0.4))

        ax.set_ylabel("Receita (R$ M)", fontsize=9)
        ax.set_title(regiao.replace("_", " "), fontsize=10, loc="left",
                     color=cor, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"R${v:.0f}M"))

    adicionar_fonte(axes[-1])
    plt.tight_layout()
    salvar_figura(fig, "17_backtest_timeseries", pasta=PASTA_GRAFICOS)
    plt.close()

    # ── 2. Payoff histórico — barras ────────────────────────
    print("📊 18_backtest_payoffs (distribuição histórica dos payoffs)...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 5.5), sharex=True, facecolor=COR_BEGE)
    fig.suptitle("Payoffs Históricos do Hedge CDD — Backtest Walk-Forward",
                 fontsize=13, fontweight="bold", color=COR_PRETA, x=0.10, ha="left")

    for ax, (regiao, df_r) in zip(axes, bt.groupby("regiao")):
        cor = PALETA_REGIOES[regiao]
        ax.set_facecolor(COR_BEGE)
        df_r = df_r.sort_values("safra")

        cores_barra = [cor if p > 0 else COR_CINZA for p in df_r["payoff"]]
        ax.bar(df_r["safra"], df_r["payoff"] / 1e3, color=cores_barra,
               alpha=0.7, width=0.7, edgecolor=COR_BRANCO, lw=0.3)

        # Linha do prêmio médio
        premio_medio = df_r["premio_pago"].mean() / 1e3
        ax.axhline(premio_medio, color=COR_VERMELHO, ls="--", lw=0.7, alpha=0.6)
        ax.text(df_r["safra"].max() + 0.5, premio_medio + 1,
                f"Prêmio médio R${premio_medio:.0f}k", fontsize=6.5,
                color=COR_VERMELHO, va="bottom")

        ax.set_ylabel("Payoff (R$ mil)", fontsize=9)
        ax.set_title(regiao.replace("_", " "), fontsize=10, loc="left",
                     color=cor, fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"R${v:.0f}k"))

        # Hit rate
        hit = (df_r["payoff"] > 0).mean() * 100
        ax.text(0.02, 0.95, f"Hit rate: {hit:.0f}%",
                transform=ax.transAxes, fontsize=9, color=cor,
                va="top", bbox=dict(facecolor=COR_BRANCO, alpha=0.7, ec="none", pad=2))

    adicionar_fonte(axes[-1])
    plt.tight_layout()
    salvar_figura(fig, "18_backtest_payoffs", pasta=PASTA_GRAFICOS)
    plt.close()

else:
    print("  ⚠ backtest_results.parquet não encontrado. Pulei gráficos 17 e 18.")


# ── 3. Diagrama conceitual — fluxo da CDD option ────────────
print("📊 19_diagrama_cdd (fluxo conceitual CDD option)...")

fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=COR_BEGE)
ax.set_xlim(0, 1200)
ax.set_ylim(0, 350)
ax.axis("off")
ax.set_facecolor(COR_BEGE)

fig.suptitle("Como Funciona uma CDD Option",
             fontsize=14, fontweight="bold", color=COR_PRETA, x=0.08, ha="left", y=0.98)


def caixa(ax, x, y, w, h, texto, cor_fundo, cor_texto=COR_BRANCO, fs=9):
    """Desenha uma caixa com texto centralizado."""
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                    facecolor=cor_fundo, edgecolor="none", alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, texto, fontsize=fs, color=cor_texto,
            ha="center", va="center", fontweight="bold")


def seta(ax, x1, y1, x2, y2, cor=COR_CINZA):
    """Desenha uma seta entre dois pontos."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=cor, lw=1.5))


# Caixas do fluxo
caixa(ax, 30, 220, 140, 50, "🌡️ Temperatura\ndiária DEZ-FEV", COR_AZUL)
caixa(ax, 230, 220, 140, 50, "CDD acumulado\nΣ max(0, T-25°C)", COR_VERDE)
caixa(ax, 430, 220, 140, 50, "Compara com\nStrike K", COR_LARANJA := "#E8751A")
caixa(ax, 630, 220, 140, 50, "Payoff\nmax(0, CDD-K)×N", COR_VERMELHO)
caixa(ax, 830, 220, 140, 50, "Receita\nHedgeada", COR_AZUL)

# Setas nível 1
seta(ax, 170, 245, 230, 245)
seta(ax, 370, 245, 430, 245)
seta(ax, 570, 245, 630, 245)
seta(ax, 770, 245, 830, 245)

# Linha do tempo inferior — safra física
caixa(ax, 30, 80, 200, 50, "🌱 Plantio\n(OUT-NOV)", COR_VERDE)
caixa(ax, 280, 80, 200, 50, "⏳ Enchimento\nde grãos (DEZ-FEV)", COR_LARANJA)
caixa(ax, 530, 80, 200, 50, "🌾 Colheita\n(MAR-MAI)", COR_VERDE)
caixa(ax, 780, 80, 200, 50, "💰 Venda da\nsafra", COR_AZUL)

# Setas nível 2
seta(ax, 230, 105, 280, 105)
seta(ax, 480, 105, 530, 105)
seta(ax, 730, 105, 780, 105)

# Conexão entre os dois níveis
ax.plot([300, 300], [170, 130], color=COR_CINZA, lw=1.0, ls="--")
ax.plot([700, 700], [170, 130], color=COR_CINZA, lw=1.0, ls="--")

# Anotação
ax.text(300, 155, "CDD determina\nprodutividade", fontsize=7, color=COR_CINZA,
        ha="center", va="center", fontstyle="italic")
ax.text(700, 155, "Hedge CDD\ncompensa perda", fontsize=7, color=COR_CINZA,
        ha="center", va="center", fontstyle="italic")

# Nota explicativa
ax.text(30, 15, "A opção CDD é liquidada financeiramente com base no CDD observado — sem perícia, sem risco moral.",
        fontsize=8, color=COR_CINZA, fontstyle="italic")

adicionar_fonte(ax)
salvar_figura(fig, "19_diagrama_cdd", pasta=PASTA_GRAFICOS)
plt.close()


# ── 4. Infográfico comparativo: seguro agrícola vs CDD option ─
print("📊 20_comparativo_seguro (seguro agrícola vs CDD option)...")

fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=COR_BEGE)
ax.set_xlim(0, 1200)
ax.set_ylim(0, 350)
ax.axis("off")
ax.set_facecolor(COR_BEGE)

fig.suptitle("Seguro Agrícola vs CDD Option — Comparativo",
             fontsize=14, fontweight="bold", color=COR_PRETA, x=0.08, ha="left")

# Cabeçalhos
ax.text(100, 290, "Seguro Agrícola (Proagro)", fontsize=11,
        color=COR_VERMELHO, ha="center", fontweight="bold")
ax.text(700, 290, "CDD Option (HEATGUARD)", fontsize=11,
        color=COR_VERDE, ha="center", fontweight="bold")

# Indicadores
indicadores = [
    ("Prêmio (% valor segurado)", "10–15%", "7,3–8,7%", 3),
    ("Inadimplência / risco moral", "≥ 30%", "0% (liquidação objetiva)", 1),
    ("Cobertura", "Perda total (multirrisco)", "Estresse térmico específico", 0),
    ("Vistoria / perícia", "Obrigatória (15–30 dias)", "Automática (CDD da estação)", 2),
    ("Prazo de indenização", "30–90 dias após vistoria", "Liquidação em D+1 do CDD", 2),
    ("Lastro", "Governo federal (Tesouro)", "Mercado de capitais / OTC", 0),
]

for i, (label, seguro, cdd, melhor) in enumerate(indicadores):
    y = 245 - i * 38
    ax.text(30, y + 5, label, fontsize=8, color=COR_PRETA, fontweight="bold")
    cor_seg = COR_VERMELHO if melhor == 3 else COR_CINZA
    cor_cdd = COR_VERDE if melhor in (1, 2) else COR_CINZA
    ax.text(100, y - 8, seguro, fontsize=8.5, color=cor_seg, ha="center")
    ax.text(700, y - 8, cdd, fontsize=8.5, color=cor_cdd, ha="center")

    # Barra visual
    ax.plot([30, 540], [y - 12, y - 12], color=COR_BEGE, lw=0.5)

# Selo "VANTAGEM" para CDD
ax.text(1030, 290, "★ VANTAGEM", fontsize=9, color=COR_VERDE,
        fontweight="bold", style="italic")

# Fonte
ax.text(30, 10, "Fontes: Proagro (BC), MAPA. CDD option: simulação HEATGUARD com parâmetros calibrados (1980–2025).",
        fontsize=6, color=COR_CINZA, fontstyle="italic")

adicionar_fonte(ax)
salvar_figura(fig, "20_comparativo_seguro", pasta=PASTA_GRAFICOS)
plt.close()

print("\n✅ Todos os gráficos de backtest e apresentação gerados!")
