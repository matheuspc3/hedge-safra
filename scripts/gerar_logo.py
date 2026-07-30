"""
Gera a identidade visual do HEATGUARD — logo + paleta + tagline.

Uso:
    python scripts/gerar_logo.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGS_OUT = Path("output/graficos")
FIGS_OUT.mkdir(parents=True, exist_ok=True)

# Paleta HEATGUARD
HG_AZUL = "#003057"       # azul profundo — solidez, hedge
HG_VERDE = "#2E8B57"      # verde soja
HG_LARANJA = "#E8751A"    # laranja calor/alerta
HG_CINZA = "#8C8C8C"      # cinza neutro
HG_BEGE = "#F5F0EB"       # fundo


def gerar_logo():
    """Gera logo HEATGUARD: termômetro com espiga de soja + texto."""
    fig = plt.figure(figsize=(6, 3), facecolor=HG_BEGE)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=HG_BEGE)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 300)
    ax.axis("off")

    # ── Termômetro ────────────────────────────────────────
    # Bulbo (inferior)
    circulo = plt.Circle((120, 80), 35, color=HG_LARANJA, zorder=3)
    ax.add_patch(circulo)
    # Haste do termômetro
    haste = plt.Rectangle((115, 80), 10, 120, color=HG_LARANJA, zorder=3)
    ax.add_patch(haste)
    # Topo arredondado
    topo = plt.Circle((120, 200), 8, color=HG_LARANJA, zorder=3)
    ax.add_patch(topo)
    # Linhas de escala
    for y, label in [(100, ""), (130, ""), (160, "CDD"), (190, "")]:
        ax.plot([90, 105], [y, y], color=HG_CINZA, lw=1.0, zorder=2)
        if label:
            ax.text(85, y, label, fontsize=7, color=HG_CINZA, ha="right", va="center")

    # ── Espiga de soja ────────────────────────────────────
    # Haste curva
    t = np.linspace(0, 1, 50)
    haste_x = 160 + 40 * t
    haste_y = 60 + 140 * t ** 0.7
    ax.plot(haste_x, haste_y, color=HG_VERDE, lw=2.5, zorder=2)

    # Folhas
    ax.plot([165, 155], [80, 60], color=HG_VERDE, lw=1.5, zorder=2)
    ax.plot([175, 190], [100, 85], color=HG_VERDE, lw=1.5, zorder=2)
    ax.plot([185, 200], [140, 130], color=HG_VERDE, lw=1.5, zorder=2)

    # Grãos (círculos verdes)
    sementes = [
        (175, 95, 5), (185, 110, 5), (192, 125, 5),
        (197, 140, 5), (200, 155, 4), (200, 168, 4),
        (197, 180, 3), (192, 190, 3),
    ]
    for sx, sy, sr in sementes:
        ax.add_patch(plt.Circle((sx, sy), sr, color=HG_VERDE, ec="#1D6B3E", lw=0.5, zorder=3))

    # ── Texto HEATGUARD ───────────────────────────────────
    ax.text(220, 210, "HEAT", fontsize=38, fontweight="bold",
            color=HG_LARANJA, fontfamily="sans-serif", zorder=4)
    ax.text(220, 160, "GUARD", fontsize=38, fontweight="bold",
            color=HG_AZUL, fontfamily="sans-serif", zorder=4)

    # Tagline
    ax.text(222, 130, "Protegendo a safra brasileira", fontsize=9,
            color=HG_CINZA, fontfamily="sans-serif", zorder=4)
    ax.text(222, 118, "do estresse térmico", fontsize=9,
            color=HG_CINZA, fontfamily="sans-serif", zorder=4)

    # ── Explicação do nome (canto inferior direito) ───────
    legenda = (
        "HEAT (calor, CDD) + GUARD (proteção):"
        "\nDerivativo climático que protege o produtor"
        "\ncontra o calor excessivo na safra de soja."
    )
    ax.text(320, 50, legenda, fontsize=6.5, color=HG_CINZA,
            fontfamily="sans-serif", linespacing=1.4, zorder=4)

    salvar_figura(fig, "heatguard_logo", pasta=FIGS_OUT)
    plt.close()
    print("📊 Logo HEATGUARD salva em output/graficos/heatguard_logo.png")


def paleta_heatguard():
    """Gera painel da paleta de cores HEATGUARD."""
    fig, ax = plt.subplots(figsize=(5, 1.5), facecolor=HG_BEGE)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 100)
    ax.axis("off")

    cores = [
        (HG_AZUL, "Azul\nHedge"),
        (HG_VERDE, "Verde\nSoja"),
        (HG_LARANJA, "Laranja\nAlerta"),
        (HG_CINZA, "Cinza\nNeutro"),
        (HG_BEGE, "Bege\nFundo"),
    ]

    for i, (cor, nome) in enumerate(cores):
        x0 = 20 + i * 95
        ret = plt.Rectangle((x0, 20), 70, 50, color=cor, ec="white", lw=1)
        ax.add_patch(ret)
        ax.text(x0 + 35, 85, nome, fontsize=7, ha="center",
                color=HG_CINZA, fontfamily="sans-serif")

    fig.savefig(FIGS_OUT / "heatguard_paleta.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("📊 Paleta HEATGUARD salva em output/graficos/heatguard_paleta.png")


def salvar_figura(fig, nome, pasta):
    """Salva figura com alta resolução."""
    fig.savefig(pasta / f"{nome}.png", dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())


if __name__ == "__main__":
    gerar_logo()
    paleta_heatguard()
    print("\n✅ Identidade visual HEATGUARD gerada com sucesso!")
