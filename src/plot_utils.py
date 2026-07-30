"""
plot_utils.py — Estilo profissional para gráficos financeiros

Uso:
    from src.plot_utils import *
    # Agora plt.rcParams está configurado
    # Use as funções helpers para gráficos prontos

Inspirado em: Bloomberg, Itaú BBA, Valor Econômico
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paleta Itaú-inspired ──────────────────────────────────
# Azul institucional + verdes/vermelhos de retorno
COR_AZUL = '#005CA9'          # Itaú azul
COR_AZUL_CLARO = '#4A90D9'
COR_AZUL_ESCURO = '#003366'
COR_VERDE = '#00995D'         # Retorno positivo
COR_VERMELHO = '#CC3333'      # Retorno negativo
COR_CINZA = '#888888'
COR_CINZA_CLARO = '#E8E8E8'
COR_CINZA_FUNDO = '#F5F5F5'
COR_BEGE = '#F7F3EB'          # Fundo de relatório
COR_PRETA = '#1A1A1A'
COR_BRANCO = '#FFFFFF'

PALETA_PRINCIPAL = [
    '#005CA9',  # Azul Itaú
    '#00995D',  # Verde
    '#CC3333',  # Vermelho
    '#F5A623',  # Laranja
    '#7B68EE',  # Roxo
    '#50B5C8',  # Ciano
    '#D4A056',  # Ouro
    '#8B8B8B',  # Cinza
]

PALETA_REGIOES = {
    'Sorriso_MT': '#CC3333',
    'Londrina_PR': '#005CA9',
    'Rio Verde_GO': '#00995D',
}

PALETA_CATEGORICA_6 = [
    '#005CA9', '#00995D', '#CC3333',
    '#F5A623', '#7B68EE', '#50B5C8',
]

# ── Configuração global de estilo ─────────────────────────
def configurar_estilo(tema='relatorio'):
    """
    Aplica estilo profissional.

    tema='relatorio' → fundo claro, sério, alta densidade de informação
    tema='apresentacao' → fundo escuro, contraste alto
    tema='bloomberg' → fundo azul escuro
    """

    if tema == 'relatorio':
        plt.rcParams.update({

            # Figura
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'savefig.transparent': False,

            # Fonte
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Calibri'],
            'font.size': 10,
            'font.weight': 'normal',

            # Eixos
            'axes.facecolor': COR_BEGE,
            'axes.edgecolor': COR_CINZA,
            'axes.linewidth': 0.6,
            'axes.grid': True,
            'axes.grid.axis': 'y',
            'axes.grid.which': 'major',
            'axes.labelcolor': COR_PRETA,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'axes.titleweight': 'bold',
            'axes.titlecolor': COR_PRETA,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,

            # Grid
            'grid.alpha': 0.3,
            'grid.color': COR_CINZA,
            'grid.linestyle': '-',
            'grid.linewidth': 0.4,

            # Ticks
            'xtick.color': COR_CINZA,
            'ytick.color': COR_CINZA,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'xtick.major.size': 3,
            'ytick.major.size': 3,
            'xtick.major.width': 0.5,
            'ytick.major.width': 0.5,

            # Legenda
            'legend.fontsize': 9,
            'legend.frameon': True,
            'legend.facecolor': COR_BRANCO,
            'legend.edgecolor': COR_CINZA_CLARO,
            'legend.framealpha': 0.95,
            'legend.loc': 'best',

            # Linhas
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
            'lines.markeredgewidth': 0.5,

            # Barras
            'patch.facecolor': COR_AZUL,
            'patch.edgecolor': COR_BRANCO,
            'patch.linewidth': 0.3,
            'patch.force_edgecolor': False,

            # Boxplot
            'boxplot.whiskerprops.color': COR_CINZA,
            'boxplot.capprops.color': COR_CINZA,
            'boxplot.boxprops.color': COR_AZUL,
            'boxplot.medianprops.color': COR_VERMELHO,
        })

        plt.style.use('default')

    elif tema == 'bloomberg':
        # Fundo azul escuro (Bloomberg Terminal)
        plt.rcParams.update({
            'figure.facecolor': '#1A2332',
            'axes.facecolor': '#1A2332',
            'axes.edgecolor': '#3A4A6B',
            'axes.labelcolor': '#CCCCCC',
            'axes.titlecolor': '#FFFFFF',
            'text.color': '#CCCCCC',
            'grid.color': '#2A3A5B',
            'xtick.color': '#8899AA',
            'ytick.color': '#8899AA',
            'legend.facecolor': '#1A2332',
            'legend.edgecolor': '#3A4A6B',
            'legend.labelcolor': '#CCCCCC',
        })


# ── Helpers ───────────────────────────────────────────────

def formatar_eixo(valor, pos):
    """Formata grandes números: 1.5M, 2.3B, etc."""
    if abs(valor) >= 1e9:
        return f'R${valor/1e9:.1f}B'
    elif abs(valor) >= 1e6:
        return f'R${valor/1e6:.1f}M'
    elif abs(valor) >= 1e3:
        return f'R${valor/1e3:.0f}K'
    else:
        return f'R${valor:.0f}'


def adicionar_fonte(ax, texto='Fonte: NASA POWER / IBGE. Elaboração própria.'):
    """Adiciona nota de fonte no canto inferior."""
    ax.text(0, -0.15, texto, transform=ax.transAxes,
            fontsize=7, color=COR_CINZA, ha='left', va='top')


def adicionar_anotacao(ax, x, y, texto, **kwargs):
    """Anotação estilizada com seta."""
    defaults = dict(
        xy=(x, y), xytext=(x, y + 0.08 * (max(ax.get_ylim()) - min(ax.get_ylim()))),
        ha='center', fontsize=8, color=COR_PRETA,
        arrowprops=dict(arrowstyle='->', color=COR_CINZA, lw=0.7),
        bbox=dict(boxstyle='round,pad=0.2', facecolor=COR_BRANCO,
                  edgecolor=COR_CINZA_CLARO, alpha=0.9)
    )
    defaults.update(kwargs)
    ax.annotate(texto, **defaults)


def salvar_figura(fig, nome, pasta=None):
    """Salva com nome padronizado e metadados."""
    if pasta is None:
        pasta = Path(__file__).resolve().parent.parent / 'output' / 'graficos'
    else:
        pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f'{nome}.png'
    fig.savefig(caminho, dpi=300, facecolor=fig.get_facecolor())
    print(f'  ✅ {caminho}')
    return caminho


# ── Gráficos prontos ─────────────────────────────────────

def grafico_serie_temporal(df, coluna='t2m', **kwargs):
    """
    Série temporal limpa, estilo Bloomberg.

    Exemplo:
        fig, ax = grafico_serie_temporal(df)
        fig.savefig('output/graficos/serie_temp.png', dpi=300)
    """
    defaults = dict(
        figsize=(12, 5),
        cor=COR_AZUL,
        media_moveis=[30, 90],
        titulo='',
        ylabel='',
    )
    defaults.update(kwargs)

    fig, ax = plt.subplots(figsize=defaults['figsize'],
                           facecolor=COR_BEGE)
    ax.set_facecolor(COR_BEGE)

    # Linha fina (dados diários)
    ax.plot(df.index, df[coluna], color=defaults['cor'],
            alpha=0.1, linewidth=0.3)

    # Médias móveis
    cores_mm = ['#E74C3C', '#2C3E50']
    for i, janela in enumerate(defaults['media_moveis']):
        suav = df[coluna].rolling(janela, center=True).mean()
        ax.plot(df.index, suav, color=cores_mm[i],
                linewidth=1.2, label=f'Média {janela}d')

    # Média geral
    media = df[coluna].mean()
    ax.axhline(media, color=COR_VERMELHO, linestyle='--',
               linewidth=0.7, alpha=0.5, label=f'Média: {media:.1f}')

    ax.set_title(defaults['titulo'], loc='left', pad=15)
    ax.set_ylabel(defaults['ylabel'])
    ax.legend(loc='upper right')
    adicionar_fonte(ax)

    plt.tight_layout()
    return fig, ax


def grafico_barras_agrupadas(df, x, valores, grupos, **kwargs):
    """
    Barras agrupadas estilo relatório de research.

    Exemplo:
        fig, ax = grafico_barras_agrupadas(df, 'regiao', 'efetividade', 'param')
    """
    defaults = dict(
        figsize=(10, 5),
        titulo='',
        ylabel='',
        paleta=PALETA_PRINCIPAL,
    )
    defaults.update(kwargs)

    fig, ax = plt.subplots(figsize=defaults['figsize'],
                           facecolor=COR_BEGE)
    ax.set_facecolor(COR_BEGE)

    grupos_unicos = df[grupos].unique()
    x_unicos = df[x].unique()
    n_grupos = len(grupos_unicos)
    n_x = len(x_unicos)
    largura = 0.7 / n_grupos

    for i, grupo in enumerate(grupos_unicos):
        sub = df[df[grupos] == grupo]
        posicoes = np.arange(n_x) + i * largura - 0.35 + largura/2
        vals = [sub[sub[x] == x_val][valores].values[0]
                if len(sub[sub[x] == x_val][valores].values) > 0
                else 0 for x_val in x_unicos]
        bars = ax.bar(posicoes, vals, largura, label=grupo,
                      color=defaults['paleta'][i % len(defaults['paleta'])],
                      edgecolor=COR_BRANCO, linewidth=0.3)

        # Data labels
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{v:.0%}' if v <= 1 else f'{v:.1f}',
                        ha='center', va='bottom', fontsize=7, color=COR_CINZA)

    ax.set_xticks(np.arange(n_x))
    ax.set_xticklabels(x_unicos)
    ax.set_title(defaults['titulo'], loc='left', pad=15)
    ax.set_ylabel(defaults['ylabel'])
    ax.legend(loc='upper right')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:.0%}' if v <= 1 else f'{v:.0f}')
    )

    plt.tight_layout()
    return fig, ax


def grafico_heatmap_correlacao(corr, **kwargs):
    """
    Heatmap de correlação no estilo dos relatórios do Itaú.

    Exemplo:
        fig, ax = grafico_heatmap_correlacao(df.corr())
    """
    defaults = dict(figsize=(9, 7), titulo='Matriz de Correlação', cmap='RdBu_r')
    defaults.update(kwargs)

    fig, ax = plt.subplots(figsize=defaults['figsize'],
                           facecolor=COR_BEGE)
    ax.set_facecolor(COR_BEGE)

    n = len(corr)
    im = ax.imshow(corr.values, cmap=defaults['cmap'], vmin=-1, vmax=1,
                   aspect='auto', interpolation='none')

    # Anota cada célula
    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            cor_texto = COR_BRANCO if abs(val) > 0.5 else COR_PRETA
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=cor_texto, fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title(defaults['titulo'], loc='left', pad=15)

    plt.colorbar(im, ax=ax, shrink=0.75, label='Correlação')
    plt.tight_layout()
    return fig, ax


# ── Decorator para aplicar estilo automaticamente ────────
def aplicar_estilo(tema='relatorio'):
    """Decorator que aplica estilo antes de executar a função de plot."""
    def decorador(func):
        def wrapper(*args, **kwargs):
            configurar_estilo(tema)
            return func(*args, **kwargs)
        return wrapper
    return decorador


if __name__ == '__main__':
    # Teste
    configurar_estilo('relatorio')
    print('✅ Estilo configurado. Use configurar_estilo() para ativar.')
    print(f'   Paletas disponíveis: {len(PALETA_PRINCIPAL)} cores')
