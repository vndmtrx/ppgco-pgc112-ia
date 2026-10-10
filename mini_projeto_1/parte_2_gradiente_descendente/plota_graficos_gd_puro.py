"""Geração de Gráficos Comparativos: Gradiente Descendente Puro.

Gera visualizações científicas de alta qualidade para a Parte 2:
1. Comparação topológica 3D (Bacia de Erro e Curvas de Nível) com trajetórias.
2. Curvas de decaimento da perda L(t) comparando eta in {0.1, 0.2, 0.4, 0.5, 0.6}.
3. Mapas de Calor (Heatmaps) mostrando a evolução temporal dos 23 bits da fita do enunciado.
"""

import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torch

from gd_puro import gradiente_descendente_puro

# Configuração visual de padrão acadêmico
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
})

DIRETORIO_GRAFICOS = "graficos"
os.makedirs(DIRETORIO_GRAFICOS, exist_ok=True)

FITA_ENUNCIADO = "00001001000011001100111"
ALVO_ENUNCIADO = "00000000000000000000000"


def gerar_grafico_bacia_3d():
    """Gera o gráfico 3D da bacia de potencial e curvas de nível para 2 dimensões."""
    x1_grid = np.linspace(-1.25, 1.25, 120)
    x2_grid = np.linspace(-1.25, 1.25, 120)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    Z = X1**2 + X2**2

    def coletar_trajetoria_2d(eta: float, passos: int = 15):
        res = gradiente_descendente_puro([1.0, 1.0], [0.0, 0.0], eta=eta, max_passos=passos)
        pts = np.array(res["trajetoria"])
        alturas = [p[0]**2 + p[1]**2 for p in pts]
        return pts, np.array(alturas)

    pts_02, z_02 = coletar_trajetoria_2d(0.2, passos=12)
    pts_06, z_06 = coletar_trajetoria_2d(0.6, passos=8)

    fig = plt.figure(figsize=(15, 6))

    # Subplot 1: Vista 3D
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X1, X2, Z, cmap="Blues", alpha=0.35, edgecolor="none")
    ax1.plot(pts_02[:, 0], pts_02[:, 1], z_02, "ro-", linewidth=2, markersize=5, label=r"Estável ($\eta = 0.2$)")
    ax1.plot(pts_06[:, 0], pts_06[:, 1], z_06, "g^--", linewidth=1.5, markersize=6, label=r"Overshoot ($\eta = 0.6$)")
    ax1.set_xlabel(r"Bit $x_1$")
    ax1.set_ylabel(r"Bit $x_2$")
    ax1.set_zlabel(r"Perda $\mathcal{L}(x)$")
    ax1.set_title("Superfície da Bacia de Erro em 3D")
    ax1.legend(loc="upper right")
    ax1.view_init(elev=32, azim=220)

    # Subplot 2: Curvas de Nível
    ax2 = fig.add_subplot(1, 2, 2)
    contorno = ax2.contour(X1, X2, Z, levels=14, cmap="Blues")
    ax2.clabel(contorno, inline=True, fontsize=8)
    ax2.plot(pts_02[:, 0], pts_02[:, 1], "ro-", linewidth=2, markersize=6, label=r"$\eta = 0.2$ (Convergência monótona)")
    ax2.plot(pts_06[:, 0], pts_06[:, 1], "g^--", linewidth=1.5, markersize=6, label=r"$\eta = 0.6$ (Overshoot com oscilação)")
    ax2.plot(0, 0, "k*", markersize=14, label=r"Alvo: Origem $(0, 0)$")
    ax2.set_xlabel(r"Bit $x_1$")
    ax2.set_ylabel(r"Bit $x_2$")
    ax2.set_title("Visão Superior (Curvas de Nível / Topografia)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    caminho = os.path.join(DIRETORIO_GRAFICOS, "01_bacia_potencial_3d.png")
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"-> Gráfico salvo: {caminho}")


def gerar_grafico_curvas_erro():
    """Gera a comparação das curvas de perda L(t) para eta in {0.1, 0.2, 0.4, 0.5, 0.6}."""
    taxas_eta = [0.1, 0.2, 0.4, 0.5, 0.6]
    cores = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    estilos = ["-", "-", "-", "-", "--"]

    plt.figure(figsize=(10, 6))

    for eta, cor, estilo in zip(taxas_eta, cores, estilos):
        res = gradiente_descendente_puro(FITA_ENUNCIADO, ALVO_ENUNCIADO, eta=eta, max_passos=35)
        perdas = res["perdas"]
        passos = list(range(len(perdas)))

        label = rf"$\eta = {eta}$"
        if eta == 0.5:
            label += " (Passo crítico perfeito)"
        elif eta == 0.6:
            label += " (Overshoot / Oscilação)"

        plt.plot(passos, perdas, marker="o" if eta in [0.5, 0.6] else None,
                 markersize=4, color=cor, linestyle=estilo, linewidth=2, label=label)

    plt.xlabel("Iteração / Passo ($t$)")
    plt.ylabel(r"Função de Perda $\mathcal{L}(\mathbf{x}^{(t)}) = \sum (x_k - y_k)^2$")
    plt.title(r"Trajetória da Função de Perda para Diferentes Taxas $\eta$ (Fita Canônica $L=23$)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.yscale("log")  # Escala logarítmica para ver claramente o coelho de Zenão até 1e-6
    plt.ylabel(r"Perda $\mathcal{L}(\mathbf{x}^{(t)})$ (Escala Logarítmica)")

    caminho = os.path.join(DIRETORIO_GRAFICOS, "02_convergencia_perda_etas.png")
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"-> Gráfico salvo: {caminho}")


def gerar_grafico_heatmaps():
    """Gera mapas de calor no tempo mostrando a evolução dos 23 bits para cada eta em subplots separados."""
    taxas_eta = [0.1, 0.2, 0.4, 0.5, 0.6]
    
    fig, axes = plt.subplots(len(taxas_eta), 1, figsize=(14, 11), sharex=True)

    for idx, (ax, eta) in enumerate(zip(axes, taxas_eta)):
        # Executa 20 passos para todos para termos a mesma janela temporal
        res = gradiente_descendente_puro(FITA_ENUNCIADO, ALVO_ENUNCIADO, eta=eta, max_passos=20)
        trajetoria = np.array(res["trajetoria"][:21])  # 21 linhas (passo 0 até 20)

        # Mapa de calor: linhas = passos, colunas = 23 bits
        # Colormap com centro em 0: azul (negativo), branco (zero), vermelho (positivo)
        im = ax.imshow(trajetoria, aspect="auto", cmap="coolwarm", vmin=-0.3, vmax=1.0, interpolation="nearest")

        ax.set_ylabel(f"Passo\n($\eta={eta}$)", fontsize=10)
        ax.set_yticks([0, 5, 10, 15, 20])

        if idx == 0:
            ax.set_title(f"Evolução Temporal dos 23 Bits da Fita Canônica (Heatmaps Comparativos)", fontsize=13)

    axes[-1].set_xlabel("Índice do Bit na Fita ($k = 0 \dots 22$)", fontsize=11)
    axes[-1].set_xticks(range(23))

    # Barra de cores lateral única
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Valor do Bit $x_k \in [-0.3, 1.0]$")

    caminho = os.path.join(DIRETORIO_GRAFICOS, "03_heatmaps_evolucao_temporal_23bits.png")
    plt.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"-> Gráfico salvo: {caminho}")


if __name__ == "__main__":
    print("Gerando suíte completa de gráficos para Gradiente Descendente Puro...")
    gerar_grafico_bacia_3d()
    gerar_grafico_curvas_erro()
    gerar_grafico_heatmaps()
    print("Todos os gráficos foram gerados com sucesso na pasta 'graficos/'!")
