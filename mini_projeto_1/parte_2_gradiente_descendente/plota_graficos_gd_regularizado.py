"""Geração de Gráficos Comparativos: Gradiente Descendente Regularizado.

Gera visualizações científicas de alta qualidade para a dinâmica acoplada J(x) = L(x) + lambda * C(x):
1. Gráfico 04: Heatmap temporal comparando o perfil dos 23 bits para lambda in {0.0, 0.5, 2.0} (eta = 0.1).
   Evidencia o comportamento de blocos contíguos (onda elástica / amortecimento conjunto) vs bits independentes.
2. Gráfico 05: Decomposição da função de perda J(t) = L(t) + lambda * C(t) ao longo das iterações,
   mostrando o trade-off entre erro em relação ao chão L(t) e tensão entre vizinhos C(t).
3. Gráfico 06: Mapa de Estabilidade Numérica / Diagrama de Fase (eta vs lambda),
   destacando a fronteira crítica teórica eta_crit = 1 / (2 + 4*lambda) e a região de 'chicotada elástica' (explosão numérica).
"""

import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torch

from gd_regularizado import gradiente_descendente_regularizado

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


def gerar_grafico_heatmaps_regularizados():
    """Gera heatmaps comparando a evolução temporal da fita de 23 bits com diferentes lambdas."""
    lambdas = [0.0, 0.5, 2.0]
    titulos = [
        r"$\lambda = 0.0$ (Sem Regularização: Bits Independentes)",
        r"$\lambda = 0.5$ (Acoplamento Moderado: Ondas de Bloco)",
        r"$\lambda = 2.0$ (Forte Tensão Elástica: Coesão Rápida)",
    ]

    passos_vis = 20
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, lam, titulo in zip(axes, lambdas, titulos):
        res = gradiente_descendente_regularizado(
            FITA_ENUNCIADO, ALVO_ENUNCIADO, eta=0.1, lam=lam, max_passos=passos_vis
        )
        # Matriz: linhas = passos (0 a passos_vis), colunas = posições da fita (0 a 22)
        traj = np.array(res["trajetoria"][:passos_vis + 1])
        
        # Mapa de calor
        im = ax.imshow(traj, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0, origin="upper")
        ax.set_title(titulo, fontsize=11, pad=10)
        ax.set_xlabel("Posição na Fita (0 a 22)")
        ax.set_xticks(range(0, 23, 2))
        ax.grid(False)

    axes[0].set_ylabel("Passo de Otimização ($t$)")
    axes[0].set_yticks(range(0, passos_vis + 1, 2))

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Ativação Contínua $x_i \in [0, 1]$", rotation=270, labelpad=15)

    caminho = os.path.join(DIRETORIO_GRAFICOS, "04_heatmaps_comparativo_lambdas_23bits.png")
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"-> Gráfico salvo: {caminho}")


def gerar_grafico_decomposicao_perdas():
    """Gera curvas de decomposição da perda total J(t) = L(t) + lambda * C(t)."""
    lambdas = [0.0, 0.5, 2.0]
    cores = ["#1f77b4", "#2ca02c", "#d62728"]
    passos_max = 25

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    for lam, cor in zip(lambdas, cores):
        res = gradiente_descendente_regularizado(
            FITA_ENUNCIADO, ALVO_ENUNCIADO, eta=0.1, lam=lam, max_passos=passos_max
        )
        passos = list(range(1, len(res["perdas_totais"]) + 1))

        # 1. Perda Total J(t)
        ax1.plot(passos, res["perdas_totais"], marker="o", markersize=4, color=cor,
                 label=rf"$\lambda = {lam}$")
        
        # 2. Perda de Alvo / Chão L(t)
        ax2.plot(passos, res["perdas_chao"], marker="s", markersize=4, color=cor,
                 label=rf"$\lambda = {lam}$")

        # 3. Tensão da Corda / Cisalhamento C(t)
        ax3.plot(passos, res["perdas_corda"], marker="^", markersize=4, color=cor,
                 label=rf"$\lambda = {lam}$")

    ax1.set_title(r"Perda Total $J(\mathbf{x}) = \mathcal{L} + \lambda \mathcal{C}$")
    ax1.set_xlabel("Passo ($t$)")
    ax1.set_ylabel(r"$J(\mathbf{x})$")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.set_title(r"Erro até o Alvo $\mathcal{L}(\mathbf{x}) = \sum (x_k - y_k)^2$")
    ax2.set_xlabel("Passo ($t$)")
    ax2.set_ylabel(r"$\mathcal{L}(\mathbf{x})$")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    ax3.set_title(r"Tensão Espacial $\mathcal{C}(\mathbf{x}) = \sum (x_{k+1} - x_k)^2$")
    ax3.set_xlabel("Passo ($t$)")
    ax3.set_ylabel(r"$\mathcal{C}(\mathbf{x})$")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    caminho = os.path.join(DIRETORIO_GRAFICOS, "05_decomposicao_perdas_regularizacao.png")
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"-> Gráfico salvo: {caminho}")


def gerar_grafico_estabilidade_fase():
    """Gera o diagrama de fase empírico e teórico: fronteira de estabilidade vs chicotada elástica."""
    etas = np.linspace(0.01, 0.45, 45)
    lams = np.linspace(0.0, 15.0, 45)

    # Malha para testar estabilidade numérica na fita do enunciado em 15 passos
    estabilidade = np.zeros((len(lams), len(etas)))

    for i, lam_val in enumerate(lams):
        for j, eta_val in enumerate(etas):
            res = gradiente_descendente_regularizado(
                FITA_ENUNCIADO, ALVO_ENUNCIADO, eta=eta_val, lam=lam_val, max_passos=15
            )
            # 1 = Divergiu / explodiu, 0 = Estável
            if res["divergiu"] or np.max(res["trajetoria"]) > 5.0 or np.isnan(res["perdas_totais"][-1]):
                estabilidade[i, j] = 1.0  # Região instável
            else:
                estabilidade[i, j] = 0.0  # Região convergente

    plt.figure(figsize=(9, 6))

    # Curva teórica de corte: eta_crit = 1.0 / (2.0 + 4.0 * lambda)
    lam_teorico = np.linspace(0.0, 15.0, 200)
    eta_teorico = 1.0 / (2.0 + 4.0 * lam_teorico)

    # Plot do mapa empírico
    E, L = np.meshgrid(etas, lams)
    plt.contourf(E, L, estabilidade, levels=[-0.5, 0.5, 1.5], colors=["#e8f5e9", "#ffebee"], alpha=0.85)

    # Linha da cota teórica
    plt.plot(eta_teorico, lam_teorico, "r--", linewidth=2.5,
             label=r"Fronteira Teórica Crítica: $\eta_{\mathrm{crit}} = \frac{1}{2 + 4\lambda}$")

    # Pontos de destaque explorados no laboratório
    plt.scatter([0.1], [0.0], color="blue", s=90, zorder=5, label=r"Teste $\lambda=0.0, \eta=0.1$ (Estável)")
    plt.scatter([0.1], [0.5], color="green", s=90, zorder=5, label=r"Teste $\lambda=0.5, \eta=0.1$ (Estável)")
    plt.scatter([0.1], [10.0], color="darkred", s=90, marker="x", linewidth=3, zorder=5,
                label=r"Teste $\lambda=10.0, \eta=0.1$ (Chicotada Elástica / Divergente)")

    plt.title(r"Diagrama de Fase: Estabilidade Numérica e Acoplamento ($\eta \times \lambda$)", fontsize=13)
    plt.xlabel(r"Taxa de Aprendizado ($\eta$)", fontsize=12)
    plt.ylabel(r"Peso da Regularização ($\lambda$)", fontsize=12)
    plt.xlim(0.01, 0.45)
    plt.ylim(0.0, 15.0)
    plt.grid(True, linestyle=":", alpha=0.6)

    # Anotações nas regiões
    plt.text(0.04, 3.0, "ZONA DE CONVERGÊNCIA\nESTÁVEL", color="darkgreen",
             fontweight="bold", fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8))
    plt.text(0.22, 10.0, "ZONA DE DIVERGÊNCIA\n(CHICOTADA ELÁSTICA)", color="darkred",
             fontweight="bold", fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

    plt.legend(loc="upper right", framealpha=0.95)

    caminho = os.path.join(DIRETORIO_GRAFICOS, "06_diagrama_fase_estabilidade_eta_lambda.png")
    plt.savefig(caminho, dpi=300)
    plt.close()
    print(f"-> Gráfico salvo: {caminho}")


if __name__ == "__main__":
    print("Iniciando geração de gráficos do Gradiente Descendente Regularizado...")
    gerar_grafico_heatmaps_regularizados()
    gerar_grafico_decomposicao_perdas()
    gerar_grafico_estabilidade_fase()
    print("Todos os gráficos de GD Regularizado foram gerados com sucesso!")
