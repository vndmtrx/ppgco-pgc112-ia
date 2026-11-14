"""Gráficos de treino do MLP (um módulo só, com o nome da perda como parâmetro).

Gera os mesmos 5 arquivos das linhas v4.2 e v5:
`<id>_perda.png`, `_acuracia.png`, `_gap_generalizacao.png`, `_tempo_execucao.png` e `_painel_geral.png`.
O ponto vermelho marca a época de menor perda de validação, que é a do checkpoint salvo.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COR_TREINO = "#1f77b4"
COR_VAL = "#ff7f0e"


def _salvar(fig, caminho: Path) -> Path:
    fig.tight_layout()
    fig.savefig(caminho, dpi=300)
    plt.close(fig)
    return caminho


def _curva_perda(ax, epocas: List[int], tr: List[float], vl: List[float], nome_perda: str) -> None:
    ax.plot(epocas, tr, label="Treino (50k)", color=COR_TREINO, linewidth=2)
    ax.plot(epocas, vl, label="Validação (10k)", color=COR_VAL, linewidth=2, linestyle="--")
    i = int(np.argmin(vl))
    ax.scatter(epocas[i], vl[i], color="red", s=60, zorder=5,
               label=f"Mín Val: {vl[i]:.4f} (Época {epocas[i]}, checkpoint salvo)")
    ax.set_xlabel("Época")
    ax.set_ylabel(f"Perda ({nome_perda})")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()


def _curva_acuracia(ax, epocas: List[int], tr: List[float], vl: List[float]) -> None:
    ax.plot(epocas, tr, label="Treino (50k)", color="#2ca02c", linewidth=2)
    ax.plot(epocas, vl, label="Validação (10k)", color="#d62728", linewidth=2, linestyle="--")
    i = int(np.argmax(vl))
    ax.scatter(epocas[i], vl[i], color="darkred", s=60, zorder=5,
               label=f"Máx Val: {vl[i]:.2f}% (Época {epocas[i]})")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")


def _curva_gap(ax, epocas: List[int], gap: List[float]) -> None:
    ax.plot(epocas, gap, color="#9467bd", linewidth=2, label="Gap: Acc(Treino) - Acc(Val)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.8)
    ax.fill_between(epocas, 0, gap, color="#9467bd", alpha=0.2)
    ax.set_xlabel("Época")
    ax.set_ylabel("Gap (pontos percentuais)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()


def _curva_tempo(ax, epocas: List[int], tempos: List[float]) -> None:
    media = float(np.mean(tempos))
    ax.plot(epocas, tempos, color="#8c564b", linewidth=1.5, marker="o", markersize=3, label="Tempo real")
    ax.axhline(media, color="black", linestyle="--", label=f"Média: {media:.2f} s/época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Segundos por época")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()


def salvar_graficos_treino(
    historico: dict,
    pasta_destino: Path,
    identificador: str,
    titulo: str,
    nome_perda: str,
    prefixo_modelo: str = "MLP",
) -> Dict[str, Path]:
    """Salva os 5 gráficos de treino/validação em `pasta_destino` (300 DPI)."""
    pasta_destino.mkdir(parents=True, exist_ok=True)
    ep = historico["epocas"]
    acc_tr = [a * 100 for a in historico["acc_treino"]]
    acc_vl = [a * 100 for a in historico["acc_val"]]
    gap = [t - v for t, v in zip(acc_tr, acc_vl)]
    cabecalho = f"{prefixo_modelo} ({titulo})"
    caminhos: Dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    _curva_perda(ax, ep, historico["perda_treino"], historico["perda_val"], nome_perda)
    ax.set_title(f"{cabecalho}: Curva de Perda", fontsize=11, fontweight="bold")
    caminhos["perda"] = _salvar(fig, pasta_destino / f"{identificador}_perda.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    _curva_acuracia(ax, ep, acc_tr, acc_vl)
    ax.set_title(f"{cabecalho}: Acurácia", fontsize=11, fontweight="bold")
    caminhos["acuracia"] = _salvar(fig, pasta_destino / f"{identificador}_acuracia.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    _curva_gap(ax, ep, gap)
    ax.set_title(f"{cabecalho}: Gap de Generalização", fontsize=11, fontweight="bold")
    caminhos["gap"] = _salvar(fig, pasta_destino / f"{identificador}_gap_generalizacao.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    _curva_tempo(ax, ep, historico["tempo_epoca"])
    ax.set_title(f"{cabecalho}: Estabilidade Temporal", fontsize=11, fontweight="bold")
    caminhos["tempo"] = _salvar(fig, pasta_destino / f"{identificador}_tempo_execucao.png")

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    _curva_perda(axs[0, 0], ep, historico["perda_treino"], historico["perda_val"], nome_perda)
    axs[0, 0].set_title("Curva de Perda")
    _curva_acuracia(axs[0, 1], ep, acc_tr, acc_vl)
    axs[0, 1].set_title("Curva de Acurácia (%)")
    _curva_gap(axs[1, 0], ep, gap)
    axs[1, 0].set_title("Gap de Generalização (Treino - Validação)")
    _curva_tempo(axs[1, 1], ep, historico["tempo_epoca"])
    axs[1, 1].set_title("Tempo de Execução por Época")
    fig.suptitle(f"Painel Integrado: {cabecalho}", fontsize=13, fontweight="bold")
    caminhos["painel"] = _salvar(fig, pasta_destino / f"{identificador}_painel_geral.png")

    print(f"\n[GRÁFICOS] 5 relatórios salvos em: {pasta_destino}")
    for p in caminhos.values():
        print(f"  - {p.name}")
    print()
    return caminhos
