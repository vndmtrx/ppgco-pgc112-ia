"""Script de Geração de Gráficos Comparativos (Linear vs Logarítmico) para o Algoritmo Guloso.

Gera 4 figuras em alta resolução (300 DPI) salvas diretamente no diretório 'bench/':
1. grafico_guloso_tempo.png: Tempo de Execução vs L (Linear vs Logarítmico).
2. grafico_guloso_nos_expandidos.png: Quantidade de Nós Expandidos vs L.
3. grafico_guloso_throughput.png: Vazão de Processamento (nós/s) vs L.
4. grafico_guloso_custo_solucao.png: Custo da Solução Encontrada vs L.

Padrão de Cores:
- Sequencial: Tons de Laranja (#E65100, #F57C00)
- PyTorch SIMD: Tons de Azul (#1565C0, #1976D2)
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def carregar_dados_csv(caminho_csv: str) -> list[dict]:
    """Lê o arquivo CSV de benchmark do Algoritmo Guloso."""
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(
            f"Arquivo de benchmark não encontrado em '{caminho_csv}'. "
            "Execute primeiramente: python bench_guloso.py"
        )

    registros = []
    with open(caminho_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            registros.append(
                {
                    "tamanho": int(row["tamanho"]),
                    "tipo_instancia": row.get("tipo_instancia", "blocos"),
                    "heuristica": row.get("heuristica", "h2"),
                    "estrategia": row["estrategia"],
                    "tempo_ms": float(row["tempo_ms"]),
                    "tempo_s": float(row["tempo_s"]),
                    "nos_expandidos": int(row["nos_expandidos"]),
                    "nos_visitados": int(row["nos_visitados"]),
                    "custo_solucao": int(row["custo_solucao"]),
                    "throughput_nos_s": float(row["throughput_nos_s"]),
                }
            )
    return registros


def plotar_paineis(caminho_csv: str = "bench/bench_guloso.csv", dir_saida: str = "bench"):
    """Gera gráficos comparativos individuais lado a lado (Linear vs Logarítmico)."""
    Path(dir_saida).mkdir(parents=True, exist_ok=True)
    dados = carregar_dados_csv(caminho_csv)

    # Separação por estratégia
    seq_dados = [d for d in dados if d["estrategia"] == "Sequencial"]
    torch_dados = [d for d in dados if d["estrategia"] == "PyTorch SIMD"]

    seq_dados.sort(key=lambda x: x["tamanho"])
    torch_dados.sort(key=lambda x: x["tamanho"])

    l_seq = [d["tamanho"] for d in seq_dados]
    t_ms_seq = [d["tempo_ms"] for d in seq_dados]
    nos_seq = [d["nos_expandidos"] for d in seq_dados]
    tp_seq = [d["throughput_nos_s"] for d in seq_dados]
    custo_seq = [d["custo_solucao"] for d in seq_dados]

    l_torch = [d["tamanho"] for d in torch_dados]
    t_ms_torch = [d["tempo_ms"] for d in torch_dados]
    nos_torch = [d["nos_expandidos"] for d in torch_dados]
    tp_torch = [d["throughput_nos_s"] for d in torch_dados]
    custo_torch = [d["custo_solucao"] for d in torch_dados]

    cor_seq = "#E65100"  # Laranja
    cor_torch = "#1565C0"  # Azul

    # -------------------------------------------------------------
    # 1. TEMPO DE EXECUÇÃO (Linear vs Log)
    # -------------------------------------------------------------
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Linear
    if l_seq:
        ax_lin.plot(l_seq, t_ms_seq, "o-", color=cor_seq, label="Sequencial (CPU)", lw=2, markersize=5)
    if l_torch:
        ax_lin.plot(l_torch, t_ms_torch, "s-", color=cor_torch, label="PyTorch SIMD", lw=2, markersize=5)
    ax_lin.set_title("Escala Linear (Tempo de Execução)", fontsize=12, fontweight="bold")
    ax_lin.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax_lin.set_ylabel("Tempo Total (ms)", fontsize=11)
    ax_lin.grid(True, linestyle="--", alpha=0.5)
    ax_lin.legend(frameon=True, facecolor="white")

    # Logarítmico
    if l_seq:
        ax_log.plot(l_seq, [max(1e-3, t) for t in t_ms_seq], "o-", color=cor_seq, label="Sequencial (CPU)", lw=2, markersize=5)
    if l_torch:
        ax_log.plot(l_torch, [max(1e-3, t) for t in t_ms_torch], "s-", color=cor_torch, label="PyTorch SIMD", lw=2, markersize=5)
    ax_log.set_yscale("log")
    ax_log.set_title("Escala Logarítmica (Tempo de Execução)", fontsize=12, fontweight="bold")
    ax_log.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax_log.set_ylabel("Tempo Total (ms) [Escala Log]", fontsize=11)
    ax_log.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_log.legend(frameon=True, facecolor="white")

    fig.suptitle("Algoritmo Guloso (h2): Tempo de Execução vs. Comprimento da Fita (L)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out1 = os.path.join(dir_saida, "grafico_guloso_tempo.png")
    fig.savefig(out1)
    plt.close(fig)
    print(f"Salvo: {out1}")

    # -------------------------------------------------------------
    # 2. NÓS EXPANDIDOS (Linear vs Log)
    # -------------------------------------------------------------
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    if l_seq:
        ax_lin.plot(l_seq, nos_seq, "o-", color=cor_seq, label="Sequencial", lw=2, markersize=5)
    if l_torch:
        ax_lin.plot(l_torch, nos_torch, "s--", color=cor_torch, label="PyTorch SIMD", lw=1.5, markersize=4)
    ax_lin.set_title("Escala Linear (Nós Expandidos)", fontsize=12, fontweight="bold")
    ax_lin.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax_lin.set_ylabel("Nós Expandidos", fontsize=11)
    ax_lin.grid(True, linestyle="--", alpha=0.5)
    ax_lin.legend(frameon=True, facecolor="white")

    if l_seq:
        ax_log.plot(l_seq, [max(1, n) for n in nos_seq], "o-", color=cor_seq, label="Sequencial", lw=2, markersize=5)
    if l_torch:
        ax_log.plot(l_torch, [max(1, n) for n in nos_torch], "s--", color=cor_torch, label="PyTorch SIMD", lw=1.5, markersize=4)
    ax_log.set_yscale("log")
    ax_log.set_title("Escala Logarítmica (Nós Expandidos)", fontsize=12, fontweight="bold")
    ax_log.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax_log.set_ylabel("Nós Expandidos [Escala Log]", fontsize=11)
    ax_log.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_log.legend(frameon=True, facecolor="white")

    fig.suptitle("Algoritmo Guloso (h2): Quantidade de Nós Expandidos vs. L", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out2 = os.path.join(dir_saida, "grafico_guloso_nos_expandidos.png")
    fig.savefig(out2)
    plt.close(fig)
    print(f"Salvo: {out2}")

    # -------------------------------------------------------------
    # 3. THROUGHPUT (Nós / segundo)
    # -------------------------------------------------------------
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    if l_seq:
        ax_lin.plot(l_seq, tp_seq, "o-", color=cor_seq, label="Sequencial (CPU)", lw=2, markersize=5)
    if l_torch:
        ax_lin.plot(l_torch, tp_torch, "s-", color=cor_torch, label="PyTorch SIMD", lw=2, markersize=5)
    ax_lin.set_title("Escala Linear (Throughput)", fontsize=12, fontweight="bold")
    ax_lin.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax_lin.set_ylabel("Throughput (nós/segundo)", fontsize=11)
    ax_lin.grid(True, linestyle="--", alpha=0.5)
    ax_lin.legend(frameon=True, facecolor="white")

    if l_seq:
        ax_log.plot(l_seq, [max(1.0, tp) for tp in tp_seq], "o-", color=cor_seq, label="Sequencial (CPU)", lw=2, markersize=5)
    if l_torch:
        ax_log.plot(l_torch, [max(1.0, tp) for tp in tp_torch], "s-", color=cor_torch, label="PyTorch SIMD", lw=2, markersize=5)
    ax_log.set_yscale("log")
    ax_log.set_title("Escala Logarítmica (Throughput)", fontsize=12, fontweight="bold")
    ax_log.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax_log.set_ylabel("Throughput (nós/s) [Escala Log]", fontsize=11)
    ax_log.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_log.legend(frameon=True, facecolor="white")

    fig.suptitle("Algoritmo Guloso (h2): Vazão de Processamento (Throughput) vs. L", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out3 = os.path.join(dir_saida, "grafico_guloso_throughput.png")
    fig.savefig(out3)
    plt.close(fig)
    print(f"Salvo: {out3}")

    # -------------------------------------------------------------
    # 4. CUSTO DA SOLUÇÃO (Qualidade da Rota)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    if l_seq:
        ax.plot(l_seq, custo_seq, "o-", color=cor_seq, label="Sequencial", lw=2, markersize=6)
    if l_torch:
        ax.plot(l_torch, custo_torch, "s--", color=cor_torch, label="PyTorch SIMD", lw=1.5, markersize=5)
    ax.set_title("Custo da Solução Encontrada (Número de Passos) vs. L", fontsize=12, fontweight="bold")
    ax.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax.set_ylabel("Custo da Solução (passos)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white")

    plt.tight_layout()
    out4 = os.path.join(dir_saida, "grafico_guloso_custo_solucao.png")
    fig.savefig(out4)
    plt.close(fig)
    print(f"Salvo: {out4}")


def main():
    parser = argparse.ArgumentParser(
        description="Gera gráficos estatísticos de escalabilidade do Algoritmo Guloso."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="bench/bench_guloso.csv",
        help="Caminho do arquivo CSV de benchmark.",
    )
    parser.add_argument(
        "--dir-saida",
        type=str,
        default="bench",
        help="Diretório de saída dos gráficos PNG.",
    )

    args = parser.parse_args()
    plotar_paineis(caminho_csv=args.csv, dir_saida=args.dir_saida)


if __name__ == "__main__":
    main()
