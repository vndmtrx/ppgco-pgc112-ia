"""Geração de Gráficos Estatísticos para o Algoritmo Guloso com Poda de Ações.

Gera 4 figuras comparativas de alta resolução (300 DPI) em 'bench/':
1. grafico_guloso_poda_tempo.png: Tempo de Execução vs L (Linear vs Logarítmico).
2. grafico_guloso_poda_nos_expandidos.png: Nós Expandidos vs L.
3. grafico_guloso_poda_throughput.png: Throughput (nós/s) vs L.
4. grafico_guloso_comparativo_com_e_sem_poda.png: Comparação Direta (Guloso Canônico vs Guloso com Poda).
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ler_csv(caminho: str) -> list[dict]:
    if not os.path.exists(caminho):
        return []
    registros = []
    with open(caminho, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            registros.append(
                {
                    "tamanho": int(row["tamanho"]),
                    "estrategia": row["estrategia"],
                    "tempo_ms": float(row["tempo_ms"]),
                    "tempo_s": float(row["tempo_s"]),
                    "nos_expandidos": int(row["nos_expandidos"]),
                    "custo_solucao": int(row["custo_solucao"]),
                    "throughput_nos_s": float(row["throughput_nos_s"]),
                }
            )
    return registros


def plotar_graficos_poda(
    csv_poda: str = "bench/bench_guloso_poda.csv",
    csv_canonico: str = "bench/bench_guloso.csv",
    dir_saida: str = "bench",
):
    Path(dir_saida).mkdir(parents=True, exist_ok=True)
    dados_poda = ler_csv(csv_poda)

    if not dados_poda:
        print(f"Erro: Arquivo '{csv_poda}' não encontrado ou vazio.")
        return

    seq_poda = [d for d in dados_poda if "Sequencial" in d["estrategia"]]
    torch_poda = [d for d in dados_poda if "PyTorch" in d["estrategia"]]

    seq_poda.sort(key=lambda x: x["tamanho"])
    torch_poda.sort(key=lambda x: x["tamanho"])

    l_seq = [d["tamanho"] for d in seq_poda]
    t_seq = [d["tempo_ms"] for d in seq_poda]
    nos_seq = [d["nos_expandidos"] for d in seq_poda]
    tp_seq = [d["throughput_nos_s"] for d in seq_poda]

    l_torch = [d["tamanho"] for d in torch_poda]
    t_torch = [d["tempo_ms"] for d in torch_poda]
    nos_torch = [d["nos_expandidos"] for d in torch_poda]
    tp_torch = [d["throughput_nos_s"] for d in torch_poda]

    cor_seq = "#E65100"    # Laranja
    cor_torch = "#1565C0"  # Azul
    cor_poda = "#2E7D32"   # Verde

    # 1. TEMPO DE EXECUÇÃO (Linear vs Log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    ax1.plot(l_seq, t_seq, "o-", color=cor_seq, label="Sequencial (Poda)", lw=2, markersize=5)
    if l_torch:
        ax1.plot(l_torch, t_torch, "s-", color=cor_torch, label="PyTorch SIMD (Poda)", lw=2, markersize=5)
    ax1.set_title("Escala Linear (Tempo de Execução)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax1.set_ylabel("Tempo Total (ms)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    ax2.plot(l_seq, [max(1e-3, t) for t in t_seq], "o-", color=cor_seq, label="Sequencial (Poda)", lw=2, markersize=5)
    if l_torch:
        ax2.plot(l_torch, [max(1e-3, t) for t in t_torch], "s-", color=cor_torch, label="PyTorch SIMD (Poda)", lw=2, markersize=5)
    ax2.set_yscale("log")
    ax2.set_title("Escala Logarítmica (Tempo de Execução)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax2.set_ylabel("Tempo Total (ms) [Log]", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True)

    fig.suptitle("Algoritmo Guloso com Poda de Ações (h2): Tempo vs. L (até L=4096)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    f1 = os.path.join(dir_saida, "grafico_guloso_poda_tempo.png")
    fig.savefig(f1)
    plt.close(fig)
    print(f"Salvo: {f1}")

    # 2. NÓS EXPANDIDOS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    ax1.plot(l_seq, nos_seq, "o-", color=cor_poda, label="Nós Expandidos (Poda)", lw=2, markersize=5)
    ax1.set_title("Escala Linear (Nós Expandidos)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax1.set_ylabel("Nós Expandidos", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    ax2.plot(l_seq, nos_seq, "o-", color=cor_poda, label="Nós Expandidos (Poda)", lw=2, markersize=5)
    ax2.set_yscale("log")
    ax2.set_title("Escala Logarítmica (Nós Expandidos)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax2.set_ylabel("Nós Expandidos [Log]", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True)

    fig.suptitle("Algoritmo Guloso com Poda de Ações (h2): Quantidade de Nós Expandidos vs. L", fontsize=14, fontweight="bold")
    plt.tight_layout()
    f2 = os.path.join(dir_saida, "grafico_guloso_poda_nos_expandidos.png")
    fig.savefig(f2)
    plt.close(fig)
    print(f"Salvo: {f2}")

    # 3. COMPARATIVO DIRETO: GULOSO CANÔNICO VS GULOSO COM PODA
    dados_canon = ler_csv(csv_canonico)
    if dados_canon:
        seq_canon = [d for d in dados_canon if d["estrategia"] == "Sequencial"]
        seq_canon.sort(key=lambda x: x["tamanho"])
        l_canon = [d["tamanho"] for d in seq_canon]
        t_canon = [d["tempo_ms"] for d in seq_canon]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
        ax1.plot(l_canon, t_canon, "o--", color="#D32F2F", label="Guloso Canônico (Sem Poda)", lw=2, markersize=5)
        ax1.plot(l_seq, t_seq, "o-", color="#2E7D32", label="Guloso com Poda de Ações", lw=2, markersize=5)
        ax1.set_title("Comparação Direta - Escala Linear", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Comprimento da Fita (L)", fontsize=11)
        ax1.set_ylabel("Tempo Total (ms)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(frameon=True)

        ax2.plot(l_canon, [max(1e-3, t) for t in t_canon], "o--", color="#D32F2F", label="Guloso Canônico (Sem Poda)", lw=2, markersize=5)
        ax2.plot(l_seq, [max(1e-3, t) for t in t_seq], "o-", color="#2E7D32", label="Guloso com Poda de Ações", lw=2, markersize=5)
        ax2.set_yscale("log")
        ax2.set_title("Comparação Direta - Escala Logarítmica", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Comprimento da Fita (L)", fontsize=11)
        ax2.set_ylabel("Tempo Total (ms) [Escala Log]", fontsize=11)
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)
        ax2.legend(frameon=True)

        fig.suptitle("Impacto da Poda de Ações: Guloso Canônico vs. Guloso com Poda (até L=4096)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        f3 = os.path.join(dir_saida, "grafico_guloso_comparativo_com_e_sem_poda.png")
        fig.savefig(f3)
        plt.close(fig)
        print(f"Salvo: {f3}")


def main():
    parser = argparse.ArgumentParser(description="Gera gráficos comparativos para o Guloso com Poda.")
    parser.add_argument("--csv-poda", type=str, default="bench/bench_guloso_poda.csv")
    parser.add_argument("--csv-canonico", type=str, default="bench/bench_guloso.csv")
    parser.add_argument("--dir-saida", type=str, default="bench")

    args = parser.parse_args()
    plotar_graficos_poda(
        csv_poda=args.csv_poda,
        csv_canonico=args.csv_canonico,
        dir_saida=args.dir_saida,
    )


if __name__ == "__main__":
    main()
