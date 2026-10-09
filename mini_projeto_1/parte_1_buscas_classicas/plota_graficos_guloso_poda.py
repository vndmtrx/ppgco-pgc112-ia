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


def suavizar_sinal(x: np.ndarray, y: np.ndarray, janela: int = 51) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Suaviza o sinal usando uma média ponderada com janela de Hanning e calcula bandas IQR."""
    if len(y) <= janela:
        return y, y, y

    # Garante tamanho ímpar para simetria
    if janela % 2 == 0:
        janela += 1

    pad = janela // 2
    y_pad = np.pad(y, pad, mode="edge")

    # Janela Hanning normalizada para transições ultra-suaves
    kernel = np.hanning(janela)
    kernel = kernel / kernel.sum()

    y_suave = np.convolve(y_pad, kernel, mode="valid")

    # Calcula desvio móvel / bandas de dispersão local
    # Janela deslizante para percentis
    y_p25 = np.zeros_like(y)
    y_p75 = np.zeros_like(y)

    for i in range(len(y)):
        inicio = max(0, i - pad)
        fim = min(len(y), i + pad + 1)
        sub = y[inicio:fim]
        y_p25[i] = np.percentile(sub, 25)
        y_p75[i] = np.percentile(sub, 75)

    y_p25_suave = np.convolve(np.pad(y_p25, pad, mode="edge"), kernel, mode="valid")
    y_p75_suave = np.convolve(np.pad(y_p75, pad, mode="edge"), kernel, mode="valid")

    return y_suave, y_p25_suave, y_p75_suave


def plotar_graficos_poda(
    csv_poda: str = "bench/bench_guloso_poda.csv",
    csv_canonico: str = "bench/bench_guloso.csv",
    dir_saida: str = "bench",
    janela_suavizacao: int = 75,
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

    l_seq = np.array([d["tamanho"] for d in seq_poda])
    t_seq = np.array([d["tempo_ms"] for d in seq_poda])
    nos_seq = np.array([d["nos_expandidos"] for d in seq_poda])

    l_torch = np.array([d["tamanho"] for d in torch_poda])
    t_torch = np.array([d["tempo_ms"] for d in torch_poda])
    nos_torch = np.array([d["nos_expandidos"] for d in torch_poda])

    cor_seq = "#E65100"    # Laranja
    cor_torch = "#1565C0"  # Azul
    cor_poda = "#2E7D32"   # Verde

    denso = len(l_seq) > 50

    if denso:
        t_seq_suave, t_seq_p25, t_seq_p75 = suavizar_sinal(l_seq, t_seq, janela=janela_suavizacao)
        nos_seq_suave, nos_p25, nos_p75 = suavizar_sinal(l_seq, nos_seq, janela=janela_suavizacao)
        if len(l_torch) > 0:
            t_torch_suave, t_torch_p25, t_torch_p75 = suavizar_sinal(l_torch, t_torch, janela=janela_suavizacao)
    else:
        t_seq_suave, t_seq_p25, t_seq_p75 = t_seq, t_seq, t_seq
        nos_seq_suave, nos_p25, nos_p75 = nos_seq, nos_seq, nos_seq
        if len(l_torch) > 0:
            t_torch_suave, t_torch_p25, t_torch_p75 = t_torch, t_torch, t_torch

    # 1. TEMPO DE EXECUÇÃO (Linear vs Log)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    if denso:
        # Fundo com os pontos originais atenuados
        ax1.scatter(l_seq, t_seq, color=cor_seq, alpha=0.08, s=3, rasterized=True)
        if len(l_torch) > 0:
            ax1.scatter(l_torch, t_torch, color=cor_torch, alpha=0.08, s=3, rasterized=True)
        # Faixa sombreada de dispersão (IQR)
        ax1.fill_between(l_seq, t_seq_p25, t_seq_p75, color=cor_seq, alpha=0.15, label="Faixa Interquartil (IQR)")
        # Curva de tendência suavizada
        ax1.plot(l_seq, t_seq_suave, "-", color=cor_seq, label="Sequencial (Tendência Suave)", lw=2.5)
        if len(l_torch) > 0:
            ax1.plot(l_torch, t_torch_suave, "-", color=cor_torch, label="PyTorch SIMD (Tendência Suave)", lw=2.5)
    else:
        ax1.plot(l_seq, t_seq, "o-", color=cor_seq, label="Sequencial (Poda)", lw=2, markersize=5)
        if len(l_torch) > 0:
            ax1.plot(l_torch, t_torch, "s-", color=cor_torch, label="PyTorch SIMD (Poda)", lw=2, markersize=5)

    ax1.set_title("Escala Linear (Tempo de Execução)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax1.set_ylabel("Tempo Total (ms)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    if denso:
        ax2.fill_between(l_seq, np.maximum(1e-3, t_seq_p25), np.maximum(1e-3, t_seq_p75), color=cor_seq, alpha=0.15)
        ax2.plot(l_seq, np.maximum(1e-3, t_seq_suave), "-", color=cor_seq, label="Sequencial (Tendência Suave)", lw=2.5)
        if len(l_torch) > 0:
            ax2.plot(l_torch, np.maximum(1e-3, t_torch_suave), "-", color=cor_torch, label="PyTorch SIMD (Tendência Suave)", lw=2.5)
    else:
        ax2.plot(l_seq, np.maximum(1e-3, t_seq), "o-", color=cor_seq, label="Sequencial (Poda)", lw=2, markersize=5)
        if len(l_torch) > 0:
            ax2.plot(l_torch, np.maximum(1e-3, t_torch), "s-", color=cor_torch, label="PyTorch SIMD (Poda)", lw=2, markersize=5)

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

    if denso:
        ax1.scatter(l_seq, nos_seq, color=cor_poda, alpha=0.08, s=3, rasterized=True)
        ax1.fill_between(l_seq, nos_p25, nos_p75, color=cor_poda, alpha=0.15, label="Faixa Interquartil (IQR)")
        ax1.plot(l_seq, nos_seq_suave, "-", color=cor_poda, label="Nós Expandidos (Tendência Suave)", lw=2.5)
    else:
        ax1.plot(l_seq, nos_seq, "o-", color=cor_poda, label="Nós Expandidos (Poda)", lw=2, markersize=5)

    ax1.set_title("Escala Linear (Nós Expandidos)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Comprimento da Fita (L)", fontsize=11)
    ax1.set_ylabel("Nós Expandidos", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    if denso:
        ax2.fill_between(l_seq, np.maximum(1, nos_p25), np.maximum(1, nos_p75), color=cor_poda, alpha=0.15)
        ax2.plot(l_seq, np.maximum(1, nos_seq_suave), "-", color=cor_poda, label="Nós Expandidos (Tendência Suave)", lw=2.5)
    else:
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
        l_canon = np.array([d["tamanho"] for d in seq_canon])
        t_canon = np.array([d["tempo_ms"] for d in seq_canon])

        fig, ax2 = plt.subplots(figsize=(7, 4.8), dpi=300)

        ax2.plot(l_canon, np.maximum(1e-3, t_canon), "o--", color="#D32F2F", label="Guloso Canônico (Sem Poda)", lw=2, markersize=5)
        if denso:
            ax2.plot(l_seq, np.maximum(1e-3, t_seq_suave), "-", color="#2E7D32", label="Guloso com Poda (Tendência Suave)", lw=2.5)
        else:
            ax2.plot(l_seq, np.maximum(1e-3, t_seq), "o-", color="#2E7D32", label="Guloso com Poda de Ações", lw=2, markersize=5)

        ax2.set_yscale("log")
        ax2.set_title("Tempo de Execução: Com vs. Sem Poda (Escala Logarítmica)", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Comprimento da Fita ($L$)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Tempo Total (ms) [Escala Log]", fontsize=11, fontweight="bold")
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)
        ax2.legend(frameon=True, fontsize=10)

        plt.tight_layout()
        f3 = os.path.join(dir_saida, "grafico_guloso_comparativo_com_e_sem_poda.png")
        fig.savefig(f3)
        plt.close(fig)
        print(f"Salvo (apenas log): {f3}")


def main():
    parser = argparse.ArgumentParser(description="Gera gráficos comparativos para o Guloso com Poda.")
    parser.add_argument("--csv-poda", type=str, default="bench/bench_guloso_poda.csv")
    parser.add_argument("--csv-canonico", type=str, default="bench/bench_guloso.csv")
    parser.add_argument("--dir-saida", type=str, default="bench")
    parser.add_argument("--janela", type=int, default=75, help="Janela de suavização para dados densos.")

    args = parser.parse_args()
    plotar_graficos_poda(
        csv_poda=args.csv_poda,
        csv_canonico=args.csv_canonico,
        dir_saida=args.dir_saida,
        janela_suavizacao=args.janela,
    )


if __name__ == "__main__":
    main()
