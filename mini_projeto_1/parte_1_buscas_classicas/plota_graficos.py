"""Script de Benchmark Randômico e Geração de Gráficos Estatísticos Individuais (BFS).

Gera e salva 4 figuras individuais limpas em alta resolução (300 DPI) com foco em Ponto Central (Média Geométrica e IQR):
1. grafico_bfs_tempo_tendencia.png: Tempo de Execução de Caso Médio por L (Sequencial em laranja vs. PyTorch em azul).
2. grafico_bfs_nos_expandidos.png: Complexidade Estrutural no Grafo (Nós Expandidos por L).
3. grafico_bfs_throughput.png: Vazão de Processamento em Nós/segundo por L (Sequencial em laranja vs. PyTorch em azul).
4. grafico_bfs_profundidade.png: Complexidade Exponencial no Grafo por Profundidade Ótima da Meta (d*).

Limpa automaticamente gráficos anteriores a cada execução e reutiliza o CSV existente
quando executado sem parâmetros.
"""

import argparse
import csv
import glob
import os
import random
import time
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from bfs_sequencial import resolver_bfs_sequencial
from bfs_torch import resolver_bfs_torch


def calcular_media_geometrica(valores: list[float]) -> float:
    """Calcula a Média Geométrica segura: exp(mean(log(x)))."""
    arr = np.array(valores, dtype=np.float64)
    arr = np.maximum(arr, 1e-6)
    return float(np.exp(np.mean(np.log(arr))))


def calcular_trimean_tukey(valores: list[float]) -> float:
    """Calcula o Trimean de Tukey: (Q1 + 2*Mediana + Q3) / 4."""
    q1 = np.percentile(valores, 25)
    mediana = np.percentile(valores, 50)
    q3 = np.percentile(valores, 75)
    return float((q1 + 2.0 * mediana + q3) / 4.0)


def gerar_fita_aleatoria(tamanho: int) -> str:
    """Gera uma fita binária aleatória uniforme de comprimento L."""
    bits = [random.choice("01") for _ in range(tamanho)]
    if all(b == "0" for b in bits):
        bits[random.randint(0, tamanho - 1)] = "1"
    return "".join(bits)


def executar_benchmark(
    num_amostras: int,
    max_l: int,
    caminho_csv: str,
    seed: int = 42,
) -> list[dict]:
    """Executa o benchmark randômico para L de 1 até max_l e salva em CSV."""
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 75)
    print(" 🚀 EXECUTANDO BENCHMARK RANDÔMICO (BFS SEQUENCIAL vs. PYTORCH SIMD)")
    print("=" * 75)
    print(f"• Amostras por L (x):   {num_amostras}")
    print(f"• Comprimento Máx (y): {max_l}")
    print(f"• Arquivo CSV de Saída: {caminho_csv}")
    print("-" * 75)

    campos = [
        "L",
        "amostra_id",
        "fita",
        "algoritmo",
        "custo",
        "nos_expandidos",
        "nos_visitados",
        "tempo_ms",
        "tempo_s",
    ]

    registros = []

    with open(caminho_csv, mode="w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=campos)
        writer.writeheader()

        for L in range(1, max_l + 1):
            print(f"▶ Processando L = {L:2d}/{max_l} ({num_amostras} amostras)...", end="", flush=True)
            inicio_l = time.perf_counter()

            for amostra_id in range(1, num_amostras + 1):
                fita = gerar_fita_aleatoria(L)

                # 1. Execução Sequencial
                res_seq, stats_seq = resolver_bfs_sequencial(fita, retornar_estatisticas=True)
                custo_seq = res_seq[1] if res_seq else -1
                t_seq_s = stats_seq["tempo_s"]
                t_seq_ms = t_seq_s * 1000.0

                reg_seq = {
                    "L": L,
                    "amostra_id": amostra_id,
                    "fita": fita,
                    "algoritmo": "Sequencial",
                    "custo": custo_seq,
                    "nos_expandidos": stats_seq["nos_expandidos"],
                    "nos_visitados": stats_seq["nos_visitados"],
                    "tempo_ms": round(t_seq_ms, 4),
                    "tempo_s": round(t_seq_s, 6),
                }
                writer.writerow(reg_seq)
                registros.append(reg_seq)

                # 2. Execução PyTorch SIMD
                res_torch, stats_torch = resolver_bfs_torch(fita, retornar_estatisticas=True)
                custo_torch = res_torch[1] if res_torch else -1
                t_torch_s = stats_torch["tempo_s"]
                t_torch_ms = t_torch_s * 1000.0

                reg_torch = {
                    "L": L,
                    "amostra_id": amostra_id,
                    "fita": fita,
                    "algoritmo": "PyTorch SIMD",
                    "custo": custo_torch,
                    "nos_expandidos": stats_torch["nos_expandidos"],
                    "nos_visitados": stats_torch["nos_visitados"],
                    "tempo_ms": round(t_torch_ms, 4),
                    "tempo_s": round(t_torch_s, 6),
                }
                writer.writerow(reg_torch)
                registros.append(reg_torch)

            tempo_l = time.perf_counter() - inicio_l
            print(f" Concluído em {tempo_l:.2f} s")

    print("-" * 75)
    print(f"✅ Benchmark concluído! Total de {len(registros)} medições salvas em '{caminho_csv}'.")
    print("=" * 75)
    return registros


def carregar_csv(caminho_csv: str) -> list[dict]:
    """Carrega os dados previamente salvos em um arquivo CSV."""
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Arquivo CSV '{caminho_csv}' não encontrado.")

    registros = []
    with open(caminho_csv, mode="r", newline="", encoding="utf-8") as f_csv:
        reader = csv.DictReader(f_csv)
        for row in reader:
            registros.append(
                {
                    "L": int(row["L"]),
                    "amostra_id": int(row["amostra_id"]),
                    "fita": row["fita"],
                    "algoritmo": row["algoritmo"],
                    "custo": int(row["custo"]),
                    "nos_expandidos": int(row["nos_expandidos"]),
                    "nos_visitados": int(row["nos_visitados"]),
                    "tempo_ms": float(row["tempo_ms"]),
                    "tempo_s": float(row["tempo_s"]),
                }
            )

    print(f"📂 Dados carregados com sucesso do CSV existente: '{caminho_csv}' ({len(registros)} registros).")
    return registros


DIR_BENCH = "bench"


def limpar_graficos_antigos(pasta: str = DIR_BENCH) -> None:
    """Remove arquivos PNG de gráficos gerados anteriormente para garantir limpeza."""
    os.makedirs(pasta, exist_ok=True)
    arquivos_antigos = glob.glob(os.path.join(pasta, "grafico_bfs_*.png"))
    for arq in arquivos_antigos:
        try:
            os.remove(arq)
        except OSError:
            pass
    if arquivos_antigos:
        print(f"🧹 Limpeza realizada: {len(arquivos_antigos)} gráficos anteriores removidos de '{pasta}/'.")


def gerar_graficos_individuais(registros: list[dict], pasta: str = DIR_BENCH) -> None:
    """Gera individualmente 4 gráficos estatísticos robustos e limpos salvos em pasta."""
    os.makedirs(pasta, exist_ok=True)
    limpar_graficos_antigos(pasta)

    dados_tempo_seq: Dict[int, List[float]] = defaultdict(list)
    dados_tempo_torch: Dict[int, List[float]] = defaultdict(list)
    dados_nos_seq: Dict[int, List[int]] = defaultdict(list)
    dados_tp_seq: Dict[int, List[float]] = defaultdict(list)
    dados_tp_torch: Dict[int, List[float]] = defaultdict(list)

    dados_nos_por_d: Dict[int, List[int]] = defaultdict(list)

    valores_l = sorted(list({r["L"] for r in registros}))

    for r in registros:
        l_val = r["L"]
        t_ms = r["tempo_ms"]
        t_s = r["tempo_s"] if r["tempo_s"] > 0 else (t_ms / 1000.0)
        nos = r["nos_expandidos"]
        d = r["custo"]
        tp = nos / max(t_s, 1e-8)  # Throughput: Nós por segundo

        if r["algoritmo"] == "Sequencial":
            dados_tempo_seq[l_val].append(t_ms)
            dados_nos_seq[l_val].append(nos)
            dados_tp_seq[l_val].append(tp)
            if d > 0:
                dados_nos_por_d[d].append(nos)
        else:
            dados_tempo_torch[l_val].append(t_ms)
            dados_tp_torch[l_val].append(tp)

    # Configuração de estilo visual refinado
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update(
        {
            "font.sans-serif": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "figure.titlesize": 14,
        }
    )

    # Pré-cálculo de medidas robustas por L
    geomean_seq = [calcular_media_geometrica(dados_tempo_seq[l]) for l in valores_l]
    p25_seq = [float(np.percentile(dados_tempo_seq[l], 25)) for l in valores_l]
    p75_seq = [float(np.percentile(dados_tempo_seq[l], 75)) for l in valores_l]

    geomean_torch = [calcular_media_geometrica(dados_tempo_torch[l]) for l in valores_l]
    p25_torch = [float(np.percentile(dados_tempo_torch[l], 25)) for l in valores_l]
    p75_torch = [float(np.percentile(dados_tempo_torch[l], 75)) for l in valores_l]

    geomean_nos = [calcular_media_geometrica(dados_nos_seq[l]) for l in valores_l]
    p25_nos = [float(np.percentile(dados_nos_seq[l], 25)) for l in valores_l]
    p75_nos = [float(np.percentile(dados_nos_seq[l], 75)) for l in valores_l]

    geomean_tp_seq = [calcular_media_geometrica(dados_tp_seq[l]) for l in valores_l]
    p25_tp_seq = [float(np.percentile(dados_tp_seq[l], 25)) for l in valores_l]
    p75_tp_seq = [float(np.percentile(dados_tp_seq[l], 75)) for l in valores_l]

    geomean_tp_torch = [calcular_media_geometrica(dados_tp_torch[l]) for l in valores_l]
    p25_tp_torch = [float(np.percentile(dados_tp_torch[l], 25)) for l in valores_l]
    p75_tp_torch = [float(np.percentile(dados_tp_torch[l], 75)) for l in valores_l]

    # =========================================================================
    # GRÁFICO 1: Tempo de Caso Médio (Média Geométrica & Corredor IQR)
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    ax1.plot(valores_l, geomean_seq, "o-", color="#d84315", linewidth=2.5, markersize=6, label="BFS Sequencial (Média Geométrica)")
    ax1.fill_between(valores_l, p25_seq, p75_seq, color="#ff9800", alpha=0.20, label="Corredor Típico Sequencial (IQR: P25 - P75)")

    ax1.plot(valores_l, geomean_torch, "s-", color="#0d47a1", linewidth=2.5, markersize=6, label="PyTorch SIMD (Média Geométrica)")
    ax1.fill_between(valores_l, p25_torch, p75_torch, color="#2196f3", alpha=0.20, label="Corredor Típico PyTorch (IQR: P25 - P75)")

    ax1.set_yscale("log")
    ax1.set_xlabel("Comprimento da Fita ($L$)", fontweight="bold")
    ax1.set_ylabel("Tempo de Execução em ms (Escala Logarítmica)", fontweight="bold")
    ax1.set_title("Curva de Tendência de Caso Médio: Evolução Temporal por Comprimento $L$", pad=15)
    ax1.set_xticks(valores_l)
    ax1.set_xticklabels([str(l) for l in valores_l])
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax1.legend(loc="upper left", frameon=True)

    caminho1 = os.path.join(pasta, "grafico_bfs_tempo_tendencia.png")
    fig1.tight_layout()
    fig1.savefig(caminho1, dpi=300)
    plt.close(fig1)
    print(f"📈 [1/4] Gráfico Individual de Tempo salvo em: '{caminho1}'")

    # =========================================================================
    # GRÁFICO 2: Complexidade Estrutural no Grafo (Nós Expandidos por L)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)
    ax2.plot(valores_l, geomean_nos, "D-", color="#37474f", linewidth=2.5, markersize=6, label="Nós Expandidos (Média Geométrica)")
    ax2.fill_between(valores_l, p25_nos, p75_nos, color="#78909c", alpha=0.25, label="Corredor Interquartil (IQR: P25 - P75)")

    ax2.set_yscale("log")
    ax2.set_xlabel("Comprimento da Fita ($L$)", fontweight="bold")
    ax2.set_ylabel("Quantidade de Nós Expandidos (Escala Logarítmica)", fontweight="bold")
    ax2.set_title("Complexidade Estrutural no Grafo de Busca: Expansão de Nós por Comprimento $L$", pad=15)
    ax2.set_xticks(valores_l)
    ax2.set_xticklabels([str(l) for l in valores_l])
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax2.legend(loc="upper left", frameon=True)

    caminho2 = os.path.join(pasta, "grafico_bfs_nos_expandidos.png")
    fig2.tight_layout()
    fig2.savefig(caminho2, dpi=300)
    plt.close(fig2)
    print(f"🌲 [2/4] Gráfico Individual de Nós Expandidos salvo em: '{caminho2}'")

    # =========================================================================
    # GRÁFICO 3: Throughput de Processamento (Vazão em Nós/segundo por L)
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=300)
    ax3.plot(valores_l, geomean_tp_seq, "o-", color="#d84315", linewidth=2.5, markersize=6, label="Throughput BFS Sequencial (Média Geométrica)")
    ax3.fill_between(valores_l, p25_tp_seq, p75_tp_seq, color="#ff9800", alpha=0.20, label="Corredor Sequencial (IQR: P25 - P75)")

    ax3.plot(valores_l, geomean_tp_torch, "s-", color="#0d47a1", linewidth=2.5, markersize=6, label="Throughput PyTorch SIMD (Média Geométrica)")
    ax3.fill_between(valores_l, p25_tp_torch, p75_tp_torch, color="#2196f3", alpha=0.20, label="Corredor PyTorch (IQR: P25 - P75)")

    ax3.set_yscale("log")
    ax3.set_xlabel("Comprimento da Fita ($L$)", fontweight="bold")
    ax3.set_ylabel("Vazão de Expansão (Nós/s, Escala Logarítmica)", fontweight="bold")
    ax3.set_title("Vazão de Processamento (Throughput): Nós Expandidos por Segundo por $L$", pad=15)
    ax3.set_xticks(valores_l)
    ax3.set_xticklabels([str(l) for l in valores_l])
    ax3.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax3.legend(loc="lower right", frameon=True)

    caminho3 = os.path.join(pasta, "grafico_bfs_throughput.png")
    fig3.tight_layout()
    fig3.savefig(caminho3, dpi=300)
    plt.close(fig3)
    print(f"🚀 [3/4] Gráfico de Throughput salvo em: '{caminho3}'")

    # =========================================================================
    # GRÁFICO 4: Complexidade por Profundidade Ótima (d*)
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(10, 6), dpi=300)
    valores_d = sorted(list(dados_nos_por_d.keys()))
    geomean_nos_d = [calcular_media_geometrica(dados_nos_por_d[d]) for d in valores_d]
    p25_nos_d = [float(np.percentile(dados_nos_por_d[d], 25)) for d in valores_d]
    p75_nos_d = [float(np.percentile(dados_nos_por_d[d], 75)) for d in valores_d]

    ax4.plot(valores_d, geomean_nos_d, "^-", color="#4a148c", linewidth=2.5, markersize=7, label="Nós Expandidos por Profundidade (Média Geométrica)")
    ax4.fill_between(valores_d, p25_nos_d, p75_nos_d, color="#7b1fa2", alpha=0.20, label="Corredor Interquartil (IQR: P25 - P75)")

    ax4.set_yscale("log")
    ax4.set_xlabel("Profundidade Ótima da Meta ($d^*$)", fontweight="bold")
    ax4.set_ylabel("Quantidade de Nós Expandidos (Escala Logarítmica)", fontweight="bold")
    ax4.set_title("Complexidade Exponencial no Grafo: Expansão de Nós por Profundidade Ótima $d^*$", pad=15)
    ax4.set_xticks(valores_d)
    ax4.set_xticklabels([str(d) for d in valores_d])
    ax4.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax4.legend(loc="upper left", frameon=True)

    caminho4 = os.path.join(pasta, "grafico_bfs_profundidade.png")
    fig4.tight_layout()
    fig4.savefig(caminho4, dpi=300)
    plt.close(fig4)
    print(f"🎯 [4/4] Gráfico de Profundidade Ótima salvo em: '{caminho4}'")
    print("=" * 75)


def main() -> None:
    """Interface CLI principal."""
    parser = argparse.ArgumentParser(
        description="Executa benchmarks randômicos do BFS e plota gráficos estatísticos individuais de Ponto Central."
    )
    parser.add_argument(
        "-x",
        "--amostras",
        type=int,
        default=None,
        help="Quantidade de instâncias randômicas testadas para cada comprimento L (padrão: 10 se novo benchmark).",
    )
    parser.add_argument(
        "-y",
        "--max-l",
        type=int,
        default=None,
        help="Comprimento máximo L da fita a ser avaliado (padrão: 15 se novo benchmark).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(DIR_BENCH, "benchmark_bfs_random.csv"),
        help="Caminho do arquivo CSV de entrada/saída (padrão: bench/benchmark_bfs_random.csv).",
    )
    parser.add_argument(
        "--pasta-saida",
        type=str,
        default=DIR_BENCH,
        help="Diretório de saída para salvar os gráficos PNG (padrão: bench/).",
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Força nova execução do benchmark mesmo se o arquivo CSV já existir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente do gerador pseudo-aleatório para reprodutibilidade (padrão: 42).",
    )

    args = parser.parse_args()

    caminho_csv = args.csv
    pasta_saida = args.pasta_saida
    os.makedirs(pasta_saida, exist_ok=True)
    csv_existe = os.path.exists(caminho_csv)

    deve_rodar = False
    if args.forcar or not csv_existe or (args.amostras is not None or args.max_l is not None):
        deve_rodar = True

    if deve_rodar:
        num_amostras = args.amostras or 10
        max_l = args.max_l or 15
        registros = executar_benchmark(
            num_amostras=num_amostras,
            max_l=max_l,
            caminho_csv=caminho_csv,
            seed=args.seed,
        )
    else:
        print(f"ℹ️  Nenhum parâmetro de reexecução fornecido e arquivo '{caminho_csv}' encontrado.")
        print("Reutilizando dados salvos para geração imediata dos 4 gráficos individuais.")
        registros = carregar_csv(caminho_csv)

    # Gera os 4 gráficos individuais na pasta de saída
    gerar_graficos_individuais(registros, pasta=pasta_saida)


if __name__ == "__main__":
    main()
