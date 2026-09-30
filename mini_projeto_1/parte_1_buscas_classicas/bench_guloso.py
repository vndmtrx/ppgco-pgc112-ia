"""Script de Benchmark de Escalabilidade do Algoritmo Guloso (Item E).

Avalia a escalabilidade do Algoritmo Guloso (Greedy Best-First Search)
em potências de 2 (L = 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096).
Compara heurísticas (h1 vs h2) e estratégias (Sequencial vs PyTorch SIMD).

Os resultados são salvos em formato tabular CSV dentro do diretório 'bench/'.
"""

import argparse
import csv
import gc
import os
import random
import sys
import time
from pathlib import Path

# Adiciona diretório corrente ao path para importação dos módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flips import carregar
from guloso_sequencial import guloso_sequencial
from guloso_torch import guloso_torch
from heuristica import h1_hamming, h2_blocos, obter_heuristica


def gerar_instancia_teste(tamanho: int, tipo: str = "blocos", seed: int = 42) -> str:
    """Gera uma fita binária de comprimento L para teste de escalabilidade.

    Tipos disponíveis:
    - 'blocos': Sequência estruturada de blocos contíguos de 1s e 0s (realista).
    - 'alternado': Pior caso ('101010...'), com L/2 blocos unitários isolados.
    - 'aleatorio': Distribuição pseudo-aleatória com 50% de densidade de 1s.
    """
    if tipo == "alternado":
        return ("10" * ((tamanho // 2) + 1))[:tamanho]
    elif tipo == "blocos":
        # Divide a fita em 4 a 8 blocos maiores de 1s e 0s
        random.seed(seed + tamanho)
        segmentos = []
        bits_restantes = tamanho
        while bits_restantes > 0:
            tam_seg = min(bits_restantes, max(1, bits_restantes // random.randint(2, 6)))
            char = "1" if len(segmentos) % 2 == 1 else "0"
            segmentos.append(char * tam_seg)
            bits_restantes -= tam_seg
        fita = "".join(segmentos)
        # Garante que não é tudo zero
        if "1" not in fita:
            fita = "1" * (tamanho // 2) + "0" * (tamanho - tamanho // 2)
        return fita
    elif tipo == "aleatorio":
        random.seed(seed + tamanho)
        return "".join(random.choice(["0", "1"]) for _ in range(tamanho))
    else:
        raise ValueError(f"Tipo de instância inválido: '{tipo}'.")


def executar_benchmark(
    tamanhos: list[int],
    tipo_instancia: str = "blocos",
    heuristica_nome: str = "h2",
    incluir_torch: bool = True,
    arquivo_csv: str = "bench/bench_guloso.csv",
    timeout_segundos: float = 300.0,
):
    """Executa a bateria de benchmarks e exporta os resultados para CSV."""
    caminho_csv = Path(arquivo_csv)
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)

    funcao_h = obter_heuristica(heuristica_nome)

    colunas = [
        "tamanho",
        "tipo_instancia",
        "heuristica",
        "estrategia",
        "tempo_ms",
        "tempo_s",
        "nos_expandidos",
        "nos_visitados",
        "custo_solucao",
        "throughput_nos_s",
    ]

    registros = []

    print("=" * 80)
    print(f"BENCHMARK DE ESCALABILIDADE GULOSO - HEURÍSTICA: {heuristica_nome.upper()}")
    print(f"Tipo de Instância: {tipo_instancia.capitalize()} | Timeout: {timeout_segundos}s")
    print("=" * 80)
    print(
        f"{'L':<6} | {'Estratégia':<14} | {'Expandidos':<12} | {'Custo':<8} | {'Tempo (ms)':<14} | {'Throughput (nó/s)':<18}"
    )
    print("-" * 80)

    for L in tamanhos:
        fita = gerar_instancia_teste(L, tipo=tipo_instancia)
        estado_inicial, _ = carregar(fita)

        # 1. Teste Sequencial
        try:
            t0 = time.perf_counter()
            no_seq, stats_seq = guloso_sequencial(
                estado_inicial=estado_inicial,
                tamanho=L,
                estado_alvo=0,
                funcao_h=funcao_h,
                retornar_estatisticas=True,
            )
            t_total = time.perf_counter() - t0
            custo_seq = no_seq[3] if no_seq else -1
            expandidos_seq = stats_seq["nos_expandidos"]
            visitados_seq = stats_seq["nos_visitados"]
            tempo_ms_seq = stats_seq["tempo_ms"]
            tempo_s_seq = stats_seq["tempo_s"]
            tp_seq = expandidos_seq / tempo_s_seq if tempo_s_seq > 0 else 0.0

            registros.append(
                {
                    "tamanho": L,
                    "tipo_instancia": tipo_instancia,
                    "heuristica": heuristica_nome,
                    "estrategia": "Sequencial",
                    "tempo_ms": round(tempo_ms_seq, 4),
                    "tempo_s": round(tempo_s_seq, 6),
                    "nos_expandidos": expandidos_seq,
                    "nos_visitados": visitados_seq,
                    "custo_solucao": custo_seq,
                    "throughput_nos_s": round(tp_seq, 2),
                }
            )

            print(
                f"{L:<6} | {'Sequencial':<14} | {expandidos_seq:<12} | {custo_seq:<8} | {tempo_ms_seq:<14.2f} | {tp_seq:<18.2f}"
            )
        except Exception as e:
            print(f"{L:<6} | {'Sequencial':<14} | ERRO: {e}")

        # 2. Teste PyTorch SIMD
        if incluir_torch:
            try:
                t0 = time.perf_counter()
                no_torch, stats_torch = guloso_torch(
                    estado_inicial=estado_inicial,
                    tamanho=L,
                    estado_alvo=0,
                    funcao_h=funcao_h,
                    retornar_estatisticas=True,
                )
                t_total = time.perf_counter() - t0
                custo_torch = no_torch[3] if no_torch else -1
                expandidos_torch = stats_torch["nos_expandidos"]
                visitados_torch = stats_torch["nos_visitados"]
                tempo_ms_torch = stats_torch["tempo_ms"]
                tempo_s_torch = stats_torch["tempo_s"]
                tp_torch = (
                    expandidos_torch / tempo_s_torch if tempo_s_torch > 0 else 0.0
                )

                registros.append(
                    {
                        "tamanho": L,
                        "tipo_instancia": tipo_instancia,
                        "heuristica": heuristica_nome,
                        "estrategia": "PyTorch SIMD",
                        "tempo_ms": round(tempo_ms_torch, 4),
                        "tempo_s": round(tempo_s_torch, 6),
                        "nos_expandidos": expandidos_torch,
                        "nos_visitados": visitados_torch,
                        "custo_solucao": custo_torch,
                        "throughput_nos_s": round(tp_torch, 2),
                    }
                )

                print(
                    f"{L:<6} | {'PyTorch SIMD':<14} | {expandidos_torch:<12} | {custo_torch:<8} | {tempo_ms_torch:<14.2f} | {tp_torch:<18.2f}"
                )
            except Exception as e:
                print(f"{L:<6} | {'PyTorch SIMD':<14} | ERRO: {e}")

        # Coleta explícita de lixo para liberar RAM entre tamanhos
        gc.collect()

    # Salva no arquivo CSV
    with open(caminho_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        writer.writerows(registros)

    print("=" * 80)
    print(f"Benchmark concluído! Resultados salvos em: {caminho_csv}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Executa benchmark de escalabilidade para o Algoritmo Guloso."
    )
    parser.add_argument(
        "--tamanhos",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64, 128, 256, 512],
        help="Lista de comprimentos de fita L a testar.",
    )
    parser.add_argument(
        "--max-l",
        type=int,
        default=None,
        help="Gera potências de 2 de L=2 até o valor especificado.",
    )
    parser.add_argument(
        "--tipo",
        type=str,
        default="blocos",
        choices=["blocos", "alternado", "aleatorio"],
        help="Tipo de padrão gerado na fita de bits.",
    )
    parser.add_argument(
        "--heuristica",
        type=str,
        default="h2",
        choices=["h1", "h2", "hamming", "blocos"],
        help="Heurística a utilizar.",
    )
    parser.add_argument(
        "--sem-torch",
        action="store_true",
        help="Desativa a execução com PyTorch SIMD.",
    )
    parser.add_argument(
        "--saida",
        type=str,
        default="bench/bench_guloso.csv",
        help="Caminho do arquivo CSV de saída.",
    )

    args = parser.parse_args()

    tamanhos = args.tamanhos
    if args.max_l is not None:
        tamanhos = []
        l = 2
        while l <= args.max_l:
            tamanhos.append(l)
            l *= 2

    executar_benchmark(
        tamanhos=tamanhos,
        tipo_instancia=args.tipo,
        heuristica_nome=args.heuristica,
        incluir_torch=not args.sem_torch,
        arquivo_csv=args.saida,
    )


if __name__ == "__main__":
    main()
