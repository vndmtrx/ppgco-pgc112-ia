"""Benchmark de Escalabilidade do Algoritmo Guloso com Poda de Ações até L = 4096.

Compara o desempenho do Guloso com Poda (Sequencial e PyTorch SIMD) em potências de 2:
L = 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096.
Salva os resultados em 'bench/bench_guloso_poda.csv'.
"""

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_guloso import gerar_instancia_teste
from flips import carregar
from guloso_poda_sequencial import guloso_poda_sequencial
from guloso_poda_torch import guloso_poda_torch
from heuristica import h2_blocos, obter_heuristica


def executar_benchmark_poda(
    tamanhos: list[int],
    tipo_instancia: str = "blocos",
    heuristica_nome: str = "h2",
    incluir_torch: bool = True,
    arquivo_csv: str = "bench/bench_guloso_poda.csv",
):
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
    total_instancias = len(tamanhos)
    modo_denso = total_instancias > 30

    print("=" * 80)
    print(f"BENCHMARK GULOSO COM PODA DE AÇÕES - HEURÍSTICA: {heuristica_nome.upper()}")
    print(
        f"Tipo de Instância: {tipo_instancia.capitalize()} | Total de Pontos: {total_instancias} (L={min(tamanhos)}..{max(tamanhos)})"
    )
    print("=" * 80)
    if not modo_denso:
        print(
            f"{'L':<6} | {'Estratégia':<14} | {'Expandidos':<12} | {'Custo':<8} | {'Tempo (ms)':<14} | {'Throughput (nó/s)':<18}"
        )
        print("-" * 80)

    t_inicio_global = time.time()

    for idx, L in enumerate(tamanhos, start=1):
        fita = gerar_instancia_teste(L, tipo=tipo_instancia)
        estado_inicial, _ = carregar(fita)

        # 1. Sequencial com Poda
        try:
            no_seq, stats_seq = guloso_poda_sequencial(
                estado_inicial=estado_inicial,
                tamanho=L,
                estado_alvo=0,
                funcao_h=funcao_h,
                retornar_estatisticas=True,
            )
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
                    "estrategia": "Sequencial (Poda)",
                    "tempo_ms": round(tempo_ms_seq, 4),
                    "tempo_s": round(tempo_s_seq, 6),
                    "nos_expandidos": expandidos_seq,
                    "nos_visitados": visitados_seq,
                    "custo_solucao": custo_seq,
                    "throughput_nos_s": round(tp_seq, 2),
                }
            )

            if not modo_denso:
                print(
                    f"{L:<6} | {'Sequencial':<14} | {expandidos_seq:<12} | {custo_seq:<8} | {tempo_ms_seq:<14.3f} | {tp_seq:<18.2f}"
                )
        except Exception as e:
            print(f"{L:<6} | {'Sequencial':<14} | ERRO: {e}")

        # 2. PyTorch SIMD com Poda
        if incluir_torch:
            try:
                no_torch, stats_torch = guloso_poda_torch(
                    estado_inicial=estado_inicial,
                    tamanho=L,
                    estado_alvo=0,
                    funcao_h=funcao_h,
                    retornar_estatisticas=True,
                )
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
                        "estrategia": "PyTorch SIMD (Poda)",
                        "tempo_ms": round(tempo_ms_torch, 4),
                        "tempo_s": round(tempo_s_torch, 6),
                        "nos_expandidos": expandidos_torch,
                        "nos_visitados": visitados_torch,
                        "custo_solucao": custo_torch,
                        "throughput_nos_s": round(tp_torch, 2),
                    }
                )

                if not modo_denso:
                    print(
                        f"{L:<6} | {'PyTorch SIMD':<14} | {expandidos_torch:<12} | {custo_torch:<8} | {tempo_ms_torch:<14.3f} | {tp_torch:<18.2f}"
                    )
            except Exception as e:
                print(f"{L:<6} | {'PyTorch SIMD':<14} | ERRO: {e}")

        if modo_denso and (idx % 250 == 0 or idx == total_instancias):
            decorrido = time.time() - t_inicio_global
            porc = (idx / total_instancias) * 100
            print(
                f"Progresso: {idx:4d}/{total_instancias} ({porc:5.1f}%) | L={L:<4d} | Tempo decorrido: {decorrido:5.1f}s"
            )

        if idx % 500 == 0:
            gc.collect()

    with open(caminho_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()
        writer.writerows(registros)

    tempo_total = time.time() - t_inicio_global
    print("=" * 80)
    print(
        f"Benchmark concluído com sucesso em {tempo_total:.2f}s! ({len(registros)} registros em {caminho_csv})"
    )
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Executa benchmark de escalabilidade para o Guloso com Poda."
    )
    parser.add_argument(
        "--tamanhos",
        type=int,
        nargs="+",
        default=None,
        help="Lista explícita de comprimentos de fita L a testar.",
    )
    parser.add_argument(
        "--min-l",
        type=int,
        default=1,
        help="Valor mínimo de L (padrão: 1).",
    )
    parser.add_argument(
        "--max-l",
        type=int,
        default=None,
        help="Valor máximo de L para varredura contínua (ex: 4096).",
    )
    parser.add_argument(
        "--passo",
        type=int,
        default=1,
        help="Passo da varredura linear de L (padrão: 1).",
    )
    parser.add_argument(
        "--tipo",
        type=str,
        default="blocos",
        choices=["blocos", "alternado", "aleatorio"],
        help="Tipo de instância.",
    )
    parser.add_argument(
        "--heuristica",
        type=str,
        default="h2",
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
        default="bench/bench_guloso_poda.csv",
        help="Caminho do CSV de saída.",
    )

    args = parser.parse_args()

    if args.tamanhos is not None:
        tamanhos = args.tamanhos
    elif args.max_l is not None:
        tamanhos = list(range(args.min_l, args.max_l + 1, args.passo))
    else:
        # Padrão: potências de 2
        tamanhos = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    executar_benchmark_poda(
        tamanhos=tamanhos,
        tipo_instancia=args.tipo,
        heuristica_nome=args.heuristica,
        incluir_torch=not args.sem_torch,
        arquivo_csv=args.saida,
    )


if __name__ == "__main__":
    main()
