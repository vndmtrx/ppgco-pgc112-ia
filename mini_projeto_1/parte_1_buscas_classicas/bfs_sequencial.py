"""Algoritmo de Busca em Largura Sequencial Canônico (BFS).

Implementa a busca em largura canônica (Russell & Norvig, 4ª ed., Figura 3.9)
utilizando fila FIFO, conjunto de visitados e nós leves em tuplas (estado, pai, acao, custo).
"""

import sys
import time
from collections import deque
from typing import Optional

from flips import CUSTO_FLIP, carregar
from utils import executar_cli_busca, gerar_operacoes, imprimir_resultado, reconstruir_caminho


def bfs_sequencial(
    estado_inicial: int,
    tamanho: int,
    estado_alvo: int = 0,
    max_nos: Optional[int] = None,
    retornar_estatisticas: bool = False,
) -> tuple[list[tuple], int] | tuple[tuple[list[tuple], int] | None, dict] | None:
    """Executa o BFS canônico para encontrar a sequência de menor custo.

    Args:
        estado_inicial: Inteiro representando a fita de bits inicial.
        tamanho: Tamanho L da fita.
        estado_alvo: Inteiro da fita desejada (padrão: 0, fita zerada).
        max_nos: Limite opcional de nós expandidos.
        retornar_estatisticas: Se True, retorna métricas detalhadas de execução.
    """
    inicio_tempo = time.perf_counter()
    no_raiz = (estado_inicial, None, None, 0)

    # Teste de objetivo imediato na raiz
    if estado_inicial == estado_alvo:
        solucao = reconstruir_caminho(no_raiz)
        if retornar_estatisticas:
            stats = {
                "nos_expandidos": 0,
                "nos_visitados": 1,
                "tempo_s": time.perf_counter() - inicio_tempo,
            }
            return solucao, stats
        return solucao

    operacoes = gerar_operacoes(tamanho)
    borda: deque[tuple] = deque([no_raiz])
    alcancados: set[int] = {estado_inicial}
    nos_expandidos = 0

    while borda:
        if max_nos is not None and nos_expandidos >= max_nos:
            break

        no_atual = borda.popleft()
        estado_atual, _, _, custo_atual = no_atual
        nos_expandidos += 1

        for mascara, acao in operacoes:
            novo_estado = estado_atual ^ mascara

            # Teste de objetivo antecipado na geração do filho
            if novo_estado == estado_alvo:
                no_filho = (novo_estado, no_atual, acao, custo_atual + CUSTO_FLIP)
                solucao = reconstruir_caminho(no_filho)
                if retornar_estatisticas:
                    stats = {
                        "nos_expandidos": nos_expandidos,
                        "nos_visitados": len(alcancados) + 1,
                        "tempo_s": time.perf_counter() - inicio_tempo,
                    }
                    return solucao, stats
                return solucao

            if novo_estado not in alcancados:
                alcancados.add(novo_estado)
                borda.append((novo_estado, no_atual, acao, custo_atual + CUSTO_FLIP))

    if retornar_estatisticas:
        stats = {
            "nos_expandidos": nos_expandidos,
            "nos_visitados": len(alcancados),
            "tempo_s": time.perf_counter() - inicio_tempo,
        }
        return None, stats
    return None


def resolver_bfs_sequencial(
    string_inicial: str,
    string_alvo: Optional[str] = None,
    max_nos: Optional[int] = None,
    retornar_estatisticas: bool = False,
) -> tuple[list[tuple], int] | tuple[tuple[list[tuple], int] | None, dict] | None:
    """Função de alto nível que recebe strings binárias diretamente."""
    estado_inicial, tamanho = carregar(string_inicial)
    if string_alvo is None:
        estado_alvo = 0
    else:
        estado_alvo, tamanho_alvo = carregar(string_alvo)
        if tamanho_alvo != tamanho:
            raise ValueError(f"Tamanho do alvo ({tamanho_alvo}) diferente do tamanho inicial ({tamanho}).")

    return bfs_sequencial(
        estado_inicial=estado_inicial,
        tamanho=tamanho,
        estado_alvo=estado_alvo,
        max_nos=max_nos,
        retornar_estatisticas=retornar_estatisticas,
    )


def executar_cli() -> None:
    """Interface de linha de comando para o BFS Sequencial."""
    executar_cli_busca("BFS Sequencial Canônico", resolver_bfs_sequencial)


if __name__ == "__main__":
    executar_cli()
