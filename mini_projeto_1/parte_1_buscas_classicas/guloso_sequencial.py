"""Implementação canônica do Algoritmo Guloso (Greedy Best-First Search) Sequencial.

Baseado estritamente em Russell & Norvig (4ª Edição, Capítulo 3, Seção 3.5.1):
- Função de avaliação puramente heurística: f(n) = h(n).
- Fila de prioridades com heapq e desempate determinístico FIFO: (h, contador, nó).
- Tabela de alcançados (reached) para poda de ciclos no grafo.
- Representação leve com tuplas nativas: (estado, pai, acao, custo).
"""

import heapq
import sys
import time
from typing import Callable, Optional

from flips import (
    CUSTO_FLIP,
    carregar,
    flip_bit,
    flip_bloco,
    formatar,
)
from heuristica import (
    h2_blocos,
    obter_heuristica,
)
from utils import (
    executar_cli_busca,
    gerar_operacoes,
    reconstruir_caminho,
)

TipoNo = tuple[int, Optional["TipoNo"], Optional[tuple], int]


def guloso_sequencial(
    estado_inicial: int,
    tamanho: int,
    estado_alvo: int = 0,
    funcao_h: Optional[Callable[[int, int, int], int]] = None,
    retornar_estatisticas: bool = False,
) -> tuple[Optional[TipoNo], dict[str, float]]:
    """Executa a Busca Gulosa (Greedy Best-First Search) sequencial.

    Args:
        estado_inicial: Inteiro representando o estado de partida.
        tamanho: Comprimento da fita (L).
        estado_alvo: Inteiro representando o estado objetivo (padrão: 0).
        funcao_h: Função heurística h(estado, tamanho, alvo). Padrão: h2_blocos.
        retornar_estatisticas: Se True, retorna métricas detalhadas de execução.

    Returns:
        Tupla contendo (nó_objetivo, dicionário_de_estatísticas).
    """
    if funcao_h is None:
        funcao_h = h2_blocos

    inicio_tempo = time.perf_counter()
    nos_expandidos = 0

    no_raiz: TipoNo = (estado_inicial, None, None, 0)

    # Verificação inicial: estado de partida já é o alvo
    if estado_inicial == estado_alvo:
        tempo_total = time.perf_counter() - inicio_tempo
        stats = {
            "tempo_s": tempo_total,
            "tempo_ms": tempo_total * 1000.0,
            "nos_expandidos": 0,
            "nos_visitados": 1,
        }
        return no_raiz, stats

    # Fila de prioridades: (h_valor, contador_fifo, no)
    contador_insercao = 0
    h_inicial = funcao_h(estado_inicial, tamanho, estado_alvo)
    fronteira: list[tuple[int, int, TipoNo]] = [
        (h_inicial, contador_insercao, no_raiz)
    ]
    heapq.heapify(fronteira)

    # Tabela de estados alcançados para prevenção de ciclos
    alcancados: set[int] = {estado_inicial}

    # Pré-computação das operações válidas para fita de tamanho L
    operacoes = gerar_operacoes(tamanho)

    no_encontrado: Optional[TipoNo] = None

    while fronteira:
        h_atual, _, no_atual = heapq.heappop(fronteira)
        estado_pai, _, _, custo_pai = no_atual
        custo_filho = custo_pai + CUSTO_FLIP
        nos_expandidos += 1

        for mascara, acao in operacoes:
            estado_filho = estado_pai ^ mascara

            # Teste de objetivo antecipado na geração do filho
            if estado_filho == estado_alvo:
                no_encontrado = (estado_filho, no_atual, acao, custo_filho)
                break

            if estado_filho not in alcancados:
                alcancados.add(estado_filho)
                no_filho: TipoNo = (estado_filho, no_atual, acao, custo_filho)
                h_filho = funcao_h(estado_filho, tamanho, estado_alvo)
                contador_insercao += 1
                heapq.heappush(
                    fronteira, (h_filho, contador_insercao, no_filho)
                )

        if no_encontrado is not None:
            break

    tempo_total = time.perf_counter() - inicio_tempo
    stats = {
        "tempo_s": tempo_total,
        "tempo_ms": tempo_total * 1000.0,
        "nos_expandidos": nos_expandidos,
        "nos_visitados": len(alcancados),
    }

    return no_encontrado, stats


def resolver_guloso_sequencial(
    string_inicial: str,
    string_alvo: Optional[str] = None,
    heuristica: str = "h2",
    retornar_estatisticas: bool = False,
) -> tuple | tuple[tuple, dict]:
    """Interface de alto nível para resolver fitas com o Algoritmo Guloso Sequencial.

    Args:
        string_inicial: Fita binária inicial (ex: '1100111').
        string_alvo: Fita binária objetivo (opcional, padrão: fita zerada).
        heuristica: Identificador da heurística ('h1', 'h2', 'hamming', 'blocos').
        retornar_estatisticas: Se True, retorna as métricas de busca.

    Returns:
        Tupla (acoes, custo) ou ((acoes, custo), stats) se retornar_estatisticas=True.
    """
    estado_inicial, L = carregar(string_inicial)

    if string_alvo is not None:
        estado_alvo, L_alvo = carregar(string_alvo)
        if L != L_alvo:
            raise ValueError(
                f"Tamanhos incompatíveis: fita inicial tem L={L} e alvo tem L={L_alvo}."
            )
    else:
        estado_alvo = 0

    funcao_h = obter_heuristica(heuristica)

    no_solucao, stats = guloso_sequencial(
        estado_inicial=estado_inicial,
        tamanho=L,
        estado_alvo=estado_alvo,
        funcao_h=funcao_h,
        retornar_estatisticas=True,
    )

    resultado = reconstruir_caminho(no_solucao)

    if retornar_estatisticas:
        return resultado, stats
    return resultado


if __name__ == "__main__":
    executar_cli_busca(
        titulo="Algoritmo Guloso (Greedy Best-First Search) Sequencial",
        funcao_resolver=resolver_guloso_sequencial,
    )
