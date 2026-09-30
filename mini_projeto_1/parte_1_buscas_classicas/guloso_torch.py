"""Algoritmo de Busca Gulosa (Greedy Best-First Search) Vetorizado com PyTorch.

Utiliza tensores PyTorch para vetorização de transições de estado, cálculo
em lote de operações bitwise e fila de prioridade com desempate determinístico,
assegurando 100% de paridade e isomorfismo com o algoritmo sequencial.
"""

import heapq
import sys
import time
from typing import Callable, Optional

import torch

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


def guloso_torch(
    estado_inicial: int,
    tamanho: int,
    estado_alvo: int = 0,
    funcao_h: Optional[Callable[[int, int, int], int]] = None,
    retornar_estatisticas: bool = False,
) -> tuple[Optional[TipoNo], dict[str, float]]:
    """Executa a Busca Gulosa vetorizada com PyTorch.

    Args:
        estado_inicial: Inteiro representando o estado de partida.
        tamanho: Comprimento da fita (L).
        estado_alvo: Inteiro representando o estado objetivo (padrão: 0).
        funcao_h: Função heurística h(estado, tamanho, alvo). Padrão: h2_blocos.
        retornar_estatisticas: Se True, retorna métricas detalhadas de busca.

    Returns:
        Tupla contendo (nó_objetivo, dicionário_de_estatísticas).
    """
    if funcao_h is None:
        funcao_h = h2_blocos

    inicio_tempo = time.perf_counter()
    nos_expandidos = 0

    no_raiz: TipoNo = (estado_inicial, None, None, 0)

    # Verificação na raiz
    if estado_inicial == estado_alvo:
        tempo_total = time.perf_counter() - inicio_tempo
        stats = {
            "tempo_s": tempo_total,
            "tempo_ms": tempo_total * 1000.0,
            "nos_expandidos": 0,
            "nos_visitados": 1,
        }
        return no_raiz, stats

    # Pré-computação das operações e máscaras tensoriais
    operacoes = gerar_operacoes(tamanho)
    mascaras_list = [op[0] for op in operacoes]
    acoes_list = [op[1] for op in operacoes]

    # Para L <= 62, podemos usar tensores int64 do PyTorch para XOR vetorizado
    usar_tensores_torch = tamanho <= 62
    if usar_tensores_torch:
        mascaras_tensor = torch.tensor(mascaras_list, dtype=torch.int64)

    # Fila de prioridades com desempate FIFO estrito
    contador_insercao = 0
    h_inicial = funcao_h(estado_inicial, tamanho, estado_alvo)
    fronteira: list[tuple[int, int, TipoNo]] = [
        (h_inicial, contador_insercao, no_raiz)
    ]
    heapq.heapify(fronteira)

    # Tabela de alcançados para prevenção de ciclos
    alcancados: set[int] = {estado_inicial}

    no_encontrado: Optional[TipoNo] = None

    while fronteira:
        h_atual, _, no_atual = heapq.heappop(fronteira)
        estado_pai, _, _, custo_pai = no_atual
        custo_filho = custo_pai + CUSTO_FLIP
        nos_expandidos += 1

        if usar_tensores_torch:
            # Geração vetorizada de todos os sucessores via PyTorch SIMD
            pai_tensor = torch.tensor([estado_pai], dtype=torch.int64)
            novos_tensor = pai_tensor ^ mascaras_tensor
            novos_estados = novos_tensor.tolist()
        else:
            # Fallback para inteiros arbitrários de alta precisão (L > 62 até L=4096)
            novos_estados = [estado_pai ^ m for m in mascaras_list]

        for i, estado_filho in enumerate(novos_estados):
            acao = acoes_list[i]

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


def resolver_guloso_torch(
    string_inicial: str,
    string_alvo: Optional[str] = None,
    heuristica: str = "h2",
    retornar_estatisticas: bool = False,
) -> tuple | tuple[tuple, dict]:
    """Interface de alto nível para o Algoritmo Guloso com PyTorch SIMD.

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

    no_solucao, stats = guloso_torch(
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
        titulo="Algoritmo Guloso (Greedy Best-First Search) PyTorch SIMD",
        funcao_resolver=resolver_guloso_torch,
    )
