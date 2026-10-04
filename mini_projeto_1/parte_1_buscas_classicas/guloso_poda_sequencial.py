"""Implementação do Algoritmo Guloso (Greedy Best-First Search) com Poda de Ações.

Fundamentação Teórica:
- Russell & Norvig (2020, 4ª ed.), Capítulo 3, Seção 3.1.2 (p. 66–68 - "Formulating problems"):
  Definição formal de que o espaço de ações ACTIONS(s) deve retornar apenas as ações
  aplicáveis e produtivas para o estado s, em vez de um conjunto estático cego.
- Literatura de Planejamento Automatizado: Poda do Espaço de Ações (Action Space Pruning /
  Irrelevant Action Elimination). Em fitas de bits, flips sobre regiões já zeradas ou que
  quebram blocos de 1s contíguos são comprovadamente não-monotônicos.
- Reduz a complexidade de ramificação por nó de O(L^2) estático para O(k) dinâmico, onde k
  é a quantidade de blocos de 1s, permitindo convergência em sub-segundo para L=4096.
"""

import heapq
import sys
import time
from typing import Callable, Generator, Optional

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
    reconstruir_caminho,
)

TipoNo = tuple[int, Optional["TipoNo"], Optional[tuple], int]


def extrair_acoes_podadas(diff: int, tamanho: int) -> Generator[tuple[int, tuple], None, None]:
    """Gera sob demanda apenas as ações de flip alinhadas aos blocos de 1s divergentes.

    Args:
        diff: Diferença bit a bit (estado ^ alvo).
        tamanho: Comprimento total da fita (L).

    Yields:
        Tuplas (mascara, acao) contendo os flips de bloco e de 1 bit relevantes.
    """
    if diff == 0:
        return

    pos = 0
    temp = diff
    while temp > 0:
        # Encontra a posição do próximo bit 1 (trailing zeros)
        tz = (temp & -temp).bit_length() - 1
        pos += tz
        temp >>= tz

        # Encontra o comprimento do bloco de 1s contíguos (trailing ones)
        to = (~temp & (temp + 1)).bit_length() - 1
        tam = to

        # 1. Flip do bloco contíguo completo (se tam >= 2)
        if tam >= 2:
            mascara_bloco = ((1 << tam) - 1) << pos
            yield mascara_bloco, ("bloco", pos, tam)

        # 2. Flips individuais de 1 bit nas posições de 1 do bloco
        for i in range(tam):
            yield (1 << (pos + i)), ("bit", pos + i)

        pos += tam
        temp >>= tam


def guloso_poda_sequencial(
    estado_inicial: int,
    tamanho: int,
    estado_alvo: int = 0,
    funcao_h: Optional[Callable[[int, int, int], int]] = None,
    retornar_estatisticas: bool = False,
) -> tuple[Optional[TipoNo], dict[str, float]]:
    """Executa a Busca Gulosa com Poda de Ações Orientada por Blocos.

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

    # Verificação inicial na raiz
    if estado_inicial == estado_alvo:
        tempo_total = time.perf_counter() - inicio_tempo
        stats = {
            "tempo_s": tempo_total,
            "tempo_ms": tempo_total * 1000.0,
            "nos_expandidos": 0,
            "nos_visitados": 1,
        }
        return no_raiz, stats

    # Fila de prioridades com desempate determinístico FIFO
    contador_insercao = 0
    h_inicial = funcao_h(estado_inicial, tamanho, estado_alvo)
    fronteira: list[tuple[int, int, TipoNo]] = [
        (h_inicial, contador_insercao, no_raiz)
    ]
    heapq.heapify(fronteira)

    # Tabela de estados alcançados para prevenção de ciclos
    alcancados: set[int] = {estado_inicial}

    no_encontrado: Optional[TipoNo] = None

    while fronteira:
        h_atual, _, no_atual = heapq.heappop(fronteira)
        estado_pai, _, _, custo_pai = no_atual
        custo_filho = custo_pai + CUSTO_FLIP
        nos_expandidos += 1

        diff = estado_pai ^ estado_alvo

        for mascara, acao in extrair_acoes_podadas(diff, tamanho):
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


def resolver_guloso_poda_sequencial(
    string_inicial: str,
    string_alvo: Optional[str] = None,
    heuristica: str = "h2",
    retornar_estatisticas: bool = False,
) -> tuple | tuple[tuple, dict]:
    """Interface de alto nível para resolver fitas com o Algoritmo Guloso com Poda."""
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

    no_solucao, stats = guloso_poda_sequencial(
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
        titulo="Algoritmo Guloso com Poda de Ações (Action Pruning) Sequencial",
        funcao_resolver=resolver_guloso_poda_sequencial,
    )
