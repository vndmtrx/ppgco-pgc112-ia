"""Algoritmo Guloso com Poda de Ações Vetorizado com PyTorch.

Utiliza tensores PyTorch para vetorização das transições de estado podadas
e cálculo em lote com desempate determinístico FIFO, mantendo paridade
estrita com a versão sequencial.
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
from guloso_poda_sequencial import extrair_acoes_podadas
from heuristica import (
    h2_blocos,
    obter_heuristica,
)
from utils import (
    executar_cli_busca,
    reconstruir_caminho,
)

TipoNo = tuple[int, Optional["TipoNo"], Optional[tuple], int]


def guloso_poda_torch(
    estado_inicial: int,
    tamanho: int,
    estado_alvo: int = 0,
    funcao_h: Optional[Callable[[int, int, int], int]] = None,
    retornar_estatisticas: bool = False,
) -> tuple[Optional[TipoNo], dict[str, float]]:
    """Executa a Busca Gulosa com Poda de Ações vetorizada com PyTorch."""
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
    usar_tensores_torch = tamanho <= 62

    while fronteira:
        h_atual, _, no_atual = heapq.heappop(fronteira)
        estado_pai, _, _, custo_pai = no_atual
        custo_filho = custo_pai + CUSTO_FLIP
        nos_expandidos += 1

        diff = estado_pai ^ estado_alvo
        acoes_podadas = list(extrair_acoes_podadas(diff, tamanho))

        if not acoes_podadas:
            continue

        mascaras_list = [op[0] for op in acoes_podadas]
        acoes_list = [op[1] for op in acoes_podadas]

        if usar_tensores_torch:
            mascaras_tensor = torch.tensor(mascaras_list, dtype=torch.int64)
            pai_tensor = torch.tensor([estado_pai], dtype=torch.int64)
            novos_tensor = pai_tensor ^ mascaras_tensor
            novos_estados = novos_tensor.tolist()
        else:
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


def resolver_guloso_poda_torch(
    string_inicial: str,
    string_alvo: Optional[str] = None,
    heuristica: str = "h2",
    retornar_estatisticas: bool = False,
) -> tuple | tuple[tuple, dict]:
    """Interface de alto nível para o Algoritmo Guloso com Poda PyTorch SIMD."""
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

    no_solucao, stats = guloso_poda_torch(
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
        titulo="Algoritmo Guloso com Poda de Ações (Action Pruning) PyTorch SIMD",
        funcao_resolver=resolver_guloso_poda_torch,
    )
