"""Funções heurísticas para algoritmos de busca informada (Guloso / A*).

Fornece funções puras e eficientes baseadas em operações de bits (O(1)):
1. h1_hamming: Distância de Hamming (contagem total de bits divergentes).
2. h2_blocos: Contagem de agrupamentos (clusters) contíguos de bits 1.
"""

from typing import Callable


def h1_hamming(estado: int, tamanho: int, alvo: int = 0) -> int:
    """Calcula a Distância de Hamming entre o estado atual e o alvo.

    Conta a quantidade total de bits divergentes (bits 1 em estado ^ alvo).

    Args:
        estado: Inteiro representando a fita atual.
        tamanho: Comprimento total da fita em bits (L).
        alvo: Inteiro representando a fita desejada (padrão: 0).

    Returns:
        Quantidade de bits divergentes.

    Examples:
        >>> h1_hamming(0b0000, 4, 0b0000)
        0
        >>> h1_hamming(0b0001, 4, 0b0000)
        1
        >>> h1_hamming(0b1111, 4, 0b0000)
        4
        >>> h1_hamming(0b1010, 4, 0b0101)
        4
    """
    diff = estado ^ alvo
    return diff.bit_count()


def h2_blocos(estado: int, tamanho: int, alvo: int = 0) -> int:
    """Calcula o número de agrupamentos (clusters) contíguos de bits 1.

    Cada bloco contíguo de bits 1 pode ser eliminado em uma única operação
    de flip de bloco. Portanto, o número de blocos contíguos de 1s fornece
    uma estimativa direta da quantidade mínima de operações de bloco.

    A detecção do início de cada bloco é calculada em O(1) via:
        inicios_de_bloco = diff & ~(diff >> 1)

    Args:
        estado: Inteiro representando a fita atual.
        tamanho: Comprimento total da fita em bits (L).
        alvo: Inteiro representando a fita desejada (padrão: 0).

    Returns:
        Quantidade de blocos contíguos de bits 1 divergentes.

    Examples:
        >>> h2_blocos(0b0000, 4, 0b0000)
        0
        >>> h2_blocos(0b0111, 4, 0b0000)
        1
        >>> h2_blocos(0b1010, 4, 0b0000)
        2
        >>> h2_blocos(0b1100111, 7, 0b0000000)
        2
        >>> h2_blocos(0b11111, 5, 0b00000)
        1
    """
    diff = estado ^ alvo
    if diff == 0:
        return 0
    inicios_de_bloco = diff & ~(diff >> 1)
    return inicios_de_bloco.bit_count()


# Dicionário canônico de heurísticas registradas
HEURISTICAS: dict[str, Callable[[int, int, int], int]] = {
    "h1": h1_hamming,
    "h2": h2_blocos,
    "hamming": h1_hamming,
    "blocos": h2_blocos,
}


def obter_heuristica(nome: str) -> Callable[[int, int, int], int]:
    """Retorna a função heurística correspondente ao nome informado.

    Args:
        nome: Identificador da heurística ('h1', 'h2', 'hamming', 'blocos').

    Returns:
        Função heurística correspondente.

    Raises:
        ValueError: Caso o nome da heurística não seja reconhecido.

    Examples:
        >>> f = obter_heuristica('h1')
        >>> f(0b101, 3, 0)
        2
        >>> f2 = obter_heuristica('h2')
        >>> f2(0b111, 3, 0)
        1
    """
    nome_normalizado = nome.strip().lower()
    if nome_normalizado not in HEURISTICAS:
        chaves_validas = ", ".join(sorted(set(HEURISTICAS.keys())))
        raise ValueError(
            f"Heurística '{nome}' não reconhecida. Opções válidas: {chaves_validas}."
        )
    return HEURISTICAS[nome_normalizado]
