"""Módulo de utilitários para algoritmos de busca e interfaces de terminal.

Contém funções auxiliares reutilizáveis compartilhadas entre todos os
algoritmos de busca (geração de operações, reconstrução de caminhos e exibição).
"""

import sys
from typing import Callable, Optional

from flips import (
    CUSTO_FLIP,
    carregar,
    flip_bit,
    flip_bloco,
    formatar,
)


def gerar_operacoes(tamanho: int) -> list[tuple[int, tuple]]:
    """Pré-computa todas as máscaras bitwise XOR e suas respectivas tuplas de ação.

    Retorna uma lista de tuplas: (mascara_xor, tupla_acao)

    >>> ops = gerar_operacoes(2)
    >>> len(ops)
    3
    >>> ops[0]
    (1, ('bit', 0))
    """
    operacoes = []

    # 1. Flips individuais de 1 bit (posição 0 até tamanho - 1)
    for pos in range(tamanho):
        operacoes.append((1 << pos, ("bit", pos)))

    # 2. Flips de blocos contíguos (tamanho >= 2)
    for tam_bloco in range(2, tamanho + 1):
        mascara_base = (1 << tam_bloco) - 1
        for pos in range(tamanho - tam_bloco + 1):
            operacoes.append((mascara_base << pos, ("bloco", pos, tam_bloco)))

    return operacoes


def reconstruir_caminho(no: tuple) -> tuple[list[tuple], int]:
    """Percorre os nós a partir do objetivo até a raiz e retorna a sequência de ações e o custo.

    O nó é representado como tupla nativa: (estado, pai, acao, custo)

    >>> raiz = (5, None, None, 0)
    >>> filho = (1, raiz, ("bit", 2), 1)
    >>> reconstruir_caminho(filho)
    ([('bit', 2)], 1)
    """
    acoes = []
    atual = no
    custo_total = no[3]

    while atual is not None and atual[1] is not None:
        acao = atual[2]
        if acao is not None:
            acoes.append(acao)
        atual = atual[1]

    acoes.reverse()
    return acoes, custo_total


def imprimir_resultado(
    titulo: str,
    string_inicial: str,
    string_alvo: Optional[str],
    resultado: Optional[tuple[list[tuple], int]],
    stats: Optional[dict] = None,
) -> None:
    """Exibe no terminal o cabeçalho, métricas de execução e o passo a passo da solução."""
    estado_inicial, L = carregar(string_inicial)
    alvo_formatado = string_alvo if string_alvo is not None else "0" * L

    print("=" * 70)
    print(f" {titulo}")
    print("=" * 70)
    print(f"• Fita Inicial:  {string_inicial} (L={L})")
    print(f"• Fita Desejada: {alvo_formatado}")

    if stats and "info_adicional" in stats:
        print(f"• {stats['info_adicional']}")

    if resultado is None:
        print("\n❌ Nenhuma solução encontrada.")
        if stats and "tempo_s" in stats:
            print(f"• Tempo total: {stats['tempo_s']:.5f} s")
        print("=" * 70)
        return

    acoes, custo = resultado

    print("\n" + "-" * 70)
    print(" ✅ Solução Ótima Encontrada com Sucesso!")
    print("-" * 70)
    print(f"• Custo Total (Passos): {custo}")

    if stats:
        if "nos_expandidos" in stats:
            print(f"• Nós Expandidos:       {stats['nos_expandidos']}")
        if "nos_visitados" in stats:
            print(f"• Estados Visitados:    {stats['nos_visitados']}")
        if "tempo_s" in stats:
            tempo_s = stats["tempo_s"]
            print(f"• Tempo de Execução:    {tempo_s * 1000:.3f} ms ({tempo_s:.5f} s)")

    print("\nPasso a Passo das Transformações:")
    estado_atual = estado_inicial
    print(f"  [0] Estado Inicial : {formatar(estado_atual, L)}")

    for i, acao in enumerate(acoes, 1):
        tipo = acao[0]
        if tipo == "bit":
            pos = acao[1]
            estado_atual = flip_bit(estado_atual, pos, L)
            desc = f"Flip 1 bit na posição {pos}"
        elif tipo == "bloco":
            pos, tam = acao[1], acao[2]
            estado_atual = flip_bloco(estado_atual, pos, tam, L)
            desc = f"Flip bloco de {tam} bits a partir da pos {pos}"
        else:
            desc = str(acao)
        print(f"  [{i}] {desc:<42} -> {formatar(estado_atual, L)}")

    print("=" * 70)


def executar_cli_busca(
    titulo: str,
    funcao_resolver: Callable,
    info_adicional: Optional[str | Callable[[], str]] = None,
) -> None:
    """Interface de linha de comando padronizada e parametrizada para os algoritmos de busca."""
    nome_script = sys.argv[0].split("/")[-1] or "script.py"

    if len(sys.argv) < 2:
        print(f"Uso: python {nome_script} <fita_inicial> [fita_alvo]")
        print(f"Exemplo: python {nome_script} 1100111")
        return

    string_inicial = sys.argv[1]
    string_alvo = sys.argv[2] if len(sys.argv) > 2 else None

    resultado, stats = funcao_resolver(
        string_inicial=string_inicial,
        string_alvo=string_alvo,
        retornar_estatisticas=True,
    )
    if stats and info_adicional is not None:
        if callable(info_adicional):
            stats["info_adicional"] = info_adicional()
        else:
            stats["info_adicional"] = info_adicional

    imprimir_resultado(titulo, string_inicial, string_alvo, resultado, stats)

