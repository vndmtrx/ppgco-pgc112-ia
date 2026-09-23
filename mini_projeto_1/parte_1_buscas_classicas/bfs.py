"""Algoritmo de Busca em Largura (BFS - Breadth-First Search).

Implementa a busca em largura canônica (Russell & Norvig, 4ª edição, Figura 3.9)
para encontrar a menor sequência de bit-flips necessária para transformar
uma sequência de bits em outra.

Adotamos tuplas nativas (estado, pai, acao, custo) para representar os nós,
garantindo baixo consumo de memória e alta performance sem a necessidade de classes.
"""

import sys
import time
from collections import deque
from typing import Optional

from flips import (
    carregar,
    formatar,
    flip_bit,
    flip_bloco,
    CUSTO_FLIP,
)


def reconstruir_caminho(no: tuple) -> tuple[list[tuple], int]:
    """Percorre os nós a partir do objetivo até a raiz e retorna a sequência de ações e o custo.

    O nó é uma tupla no formato: (estado, pai, acao, custo)

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


def gerar_sucessores(estado: int, tamanho: int) -> list[tuple[int, tuple, int]]:
    """Gera todos os estados sucessores válidos a partir de um estado numérico.

    Retorna uma lista de tuplas contendo:
    (novo_estado, acao_executada, custo_da_acao)

    >>> sucessores = gerar_sucessores(estado=0, tamanho=2)
    >>> len(sucessores)
    3
    """
    sucessores = []

    # 1. Flips individuais de 1 bit para cada posição (0 até tamanho - 1)
    for pos in range(tamanho):
        novo_estado = flip_bit(estado, pos, tamanho)
        sucessores.append((novo_estado, ("bit", pos), CUSTO_FLIP))

    # 2. Flips de blocos de bits consecutivos (tamanho de bloco >= 2)
    for tam_bloco in range(2, tamanho + 1):
        for pos in range(tamanho - tam_bloco + 1):
            novo_estado = flip_bloco(estado, pos, tam_bloco, tamanho)
            sucessores.append((novo_estado, ("bloco", pos, tam_bloco), CUSTO_FLIP))

    return sucessores


def bfs(
    estado_inicial: int,
    tamanho: int,
    estado_alvo: int = 0,
    max_nos: Optional[int] = None,
    retornar_estatisticas: bool = False,
) -> tuple[list[tuple], int] | tuple[tuple[list[tuple], int] | None, dict] | None:
    """Executa a busca em largura canônica para encontrar o caminho de menor custo.

    Representamos cada nó como uma tupla simples:
        (estado, pai, acao, custo)

    Args:
        estado_inicial: Inteiro representando a fita de bits inicial.
        tamanho: Tamanho L da fita de bits.
        estado_alvo: Inteiro representando a fita desejada (padrão: 0, fita zerada).
        max_nos: Limite opcional de nós expandidos para evitar estouro de memória.
        retornar_estatisticas: Se True, retorna também métricas de nós e tempo.

    Retorna:
        (acoes, custo) ou ((acoes, custo), estatisticas) se solicitado.

    >>> acoes, custo = bfs(estado_inicial=5, tamanho=3, estado_alvo=0)
    >>> custo
    2
    """
    inicio_tempo = time.perf_counter()

    # Cria o nó raiz: (estado, pai, acao, custo)
    no_raiz = (estado_inicial, None, None, 0)

    # Teste de objetivo inicial (se já começa no alvo)
    if estado_inicial == estado_alvo:
        solucao = reconstruir_caminho(no_raiz)
        if retornar_estatisticas:
            fim_tempo = time.perf_counter()
            stats = {
                "nos_expandidos": 0,
                "nos_visitados": 1,
                "tempo_s": fim_tempo - inicio_tempo,
            }
            return solucao, stats
        return solucao

    # Fila FIFO (fronteira/borda) e conjunto de visitados (alcançados)
    borda: deque[tuple] = deque([no_raiz])
    alcancados: set[int] = {estado_inicial}
    nos_expandidos = 0

    while borda:
        if max_nos is not None and nos_expandidos >= max_nos:
            if retornar_estatisticas:
                fim_tempo = time.perf_counter()
                stats = {
                    "nos_expandidos": nos_expandidos,
                    "nos_visitados": len(alcancados),
                    "tempo_s": fim_tempo - inicio_tempo,
                }
                return None, stats
            return None

        no_atual = borda.popleft()
        estado_atual, _, _, custo_atual = no_atual
        nos_expandidos += 1

        # Expande os sucessores do estado atual
        for novo_estado, acao, custo_acao in gerar_sucessores(estado_atual, tamanho):
            # Teste de objetivo antecipado (ao gerar o filho)
            if novo_estado == estado_alvo:
                no_filho = (novo_estado, no_atual, acao, custo_atual + custo_acao)
                solucao = reconstruir_caminho(no_filho)
                if retornar_estatisticas:
                    fim_tempo = time.perf_counter()
                    stats = {
                        "nos_expandidos": nos_expandidos,
                        "nos_visitados": len(alcancados) + 1,
                        "tempo_s": fim_tempo - inicio_tempo,
                    }
                    return solucao, stats
                return solucao

            # Se o estado ainda não foi alcançado, adiciona aos visitados e enfileira
            if novo_estado not in alcancados:
                alcancados.add(novo_estado)
                no_filho = (novo_estado, no_atual, acao, custo_atual + custo_acao)
                borda.append(no_filho)

    if retornar_estatisticas:
        fim_tempo = time.perf_counter()
        stats = {
            "nos_expandidos": nos_expandidos,
            "nos_visitados": len(alcancados),
            "tempo_s": fim_tempo - inicio_tempo,
        }
        return None, stats
    return None


def resolver_bfs(
    string_inicial: str,
    string_alvo: Optional[str] = None,
    max_nos: Optional[int] = None,
    retornar_estatisticas: bool = False,
) -> tuple[list[tuple], int] | tuple[tuple[list[tuple], int] | None, dict] | None:
    """Função utilitária de alto nível que recebe strings binárias diretamente.

    >>> resolver_bfs("111", "000")
    ([('bloco', 0, 3)], 1)
    """
    estado_inicial, tamanho = carregar(string_inicial)

    if string_alvo is None:
        estado_alvo = 0
    else:
        estado_alvo, tamanho_alvo = carregar(string_alvo)
        if tamanho_alvo != tamanho:
            raise ValueError(
                f"Tamanho do alvo ({tamanho_alvo}) diferente do tamanho inicial ({tamanho})."
            )

    return bfs(
        estado_inicial=estado_inicial,
        tamanho=tamanho,
        estado_alvo=estado_alvo,
        max_nos=max_nos,
        retornar_estatisticas=retornar_estatisticas,
    )


def executar_cli() -> None:
    """Executa a busca a partir dos argumentos da linha de comando."""
    argumentos = sys.argv[1:]

    # Se nenhum argumento for passado, exibe a documentação de uso e exemplos
    if not argumentos:
        print("=" * 70)
        print(" Algoritmo de Busca em Largura (BFS) - Instruções de Uso")
        print("=" * 70)
        print("\nUso via terminal:")
        print("  python bfs.py <sequencia_inicial> [sequencia_alvo]")
        print("\nParâmetros:")
        print("  <sequencia_inicial>  String binária (ex: 1100111 ou 0000100100001100111)")
        print("  [sequencia_alvo]     (Opcional) String binária desejada (padrão: zeros)")
        print("\nExemplos de execução:")
        print("  1. Fita pequena:")
        print("     python bfs.py 1100111")
        print("\n  2. Fita pequena com alvo customizado:")
        print("     python bfs.py 1100000 0000011")
        print("\n  3. Fita da instância do enunciado (L=23):")
        print("     python bfs.py 00001001000011001100111")
        print("=" * 70)
        return

    string_inicial = argumentos[0]
    string_alvo = argumentos[1] if len(argumentos) > 1 else None

    try:
        estado_inicial, L = carregar(string_inicial)
        alvo_formatado = string_alvo if string_alvo is not None else "0" * L

        print("=" * 70)
        print(f" Executando BFS (Busca em Largura)")
        print("=" * 70)
        print(f"• Fita Inicial:  {string_inicial} (L={L})")
        print(f"• Fita Desejada: {alvo_formatado}")
        print("• Processando busca...")

        resultado, stats = resolver_bfs(
            string_inicial=string_inicial,
            string_alvo=string_alvo,
            retornar_estatisticas=True,
        )

        if resultado is None:
            print("\n❌ Nenhuma solução encontrada dentro dos limites.")
            print(f"• Nós expandidos: {stats['nos_expandidos']}")
            print(f"• Tempo total:    {stats['tempo_s']:.4f} s")
            print("=" * 70)
            return

        acoes, custo = resultado

        print("\n" + "-" * 70)
        print(" ✅ Solução Ótima Encontrada!")
        print("-" * 70)
        print(f"• Custo Total (Passos): {custo}")
        print(f"• Nós Expandidos:       {stats['nos_expandidos']}")
        print(f"• Estados Visitados:    {stats['nos_visitados']}")
        print(f"• Tempo de Execução:    {stats['tempo_s'] * 1000:.3f} ms ({stats['tempo_s']:.5f} s)")

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
            print(f"  [{i}] {desc:<42} -> {formatar(estado_atual, L)}")

        print("=" * 70)

    except Exception as erro:
        print(f"\n❌ Erro na execução: {erro}")
        print("Execute sem argumentos para ver instruções: python bfs.py")


if __name__ == "__main__":
    executar_cli()
