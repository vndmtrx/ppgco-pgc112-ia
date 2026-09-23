"""Suíte unificada de testes para algoritmos BFS utilizando o padrão Strategy.

Garante que todas as implementações de BFS (Sequencial e PyTorch SIMD)
respeitem rigorosamente o mesmo contrato comportamental, produzam
soluções de custo ótimo idênticas e executem a mesma sequência canônica de ações.
"""

import unittest
from typing import Callable

from bfs_sequencial import (
    bfs_sequencial,
    resolver_bfs_sequencial,
)
from bfs_torch import (
    bfs_torch,
    resolver_bfs_torch,
)
from flips import (
    carregar,
    flip_bit,
    flip_bloco,
    formatar,
)


class BaseTesteBFS(unittest.TestCase):
    """Classe base abstrata que define o contrato Strategy para testes de BFS."""

    __test__ = False

    bfs_func: Callable
    resolver: Callable

    def test_estado_inicial_ja_e_objetivo(self) -> None:
        """Prova o teste de objetivo na raiz para a estratégia."""
        resultado = self.resolver("0000", "0000")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(acoes, [])
        self.assertEqual(custo, 0)

    def test_solucao_1_passo_flip_bit(self) -> None:
        """Prova a descoberta de solução ótima com flip de 1 bit."""
        resultado = self.resolver("0001", "0000")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(acoes, [("bit", 0)])
        self.assertEqual(custo, 1)

    def test_solucao_1_passo_flip_bloco(self) -> None:
        """Prova a descoberta de solução ótima com flip de bloco contíguo."""
        resultado = self.resolver("01110", "00000")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(acoes, [("bloco", 1, 3)])
        self.assertEqual(custo, 1)

    def test_escolha_otima_bloco_vs_bits(self) -> None:
        """Prova a garantia fundamental de otimalidade: bloco (custo 1) vs 3 bits (custo 3)."""
        resultado = self.resolver("111", "000")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(custo, 1)
        self.assertEqual(acoes, [("bloco", 0, 3)])

    def test_solucao_multiplos_passos(self) -> None:
        """Prova a expansão em múltiplos níveis para '101' -> '000' (custo 2)."""
        resultado = self.resolver("101", "000")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(custo, 2)
        self.assertEqual(len(acoes), 2)

    def test_alvo_arbitrario_diferente_de_zero(self) -> None:
        """Prova a generalidade para alvos arbitrários não-nulos."""
        resultado = self.resolver("0000", "0110")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(custo, 1)
        self.assertEqual(acoes, [("bloco", 1, 2)])

    def test_validacao_execucao_das_acoes_atinge_alvo(self) -> None:
        """Aplica as ações encontradas e valida que o estado final é matematicamente idêntico ao alvo."""
        string_inicial = "110011"
        string_alvo = "000000"
        estado_inicial, L = carregar(string_inicial)
        estado_alvo, _ = carregar(string_alvo)

        resultado = self.bfs_func(estado_inicial, L, estado_alvo)
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(custo, 2)

        estado_atual = estado_inicial
        for acao in acoes:
            if acao[0] == "bit":
                estado_atual = flip_bit(estado_atual, acao[1], L)
            elif acao[0] == "bloco":
                estado_atual = flip_bloco(estado_atual, acao[1], acao[2], L)

        self.assertEqual(estado_atual, estado_alvo)
        self.assertEqual(formatar(estado_atual, L), string_alvo)

    def test_validacao_tamanhos_incompativeis(self) -> None:
        """Prova a rejeição com ValueError quando fita inicial e alvo possuem comprimentos distintos."""
        with self.assertRaises(ValueError):
            self.resolver("010", "0100")


class TesteBFSSequencial(BaseTesteBFS):
    """Testa a estratégia de BFS Sequencial Canônico."""

    __test__ = True
    bfs_func = staticmethod(bfs_sequencial)
    resolver = staticmethod(resolver_bfs_sequencial)


class TesteBFSPyTorchSIMD(BaseTesteBFS):
    """Testa a estratégia de BFS Vetorizado com PyTorch SIMD."""

    __test__ = True
    bfs_func = staticmethod(bfs_torch)
    resolver = staticmethod(resolver_bfs_torch)


class TesteParidadeEIsomorfismoBFS(unittest.TestCase):
    """Comprova a paridade estrita e isomorfismo de busca entre as estratégias."""

    def test_paridade_e_isomorfismo_estrito(self) -> None:
        """Valida que Sequencial e PyTorch geram as exatas mesmas métricas e caminhos canônicos."""
        instancias_teste = [
            ("1", "0"),
            ("10", "00"),
            ("101", "000"),
            ("1010", "0000"),
            ("11111", "00000"),
            ("010101", "000000"),
            ("11001100", "00000000"),
            ("1010101010", "0000000000"),
        ]

        for s_inicial, s_alvo in instancias_teste:
            res_seq, stats_seq = resolver_bfs_sequencial(s_inicial, s_alvo, retornar_estatisticas=True)
            res_torch, stats_torch = resolver_bfs_torch(s_inicial, s_alvo, retornar_estatisticas=True)

            self.assertIsNotNone(res_seq)
            self.assertIsNotNone(res_torch)

            acoes_seq, custo_seq = res_seq
            acoes_torch, custo_torch = res_torch

            # Custo ótimo idêntico
            self.assertEqual(custo_seq, custo_torch)
            # Sequência de transformações rigorosamente idêntica (isomorfismo de busca)
            self.assertEqual(acoes_seq, acoes_torch)
            # Paridade exata de nós expandidos e estados visitados
            self.assertEqual(stats_seq["nos_expandidos"], stats_torch["nos_expandidos"])
            self.assertEqual(stats_seq["nos_visitados"], stats_torch["nos_visitados"])


if __name__ == "__main__":
    unittest.main()


