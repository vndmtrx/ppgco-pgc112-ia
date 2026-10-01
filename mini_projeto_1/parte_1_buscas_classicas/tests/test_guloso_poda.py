"""Suíte unificada de testes para o Algoritmo Guloso com Poda de Ações.

Valida a exatidão, preservação de invariantes e paridade estrita entre as
implementações Sequencial e PyTorch SIMD com poda de ações orientada a blocos.
"""

import unittest
from typing import Callable

from flips import carregar
from guloso_poda_sequencial import (
    guloso_poda_sequencial,
    resolver_guloso_poda_sequencial,
)
from guloso_poda_torch import (
    guloso_poda_torch,
    resolver_guloso_poda_torch,
)
from guloso_sequencial import resolver_guloso_sequencial
from heuristica import h1_hamming, h2_blocos


class BaseTesteGulosoPoda(unittest.TestCase):
    """Classe base abstrata com os contratos comportamentais da busca com poda."""

    __test__ = False
    resolver: Callable

    def test_estado_inicial_ja_e_objetivo(self) -> None:
        resultado = self.resolver("0000", "0000")
        acoes, custo = resultado
        self.assertEqual(acoes, [])
        self.assertEqual(custo, 0)

    def test_solucao_1_passo_flip_bit(self) -> None:
        resultado = self.resolver("0001", "0000", heuristica="h2")
        acoes, custo = resultado
        self.assertEqual(acoes, [("bit", 0)])
        self.assertEqual(custo, 1)

    def test_solucao_1_passo_flip_bloco(self) -> None:
        resultado = self.resolver("01110", "00000", heuristica="h2")
        acoes, custo = resultado
        self.assertEqual(acoes, [("bloco", 1, 3)])
        self.assertEqual(custo, 1)

    def test_solucao_multiplos_passos(self) -> None:
        resultado = self.resolver("1100111", "0000000", heuristica="h2")
        acoes, custo = resultado
        self.assertEqual(custo, 2)
        self.assertEqual(len(acoes), 2)

    def test_alvo_arbitrario_diferente_de_zero(self) -> None:
        resultado = self.resolver("1010", "0101", heuristica="h2")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertGreater(custo, 0)

    def test_paridade_com_guloso_classico(self) -> None:
        """Verifica que o guloso com poda encontra o mesmo custo que o canônico."""
        fita = "00001001000011001100111"
        res_canonico = resolver_guloso_sequencial(fita, heuristica="h2")
        res_poda = self.resolver(fita, heuristica="h2")
        self.assertEqual(res_poda[1], res_canonico[1])


class TesteGulosoPodaSequencial(BaseTesteGulosoPoda):
    __test__ = True
    resolver = staticmethod(resolver_guloso_poda_sequencial)


class TesteGulosoPodaPyTorch(BaseTesteGulosoPoda):
    __test__ = True
    resolver = staticmethod(resolver_guloso_poda_torch)


class TesteParidadeEIsomorfismoGulosoPoda(unittest.TestCase):
    """Testa a paridade exata entre Sequencial e PyTorch com Poda."""

    def test_isomorfismo_fita_professor(self) -> None:
        fita = "00001001000011001100111"
        res_seq, stats_seq = resolver_guloso_poda_sequencial(fita, retornar_estatisticas=True)
        res_torch, stats_torch = resolver_guloso_poda_torch(fita, retornar_estatisticas=True)
        self.assertEqual(res_seq, res_torch)
        self.assertEqual(stats_seq["nos_expandidos"], stats_torch["nos_expandidos"])


if __name__ == "__main__":
    unittest.main()
