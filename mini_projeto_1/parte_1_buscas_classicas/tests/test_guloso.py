"""Suíte unificada de testes para o Algoritmo Guloso utilizando o padrão Strategy.

Garante que todas as implementações do Algoritmo Guloso (Sequencial e PyTorch SIMD)
respeitem rigorosamente o mesmo contrato comportamental, produzam
soluções de custo idêntico e executem a mesma sequência canônica de ações.
"""

import unittest
from typing import Callable

from flips import (
    carregar,
    flip_bit,
    flip_bloco,
    formatar,
)
from guloso_sequencial import (
    guloso_sequencial,
    resolver_guloso_sequencial,
)
from guloso_torch import (
    guloso_torch,
    resolver_guloso_torch,
)
from heuristica import (
    h1_hamming,
    h2_blocos,
)


class BaseTesteGuloso(unittest.TestCase):
    """Classe base abstrata que define o contrato Strategy para o Algoritmo Guloso."""

    __test__ = False

    guloso_func: Callable
    resolver: Callable

    def test_estado_inicial_ja_e_objetivo(self) -> None:
        """Prova o teste de objetivo na raiz para o algoritmo guloso."""
        resultado = self.resolver("0000", "0000")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(acoes, [])
        self.assertEqual(custo, 0)

    def test_solucao_1_passo_flip_bit(self) -> None:
        """Prova a descoberta de solução com flip de 1 bit."""
        resultado = self.resolver("0001", "0000", heuristica="h2")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(acoes, [("bit", 0)])
        self.assertEqual(custo, 1)

    def test_solucao_1_passo_flip_bloco(self) -> None:
        """Prova a descoberta de solução com flip de bloco contíguo."""
        resultado = self.resolver("01110", "00000", heuristica="h2")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(acoes, [("bloco", 1, 3)])
        self.assertEqual(custo, 1)

    def test_escolha_otima_bloco_vs_bits(self) -> None:
        """Prova a escolha direta de flip de bloco (custo 1) guiado por h2."""
        resultado = self.resolver("111", "000", heuristica="h2")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(custo, 1)
        self.assertEqual(acoes, [("bloco", 0, 3)])

    def test_solucao_multiplos_passos(self) -> None:
        """Prova a resolução de instâncias com múltiplos passos."""
        resultado = self.resolver("101", "000", heuristica="h2")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(custo, 2)
        self.assertEqual(len(acoes), 2)

    def test_alvo_arbitrario_diferente_de_zero(self) -> None:
        """Prova a generalidade para alvos arbitrários não-nulos."""
        resultado = self.resolver("0000", "0110", heuristica="h2")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(custo, 1)
        self.assertEqual(acoes, [("bloco", 1, 2)])

    def test_validacao_execucao_das_acoes_atinge_alvo(self) -> None:
        """Aplica as ações encontradas e valida que o estado final atinge o alvo."""
        string_inicial = "110011"
        string_alvo = "000000"
        estado_inicial, L = carregar(string_inicial)
        estado_alvo, _ = carregar(string_alvo)

        no_res, stats = self.guloso_func(
            estado_inicial, L, estado_alvo, funcao_h=h2_blocos, retornar_estatisticas=True
        )
        self.assertIsNotNone(no_res)

        # Reconstrução
        atual = no_res
        acoes = []
        while atual is not None and atual[1] is not None:
            acoes.append(atual[2])
            atual = atual[1]
        acoes.reverse()

        estado_atual = estado_inicial
        for acao in acoes:
            if acao[0] == "bit":
                estado_atual = flip_bit(estado_atual, acao[1], L)
            elif acao[0] == "bloco":
                estado_atual = flip_bloco(estado_atual, acao[1], acao[2], L)

        self.assertEqual(estado_atual, estado_alvo)
        self.assertEqual(formatar(estado_atual, L), string_alvo)

    def test_validacao_tamanhos_incompativeis(self) -> None:
        """Prova a rejeição com ValueError quando inicial e alvo têm tamanhos distintos."""
        with self.assertRaises(ValueError):
            self.resolver("010", "0100")

    def test_comparacao_h1_vs_h2_eficiencia_em_blocos(self) -> None:
        """Comprova que h2 (blocos) expande muito menos nós que h1 em fitas com clusters."""
        # Fita com blocos contíguos de 1s
        fita = "00001111000011110000"
        _, stats_h1 = self.resolver(fita, heuristica="h1", retornar_estatisticas=True)
        _, stats_h2 = self.resolver(fita, heuristica="h2", retornar_estatisticas=True)

        # h2 deve guiar diretamente para os 2 flips de bloco (2 expansões)
        self.assertEqual(stats_h2["nos_expandidos"], 2)
        # h1 não enxerga blocos e expande mais nós explorando flips parciais
        self.assertGreaterEqual(stats_h1["nos_expandidos"], stats_h2["nos_expandidos"])


class TesteGulosoSequencial(BaseTesteGuloso):
    """Testa a estratégia do Algoritmo Guloso Sequencial."""

    __test__ = True
    guloso_func = staticmethod(guloso_sequencial)
    resolver = staticmethod(resolver_guloso_sequencial)


class TesteGulosoPyTorchSIMD(BaseTesteGuloso):
    """Testa a estratégia do Algoritmo Guloso com PyTorch SIMD."""

    __test__ = True
    guloso_func = staticmethod(guloso_torch)
    resolver = staticmethod(resolver_guloso_torch)


class TesteParidadeEIsomorfismoGuloso(unittest.TestCase):
    """Comprova a paridade estrita e isomorfismo entre as abordagens do Guloso."""

    def test_paridade_e_isomorfismo_estrito_h2(self) -> None:
        """Valida que Sequencial e PyTorch produzem as mesmas métricas e ações com h2."""
        instancias_teste = [
            ("1", "0"),
            ("10", "00"),
            ("101", "000"),
            ("1010", "0000"),
            ("11111", "00000"),
            ("010101", "000000"),
            ("11001100", "00000000"),
            ("1010101010", "0000000000"),
            ("00001001000011001100111", "00000000000000000000000"),  # Instância oficial L=23
        ]

        for s_inicial, s_alvo in instancias_teste:
            res_seq, stats_seq = resolver_guloso_sequencial(
                s_inicial, s_alvo, heuristica="h2", retornar_estatisticas=True
            )
            res_torch, stats_torch = resolver_guloso_torch(
                s_inicial, s_alvo, heuristica="h2", retornar_estatisticas=True
            )

            self.assertIsNotNone(res_seq)
            self.assertIsNotNone(res_torch)

            acoes_seq, custo_seq = res_seq
            acoes_torch, custo_torch = res_torch

            # Custo idêntico
            self.assertEqual(custo_seq, custo_torch)
            # Ações idênticas
            self.assertEqual(acoes_seq, acoes_torch)
            # Nós expandidos e visitados idênticos
            self.assertEqual(stats_seq["nos_expandidos"], stats_torch["nos_expandidos"])
            self.assertEqual(stats_seq["nos_visitados"], stats_torch["nos_visitados"])


if __name__ == "__main__":
    unittest.main()
