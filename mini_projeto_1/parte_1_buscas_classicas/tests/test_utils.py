"""Testes unitários para as funções utilitárias em utils.py."""

import unittest
from unittest.mock import patch
import io
import sys
from flips import CUSTO_FLIP
from utils import executar_cli_busca, gerar_operacoes, imprimir_resultado, reconstruir_caminho


class TesteUtils(unittest.TestCase):
    """Testa as rotinas de pré-computação de operações, reconstrução de caminhos e utilitários de CLI."""

    def test_gerar_operacoes_tamanho_3(self) -> None:
        """Prova que gerar_operacoes(3) gera exatamente 6 operações com custo 1."""
        ops = gerar_operacoes(3)
        self.assertEqual(len(ops), 6)
        mascaras = [m for m, _ in ops]
        # Posições 0, 1, 2 (1, 2, 4) + Blocos tam 2 (3, 6) + Bloco tam 3 (7)
        self.assertEqual(mascaras, [1, 2, 4, 3, 6, 7])

    def test_gerar_operacoes_formula_geral(self) -> None:
        """Prova que a quantidade de operações para qualquer L segue L*(L+1)/2."""
        for L in range(1, 10):
            ops = gerar_operacoes(L)
            self.assertEqual(len(ops), (L * (L + 1)) // 2)

    def test_reconstruir_caminho_raiz(self) -> None:
        """Prova que a raiz tem caminho vazio e custo zero."""
        raiz = (5, None, None, 0)
        acoes, custo = reconstruir_caminho(raiz)
        self.assertEqual(acoes, [])
        self.assertEqual(custo, 0)

    def test_reconstruir_caminho_encadeado(self) -> None:
        """Prova que a ordem cronológica das ações é restaurada."""
        raiz = (5, None, None, 0)
        no1 = (1, raiz, ("bit", 2), 1)
        no2 = (0, no1, ("bloco", 0, 2), 2)
        acoes, custo = reconstruir_caminho(no2)
        self.assertEqual(acoes, [("bit", 2), ("bloco", 0, 2)])
        self.assertEqual(custo, 2)

    def test_imprimir_resultado_formatado(self) -> None:
        """Prova que imprimir_resultado exibe a fita e os passos sem erros."""
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            imprimir_resultado(
                titulo="Teste BFS",
                string_inicial="10",
                string_alvo="00",
                resultado=([("bit", 1)], 1),
                stats={"nos_expandidos": 1, "nos_visitados": 2, "tempo_s": 0.001},
            )
        saida = buffer.getvalue()
        self.assertIn("Teste BFS", saida)
        self.assertIn("Fita Inicial:  10", saida)
        self.assertIn("Custo Total (Passos): 1", saida)

    def test_executar_cli_busca_parametrizada(self) -> None:
        """Prova que executar_cli_busca parseia argumentos e executa a função resolver."""
        mock_resolver = lambda string_inicial, string_alvo, retornar_estatisticas: (
            ([("bit", 0)], 1),
            {"nos_expandidos": 1, "nos_visitados": 1, "tempo_s": 0.0001},
        )
        buffer = io.StringIO()
        with patch.object(sys, "argv", ["script_teste.py", "1"]), patch("sys.stdout", buffer):
            executar_cli_busca(
                titulo="Algoritmo Parametrizado",
                funcao_resolver=mock_resolver,
                info_adicional="Info Extra",
            )
        saida = buffer.getvalue()
        self.assertIn("Algoritmo Parametrizado", saida)
        self.assertIn("Info Extra", saida)


if __name__ == "__main__":
    unittest.main()

