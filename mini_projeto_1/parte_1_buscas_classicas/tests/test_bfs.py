"""Testes unitários para validar a busca em largura (BFS) com tuplas.

Cada método de teste documenta formalmente a hipótese/propriedade que está sendo
provada, garantindo a corretude teórica e prática do algoritmo canônico.
"""

import unittest
from bfs import (
    reconstruir_caminho,
    gerar_sucessores,
    bfs,
    resolver_bfs,
)
from flips import (
    carregar,
    formatar,
    flip_bit,
    flip_bloco,
    CUSTO_FLIP,
)


class TesteEstruturaNoECaminho(unittest.TestCase):
    """Testa a integridade da estrutura em tupla (estado, pai, acao, custo) e a reconstrução do caminho."""

    def test_reconstruir_caminho_raiz(self) -> None:
        """Prova que a reconstrução do caminho a partir do nó raiz retorna custo zero e lista vazia.

        Hipótese: Um nó raiz não possui nó pai e nenhuma ação foi executada para gerá-lo.
        Portanto, a solução encontrada na raiz deve conter 0 passos e custo 0.
        """
        raiz = (5, None, None, 0)
        acoes, custo = reconstruir_caminho(raiz)
        self.assertEqual(acoes, [])
        self.assertEqual(custo, 0)

    def test_reconstruir_caminho_multiplos_passos(self) -> None:
        """Prova que a travessia reversa dos ponteiros de pai restaura a ordem cronológica correta das ações.

        Hipótese: Ao rastrear da folha até a raiz e inverter a lista, a sequência de ações
        deve ser exatamente [ação1, ação2], acumulando o custo total do nó folha.
        """
        raiz = (5, None, None, 0)
        no1 = (1, raiz, ("bit", 2), 1)
        no2 = (0, no1, ("bit", 0), 2)

        acoes, custo = reconstruir_caminho(no2)
        self.assertEqual(acoes, [("bit", 2), ("bit", 0)])
        self.assertEqual(custo, 2)


class TesteGeracaoDeSucessores(unittest.TestCase):
    """Valida a completude da expansão e o cálculo do fator de ramificação."""

    def test_quantidade_de_sucessores_tamanho_3(self) -> None:
        """Prova a contagem exata de sucessores gerados para uma fita de comprimento L = 3.

        Hipótese: Para L=3, o espaço de ações possíveis a partir de qualquer estado é composto por:
        - 3 flips de 1 bit (posições 0, 1 e 2);
        - 2 flips de bloco de tamanho 2 (posições 0 e 1);
        - 1 flip de bloco de tamanho 3 (posição 0);
        Totalizando exatamente 6 sucessores, todos com custo de transição unitário c = 1.
        """
        sucessores = gerar_sucessores(estado=0, tamanho=3)
        self.assertEqual(len(sucessores), 6)

        # Valida que todas as transições geradas respeitam o custo unitário fixo
        for _, _, custo in sucessores:
            self.assertEqual(custo, CUSTO_FLIP)

    def test_quantidade_de_sucessores_formula_geral(self) -> None:
        """Prova analiticamente a fórmula fechada do fator de ramificação: b = L*(L+1)/2.

        Hipótese: Para qualquer tamanho L, a soma dos flips individuais (L) com todas as janelas
        de blocos possíveis de tamanho 2 até L (L*(L-1)/2) é identicamente igual a L*(L+1)/2.
        """
        for L in range(1, 10):
            sucessores = gerar_sucessores(estado=0, tamanho=L)
            esperado = (L * (L + 1)) // 2
            self.assertEqual(len(sucessores), esperado)


class TesteBFSCorretudeEOtimalidade(unittest.TestCase):
    """Valida as propriedades teóricas de otimalidade, completude e terminação do BFS."""

    def test_estado_inicial_ja_e_objetivo(self) -> None:
        """Prova o teste de objetivo inicial antecipado na raiz (Russell & Norvig, Fig. 3.9).

        Hipótese: Se o estado inicial já for idêntico ao estado alvo, o algoritmo encerra
        imediatamente sem enfileirar ou expandir filhos, retornando custo 0 e ações vazias.
        """
        resultado = resolver_bfs("0000", "0000")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(acoes, [])
        self.assertEqual(custo, 0)

    def test_solucao_1_passo_flip_bit(self) -> None:
        """Prova a capacidade do BFS de identificar soluções de profundidade 1 via flip de 1 bit.

        Hipótese: '0001' -> '0000' deve ser resolvido com 1 única operação ('bit', 0) de custo 1.
        """
        resultado = resolver_bfs("0001", "0000")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(acoes, [("bit", 0)])
        self.assertEqual(custo, 1)

    def test_solucao_1_passo_flip_bloco(self) -> None:
        """Prova a capacidade do BFS de identificar soluções de profundidade 1 via flip de bloco.

        Hipótese: '01110' -> '00000' possui 3 bits contíguos na janela [1, 4), devendo ser
        resolvido em 1 único passo ('bloco', 1, 3) com custo 1.
        """
        resultado = resolver_bfs("01110", "00000")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(acoes, [("bloco", 1, 3)])
        self.assertEqual(custo, 1)

    def test_escolha_otima_bloco_vs_bits(self) -> None:
        """Prova a garantia de otimalidade do BFS em custos uniformes.

        Hipótese: Para a sequência '111', existem dois caminhos possíveis:
        1. Três flips individuais de 1 bit (custo total = 3).
        2. Um único flip de bloco de tamanho 3 (custo total = 1).
        Como o BFS expande em largura nível a nível, ele DEVE encontrar primeiro a solução
        de menor profundidade/custo, escolhendo o flip de bloco com custo 1.
        """
        resultado = resolver_bfs("111", "000")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(custo, 1)
        self.assertEqual(acoes, [("bloco", 0, 3)])

    def test_solucao_multiplos_passos(self) -> None:
        """Prova a exploração multinível em largura quando o objetivo está em profundidade d > 1.

        Hipótese: '101' possui dois bits 1 separados por um zero. Como nenhum bloco contíguo
        pode zerá-los sem inverter o bit central, a solução ótima requer exatamente 2 operações
        de bit individuais (custo 2).
        """
        resultado = resolver_bfs("101", "000")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(custo, 2)
        self.assertEqual(len(acoes), 2)

    def test_alvo_arbitrario_diferente_de_zero(self) -> None:
        """Prova a generalidade do BFS para transformar uma fita em qualquer alvo arbitrário.

        Hipótese: O algoritmo não assume que o alvo é apenas zeros; transformar '0000' em '0110'
        deve encontrar a operação ('bloco', 1, 2) de custo 1 com sucesso.
        """
        resultado = resolver_bfs("0000", "0110")
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(custo, 1)
        self.assertEqual(acoes, [("bloco", 1, 2)])

    def test_validacao_execucao_das_acoes_atinge_alvo(self) -> None:
        """Prova a corretude empírica aplicando a sequência de passos encontrada sobre o estado original.

        Hipótese: Executar passo a passo as ações devolvidas pelo BFS a partir do estado inicial
        deve resultar exatamente no estado alvo, sem resíduos ou erros de convenção posicional.
        """
        string_inicial = "110011"
        string_alvo = "000000"
        estado_inicial, L = carregar(string_inicial)
        estado_alvo, _ = carregar(string_alvo)

        resultado = bfs(estado_inicial, L, estado_alvo)
        self.assertIsNotNone(resultado)
        acoes, custo = resultado
        self.assertEqual(custo, 2)

        # Execução das ações no estado inicial
        estado_atual = estado_inicial
        for acao in acoes:
            if acao[0] == "bit":
                estado_atual = flip_bit(estado_atual, acao[1], L)
            elif acao[0] == "bloco":
                estado_atual = flip_bloco(estado_atual, acao[1], acao[2], L)

        self.assertEqual(estado_atual, estado_alvo)
        self.assertEqual(formatar(estado_atual, L), string_alvo)

    def test_validacao_tamanhos_incompativeis(self) -> None:
        """Prova a rejeição imediata quando fita inicial e fita alvo possuem comprimentos distintos.

        Hipótese: Transformar uma fita de tamanho 3 em uma fita de tamanho 4 é uma operação
        estruturalmente inválida e deve disparar ValueError.
        """
        with self.assertRaises(ValueError):
            resolver_bfs("010", "0100")

    def test_limite_max_nos(self) -> None:
        """Prova a parada segura do BFS quando o orçamento de expansão (max_nos) é excedido.

        Hipótese: Para instâncias onde a solução requer mais expansões do que o limite permitido,
        o BFS encerra com segurança retornando None, prevenindo exaustão de memória RAM.
        """
        resultado = resolver_bfs("10101", "00000", max_nos=1)
        self.assertIsNone(resultado)


if __name__ == "__main__":
    unittest.main()
