"""Testes unitários para validar as sub-rotinas de bit-flip e detecção de overflow.

Cada teste valida uma propriedade formal específica das operações binárias e
das regras estabelecidas no enunciado.
"""

import unittest
from flips import (
    carregar,
    formatar,
    flip_bit,
    flip_bloco,
    CUSTO_FLIP,
)


class TesteFlipsSimples(unittest.TestCase):
    """Testes básicos de conversão e propriedades fundamentais de bit-flip."""

    def test_carregar_e_formatar(self) -> None:
        """Prova o isomorfismo entre a representação textual e numérica inteira.

        Hipótese: A conversão de uma string binária com zeros à esquerda para inteiro
        e sua posterior formatação preserva exatamente o comprimento original L e
        os valores posicionais dos bits.
        """
        estado, L = carregar("0000101")
        self.assertEqual(L, 7)
        self.assertEqual(estado, 5)
        self.assertEqual(formatar(estado, L), "0000101")

    def test_custo_fixo(self) -> None:
        """Prova a uniformidade de custos conforme definido no enunciado.

        Hipótese: Qualquer operação de flip (seja de 1 bit ou de múltiplos bits em bloco)
        possui peso unitário c = 1.
        """
        self.assertEqual(CUSTO_FLIP, 1)

    def test_flip_bit_inverte_corretamente(self) -> None:
        """Prova a corretude e a propriedade de involutividade (reversibilidade) do flip de 1 bit.

        Hipótese:
        1. Inverter o bit 0 de '000' gera '001' (1 em decimal).
        2. Inverter o bit 2 de '001' gera '101' (5 em decimal).
        3. Aplicar XOR com a mesma máscara novamente restaura o estado original (x ^ m ^ m = x).
        """
        # Estado inicial com 3 zeros: '000' (tamanho 3)
        novo = flip_bit(0, 0, 3)
        self.assertEqual(novo, 1)
        self.assertEqual(formatar(novo, 3), "001")

        # Invertendo o bit 2 (mais à esquerda) vira '101' (5)
        novo2 = flip_bit(novo, 2, 3)
        self.assertEqual(novo2, 5)
        self.assertEqual(formatar(novo2, 3), "101")

        # Involutividade: inverter o mesmo bit restaura o valor
        restaurado = flip_bit(novo2, 2, 3)
        self.assertEqual(restaurado, novo)

    def test_flip_bloco_inverte_corretamente(self) -> None:
        """Prova a corretude da inversão por máscara de bloco e sua reversibilidade.

        Hipótese: Inverter um bloco contíguo de tamanho 3 a partir da posição 1
        sobre '00000' deve alterar apenas os bits das posições 1, 2 e 3 ('01110').
        Reaplicar a mesma operação deve retornar ao estado inicial '00000'.
        """
        # '00000' invertendo 3 bits a partir da posição 1 -> '01110'
        novo = flip_bloco(0, 1, 3, 5)
        self.assertEqual(formatar(novo, 5), "01110")

        # Reversibilidade do bloco
        restaurado = flip_bloco(novo, 1, 3, 5)
        self.assertEqual(restaurado, 0)

    def test_flip_bloco_tamanho_um_equivale_ao_flip_bit(self) -> None:
        """Prova a consistência entre as operações de 1 bit e blocos unitários.

        Hipótese: Para qualquer posição válida, flip_bloco(posicao, tamanho_bloco=1)
        produz o mesmo resultado binário que flip_bit(posicao).
        """
        for pos in range(4):
            self.assertEqual(flip_bloco(0, pos, 1, 4), flip_bit(0, pos, 4))


class TesteLimitesEOverflow(unittest.TestCase):
    """Testes de robustez para garantir integridade e prevenção de overflow."""

    def test_overflow_posicao_flip_bit(self) -> None:
        """Prova que acessos a posições negativas ou maiores/iguais a L disparam IndexError.

        Hipótese: Nenhuma operação deve modificar bits fantasmas além da largura da fita.
        """
        with self.assertRaises(IndexError):
            flip_bit(0, 3, 3)

        with self.assertRaises(IndexError):
            flip_bit(0, -1, 3)

    def test_overflow_janela_flip_bloco(self) -> None:
        """Prova que blocos cuja janela [pos, pos + tamanho_bloco) ultrapassa L são rejeitados.

        Hipótese: Tentar aplicar um bloco de 2 bits a partir da posição 2 em uma fita
        de tamanho 3 exigiria as posições 2 e 3 (estourando o limite máximo de índice 2).
        """
        with self.assertRaises(IndexError):
            flip_bloco(0, 2, 2, 3)

    def test_overflow_formatar_estado_invalido(self) -> None:
        """Prova que a formatação rejeita estados numéricos maiores que 2^L - 1.

        Hipótese: O valor decimal 8 ('1000'b) não cabe em uma fita de 3 bits.
        """
        with self.assertRaises(IndexError):
            formatar(8, 3)

    def test_caractere_invalido(self) -> None:
        """Prova que a leitura rejeita strings vazias ou caracteres não binários.

        Hipótese: Entradas contendo caracteres diferentes de '0' e '1' devem disparar ValueError.
        """
        with self.assertRaises(ValueError):
            carregar("001201")
        with self.assertRaises(ValueError):
            carregar("")


class TesteInstanciaDoProfessor(unittest.TestCase):
    """Testa a conformidade direta com a sequência fornecida no enunciado do problema."""

    def test_sequencia_enunciado(self) -> None:
        """Prova a correta interpretação da instância de teste oficial (L=23).

        Hipótese: A sequência inicial '00001001000011001100111' possui L=23 e,
        ao aplicarmos um flip de bloco de tamanho 3 na base (posições 0, 1 e 2),
        o sufixo '111' é transformado em '000' com custo unitário 1.
        """
        string_inicial = "00001001000011001100111"
        estado_inicial, L = carregar(string_inicial)

        self.assertEqual(L, 23)
        self.assertEqual(formatar(estado_inicial, L), string_inicial)

        # Invertendo o bloco de 3 bits na posição 0
        passo1 = flip_bloco(estado_inicial, 0, 3, L)
        self.assertEqual(formatar(passo1, L), "00001001000011001100000")


if __name__ == "__main__":
    unittest.main()
