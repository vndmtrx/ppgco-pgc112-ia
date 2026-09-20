"""Testes unitários simples para validar as funções de flip e overflow."""

import unittest
from flips import (
    carregar,
    formatar,
    flip_bit,
    flip_bloco,
    CUSTO_FLIP,
)


class TesteFlipsSimples(unittest.TestCase):
    """Testes básicos de conversão e operações de bit-flip."""

    def test_carregar_e_formatar(self) -> None:
        estado, L = carregar("0000101")
        self.assertEqual(L, 7)
        self.assertEqual(estado, 5)
        self.assertEqual(formatar(estado, L), "0000101")

    def test_custo_fixo(self) -> None:
        self.assertEqual(CUSTO_FLIP, 1)

    def test_flip_bit_inverte_corretamente(self) -> None:
        # Estado inicial com 3 zeros: '000' (tamanho 3)
        # Invertendo o bit 0 (da direita) vira '001' (1)
        novo = flip_bit(0, 0, 3)
        self.assertEqual(novo, 1)
        self.assertEqual(formatar(novo, 3), "001")

        # Invertendo o bit 2 (mais à esquerda) vira '101' (5)
        novo2 = flip_bit(novo, 2, 3)
        self.assertEqual(novo2, 5)
        self.assertEqual(formatar(novo2, 3), "101")

        # Inverter de novo restaura o valor original
        restaurado = flip_bit(novo2, 2, 3)
        self.assertEqual(restaurado, novo)

    def test_flip_bloco_inverte_corretamente(self) -> None:
        # '00000' invertendo 3 bits a partir da posição 1 -> '01110'
        novo = flip_bloco(0, 1, 3, 5)
        self.assertEqual(formatar(novo, 5), "01110")

        # Inverter o mesmo bloco desfaz a operação
        restaurado = flip_bloco(novo, 1, 3, 5)
        self.assertEqual(restaurado, 0)

    def test_flip_bloco_tamanho_um_equivale_ao_flip_bit(self) -> None:
        for pos in range(4):
            self.assertEqual(flip_bloco(0, pos, 1, 4), flip_bit(0, pos, 4))


class TesteLimitesEOverflow(unittest.TestCase):
    """Testes para garantir que operações fora da fita sejam barradas com IndexError."""

    def test_overflow_posicao_flip_bit(self) -> None:
        # Posição 3 em fita de tamanho 3 é fora (válidas são 0, 1, 2)
        with self.assertRaises(IndexError):
            flip_bit(0, 3, 3)

        with self.assertRaises(IndexError):
            flip_bit(0, -1, 3)

    def test_overflow_janela_flip_bloco(self) -> None:
        # Começa na posição 2 com tamanho 2 -> precisa de posições 2 e 3 (estoura limite 3)
        with self.assertRaises(IndexError):
            flip_bloco(0, 2, 2, 3)

    def test_overflow_formatar_estado_invalido(self) -> None:
        # O número 8 (1000 em binário) precisa de 4 bits. Tentar formatar com tamanho 3 deve estourar
        with self.assertRaises(IndexError):
            formatar(8, 3)

    def test_caractere_invalido(self) -> None:
        with self.assertRaises(ValueError):
            carregar("001201")
        with self.assertRaises(ValueError):
            carregar("")


class TesteInstanciaDoProfessor(unittest.TestCase):
    """Testa diretamente a sequência dada no enunciado."""

    def test_sequencia_enunciado(self) -> None:
        texto_inicial = "00001001000011001100111"
        estado_inicial, L = carregar(texto_inicial)

        self.assertEqual(L, 23)
        self.assertEqual(formatar(estado_inicial, L), texto_inicial)

        # O final do texto é '111', que na nossa convenção são os primeiros bits (posições 0, 1 e 2)
        # Invertendo esse bloco de 3 bits com custo 1:
        passo1 = flip_bloco(estado_inicial, 0, 3, L)
        self.assertEqual(formatar(passo1, L), "00001001000011001100000")


if __name__ == "__main__":
    unittest.main()
