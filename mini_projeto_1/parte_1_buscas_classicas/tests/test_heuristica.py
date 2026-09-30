"""Testes unitários para as funções heurísticas h1 e h2."""

import unittest

from heuristica import (
    h1_hamming,
    h2_blocos,
    obter_heuristica,
)


class TesteHeuristicas(unittest.TestCase):
    """Testa o comportamento e propriedades das funções heurísticas."""

    def test_h1_hamming_casos_basicos(self) -> None:
        """Valida o cálculo da distância de Hamming."""
        self.assertEqual(h1_hamming(0b0000, 4, 0b0000), 0)
        self.assertEqual(h1_hamming(0b0001, 4, 0b0000), 1)
        self.assertEqual(h1_hamming(0b1010, 4, 0b0000), 2)
        self.assertEqual(h1_hamming(0b1111, 4, 0b0000), 4)
        self.assertEqual(h1_hamming(0b1100, 4, 0b0011), 4)

    def test_h2_blocos_casos_basicos(self) -> None:
        """Valida a contagem de blocos contíguos de 1s."""
        self.assertEqual(h2_blocos(0b0000, 4, 0b0000), 0)
        self.assertEqual(h2_blocos(0b0001, 4, 0b0000), 1)
        self.assertEqual(h2_blocos(0b01110, 5, 0b00000), 1)
        self.assertEqual(h2_blocos(0b11111, 5, 0b00000), 1)
        self.assertEqual(h2_blocos(0b10101, 5, 0b00000), 3)
        self.assertEqual(h2_blocos(0b1100111, 7, 0b0000000), 2)

    def test_obter_heuristica_validas(self) -> None:
        """Valida a recuperação de heurísticas pelos apelidos válidos."""
        self.assertEqual(obter_heuristica("h1"), h1_hamming)
        self.assertEqual(obter_heuristica("hamming"), h1_hamming)
        self.assertEqual(obter_heuristica("H1"), h1_hamming)
        self.assertEqual(obter_heuristica("h2"), h2_blocos)
        self.assertEqual(obter_heuristica("blocos"), h2_blocos)
        self.assertEqual(obter_heuristica("H2"), h2_blocos)

    def test_obter_heuristica_invalida(self) -> None:
        """Valida que heurísticas inexistentes disparam ValueError."""
        with self.assertRaises(ValueError):
            obter_heuristica("heuristica_inexistente")


if __name__ == "__main__":
    unittest.main()
