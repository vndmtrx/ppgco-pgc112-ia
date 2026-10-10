"""Suíte de Testes Automatizados para o Gradiente Descendente Puro (Sem Custo).

Cobre os requisitos formais da Seção III.C do enunciado:
1. Equivalência estrita entre gradiente analítico (2x) e autograd do PyTorch.
2. Ponto crítico analítico eta = 0.5 convergindo em exatamente 1 iteração.
3. Detecção e comprovação de overshoot negativo para eta > 0.5 (ex.: eta = 0.6).
4. Convergência da fita canônica de 23 bits com tolerância numérica < 1e-6.
5. Invariância e robustez quanto aos tipos de entrada (str, list, torch.Tensor).
"""

import pytest
import torch
from gd_puro import gradiente_descendente_puro


class TestGradienteDescendentePuro:
    """Conjunto de testes unitários para a convergência e dinâmica de gd_puro."""

    FITA_ENUNCIADO = "00001001000011001100111"
    ALVO_ENUNCIADO = "00000000000000000000000"

    def test_gradiente_analitico_vs_autograd(self):
        """Verifica se o autograd do PyTorch bate exatamente com a derivada analítica 2*(x - y)."""
        x_val = [0.1, 0.4, 0.8, 1.0]
        y_val = [0.0, 0.0, 0.0, 0.0]

        x_tensor = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
        y_tensor = torch.tensor(y_val, dtype=torch.float32)

        perda = torch.sum((x_tensor - y_tensor) ** 2)
        perda.backward()

        grad_esperado = [2.0 * v for v in x_val]
        for obtido, esperado in zip(x_tensor.grad.tolist(), grad_esperado):
            assert pytest.approx(esperado, abs=1e-6) == obtido

    def test_ponto_critico_passo_unico(self):
        """Para eta = 0.5, o algoritmo deve atingir o mínimo global exato no primeiro passo."""
        resultado = gradiente_descendente_puro(self.FITA_ENUNCIADO, self.ALVO_ENUNCIADO, eta=0.5)

        assert resultado["convergiu"] is True
        assert resultado["passos_executados"] == 1
        assert resultado["ocorreu_overshoot"] is False
        assert pytest.approx(0.0, abs=1e-6) == resultado["perdas"][-1]

        # Todos os elementos devem ser exatamente zero
        x_final = resultado["x_final"]
        assert torch.all(torch.abs(x_final) < 1e-6)

    def test_deteccao_overshoot_negativo_para_eta_maior_meio(self):
        """Para eta = 0.6 (> 1/2), deve produzir valores negativos na primeira iteração."""
        resultado = gradiente_descendente_puro(self.FITA_ENUNCIADO, self.ALVO_ENUNCIADO, eta=0.6, max_passos=10)

        assert resultado["ocorreu_overshoot"] is True

        # O primeiro passo após a inicialização (índice 1 da trajetória)
        primeiro_passo = resultado["trajetoria"][1]

        # Índices 4 e 7 eram 1.0 inicialmente: 1.0 - 0.6 * (2 * 1.0) = -0.2
        assert pytest.approx(-0.2, abs=1e-6) == primeiro_passo[4]
        assert pytest.approx(-0.2, abs=1e-6) == primeiro_passo[7]

        # Índices que eram zero devem continuar rigorosamente em zero
        assert pytest.approx(0.0, abs=1e-6) == primeiro_passo[0]
        assert pytest.approx(0.0, abs=1e-6) == primeiro_passo[1]

    def test_convergencia_fita_enunciado_taxas_estaveis(self):
        """Verifica convergência abaixo de 1e-6 para diferentes taxas estáveis."""
        for eta in [0.1, 0.2, 0.4]:
            res = gradiente_descendente_puro(self.FITA_ENUNCIADO, self.ALVO_ENUNCIADO, eta=eta, max_passos=100)
            assert res["convergiu"] is True
            assert res["ocorreu_overshoot"] is False
            assert res["perdas"][-1] < 1e-6

            # O número de passos deve ser estritamente decrescente conforme eta cresce
            if eta == 0.1:
                assert res["passos_executados"] > 25
            elif eta == 0.4:
                assert res["passos_executados"] < 15

    def test_invariancia_tipos_entrada(self):
        """Garante que a função aceita str, list e torch.Tensor com comportamento idêntico."""
        res_str = gradiente_descendente_puro("101", "000", eta=0.2)
        res_list = gradiente_descendente_puro([1.0, 0.0, 1.0], [0.0, 0.0, 0.0], eta=0.2)
        res_tensor = gradiente_descendente_puro(torch.tensor([1., 0., 1.]), torch.tensor([0., 0., 0.]), eta=0.2)

        assert res_str["passos_executados"] == res_list["passos_executados"]
        assert res_str["passos_executados"] == res_tensor["passos_executados"]
        assert pytest.approx(res_str["perdas"][-1], abs=1e-7) == res_list["perdas"][-1]

    def test_validacao_parametros_invalidos(self):
        """Dispara exceções para eta não positivo ou dimensões incompatíveis."""
        with pytest.raises(ValueError, match="estritamente positiva"):
            gradiente_descendente_puro("101", eta=0.0)

        with pytest.raises(ValueError, match="estritamente positiva"):
            gradiente_descendente_puro("101", eta=-0.1)

        with pytest.raises(ValueError, match="Dimensões incompatíveis"):
            gradiente_descendente_puro("101", alvo="00")
