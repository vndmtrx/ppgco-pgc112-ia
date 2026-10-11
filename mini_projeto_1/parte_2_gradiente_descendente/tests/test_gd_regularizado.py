"""Suíte de Testes Automatizados para o Gradiente Descendente Regularizado.

Testa:
1. Coerência analítica vs. PyTorch autograd para dJ/dx.
2. Identificação de preservação de blocos: miolo livre vs. borda tracionada.
3. Detecção e comportamento frente a divergência numérica (chicotada elástica).
4. Suporte a diferentes formatos de entrada (str, list, Tensor).
5. Monotonicidade e convergência da perda total para pares (eta, lambda) estáveis.
"""

import pytest
import torch
from gd_regularizado import gradiente_descendente_regularizado


def test_gradiente_analitico_vs_autograd():
    """Valida se o gradiente autograd coincide com a fórmula analítica do enunciado."""
    # Vetor de teste: [x0, x1, x2, x3]
    x_val = [0.2, 0.8, 0.7, 0.1]
    y_val = [0.0, 0.0, 0.0, 0.0]
    lam = 0.5

    x = torch.tensor(x_val, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(y_val, dtype=torch.float32)

    # Perda combinada J(x)
    p_chao = torch.sum((x - y) ** 2)
    p_corda = torch.sum((x[1:] - x[:-1]) ** 2)
    p_total = p_chao + lam * p_corda
    p_total.backward()

    grad_autograd = x.grad.clone()

    # Gradiente analítico elemento a elemento
    # i = 0 (borda esquerda): 2*(x0 - y0) - 2*lam*(x1 - x0) = 2*x0 + 2*lam*(x0 - x1)
    grad_0 = 2 * x_val[0] + 2 * lam * (x_val[0] - x_val[1])
    # i = 1 (interno): 2*x1 + 2*lam*(2*x1 - x0 - x2)
    grad_1 = 2 * x_val[1] + 2 * lam * (2 * x_val[1] - x_val[0] - x_val[2])
    # i = 2 (interno): 2*x2 + 2*lam*(2*x2 - x1 - x3)
    grad_2 = 2 * x_val[2] + 2 * lam * (2 * x_val[2] - x_val[1] - x_val[3])
    # i = 3 (borda direita): 2*x3 + 2*lam*(x3 - x2)
    grad_3 = 2 * x_val[3] + 2 * lam * (x_val[3] - x_val[2])

    grad_analitico = torch.tensor([grad_0, grad_1, grad_2, grad_3], dtype=torch.float32)

    assert torch.allclose(grad_autograd, grad_analitico, atol=1e-5), (
        f"Divergência entre autograd e gradiente analítico: {grad_autograd} vs {grad_analitico}"
    )


def test_miolo_livre_vs_borda_bloco():
    """Valida a hipótese física dos paraquedistas: miolo de bloco idêntico não sofre cisalhamento."""
    # Bloco central idêntico: [0, 1, 1, 1, 0]
    # Pos 0: 0
    # Pos 1: borda (1, vizinhos 0 e 1)
    # Pos 2: miolo (1, vizinhos 1 e 1) -> Tensão de corda deve ser zero!
    # Pos 3: borda (1, vizinhos 1 e 0)
    # Pos 4: 0
    x = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32, requires_grad=True)
    lam = 1.0

    p_chao = torch.sum(x ** 2)
    p_corda = torch.sum((x[1:] - x[:-1]) ** 2)
    p_total = p_chao + lam * p_corda
    p_total.backward()

    # Para a posição 2 (miolo):
    # dL/dx2 = 2 * 1.0 = 2.0
    # dC/dx2 = 2 * (2*1 - 1 - 1) = 0.0
    # Força total no miolo deve ser exatamente 2.0
    assert torch.isclose(x.grad[2], torch.tensor(2.0), atol=1e-5)

    # Para a posição 1 (borda):
    # dC/dx1 = 2 * (2*1 - 0 - 1) = 2.0
    # Força total na borda deve ser 2.0 (chão) + 1.0 * 2.0 (corda) = 4.0
    assert torch.isclose(x.grad[1], torch.tensor(4.0), atol=1e-5)


def test_deteccao_chicotada_elastica_divergencia():
    """Valida se o algoritmo sinaliza divergência numérica quando eta e lambda violam a estabilidade."""
    fita_teste = "01001110"
    # Com eta=0.1 e lambda=20.0, sabemos empiricamente e analiticamente que a perda explode para infinito
    res = gradiente_descendente_regularizado(fita_teste, eta=0.1, lam=20.0, max_passos=30)
    assert res["divergiu"] is True
    assert res["convergiu"] is False


def test_convergencia_estavel_fita_enunciado():
    """Valida que para eta=0.1 e lambda=0.5 a fita do enunciado converge com sucesso para zeros."""
    fita_enunciado = "00001001000011001100111"
    res = gradiente_descendente_regularizado(fita_enunciado, eta=0.1, lam=0.5, max_passos=60)
    assert res["divergiu"] is False
    assert res["convergiu"] is True
    assert torch.all(torch.abs(res["x_final"]) < 1e-2)


def test_tipos_entrada_suportados():
    """Valida compatibilidade de tipos: string de bits, lista de floats e tensor PyTorch."""
    fita_str = "0110"
    fita_lista = [0.0, 1.0, 1.0, 0.0]
    fita_tensor = torch.tensor([0.0, 1.0, 1.0, 0.0])

    r_str = gradiente_descendente_regularizado(fita_str, eta=0.1, lam=0.5, max_passos=10)
    r_list = gradiente_descendente_regularizado(fita_lista, eta=0.1, lam=0.5, max_passos=10)
    r_tens = gradiente_descendente_regularizado(fita_tensor, eta=0.1, lam=0.5, max_passos=10)

    assert torch.allclose(r_str["x_final"], r_list["x_final"], atol=1e-5)
    assert torch.allclose(r_list["x_final"], r_tens["x_final"], atol=1e-5)
