"""Gradiente Descendente Regularizado (Com Função de Custo de Vizinhança Contígua).

Fundamentação Teórica:
- Smola & Zhang (2023), "Dive into Deep Learning", Capítulo 12.
- Mini-Projeto 1, Seção III.D: Inclusão da função custo C(z) que penaliza mudanças não consecutivas.
- Função Objetivo Combinada: J(x) = L(x) + lambda * C(x), onde:
    * L(x) = sum_{k=1}^L (x_k - y_k)^2 (Erro em relação ao alvo)
    * C(x) = sum_{k=1}^{L-1} (x_{k+1} - x_k)^2 (Penalidade de cisalhamento/tensão espacial)
- Regra da Cadeia e Acoplamento de Vizinhança:
    dJ / dx_i = 2(x_i - y_i) + 2*lambda * (2*x_i - x_{i-1} - x_{i+1})  [para elementos internos]
- Cota de Estabilidade Numérica:
    O gradiente máximo escala com (2 + 4*lambda). Para evitar a chicotada elástica divergente,
    a taxa de aprendizado efetiva deve respeitar eta <= 1 / (2 + 4*lambda).
"""

from typing import Dict, List, Optional, Union
import torch


def gradiente_descendente_regularizado(
    fita_inicial: Union[str, List[float], torch.Tensor],
    alvo: Optional[Union[str, List[float], torch.Tensor]] = None,
    eta: float = 0.1,
    lam: float = 0.5,
    max_passos: int = 200,
    tolerancia: float = 1e-6,
) -> Dict[str, Union[torch.Tensor, List[List[float]], List[float], bool, int]]:
    """Executa o Gradiente Descendente minimizando J(x) = L(x) + lambda * C(x).

    Args:
        fita_inicial: Sequência inicial (str de bits, lista de floats ou tensor PyTorch).
        alvo: Sequência alvo desejada (padrão: zeros).
        eta: Taxa de aprendizado (passo).
        lam: Peso de acoplamento da função de custo regularizadora (lambda >= 0).
        max_passos: Número máximo de iterações permitidas.
        tolerancia: Limiar para critério de parada por convergência da perda.

    Returns:
        Dicionário com x_final, trajetoria, perdas_totais, perdas_chao, perdas_corda,
        convergiu, divergiu, passos_executados e ocorreu_overshoot.
    """
    if eta <= 0:
        raise ValueError(f"A taxa de aprendizado eta deve ser estritamente positiva, recebido {eta}.")
    if lam < 0:
        raise ValueError(f"O parâmetro lambda deve ser não-negativo, recebido {lam}.")

    # Normalização do tensor inicial x
    if isinstance(fita_inicial, str):
        valores_x = [float(c) for c in fita_inicial]
        x = torch.tensor(valores_x, dtype=torch.float32)
    elif isinstance(fita_inicial, list):
        x = torch.tensor(fita_inicial, dtype=torch.float32)
    elif isinstance(fita_inicial, torch.Tensor):
        x = fita_inicial.clone().detach().to(dtype=torch.float32)
    else:
        raise TypeError("fita_inicial deve ser str, list de floats ou torch.Tensor.")

    # Normalização do tensor alvo y
    if alvo is None:
        y = torch.zeros_like(x)
    elif isinstance(alvo, str):
        valores_y = [float(c) for c in alvo]
        y = torch.tensor(valores_y, dtype=torch.float32)
    elif isinstance(alvo, list):
        y = torch.tensor(alvo, dtype=torch.float32)
    elif isinstance(alvo, torch.Tensor):
        y = alvo.clone().detach().to(dtype=torch.float32)
    else:
        raise TypeError("alvo deve ser str, list de floats ou torch.Tensor.")

    if x.shape != y.shape:
        raise ValueError(f"Dimensões incompatíveis: {x.shape} != {y.shape}")

    x = x.clone().detach().requires_grad_(True)

    trajetoria: List[List[float]] = [x.detach().clone().tolist()]
    perdas_totais: List[float] = []
    perdas_chao: List[float] = []
    perdas_corda: List[float] = []

    convergiu: bool = False
    divergiu: bool = False
    ocorreu_overshoot: bool = False
    passos_executados: int = 0

    for passo in range(1, max_passos + 1):
        # 1. Componente da perda até o alvo: L(x) = sum (x_k - y_k)^2
        p_chao = torch.sum((x - y) ** 2)

        # 2. Componente de custo de vizinhança contígua: C(x) = sum (x_{k+1} - x_k)^2
        if len(x) > 1:
            p_corda = torch.sum((x[1:] - x[:-1]) ** 2)
        else:
            p_corda = torch.tensor(0.0, device=x.device)

        # 3. Perda combinada J(x)
        p_total = p_chao + lam * p_corda

        val_total = float(p_total.item())
        val_chao = float(p_chao.item())
        val_corda = float(p_corda.item())

        perdas_totais.append(val_total)
        perdas_chao.append(val_chao)
        perdas_corda.append(val_corda)

        # Detecção de divergência numérica (inf ou NaN)
        if torch.isnan(p_total) or torch.isinf(p_total) or val_total > 1e12:
            divergiu = True
            passos_executados = passo
            break

        # Critério de parada por tolerância (quando o erro em relação ao alvo zera)
        if val_chao < tolerancia:
            convergiu = True
            passos_executados = passo - 1
            break

        # Diferenciação automática (backward propaga por L e por C via regra da cadeia)
        p_total.backward()

        with torch.no_grad():
            x -= eta * x.grad

            if torch.any(x < 0.0) or torch.any(x > 1.0):
                ocorreu_overshoot = True

            x.grad.zero_()

        passos_executados = passo
        trajetoria.append(x.detach().clone().tolist())

        # Checagem de convergência após o passo
        if torch.sum((x - y) ** 2).item() < tolerancia:
            convergiu = True
            break

    return {
        "x_final": x.detach().clone(),
        "trajetoria": trajetoria,
        "perdas_totais": perdas_totais,
        "perdas_chao": perdas_chao,
        "perdas_corda": perdas_corda,
        "convergiu": convergiu,
        "divergiu": divergiu,
        "passos_executados": passos_executados,
        "ocorreu_overshoot": ocorreu_overshoot,
    }


if __name__ == "__main__":
    fita_enunciado = "00001001000011001100111"
    print("=" * 65)
    print("TESTE DE PRODUÇÃO: GRADIENTE DESCENDENTE REGULARIZADO")
    print(f"Fita Enunciado (L=23): {fita_enunciado}")
    print("=" * 65)

    for lam in [0.0, 0.5, 2.0, 20.0]:
        res = gradiente_descendente_regularizado(fita_enunciado, eta=0.05, lam=lam, max_passos=40)
        status = "DIVERGIU" if res["divergiu"] else ("CONVERGIU" if res["convergiu"] else "RODOU")
        p_chao_fim = res["perdas_chao"][-1]
        print(f"lambda={lam:4.1f} | eta=0.05 | Status: {status:9s} | Passos: {res['passos_executados']:2d} | Perda Chão Final: {p_chao_fim:.6f}")
