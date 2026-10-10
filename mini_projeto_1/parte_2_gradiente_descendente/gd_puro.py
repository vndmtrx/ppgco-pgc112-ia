"""Gradiente Descendente Puro (Sem Custo) para Otimização de Sequências Contínuas.

Fundamentação Teórica:
- Smola & Zhang (2023), "Dive into Deep Learning", Capítulo 12, Seção 12.3:
  Gradiente Descendente em uma e múltiplas dimensões aplicado sobre funções de perda quadráticas.
- Formulação contínua (relaxamento fuzzy): cada bit da sequência discreta de tamanho L
  é relaxado para uma coordenada x_k em [0.0, 1.0].
- Função de Erro Quadrático Euclidiano: L(x) = sum_{k=1}^L (x_k - y_k)^2.
- Regra de atualização: x_i <- x_i - eta * (dL / dx_i).
- Vetores de transformação entre passos consecutivos: Delta x^(t) = x^(t) - x^(t-1).
"""

from typing import Dict, List, Optional, Tuple, Union
import torch


def gradiente_descendente_puro(
    fita_inicial: Union[str, List[float], torch.Tensor],
    alvo: Optional[Union[str, List[float], torch.Tensor]] = None,
    eta: float = 0.1,
    max_passos: int = 200,
    tolerancia: float = 1e-6,
) -> Dict[str, Union[torch.Tensor, List[List[float]], List[float], bool, int]]:
    """Executa o algoritmo de Gradiente Descendente para minimizar a função de erro quadrático.

    Args:
        fita_inicial: Sequência inicial (string de bits, lista de floats ou tensor PyTorch).
        alvo: Sequência alvo desejada. Se None, assume o vetor nulo (todos zeros),
            conforme especificado no enunciado do mini-projeto.
        eta: Taxa de aprendizado (learning rate, incremento).
        max_passos: Número máximo de iterações permitidas.
        tolerancia: Limiar de tolerância para parada por convergência (||L(x)|| < tolerancia).

    Returns:
        Um dicionário contendo:
        - "x_final": Tensor PyTorch com o estado final atingido.
        - "trajetoria": Lista de estados da fita a cada iteração (como listas de floats).
        - "transformacoes": Lista dos vetores de diferença Delta x^(t) entre passos consecutivos.
        - "perdas": Lista dos valores da função de erro L(x) ao longo das iterações.
        - "convergiu": Booleano indicando se o critério de parada por tolerância foi atingido.
        - "passos_executados": Inteiro com a quantidade total de iterações executadas.
        - "ocorreu_overshoot": Booleano indicando se algum elemento saiu do domínio [0, 1].
    """
    if eta <= 0:
        raise ValueError(f"A taxa de aprendizado eta deve ser estritamente positiva, recebido {eta}.")

    # 1. Normalização do vetor inicial x^(0)
    if isinstance(fita_inicial, str):
        valores_x = [float(caractere) for caractere in fita_inicial]
        x = torch.tensor(valores_x, dtype=torch.float32)
    elif isinstance(fita_inicial, list):
        x = torch.tensor(fita_inicial, dtype=torch.float32)
    elif isinstance(fita_inicial, torch.Tensor):
        x = fita_inicial.clone().detach().to(dtype=torch.float32)
    else:
        raise TypeError("fita_inicial deve ser str, list de floats ou torch.Tensor.")

    # 2. Normalização do vetor alvo y
    if alvo is None:
        y = torch.zeros_like(x)
    elif isinstance(alvo, str):
        valores_y = [float(caractere) for caractere in alvo]
        y = torch.tensor(valores_y, dtype=torch.float32)
    elif isinstance(alvo, list):
        y = torch.tensor(alvo, dtype=torch.float32)
    elif isinstance(alvo, torch.Tensor):
        y = alvo.clone().detach().to(dtype=torch.float32)
    else:
        raise TypeError("alvo deve ser str, list de floats ou torch.Tensor.")

    if x.shape != y.shape:
        raise ValueError(f"Dimensões incompatíveis: fita_inicial {x.shape} != alvo {y.shape}")

    # Habilita o rastreamento do autograd no tensor
    x = x.clone().detach().requires_grad_(True)

    trajetoria: List[List[float]] = [x.detach().clone().tolist()]
    transformacoes: List[List[float]] = []
    perdas: List[float] = []
    ocorreu_overshoot: bool = False

    convergiu: bool = False
    passos_executados: int = 0

    for passo in range(1, max_passos + 1):
        # Avaliação da função de perda quadrática euclidiana: L(x) = sum (x_k - y_k)^2
        perda = torch.sum((x - y) ** 2)
        perdas.append(float(perda.item()))

        # Critério de parada por tolerância numérica
        if perda.item() < tolerancia:
            convergiu = True
            passos_executados = passo - 1
            break

        # Cálculo da inclinação exata da superfície via backpropagation
        perda.backward()

        # Atualização do estado com torch.no_grad()
        with torch.no_grad():
            x_anterior = x.clone()
            x -= eta * x.grad

            # O vetor diferença entre passos consecutivos representa a transformação contínua
            delta_x = (x - x_anterior).tolist()
            transformacoes.append(delta_x)

            # Verifica violação de domínio [0.0, 1.0] (overshoot negativo ou explosão)
            if torch.any(x < 0.0) or torch.any(x > 1.0):
                ocorreu_overshoot = True

            # Limpa o acumulador de gradientes para a próxima iteração
            x.grad.zero_()

        passos_executados = passo
        trajetoria.append(x.detach().clone().tolist())

        # Checagem adicional de convergência após o passo
        perda_atual = torch.sum((x - y) ** 2).item()
        if perda_atual < tolerancia:
            convergiu = True
            perdas.append(float(perda_atual))
            break

    return {
        "x_final": x.detach().clone(),
        "trajetoria": trajetoria,
        "transformacoes": transformacoes,
        "perdas": perdas,
        "convergiu": convergiu,
        "passos_executados": passos_executados,
        "ocorreu_overshoot": ocorreu_overshoot,
    }


if __name__ == "__main__":
    # Teste canônico direto com a fita oficial do enunciado (23 bits)
    fita_enunciado = "00001001000011001100111"
    alvo_enunciado = "00000000000000000000000"

    print("=" * 60)
    print("TESTE CANÔNICO DO ENUNCIADO: GRADIENTE DESCENDENTE PURO")
    print(f"Fita Inicial (L={len(fita_enunciado)}): {fita_enunciado}")
    print(f"Fita Alvo    (L={len(alvo_enunciado)}): {alvo_enunciado}")
    print("=" * 60)

    # 1. Execução com eta = 0.2 (estável)
    resultado_estavel = gradiente_descendente_puro(fita_enunciado, alvo_enunciado, eta=0.2)
    print(f"\n[eta = 0.2] Convergiu em {resultado_estavel['passos_executados']} passos.")
    print(f"Perda inicial: {resultado_estavel['perdas'][0]:.4f} -> Perda final: {resultado_estavel['perdas'][-1]:.6f}")
    print(f"Ocorreu overshoot / negativo? {resultado_estavel['ocorreu_overshoot']}")

    # 2. Execução com eta = 0.6 (overshoot esperado do enunciado)
    resultado_overshoot = gradiente_descendente_puro(fita_enunciado, alvo_enunciado, eta=0.6, max_passos=10)
    print(f"\n[eta = 0.6] Ocorreu overshoot / negativo? {resultado_overshoot['ocorreu_overshoot']}")
    primeiro_passo = resultado_overshoot['trajetoria'][1]
    print(f"Valores após Passo 1 (índices 4 e 7): x[4]={primeiro_passo[4]:.2f}, x[7]={primeiro_passo[7]:.2f}")
