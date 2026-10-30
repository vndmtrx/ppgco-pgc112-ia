"""Laboratório Didático: O Backpropagation na Prática com PyTorch.

Este script demonstra passo a passo:
1. A ida (Forward Pass) através de matrizes;
2. A medição da tensão do erro (Loss);
3. A volta (Backward Pass via autograd);
4. O ajuste dos pesos pelo Gradiente Descendente;
5. A comprovação de que o novo erro diminuiu!
"""

import torch

def executar_laboratorio():
    print("=" * 65)
    print("LABORATÓRIO DIDÁTICO: O BACKPROPAGATION NA PRÁTICA")
    print("=" * 65)

    # 1. ENTRADAS E PESOS INICIAIS
    # Entrada com 2 valores
    x = torch.tensor([[2.0, 1.0]])

    # Matriz da Camada Oculta (2 neurônios x 2 entradas)
    # requires_grad=True avisa o PyTorch para registrar o grafo de operações
    W1 = torch.tensor([[1.0, 2.0], 
                       [3.0, 0.0]], requires_grad=True)

    # Matriz da Camada de Saída (1 neurônio de saída x 2 entradas da camada anterior)
    W2 = torch.tensor([[2.0, 1.0]], requires_grad=True)

    # Alvo real desejado
    y_alvo = torch.tensor([[10.0]])

    print("\n[1] CONFIGURAÇÃO INICIAL:")
    print(f"    Entrada x:              {x.tolist()[0]}")
    print(f"    Pesos Camada 1 (W1):    {W1.tolist()}")
    print(f"    Pesos Saída (W2):       {W2.tolist()[0]}")
    print(f"    Alvo Desejado (y_alvo): {y_alvo.item()}")

    # 2. FORWARD PASS (A IDA)
    # Multiplicação matricial: x @ W1.T
    z1 = torch.matmul(x, W1.t())
    # Função de ativação ReLU: max(0, z)
    a1 = torch.relu(z1)
    # Saída da rede: a1 @ W2.T
    y_pred = torch.matmul(a1, W2.t())

    # 3. CÁLCULO DA PERDA (A TENSÃO DA MOLA)
    loss = 0.5 * (y_pred - y_alvo) ** 2

    print("\n[2] FORWARD PASS (A IDA):")
    print(f"    Combinação linear z1:   {z1.tolist()[0]}")
    print(f"    Ativação ReLU a1:       {a1.tolist()[0]}")
    print(f"    Predição da rede y:     {y_pred.item():.2f}")
    print(f"    Deformação (y - alvo):  {y_pred.item() - y_alvo.item():+.2f}")
    print(f"    Energia do Erro (Loss): {loss.item():.4f}")

    # 4. BACKWARD PASS (A VOLTA)
    # O PyTorch percorre o grafo de trás para frente usando as matrizes transpostas
    loss.backward()

    print("\n[3] BACKWARD PASS (A VOLTA - GRADIENTES):")
    print(f"    Gradiente de W2 (Saída):    {W2.grad.tolist()[0]}")
    print(f"    Gradiente de W1 (Camada 1): {W1.grad.tolist()}")

    # 5. O PASSO DO GRADIENTE DESCENDENTE (ALIVIANDO A MOLA)
    taxa_aprendizado = 0.01  # eta (tamanho do passo)

    # Atualizamos os pesos sem rastrear gradiente nesta etapa
    with torch.no_grad():
        W1_novo = W1 - taxa_aprendizado * W1.grad
        W2_novo = W2 - taxa_aprendizado * W2.grad

    print(f"\n[4] ATUALIZAÇÃO DOS PESOS (Passo eta = {taxa_aprendizado}):")
    print(f"    W1 Novo: {W1_novo.tolist()}")
    print(f"    W2 Novo: {W2_novo.tolist()[0]}")

    # 6. COMPROVAÇÃO: O NOVO FORWARD PASS
    with torch.no_grad():
        z1_novo = torch.matmul(x, W1_novo.t())
        a1_novo = torch.relu(z1_novo)
        y_pred_novo = torch.matmul(a1_novo, W2_novo.t())
        loss_novo = 0.5 * (y_pred_novo - y_alvo) ** 2

    print("\n[5] COMPROVAÇÃO DO RESULTADO:")
    print(f"    Predição Antiga: {y_pred.item():.4f}  ->  Nova Predição: {y_pred_novo.item():.4f} (Mais perto de {y_alvo.item()}!)")
    print(f"    Erro Antigo:     {loss.item():.4f}  ->  Novo Erro:     {loss_novo.item():.4f} (O erro caiu!)")
    print("=" * 65)

if __name__ == "__main__":
    executar_laboratorio()
