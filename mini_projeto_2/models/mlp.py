"""Implementação da arquitetura Multilayer Perceptron (MLP) para o MNIST.

Atende à Seção II do enunciado oficial:
- Rede totalmente conectada (feedforward);
- Achatamento da imagem espacial (28x28 -> 784);
- Funções de ativação ReLU nas camadas ocultas;
- Camada de saída com 10 logits para classificação multisseletiva (dígitos 0 a 9);
- Número de camadas e neurônios configuráveis como hiperparâmetros.
"""

from typing import List
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Rede Neural Totalmente Conectada (Multilayer Perceptron).

    Args:
        input_dim: Dimensão do vetor de entrada achatado (padrão MNIST: 28*28 = 784).
        hidden_dims: Lista com a quantidade de neurônios em cada camada oculta.
                     Exemplo v1 (básico): [128] (1 camada oculta).
        num_classes: Quantidade de classes na saída (padrão MNIST: 10 dígitos).
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = None,
        num_classes: int = 10,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128]  # Baseline v1

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Camada linear final (gera 10 logits)
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a passagem para frente (Forward Pass).

        Args:
            x: Tensor de entrada com formato (Batch, 1, 28, 28) ou (Batch, 784).

        Returns:
            Logits de saída com dimensão (Batch, 10).
        """
        # Achatamento da estrutura espacial 2D para 1D: (B, 1, 28, 28) -> (B, 784)
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return self.network(x)
