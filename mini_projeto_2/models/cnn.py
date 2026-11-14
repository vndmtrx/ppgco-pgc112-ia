"""Implementação da Rede Neural Convolucional (CNN estilo LeNet) para o MNIST.

Atende à Seção III do enunciado oficial:
- Preserva a estrutura espacial 2D da imagem (1 x 28 x 28);
- Dois blocos convolucionais (Conv2d 5x5 + ReLU + MaxPool2d 2x2) com compartilhamento de pesos;
- Classificador denso com camadas lineares (256 -> 120 -> 84 -> 10 logits);
- Saída em logits puros (compatível com nn.CrossEntropyLoss).
"""

from typing import Dict
import torch
import torch.nn as nn


class LeNet(nn.Module):
    """Arquitetura convolucional no estilo LeNet adaptada para imagens 28x28 do MNIST.

    Fluxo dimensional:
        Entrada: (B, 1, 28, 28)
        Conv1 (5x5, 6 filtros):   (B, 6, 24, 24)
        MaxPool1 (2x2, stride 2): (B, 6, 12, 12)
        Conv2 (5x5, 16 filtros):  (B, 16, 8, 8)
        MaxPool2 (2x2, stride 2): (B, 16, 4, 4)
        Flatten:                  (B, 256)
        Linear1 (256 -> 120):     (B, 120)
        Linear2 (120 -> 84):      (B, 84)
        Linear3 (84 -> 10):       (B, 10)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa a passagem para frente retornando logits (Batch, num_classes)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError(
                f"Entrada esperada com 4 dimensões (B, C, H, W), recebido shape {tuple(x.shape)}."
            )
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

    def resumo_parametros(self) -> Dict[str, int]:
        """Retorna a contagem exata de parâmetros treináveis por camada e bloco."""
        por_modulo = {
            "conv1": sum(p.numel() for p in self.features[0].parameters()),
            "conv2": sum(p.numel() for p in self.features[3].parameters()),
            "fc1": sum(p.numel() for p in self.classifier[0].parameters()),
            "fc2": sum(p.numel() for p in self.classifier[2].parameters()),
            "fc3": sum(p.numel() for p in self.classifier[4].parameters()),
        }
        por_modulo["convolucionais"] = por_modulo["conv1"] + por_modulo["conv2"]
        por_modulo["lineares"] = por_modulo["fc1"] + por_modulo["fc2"] + por_modulo["fc3"]
        por_modulo["total"] = por_modulo["convolucionais"] + por_modulo["lineares"]
        return por_modulo
