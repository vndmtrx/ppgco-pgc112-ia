"""Testes unitários para validação da arquitetura MLP (models/mlp.py).

Verifica integridade dimensional, contagem exata de parâmetros teóricos,
fluxo de gradientes via backpropagation e validação de persistência se o modelo salvo existir.
"""

from pathlib import Path
import unittest
import torch
import torch.nn as nn
from models.mlp import MLP

CAMINHO_CHECKPOINT = Path(__file__).resolve().parent.parent / "modelo" / "mlp_treinado.pth"


class TestMLPArchitecture(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shape_spatial_input(self):
        """Verifica se entrada com dimensões de imagem (B, 1, 28, 28) produz saída (B, 10)."""
        model = MLP(input_dim=784, hidden_dims=[128], num_classes=10)
        batch_size = 16
        x = torch.randn(batch_size, 1, 28, 28)

        output = model(x)
        self.assertEqual(output.shape, (batch_size, 10))

    def test_forward_shape_flattened_input(self):
        """Verifica se entrada já achatada (B, 784) produz saída (B, 10)."""
        model = MLP(input_dim=784, hidden_dims=[128], num_classes=10)
        batch_size = 8
        x = torch.randn(batch_size, 784)

        output = model(x)
        self.assertEqual(output.shape, (batch_size, 10))

    def test_parameter_count_lecun_1_layer(self):
        """Valida contagem analítica para a arquitetura 784 -> 300 -> 10 de LeCun (1998).

        Camada 1: 784 * 300 (pesos) + 300 (bias) = 235.500
        Camada 2: 300 * 10  (pesos) + 10  (bias) =   3.010
        Total: 238.510 parâmetros.
        """
        model = MLP(input_dim=784, hidden_dims=[300], num_classes=10)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 238510)

    def test_parameter_count_lecun_2_layers(self):
        """Valida contagem analítica para a arquitetura 784 -> 300 -> 100 -> 10.

        Camada 1: 784 * 300 + 300 = 235.500
        Camada 2: 300 * 100 + 100 =  30.100
        Camada 3: 100 * 10  + 10  =   1.010
        Total: 266.610 parâmetros.
        """
        model = MLP(input_dim=784, hidden_dims=[300, 100], num_classes=10)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 266610)

    def test_gradient_flow_backward(self):
        """Garante que a passagem para trás calcula gradientes não-nulos em todas as camadas."""
        model = MLP(input_dim=784, hidden_dims=[64], num_classes=10)
        criterio = nn.MSELoss()

        x = torch.randn(4, 1, 28, 28)
        target = torch.randn(4, 10)

        out = model(x)
        loss = criterio(out, target)
        loss.backward()

        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradiente nulo encontrado em {name}")
            self.assertTrue(
                torch.any(param.grad != 0),
                f"Gradiente idêntico a zero em {name} (sinal de camada desconectada)",
            )

    @unittest.skipIf(
        not CAMINHO_CHECKPOINT.exists(),
        "Modelo persistido não encontrado em modelo/mlp_treinado.pth (execute mlp_v2_treinar.py)",
    )
    def test_modelo_persistido_acuracia_minima(self):
        """Verifica se o modelo salvo no disco pode ser carregado e alcança acurácia mínima (>90%)."""
        from dataset import get_mnist_dataloaders
        from mlp_v2_treinar import carregar_checkpoint

        modelo = carregar_checkpoint(CAMINHO_CHECKPOINT, hidden_dims=[300])
        _, _, test_loader = get_mnist_dataloaders(batch_size=128, num_workers=0)

        imagens, rotulos = next(iter(test_loader))
        with torch.no_grad():
            saidas = modelo(imagens)
            predicoes = saidas.argmax(dim=-1)
            acuracia_lote = (predicoes == rotulos).float().mean().item()

        self.assertGreaterEqual(
            acuracia_lote,
            0.90,
            f"Acurácia do modelo carregado ({acuracia_lote * 100:.2f}%) abaixo do esperado (>90%)",
        )

    def test_calcular_metricas_multiclasse(self):
        """Valida o cálculo analítico da matriz de confusão, precisão, recall e macro-F1."""
        from mlp_v3_testar import calcular_metricas

        reais = [0, 0, 1, 1, 2, 2]
        preditas = [0, 1, 1, 1, 2, 0]
        # Classe 0: TP=1, FP=1 (pred 0 qdo era 2), FN=1 (pred 1 qdo era 0) -> Prec=1/2, Rec=1/2, F1=0.5
        # Classe 1: TP=2, FP=1 (pred 1 qdo era 0), FN=0 -> Prec=2/3, Rec=1.0, F1=0.8
        # Classe 2: TP=1, FP=0, FN=1 (pred 0 qdo era 2) -> Prec=1.0, Rec=0.5, F1=2/3
        res = calcular_metricas(reais, preditas)
        self.assertEqual(res["matriz"].shape, (10, 10))
        self.assertAlmostEqual(res["acuracia"], 4 / 6)
        self.assertAlmostEqual(res["precisao"][0], 0.5)
        self.assertAlmostEqual(res["recall"][1], 1.0)
        self.assertAlmostEqual(res["f1"][0], 0.5)
        self.assertGreater(len(res["top_confusoes"]), 0)


if __name__ == "__main__":
    unittest.main()
