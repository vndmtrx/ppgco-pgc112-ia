"""Treinamento e avaliacao do modelo MLP seguindo o benchmark de LeCun et al. (1998).

Configuracoes classicas avaliadas:
1. LeCun 1 camada oculta: 784 -> 300 (ReLU) -> 10 (Linear);
2. LeCun 2 camadas ocultas: 784 -> 300 -> 100 -> 10.
"""

import time
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import get_mnist_dataloaders
from models.mlp import MLP


def treinar_e_avaliar_modelo(
    nome_modelo: str,
    hidden_dims: List[int],
    num_epocas: int = 3,
    batch_size: int = 64,
    learning_rate: float = 0.1,
):
    print("=" * 70)
    print(f"EXPERIMENTO: {nome_modelo}")
    print("=" * 70)

    # 1. Pipeline de Dados
    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir="./data",
        batch_size=batch_size,
        val_size=10000,
        seed=42,
    )
    total_treino = len(train_loader.dataset)
    total_val = len(val_loader.dataset)
    total_teste = len(test_loader.dataset)

    # 2. Inicializacao da Rede e Otimizador
    modelo = MLP(input_dim=784, hidden_dims=hidden_dims, num_classes=10)
    total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)

    criterio = nn.MSELoss()
    otimizador = torch.optim.SGD(modelo.parameters(), lr=learning_rate)

    arquitetura_str = " -> ".join(["784"] + [str(h) for h in hidden_dims] + ["10"])
    print(f"Arquitetura:       {arquitetura_str}")
    print(f"Total Parametros:  {total_params:,}")
    print(f"Funcao de Perda:   MSELoss (One-Hot)")
    print(f"Otimizador:        SGD (lr = {learning_rate})")
    print(f"Divisao dos Dados: {total_treino:,} treino | {total_val:,} validacao | {total_teste:,} teste")
    print("-" * 70)

    # 3. Treinamento
    tempo_inicio = time.time()

    for epoca in range(1, num_epocas + 1):
        modelo.train()
        soma_perda_treino = 0.0
        acertos_treino = 0

        for imagens, rotulos in train_loader:
            otimizador.zero_grad()

            saidas = modelo(imagens)
            alvos_one_hot = F.one_hot(rotulos, num_classes=10).float()
            perda = criterio(saidas, alvos_one_hot)

            perda.backward()
            otimizador.step()

            soma_perda_treino += perda.item() * imagens.size(0)
            predicoes = torch.argmax(saidas, dim=1)
            acertos_treino += (predicoes == rotulos).sum().item()

        perda_media_treino = soma_perda_treino / total_treino
        acc_treino = (acertos_treino / total_treino) * 100.0

        # Validacao
        modelo.eval()
        soma_perda_val = 0.0
        acertos_val = 0

        with torch.no_grad():
            for imagens, rotulos in val_loader:
                saidas = modelo(imagens)
                alvos_one_hot = F.one_hot(rotulos, num_classes=10).float()
                perda = criterio(saidas, alvos_one_hot)

                soma_perda_val += perda.item() * imagens.size(0)
                predicoes = torch.argmax(saidas, dim=1)
                acertos_val += (predicoes == rotulos).sum().item()

        perda_media_val = soma_perda_val / total_val
        acc_val = (acertos_val / total_val) * 100.0

        print(
            f"Epoca [{epoca}/{num_epocas}] | "
            f"Perda Treino: {perda_media_treino:.5f} (Acc: {acc_treino:5.2f}%) | "
            f"Perda Val: {perda_media_val:.5f} (Acc: {acc_val:5.2f}%)"
        )

    tempo_treino = time.time() - tempo_inicio
    print(f"Tempo de Treinamento: {tempo_treino:.2f} s")
    print("-" * 70)

    # 4. Avaliacao no Conjunto de Teste Oficial (10.000 imagens)
    modelo.eval()
    total_acertos_teste = 0
    exemplos_erros = []
    indice_global = 0

    with torch.no_grad():
        for imagens, rotulos in test_loader:
            saidas = modelo(imagens)
            predicoes = torch.argmax(saidas, dim=1)

            for i in range(imagens.size(0)):
                real = rotulos[i].item()
                pred = predicoes[i].item()
                indice_global += 1

                if pred == real:
                    total_acertos_teste += 1
                else:
                    if len(exemplos_erros) < 6:
                        exemplos_erros.append({
                            "id": indice_global,
                            "real": real,
                            "chute": pred,
                            "saida_chute": saidas[i][pred].item(),
                            "saida_real": saidas[i][real].item(),
                        })

    total_erros_teste = total_teste - total_acertos_teste
    taxa_acerto = (total_acertos_teste / total_teste) * 100.0
    taxa_erro = (total_erros_teste / total_teste) * 100.0

    print(f"RESULTADO NO CONJUNTO DE TESTE ({total_teste:,} IMAGENS):")
    print(f"• Acertos: {total_acertos_teste:,} ({taxa_acerto:.2f}%)")
    print(f"• Erros:   {total_erros_teste:,} ({taxa_erro:.2f}%)")
    print("-" * 70)

    print("Amostras de Erros (Real vs Predito):")
    for erro in exemplos_erros:
        print(
            f"  [#{erro['id']:<5}] Real: {erro['real']} | Chutou: {erro['chute']} "
            f"(Score Real: {erro['saida_real']:+.2f} vs Score Chute: {erro['saida_chute']:+.2f})"
        )
    print("=" * 70 + "\n")

    return {
        "modelo": nome_modelo,
        "arquitetura": arquitetura_str,
        "parametros": total_params,
        "acertos": total_acertos_teste,
        "erros": total_erros_teste,
        "taxa_acerto": taxa_acerto,
        "taxa_erro": taxa_erro,
        "tempo_s": tempo_treino,
    }


def executar_experimentos_lecun():
    # 1. LeCun 1 Camada Oculta (300 unidades)
    res_300 = treinar_e_avaliar_modelo(
        nome_modelo="LeCun (1998) - 1 Camada Oculta (300 neurônios)",
        hidden_dims=[300],
        num_epocas=3,
        batch_size=64,
        learning_rate=0.1,
    )

    # 2. LeCun 2 Camadas Ocultas (300 + 100 unidades)
    res_300_100 = treinar_e_avaliar_modelo(
        nome_modelo="LeCun (1998) - 2 Camadas Ocultas (300 + 100 neurônios)",
        hidden_dims=[300, 100],
        num_epocas=3,
        batch_size=64,
        learning_rate=0.1,
    )

    # Tabela Comparativa Final
    print("=" * 70)
    print("TABELA COMPARATIVA: EXPERIMENTOS LECUN (1998)")
    print("=" * 70)
    print(f"{'Modelo':<32} | {'Parametros':<12} | {'Erro Teste':<12} | {'Acuracia':<10}")
    print("-" * 70)
    for r in [res_300, res_300_100]:
        print(f"{r['modelo'][:32]:<32} | {r['parametros']:<12,} | {r['taxa_erro']:5.2f}%       | {r['taxa_acerto']:5.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    executar_experimentos_lecun()
