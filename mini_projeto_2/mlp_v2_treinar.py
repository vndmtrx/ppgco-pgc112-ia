"""Experimento MLP v2: Modelo LeCun (1998) com Persistência em Disco.

Esta versão mantém a base conceitual de LeCun (1998) com MSE e One-Hot,
adicionando a infraestrutura de salvamento e carregamento de pesos:
1. Pasta de armazenamento: `modelo/`;
2. Verificação de existência: cria a pasta se não existir e substitui o arquivo caso já exista;
3. Funções modulares de `salvar_checkpoint` e `carregar_checkpoint`;
4. Demonstração de recarga a partir do disco para validação de testes sem necessidade de retreino.
"""

from pathlib import Path
import time
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_mnist_dataloaders
from models.mlp import MLP

# Definição dos caminhos
DIRETORIO_PROJETO = Path(__file__).resolve().parent
PASTA_MODELO = DIRETORIO_PROJETO / "modelo"
CAMINHO_CHECKPOINT = PASTA_MODELO / "mlp_treinado.pth"


def rotulos_para_one_hot(rotulos: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Converte vetor de rótulos inteiros (B,) em matriz One-Hot (B, num_classes)."""
    return torch.zeros(rotulos.size(0), num_classes, device=rotulos.device).scatter_(
        1, rotulos.unsqueeze(1), 1.0
    )


def treinar_epoca_mse(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterio: nn.Module,
    otimizador: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Executa uma época de treino com MSELoss e alvos One-Hot."""
    modelo.train()
    perda_total = 0.0
    acertos = 0
    total = 0

    for imagens, rotulos in dataloader:
        imagens = imagens.to(device)
        rotulos = rotulos.to(device)
        alvos_one_hot = rotulos_para_one_hot(rotulos, num_classes=10)

        otimizador.zero_grad()
        saidas = modelo(imagens)
        perda = criterio(saidas, alvos_one_hot)
        perda.backward()
        otimizador.step()

        perda_total += perda.item() * imagens.size(0)
        predicoes = saidas.argmax(dim=-1)
        acertos += (predicoes == rotulos).sum().item()
        total += imagens.size(0)

    return perda_total / total, acertos / total


def avaliar_mse(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterio: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Avalia o modelo sem atualizar gradientes."""
    modelo.eval()
    perda_total = 0.0
    acertos = 0
    total = 0
    todas_reais = []
    todas_preditas = []

    with torch.no_grad():
        for imagens, rotulos in dataloader:
            imagens = imagens.to(device)
            rotulos = rotulos.to(device)
            alvos_one_hot = rotulos_para_one_hot(rotulos, num_classes=10)

            saidas = modelo(imagens)
            perda = criterio(saidas, alvos_one_hot)

            perda_total += perda.item() * imagens.size(0)
            predicoes = saidas.argmax(dim=-1)
            acertos += (predicoes == rotulos).sum().item()
            total += imagens.size(0)

            todas_reais.extend(rotulos.cpu().tolist())
            todas_preditas.extend(predicoes.cpu().tolist())

    return perda_total / total, acertos / total, todas_reais, todas_preditas


def salvar_checkpoint(modelo: nn.Module, caminho: Path) -> None:
    """Salva os pesos do modelo no disco, garantindo a criação da pasta e substituição controlada."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if caminho.exists():
        print(f"[PERSISTÊNCIA] Arquivo existente detectado. Substituindo: {caminho.name}")
    else:
        print(f"[PERSISTÊNCIA] Criando novo arquivo de pesos em: {caminho}")

    torch.save(modelo.state_dict(), caminho)
    tamanho_kb = caminho.stat().st_size / 1024
    print(f"[PERSISTÊNCIA] Salvo com sucesso! Tamanho do arquivo: {tamanho_kb:.2f} KB\n")


def carregar_checkpoint(caminho: Path, hidden_dims: List[int] = [300]) -> MLP:
    """Instancia a arquitetura vazia e injeta os pesos salvos do arquivo."""
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de checkpoint não encontrado: {caminho}")

    modelo = MLP(input_dim=784, hidden_dims=hidden_dims, num_classes=10)
    estado = torch.load(caminho, weights_only=True)
    modelo.load_state_dict(estado)
    modelo.eval()
    return modelo


def executar_treinamento_v2(
    hidden_dims: List[int] = [300],
    num_epocas: int = 3,
    taxa_aprendizado: float = 0.1,
    batch_size: int = 64,
) -> None:
    """Treina o modelo LeCun (1998), salva em disco e valida a persistência."""
    print("=" * 70)
    print("EXPERIMENTO MLP v2: LeCun (1998) com Persistência em Disco")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo:           {device}")
    print(f"Arquitetura:           784 -> {' -> '.join(map(str, hidden_dims))} -> 10")
    print("Função de Perda:       MSELoss (One-Hot)")
    print(f"Otimizador:            SGD (lr = {taxa_aprendizado})")
    print(f"Pasta de Persistência: {PASTA_MODELO}")
    print(f"Arquivo de Saída:      {CAMINHO_CHECKPOINT.name}")
    print("-" * 70)

    # 1. Pipeline de dados
    loader_treino, loader_val, loader_teste = get_mnist_dataloaders(
        batch_size=batch_size, num_workers=2
    )

    # 2. Inicialização do modelo em memória
    modelo = MLP(input_dim=784, hidden_dims=hidden_dims, num_classes=10).to(device)
    criterio = nn.MSELoss()
    otimizador = torch.optim.SGD(modelo.parameters(), lr=taxa_aprendizado)

    # 3. Loop de treinamento
    tempo_inicio = time.time()
    for epoca in range(1, num_epocas + 1):
        perda_tr, acc_tr = treinar_epoca_mse(modelo, loader_treino, criterio, otimizador, device)
        perda_vl, acc_vl, _, _ = avaliar_mse(modelo, loader_val, criterio, device)

        print(
            f"Época [{epoca}/{num_epocas}] | "
            f"Perda Treino: {perda_tr:.5f} (Acc: {acc_tr * 100:.2f}%) | "
            f"Perda Val: {perda_vl:.5f} (Acc: {acc_vl * 100:.2f}%)"
        )

    tempo_treino = time.time() - tempo_inicio
    print("-" * 70)
    print(f"Tempo de Treinamento: {tempo_treino:.2f} s")
    print("-" * 70)

    # 4. Salvar modelo na pasta modelo/ (Persistência)
    salvar_checkpoint(modelo, CAMINHO_CHECKPOINT)

    # 5. Teste de Persistência: Carregar do disco em novo objeto
    print("TESTE DE PERSISTÊNCIA: Carregando modelo diretamente do arquivo .pth...")
    modelo_carregado = carregar_checkpoint(CAMINHO_CHECKPOINT, hidden_dims=hidden_dims).to(device)

    # 6. Avaliação final com o modelo recarregado
    _, acc_teste, reais, preditas = avaliar_mse(modelo_carregado, loader_teste, criterio, device)
    total_testes = len(reais)
    acertos = sum(1 for r, p in zip(reais, preditas) if r == p)
    erros = total_testes - acertos

    print("=" * 70)
    print(f"RESULTADO NO CONJUNTO DE TESTE COM MODELO CARREGADO ({total_testes:,} IMAGENS):")
    print(f"• Acertos: {acertos:,} ({acc_teste * 100:.2f}%)")
    print(f"• Erros:   {erros:,} ({(erros / total_testes) * 100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    executar_treinamento_v2(hidden_dims=[300], num_epocas=3, taxa_aprendizado=0.1)
