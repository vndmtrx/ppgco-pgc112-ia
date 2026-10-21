"""Módulo de carregamento, pré-processamento e exploração do dataset MNIST.

Este módulo implementa o pipeline de dados da Seção I do mini-projeto:
- Download automático do MNIST via torchvision;
- Cálculo empírico da média e desvio padrão dos pixels de treino;
- Normalização afim Z-score: (x - mean) / std;
- Divisão determinística em Treino (50.000), Validação (10.000) e Teste (10.000);
- Construção de DataLoaders para iteração em mini-batches;
- Geração de gráficos de distribuição de classes e visualização de exemplos.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

# Constantes canônicas da literatura para o MNIST (PyTorch/Torchvision)
CANONICAL_MEAN = 0.1307
CANONICAL_STD = 0.3081


def compute_mean_std(data_dir: str = "./data") -> Tuple[float, float]:
    """Calcula a média e o desvio padrão exatos dos pixels do conjunto de treino original.

    Esta análise atende à Seção I do enunciado:
    "Faça uma breve descrição dos dados: (...) a média e o desvio padrão
    dos pixels usados na normalização."

    Args:
        data_dir: Diretório raiz onde os dados serão armazenados.

    Returns:
        Tupla (media, desvio_padrao) calculada sobre todas as 60.000 imagens
        em escala [0.0, 1.0].
    """
    # Carrega o conjunto cru de treino convertendo apenas para tensor [0, 1]
    raw_train = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Carrega todas as 60.000 imagens para a memória em batch único
    loader = DataLoader(raw_train, batch_size=len(raw_train), shuffle=False)
    images, _ = next(iter(loader))

    mean = float(images.mean().item())
    std = float(images.std().item())
    return mean, std


def get_mnist_datasets(
    data_dir: str = "./data",
    val_size: int = 10000,
    seed: int = 42,
    mean: float = CANONICAL_MEAN,
    std: float = CANONICAL_STD,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Carrega os datasets de Treino, Validação e Teste do MNIST normalizados.

    Aplica a normalização Z-score e realiza a partição determinística:
    60.000 originais -> (60.000 - val_size) para Treino e val_size para Validação.
    O conjunto de Teste permanece com suas 10.000 imagens intocadas.

    Args:
        data_dir: Diretório de armazenamento dos dados brutos.
        val_size: Quantidade de amostras para o split de validação (padrão: 10.000).
        seed: Semente pseudoaleatória para reprodutibilidade da divisão.
        mean: Média dos pixels para centralização em zero.
        std: Desvio padrão para reescalonamento unitário.

    Returns:
        Tupla (train_dataset, val_dataset, test_dataset).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    total_train = len(full_train_dataset)
    train_size = total_train - val_size

    if train_size <= 0:
        raise ValueError(
            f"Tamanho de validação ({val_size}) deve ser menor que o total de treino ({total_train})."
        )

    # Divisão determinística utilizando gerador com seed fixa
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def get_mnist_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_size: int = 10000,
    seed: int = 42,
    mean: float = CANONICAL_MEAN,
    std: float = CANONICAL_STD,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Cria os DataLoaders para iteração em batches durante o treinamento.

    Args:
        data_dir: Diretório de armazenamento dos dados.
        batch_size: Tamanho do mini-batch (padrão: 64).
        val_size: Quantidade de imagens de validação.
        seed: Semente determinística para a partição.
        mean: Média de normalização.
        std: Desvio padrão de normalização.
        num_workers: Número de subprocessos para carregamento de dados.

    Returns:
        Tupla (train_loader, val_loader, test_loader).
    """
    train_dataset, val_dataset, test_dataset = get_mnist_datasets(
        data_dir=data_dir,
        val_size=val_size,
        seed=seed,
        mean=mean,
        std=std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def get_class_distribution(dataset: Dataset) -> Dict[int, int]:
    """Computa o histograma de ocorrências de cada dígito (0 a 9) no dataset.

    Args:
        dataset: Instância de Dataset ou Subset do PyTorch.

    Returns:
        Dicionário mapeando a classe (0..9) para a contagem de amostras.
    """
    counts = {i: 0 for i in range(10)}

    # Se for um Subset (gerado pelo random_split), acessamos os alvos através dos índices
    if isinstance(dataset, Subset):
        base_targets = dataset.dataset.targets
        indices = dataset.indices
        for idx in indices:
            label = int(base_targets[idx])
            counts[label] += 1
    elif hasattr(dataset, "targets"):
        for label in dataset.targets:
            counts[int(label)] += 1
    else:
        for _, label in dataset:
            counts[int(label)] += 1

    return counts


def plot_exploratory_data(
    data_dir: str = "./data",
    output_dir: str = "./figuras",
    seed: int = 42,
) -> Tuple[str, str]:
    """Gera e salva os gráficos exploratórios exigidos na Seção I.

    - figuras/distribuicao_classes.png: Histograma de contagem por classe nos splits.
    - figuras/exemplos_digitos.png: Grade de exemplos visuais de cada dígito.

    Args:
        data_dir: Diretório dos dados.
        output_dir: Diretório de saída para salvar os gráficos em alta resolução.
        seed: Semente determinística.

    Returns:
        Tupla com os caminhos dos arquivos de imagem gerados.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = get_mnist_datasets(
        data_dir=data_dir, val_size=10000, seed=seed
    )

    train_counts = get_class_distribution(train_ds)
    val_counts = get_class_distribution(val_ds)
    test_counts = get_class_distribution(test_ds)

    # 1. Gráfico de Distribuição das Classes (Balanceamento)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    classes = np.arange(10)
    width = 0.25

    ax.bar(classes - width, [train_counts[c] for c in classes], width, label="Treino (50k)", color="#2b5c8f")
    ax.bar(classes, [val_counts[c] for c in classes], width, label="Validação (10k)", color="#f28e2b")
    ax.bar(classes + width, [test_counts[c] for c in classes], width, label="Teste (10k)", color="#59a14f")

    ax.set_xlabel("Classe (Dígito)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Número de Amostras", fontsize=11, fontweight="bold")
    ax.set_title("Distribuição de Amostras por Classe nos Conjuntos de Dados", fontsize=13, fontweight="bold")
    ax.set_xticks(classes)
    ax.set_xticklabels([str(c) for c in classes])
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    dist_file = out_path / "distribuicao_classes.png"
    plt.tight_layout()
    plt.savefig(dist_file, dpi=300)
    plt.close()

    # 2. Grade de Exemplos dos Dígitos (amostras reais do dataset cru)
    raw_dataset = datasets.MNIST(root=data_dir, train=True, download=True)
    
    # Coleta 5 exemplos distintos de cada dígito (0..9)
    samples_per_digit: Dict[int, list] = {i: [] for i in range(10)}
    for img, label in raw_dataset:
        if len(samples_per_digit[label]) < 5:
            samples_per_digit[label].append(img)
        if all(len(v) == 5 for v in samples_per_digit.values()):
            break

    fig, axes = plt.subplots(10, 5, figsize=(6, 12), dpi=300)
    for digit in range(10):
        for col in range(5):
            ax = axes[digit, col]
            ax.imshow(samples_per_digit[digit][col], cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"Dígito {digit}", fontsize=11, fontweight="bold", rotation=0, labelpad=40, va="center")

    plt.suptitle("Amostras Reais do Conjunto MNIST (28x28 pixels)", fontsize=13, fontweight="bold", y=0.995)
    samples_file = out_path / "exemplos_digitos.png"
    plt.tight_layout()
    plt.savefig(samples_file, dpi=300)
    plt.close()

    return str(dist_file), str(samples_file)


if __name__ == "__main__":
    print("=" * 60)
    print("Pipeline de Dados MNIST - Seção I do Mini-Projeto 2")
    print("=" * 60)

    print("\n1. Calculando média e desvio padrão empíricos dos pixels...")
    emp_mean, emp_std = compute_mean_std()
    print(f"   -> Média empírica dos pixels: {emp_mean:.4f} (Canônica: {CANONICAL_MEAN:.4f})")
    print(f"   -> Desvio padrão empírico:   {emp_std:.4f} (Canônica: {CANONICAL_STD:.4f})")

    print("\n2. Construindo datasets com split 50.000 / 10.000 / 10.000...")
    train_ds, val_ds, test_ds = get_mnist_datasets()
    print(f"   -> Treino:    {len(train_ds):,} imagens")
    print(f"   -> Validação: {len(val_ds):,} imagens")
    print(f"   -> Teste:     {len(test_ds):,} imagens")
    print(f"   -> TOTAL:     {len(train_ds) + len(val_ds) + len(test_ds):,} imagens")

    print("\n3. Gerando gráficos exploratórios em './figuras'...")
    f_dist, f_samp = plot_exploratory_data()
    print(f"   -> Gráfico de distribuição: {f_dist}")
    print(f"   -> Grade de amostras:       {f_samp}")

    print("\nConcluído com sucesso!")
