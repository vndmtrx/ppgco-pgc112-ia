"""Experimento MLP v4.1 — Treinamento Otimizado com Acumulação Assíncrona e Pin Memory.

Evolução de Engenharia da linha v4 (Softmax + CrossEntropy):
1. Mantém Rigor Metodológico (Ceteris Paribus):
   - Mesma arquitetura: 784 -> 300 -> 10 (Linear -> ReLU -> Logits)
   - Mesmo batch size: 64
   - Mesmo otimizador: SGD (lr = 0.1)
   - Mesmos dados em RAM e mesma semente determinística
2. Otimização de Pipeline de Hardware (Zero CUDA Synchronization Barriers):
   - Elimina as chamadas repetidas de `.item()` dentro do loop de batch.
   - Perda e acurácia são acumuladas diretamente como tensores na ULA/VRAM do dispositivo (CPU ou GPU).
   - Ocorre exatamente 1 única sincronização com o host ao final da época inteira.
   - Ativa `pin_memory=True` e `non_blocking=True` para transferências DMA diretas no barramento PCIe.

Uso:
    python mlp_v4_1_treinar.py -e 50
"""

import argparse
from pathlib import Path
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from dataset import get_mnist_datasets
from models.mlp import MLP

# Definição dos caminhos
DIRETORIO_PROJETO = Path(__file__).resolve().parent
PASTA_MODELO = DIRETORIO_PROJETO / "modelo"
PASTA_FIGURAS = DIRETORIO_PROJETO / "figuras"


def dataset_para_ram(dataset: Dataset) -> TensorDataset:
    """Converte um Dataset do torchvision em Tensores contíguos na memória RAM."""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    imagens, rotulos = next(iter(loader))
    return TensorDataset(imagens, rotulos)


def treinar_epoca_ce_otimizada(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterio: nn.Module,
    otimizador: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Executa uma época com acumulação assíncrona no dispositivo (Zero GPU Syncs no loop)."""
    modelo.train()
    perda_acumulada = torch.tensor(0.0, device=device)
    acertos_acumulados = torch.tensor(0, dtype=torch.long, device=device)
    total = 0

    for imagens, rotulos in dataloader:
        imagens = imagens.to(device, non_blocking=True)
        rotulos = rotulos.to(device, non_blocking=True)

        otimizador.zero_grad()
        saidas = modelo(imagens)
        perda = criterio(saidas, rotulos)
        perda.backward()
        otimizador.step()

        # Acumula direto no dispositivo sem barreira de sincronização
        perda_acumulada += perda.detach() * imagens.size(0)
        predicoes = saidas.argmax(dim=-1)
        acertos_acumulados += (predicoes == rotulos).sum()
        total += imagens.size(0)

    # Uma única sincronização ao término de todos os lotes da época
    return perda_acumulada.item() / total, acertos_acumulados.item() / total


def avaliar_val_ce_otimizada(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterio: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Avaliação de validação com acumulação assíncrona no dispositivo."""
    modelo.eval()
    perda_acumulada = torch.tensor(0.0, device=device)
    acertos_acumulados = torch.tensor(0, dtype=torch.long, device=device)
    total = 0

    with torch.no_grad():
        for imagens, rotulos in dataloader:
            imagens = imagens.to(device, non_blocking=True)
            rotulos = rotulos.to(device, non_blocking=True)

            saidas = modelo(imagens)
            perda = criterio(saidas, rotulos)

            perda_acumulada += perda.detach() * imagens.size(0)
            predicoes = saidas.argmax(dim=-1)
            acertos_acumulados += (predicoes == rotulos).sum()
            total += imagens.size(0)

    return perda_acumulada.item() / total, acertos_acumulados.item() / total


def salvar_checkpoint(modelo: nn.Module, caminho: Path) -> None:
    """Salva os pesos do modelo no disco."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if caminho.exists():
        print(f"[PERSISTÊNCIA] Substituindo checkpoint existente: {caminho.name}")
    else:
        print(f"[PERSISTÊNCIA] Criando novo checkpoint em: {caminho.name}")

    torch.save(modelo.state_dict(), caminho)
    tamanho_kb = caminho.stat().st_size / 1024
    print(f"[PERSISTÊNCIA] Salvo com sucesso! Tamanho do arquivo: {tamanho_kb:.2f} KB\n")


def salvar_graficos_treinamento(
    historico: dict,
    pasta_figuras: Path,
    num_epocas: int,
) -> Dict[str, Path]:
    """Gera suíte de 5 relatórios visuais modulares em 300 DPI organizados por regime de épocas."""
    pasta_destino = pasta_figuras / f"e{num_epocas}"
    pasta_destino.mkdir(parents=True, exist_ok=True)

    epocas = historico["epocas"]
    perda_tr = historico["perda_treino"]
    perda_vl = historico["perda_val"]
    acc_tr = [a * 100 for a in historico["acc_treino"]]
    acc_vl = [a * 100 for a in historico["acc_val"]]
    gap_gen = [t - v for t, v in zip(acc_tr, acc_vl)]
    tempos = historico["tempo_epoca"]

    caminhos = {}

    # 1. Curva de Perda
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epocas, perda_tr, label="Treino (50k)", color="#1f77b4", linewidth=2)
    ax.plot(epocas, perda_vl, label="Validação (10k)", color="#ff7f0e", linewidth=2, linestyle="--")
    min_val_idx = int(np.argmin(perda_vl))
    ax.scatter(epocas[min_val_idx], perda_vl[min_val_idx], color="red", s=60, zorder=5,
               label=f"Mín Val: {perda_vl[min_val_idx]:.4f} (Época {epocas[min_val_idx]})")
    ax.set_title(f"MLP v4.1 (Otimizado): Curva de Perda — Cross-Entropy ({num_epocas} Épocas)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda (Cross-Entropy)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    p_perda = pasta_destino / f"mlp_treinado_v4_1_e{num_epocas}_perda.png"
    fig.savefig(p_perda, dpi=300)
    plt.close(fig)
    caminhos["perda"] = p_perda

    # 2. Curva de Acurácia
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epocas, acc_tr, label="Treino (50k)", color="#2ca02c", linewidth=2)
    ax.plot(epocas, acc_vl, label="Validação (10k)", color="#d62728", linewidth=2, linestyle="--")
    max_val_idx = int(np.argmax(acc_vl))
    ax.scatter(epocas[max_val_idx], acc_vl[max_val_idx], color="darkred", s=60, zorder=5,
               label=f"Máx Val: {acc_vl[max_val_idx]:.2f}% (Época {epocas[max_val_idx]})")
    ax.set_title(f"MLP v4.1 (Otimizado): Evolução da Acurácia ({num_epocas} Épocas)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    p_acc = pasta_destino / f"mlp_treinado_v4_1_e{num_epocas}_acuracia.png"
    fig.savefig(p_acc, dpi=300)
    plt.close(fig)
    caminhos["acuracia"] = p_acc

    # 3. Gap de Generalização
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epocas, gap_gen, color="#9467bd", linewidth=2, label="Gap: Acc(Treino) - Acc(Val)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.8)
    ax.fill_between(epocas, 0, gap_gen, color="#9467bd", alpha=0.2)
    ax.set_title(f"MLP v4.1: Gap de Generalização ({num_epocas} Épocas)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Gap de Acurácia (% pontos)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    p_gap = pasta_destino / f"mlp_treinado_v4_1_e{num_epocas}_gap_generalizacao.png"
    fig.savefig(p_gap, dpi=300)
    plt.close(fig)
    caminhos["gap"] = p_gap

    # 4. Estabilidade Temporal
    fig, ax = plt.subplots(figsize=(8, 5))
    media_tempo = float(np.mean(tempos))
    ax.plot(epocas, tempos, color="#8c564b", linewidth=1.5, marker="o", markersize=3, label="Tempo Real")
    ax.axhline(media_tempo, color="black", linestyle="--", label=f"Média: {media_tempo:.2f} s/época")
    ax.set_title(f"MLP v4.1: Estabilidade Temporal por Época ({num_epocas} Épocas)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Segundos por Época")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    p_tempo = pasta_destino / f"mlp_treinado_v4_1_e{num_epocas}_tempo_execucao.png"
    fig.savefig(p_tempo, dpi=300)
    plt.close(fig)
    caminhos["tempo"] = p_tempo

    # 5. Painel Integrado 2x2
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs[0, 0].plot(epocas, perda_tr, label="Treino", color="#1f77b4")
    axs[0, 0].plot(epocas, perda_vl, label="Validação", color="#ff7f0e", linestyle="--")
    axs[0, 0].scatter(epocas[min_val_idx], perda_vl[min_val_idx], color="red", s=40, zorder=5)
    axs[0, 0].set_title("Curva de Perda (Cross-Entropy)")
    axs[0, 0].set_xlabel("Época")
    axs[0, 0].set_ylabel("Perda")
    axs[0, 0].grid(True, linestyle=":", alpha=0.6)
    axs[0, 0].legend()

    axs[0, 1].plot(epocas, acc_tr, label="Treino", color="#2ca02c")
    axs[0, 1].plot(epocas, acc_vl, label="Validação", color="#d62728", linestyle="--")
    axs[0, 1].scatter(epocas[max_val_idx], acc_vl[max_val_idx], color="darkred", s=40, zorder=5)
    axs[0, 1].set_title("Curva de Acurácia (%)")
    axs[0, 1].set_xlabel("Época")
    axs[0, 1].set_ylabel("Acurácia (%)")
    axs[0, 1].grid(True, linestyle=":", alpha=0.6)
    axs[0, 1].legend()

    axs[1, 0].plot(epocas, gap_gen, color="#9467bd")
    axs[1, 0].axhline(0, color="gray", linestyle=":", alpha=0.8)
    axs[1, 0].fill_between(epocas, 0, gap_gen, color="#9467bd", alpha=0.2)
    axs[1, 0].set_title("Gap de Generalização (Treino - Validação)")
    axs[1, 0].set_xlabel("Época")
    axs[1, 0].set_ylabel("Diferença (% pontos)")
    axs[1, 0].grid(True, linestyle=":", alpha=0.6)

    axs[1, 1].plot(epocas, tempos, color="#8c564b", marker="o", markersize=2.5)
    axs[1, 1].axhline(media_tempo, color="black", linestyle="--", label=f"Média: {media_tempo:.2f}s")
    axs[1, 1].set_title("Tempo de Execução por Época")
    axs[1, 1].set_xlabel("Época")
    axs[1, 1].set_ylabel("Segundos")
    axs[1, 1].grid(True, linestyle=":", alpha=0.6)
    axs[1, 1].legend()

    plt.suptitle(
        f"Painel Integrado: MLP v4.1 Otimizada ({num_epocas} Épocas) — Cross-Entropy & Softmax",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    p_painel = pasta_destino / f"mlp_treinado_v4_1_e{num_epocas}_painel_geral.png"
    fig.savefig(p_painel, dpi=300)
    plt.close(fig)
    caminhos["painel"] = p_painel

    print(f"\n[GRÁFICOS] 5 arquivos gerados com sucesso na pasta 'figuras/e{num_epocas}/':")
    for k, p in caminhos.items():
        print(f"  • {p.name}")
    print()

    return caminhos


def executar_treinamento(
    num_epocas: int = 50,
    batch_size: int = 64,
    taxa_aprendizado: float = 0.1,
    hidden_dims: list = [300],
) -> Tuple[Path, dict]:
    """Fluxo principal de treinamento isolado da MLP v4.1 com pipeline otimizado."""
    torch.manual_seed(42)
    np.random.seed(42)

    caminho_checkpoint = PASTA_MODELO / f"mlp_treinado_v4_1_e{num_epocas}.pth"
    pasta_destino = PASTA_FIGURAS / f"e{num_epocas}"

    print("=" * 70)
    print(f"EXPERIMENTO MLP v4.1: Softmax & Entropia Cruzada OTIMIZADA ({num_epocas} Épocas)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo:           {device}")
    print(f"Arquitetura:           784 -> 300 -> 10 (Linear -> ReLU -> Logits)")
    print(f"Função de Perda:       CrossEntropyLoss (Softmax Integrado)")
    print(f"Otimização Aplicada:   Acumulação Assíncrona no Dispositivo (Zero Syncs no Loop)")
    print(f"Épocas Solicitadas:    {num_epocas}")
    print(f"Taxa de Aprendizado:   {taxa_aprendizado}")
    print(f"Tamanho do Batch:      {batch_size}")
    print(f"Pasta de Persistência: {PASTA_MODELO}")
    print(f"Pasta de Gráficos:     {pasta_destino}")
    print(f"Arquivo de Pesos:      {caminho_checkpoint.name}")
    print(f"Padrão de Gráficos:    mlp_treinado_v4_1_e{num_epocas}_*.png (em figuras/e{num_epocas}/)")
    print("-" * 70)

    # 1. Pipeline de dados: Treino e Validação em RAM com Pin Memory para DMA
    t_load = time.time()
    ds_treino_cru, ds_val_cru, _ = get_mnist_datasets()

    print("Carregando partições de Treino e Validação na memória RAM...")
    mem_treino = dataset_para_ram(ds_treino_cru)
    mem_val = dataset_para_ram(ds_val_cru)

    tamanho_mb = (
        (mem_treino.tensors[0].element_size() * mem_treino.tensors[0].nelement()) +
        (mem_val.tensors[0].element_size() * mem_val.tensors[0].nelement())
    ) / (1024 * 1024)
    print(f"-> Dados em RAM em {time.time() - t_load:.2f}s | Pegada de Memória: ~{tamanho_mb:.1f} MB")
    print("-> Conjunto de Teste: PERMANECE LACRADO (não carregado)")
    print("-" * 70)

    pin_mem = (device.type == "cuda")
    loader_treino = DataLoader(mem_treino, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_mem)
    loader_val = DataLoader(mem_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem)

    # 2. Inicialização do modelo
    modelo = MLP(input_dim=784, hidden_dims=hidden_dims, num_classes=10).to(device)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.SGD(modelo.parameters(), lr=taxa_aprendizado)

    # 3. Loop de treinamento
    historico = {
        "epocas": [],
        "perda_treino": [],
        "perda_val": [],
        "acc_treino": [],
        "acc_val": [],
        "tempo_epoca": [],
    }

    tempo_inicio = time.time()
    for epoca in range(1, num_epocas + 1):
        t_ep_inicio = time.time()
        perda_tr, acc_tr = treinar_epoca_ce_otimizada(modelo, loader_treino, criterio, otimizador, device)
        perda_vl, acc_vl = avaliar_val_ce_otimizada(modelo, loader_val, criterio, device)
        t_ep = time.time() - t_ep_inicio

        historico["epocas"].append(epoca)
        historico["perda_treino"].append(perda_tr)
        historico["perda_val"].append(perda_vl)
        historico["acc_treino"].append(acc_tr)
        historico["acc_val"].append(acc_vl)
        historico["tempo_epoca"].append(t_ep)

        print(
            f"Época [{epoca:02d}/{num_epocas:02d}] ({t_ep:.2f}s) | "
            f"Treino: {perda_tr:.4f} (Acc: {acc_tr * 100:.2f}%) | "
            f"Val: {perda_vl:.4f} (Acc: {acc_vl * 100:.2f}%)"
        )

    tempo_treino = time.time() - tempo_inicio
    print("-" * 70)
    print(f"Tempo Total de Treinamento: {tempo_treino:.2f} s ({tempo_treino / num_epocas:.2f} s/época)")
    print("-" * 70)

    # 4. Salvar modelo no disco
    salvar_checkpoint(modelo, caminho_checkpoint)

    # 5. Salvar conjunto de gráficos pareados
    graficos = salvar_graficos_treinamento(historico, PASTA_FIGURAS, num_epocas)

    print("=" * 70)
    print("TREINAMENTO MLP v4.1 (OTIMIZADO) CONCLUÍDO COM SUCESSO!")
    print(f"• Arquivo de Pesos: {caminho_checkpoint.name} ({PASTA_MODELO})")
    print(f"• Gráficos Gerados: 5 arquivos em {pasta_destino}")
    print("-" * 70)
    print("Para avaliar este modelo no conjunto de teste oficial, execute:")
    print(f"    python mlp_v4_testar.py -c {caminho_checkpoint.name}")
    print("=" * 70)

    return caminho_checkpoint, graficos


def main():
    parser = argparse.ArgumentParser(
        description="mlp_v4_1_treinar.py: Treinamento Otimizado da MLP (Zero CUDA Synchronization)."
    )
    parser.add_argument(
        "-e", "--epocas",
        type=int,
        default=50,
        help="Número de épocas de treinamento (padrão: 50)",
    )

    args = parser.parse_args()
    executar_treinamento(num_epocas=args.epocas)


if __name__ == "__main__":
    main()
