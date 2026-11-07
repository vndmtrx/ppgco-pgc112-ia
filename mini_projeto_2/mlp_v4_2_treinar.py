"""Experimento MLP v4.2 — Treinamento Parametrizável para Exploração de Hiperparâmetros.

Permite explorar variações além das travas históricas de LeCun (1998):
1. Tamanho do lote (Batch Size): 64, 128, 256, 512, 1024;
2. Taxa de aprendizado (Learning Rate): 0.1, 0.01, 0.2, etc.;
3. Momentum de SGD: 0.0, 0.9;
4. Otimizadores alternativos: SGD, Adam, AdamW;
5. Pipeline de Hardware Otimizado: Acumulação assíncrona no dispositivo e DMA pin_memory.

Comportamento:
    Se executado sem nenhum parâmetro (`python mlp_v4_2_treinar.py`), o script NÃO
    executa nada nem cria arquivos; apenas exibe a ajuda formatada dos argumentos.

Uso:
    python mlp_v4_2_treinar.py -e 50 -b 256 -l 0.1
    python mlp_v4_2_treinar.py -e 50 -b 512 -l 0.2 -m 0.9
"""

import argparse
from pathlib import Path
import sys
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


def treinar_epoca(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterio: nn.Module,
    otimizador: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Executa uma época de treino com acumulação assíncrona no dispositivo."""
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

        perda_acumulada += perda.detach() * imagens.size(0)
        predicoes = saidas.argmax(dim=-1)
        acertos_acumulados += (predicoes == rotulos).sum()
        total += imagens.size(0)

    return perda_acumulada.item() / total, acertos_acumulados.item() / total


def avaliar_validacao(
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


def salvar_graficos(
    historico: dict,
    pasta_destino: Path,
    identificador: str,
    titulo_experimento: str,
) -> Dict[str, Path]:
    """Gera suíte completa de 5 gráficos comparativos em 300 DPI."""
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
    ax.set_title(f"MLP v4.2 ({titulo_experimento}): Curva de Perda", fontsize=11, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Perda (Cross-Entropy)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    p_perda = pasta_destino / f"{identificador}_perda.png"
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
    ax.set_title(f"MLP v4.2 ({titulo_experimento}): Acurácia", fontsize=11, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    p_acc = pasta_destino / f"{identificador}_acuracia.png"
    fig.savefig(p_acc, dpi=300)
    plt.close(fig)
    caminhos["acuracia"] = p_acc

    # 3. Gap de Generalização
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epocas, gap_gen, color="#9467bd", linewidth=2, label="Gap: Acc(Treino) - Acc(Val)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.8)
    ax.fill_between(epocas, 0, gap_gen, color="#9467bd", alpha=0.2)
    ax.set_title(f"MLP v4.2 ({titulo_experimento}): Gap de Generalização", fontsize=11, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Gap (% pontos)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    p_gap = pasta_destino / f"{identificador}_gap_generalizacao.png"
    fig.savefig(p_gap, dpi=300)
    plt.close(fig)
    caminhos["gap"] = p_gap

    # 4. Estabilidade Temporal
    fig, ax = plt.subplots(figsize=(8, 5))
    media_tempo = float(np.mean(tempos))
    ax.plot(epocas, tempos, color="#8c564b", linewidth=1.5, marker="o", markersize=3, label="Tempo Real")
    ax.axhline(media_tempo, color="black", linestyle="--", label=f"Média: {media_tempo:.2f} s/época")
    ax.set_title(f"MLP v4.2 ({titulo_experimento}): Estabilidade Temporal", fontsize=11, fontweight="bold")
    ax.set_xlabel("Época")
    ax.set_ylabel("Segundos / Época")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    p_tempo = pasta_destino / f"{identificador}_tempo_execucao.png"
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

    plt.suptitle(f"Painel Integrado: MLP v4.2 ({titulo_experimento})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p_painel = pasta_destino / f"{identificador}_painel_geral.png"
    fig.savefig(p_painel, dpi=300)
    plt.close(fig)
    caminhos["painel"] = p_painel

    print(f"\n[GRÁFICOS] 5 relatórios salvos em: {pasta_destino.relative_to(DIRETORIO_PROJETO)}/")
    for k, p in caminhos.items():
        print(f"  • {p.name}")
    print()

    return caminhos


def criar_otimizador(
    nome: str,
    modelo: nn.Module,
    lr: float,
    momentum: float,
) -> torch.optim.Optimizer:
    """Instancia o otimizador selecionado via linha de comando."""
    nome = nome.lower()
    if nome == "sgd":
        return torch.optim.SGD(modelo.parameters(), lr=lr, momentum=momentum)
    elif nome == "adam":
        return torch.optim.Adam(modelo.parameters(), lr=lr)
    elif nome == "adamw":
        return torch.optim.AdamW(modelo.parameters(), lr=lr)
    else:
        raise ValueError(f"Otimizador '{nome}' não suportado. Opções válidas: sgd, adam, adamw")


def executar_treinamento(
    num_epocas: int,
    batch_size: int,
    lr: float,
    momentum: float,
    otimizador_nome: str,
    hidden_dims: List[int],
) -> Tuple[Path, dict]:
    """Fluxo principal do experimento parametrizado v4.2."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Nomenclatura descritiva única
    dims_str = "-".join(str(d) for d in hidden_dims)
    mom_str = f"_m{momentum}" if momentum > 0 else ""
    opt_str = f"_{otimizador_nome}" if otimizador_nome != "sgd" else ""
    id_exp = f"mlp_treinado_v4_2_e{num_epocas}_b{batch_size}_lr{lr}{mom_str}{opt_str}"
    
    caminho_checkpoint = PASTA_MODELO / f"{id_exp}.pth"
    pasta_destino = PASTA_FIGURAS / f"v4_2_e{num_epocas}_b{batch_size}"

    titulo_exp = f"E={num_epocas}, B={batch_size}, LR={lr}, Opt={otimizador_nome.upper()}"
    if momentum > 0:
        titulo_exp += f", Mom={momentum}"

    print("=" * 70)
    print(f"EXPERIMENTO MLP v4.2: Treinamento Parametrizável ({titulo_exp})")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo:           {device}")
    print(f"Arquitetura:           784 -> {dims_str} -> 10 (Linear -> ReLU -> Logits)")
    print(f"Função de Perda:       CrossEntropyLoss (Softmax Integrado)")
    print(f"Épocas Solicitadas:    {num_epocas}")
    print(f"Tamanho do Batch:      {batch_size} (Passos/época: {50000 // batch_size})")
    print(f"Taxa de Aprendizado:   {lr}")
    print(f"Otimizador:            {otimizador_nome.upper()}" + (f" (Momentum: {momentum})" if momentum > 0 else ""))
    print(f"Pasta de Persistência: {PASTA_MODELO}")
    print(f"Pasta de Gráficos:     {pasta_destino}")
    print(f"Arquivo de Pesos:      {caminho_checkpoint.name}")
    print("-" * 70)

    # 1. Pipeline de dados: Treino e Validação em RAM com Pin Memory
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

    # 2. Inicialização do modelo e otimizador
    modelo = MLP(input_dim=784, hidden_dims=hidden_dims, num_classes=10).to(device)
    criterio = nn.CrossEntropyLoss()
    otimizador = criar_otimizador(otimizador_nome, modelo, lr, momentum)

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
        perda_tr, acc_tr = treinar_epoca(modelo, loader_treino, criterio, otimizador, device)
        perda_vl, acc_vl = avaliar_validacao(modelo, loader_val, criterio, device)
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
    print(f"Tempo Total de Treinamento: {tempo_treino:.2f} s ({tempo_treino / num_epocas:.3f} s/época)")
    print("-" * 70)

    # 4. Salvar checkpoint
    salvar_checkpoint(modelo, caminho_checkpoint)

    # 5. Salvar gráficos
    graficos = salvar_graficos(historico, pasta_destino, id_exp, titulo_exp)

    print("=" * 70)
    print(f"TREINAMENTO MLP v4.2 ({titulo_exp}) CONCLUÍDO COM SUCESSO!")
    print(f"• Arquivo de Pesos: {caminho_checkpoint.name} ({PASTA_MODELO})")
    print(f"• Gráficos Gerados: 5 arquivos em {pasta_destino}")
    print("-" * 70)
    print("Para avaliar este modelo no conjunto de teste oficial, execute:")
    print(f"    python mlp_v4_testar.py -c {caminho_checkpoint.name}")
    print("=" * 70)

    return caminho_checkpoint, graficos


def main():
    parser = argparse.ArgumentParser(
        prog="mlp_v4_2_treinar.py",
        description="Experimento MLP v4.2: Treinador Parametrizável para Exploração de Hiperparâmetros (Mini-Projeto 2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de Uso:
  python mlp_v4_2_treinar.py -e 50 -b 64 -l 0.1             # Baseline LeCun (batch 64)
  python mlp_v4_2_treinar.py -e 50 -b 256 -l 0.1            # Batch maior (256) na RTX 3070
  python mlp_v4_2_treinar.py -e 50 -b 512 -l 0.2 -m 0.9     # Batch grande (512) com Momentum (0.9)
  python mlp_v4_2_treinar.py -e 50 -b 256 --otimizador adam # Adam com batch 256
        """,
    )

    parser.add_argument(
        "-e", "--epocas",
        type=int,
        required=True,
        help="Número de épocas de treinamento (obrigatório, ex: -e 50)",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=64,
        help="Tamanho do batch/lote de imagens (padrão: 64, opções: 64, 128, 256, 512, 1024)",
    )
    parser.add_argument(
        "-l", "--lr",
        type=float,
        default=0.1,
        help="Taxa de aprendizado inicial (padrão: 0.1)",
    )
    parser.add_argument(
        "-m", "--momentum",
        type=float,
        default=0.0,
        help="Fator de momento para o SGD (padrão: 0.0, ex: -m 0.9)",
    )
    parser.add_argument(
        "--otimizador",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "adamw"],
        help="Algoritmo de otimização (padrão: sgd, opções: sgd, adam, adamw)",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=int,
        default=[300],
        help="Lista de neurônios nas camadas ocultas (padrão: 300, ex: --hidden-dims 300 100)",
    )

    # Se nenhum argumento for passado, NÃO executa nada: apenas exibe o help e sai
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    executar_treinamento(
        num_epocas=args.epocas,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        otimizador_nome=args.otimizador,
        hidden_dims=args.hidden_dims,
    )


if __name__ == "__main__":
    main()
