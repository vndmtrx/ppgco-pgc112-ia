"""Experimento MLP v4 — Módulo de Treinamento com Softmax e Entropia Cruzada.

Atende à Seção II do enunciado oficial (Mini-Projeto 2):
1. Arquitetura MLP: 784 -> 300 -> 10 com ativações ReLU;
2. Classificador de Saída: Softmax nativo integrado via CrossEntropyLoss;
3. Otimizador: SGD (taxa de aprendizado 0.1, tamanho de lote 64);
4. Comparativo Direto (Ceteris Paribus): Mantém a mesma arquitetura, lote e dados em RAM
   do experimento v3 para isolar com rigor científico o impacto da transição de MSE para Cross-Entropy.

Uso:
    python mlp_v4_treinar.py -e 3
    python mlp_v4_treinar.py --epocas 50
    python mlp_v4_treinar.py -e 100
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


def treinar_epoca_ce(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterio: nn.Module,
    otimizador: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Executa uma época de treino com CrossEntropyLoss (Softmax nativo em C++/CUDA)."""
    modelo.train()
    perda_total = 0.0
    acertos = 0
    total = 0

    for imagens, rotulos in dataloader:
        imagens = imagens.to(device)
        rotulos = rotulos.to(device)

        otimizador.zero_grad()
        saidas = modelo(imagens)  # Logits lineares (B, 10)
        perda = criterio(saidas, rotulos)  # CrossEntropyLoss recebe logits e índices inteiros!
        perda.backward()
        otimizador.step()

        perda_total += perda.item() * imagens.size(0)
        predicoes = saidas.argmax(dim=-1)
        acertos += (predicoes == rotulos).sum().item()
        total += imagens.size(0)

    return perda_total / total, acertos / total


def avaliar_val_ce(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterio: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Avalia o modelo estritamente no conjunto de validação com CrossEntropyLoss."""
    modelo.eval()
    perda_total = 0.0
    acertos = 0
    total = 0

    with torch.no_grad():
        for imagens, rotulos in dataloader:
            imagens = imagens.to(device)
            rotulos = rotulos.to(device)

            saidas = modelo(imagens)
            perda = criterio(saidas, rotulos)

            perda_total += perda.item() * imagens.size(0)
            predicoes = saidas.argmax(dim=-1)
            acertos += (predicoes == rotulos).sum().item()
            total += imagens.size(0)

    return perda_total / total, acertos / total


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
    """Gera e salva o conjunto de gráficos de acompanhamento do treinamento v4.

    Os gráficos são organizados na subpasta de épocas: figuras/e{num_epocas}/
    Arquivos gerados:
    1. mlp_treinado_v4_e{E}_perda.png
    2. mlp_treinado_v4_e{E}_acuracia.png
    3. mlp_treinado_v4_e{E}_gap_generalizacao.png
    4. mlp_treinado_v4_e{E}_tempo_execucao.png
    5. mlp_treinado_v4_e{E}_painel_geral.png
    """
    subpasta = f"e{num_epocas}"
    pasta_destino = pasta_figuras / subpasta
    pasta_destino.mkdir(parents=True, exist_ok=True)
    prefixo = f"mlp_treinado_v4_e{num_epocas}"

    epocas = historico["epocas"]
    perda_tr = historico["perda_treino"]
    perda_vl = historico["perda_val"]
    acc_tr_pct = [a * 100 for a in historico["acc_treino"]]
    acc_vl_pct = [a * 100 for a in historico["acc_val"]]
    tempos = historico.get("tempo_epoca", [0.5] * len(epocas))

    usar_markers = num_epocas <= 15
    marker_tr = "o" if usar_markers else None
    marker_vl = "s" if usar_markers else None

    # 1. Gráfico de Perda (Cross-Entropy Loss)
    caminho_perda = pasta_destino / f"{prefixo}_perda.png"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(epocas, perda_tr, label="Treino (Cross-Entropy)", color="#2b5c8f", lw=2, marker=marker_tr, markersize=4)
    ax.plot(epocas, perda_vl, label="Validação (Cross-Entropy)", color="#e15759", lw=2, marker=marker_vl, markersize=4)
    idx_min_vl = int(torch.tensor(perda_vl).argmin().item())
    ep_min = epocas[idx_min_vl]
    val_min = perda_vl[idx_min_vl]
    ax.scatter([ep_min], [val_min], color="#e15759", s=60, zorder=5)
    ax.annotate(
        f"Mínimo Validação: {val_min:.4f}\n(Época {ep_min})",
        xy=(ep_min, val_min),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#8b1c1c",
        arrowprops=dict(arrowstyle="->", color="#8b1c1c", lw=1),
    )
    if num_epocas > 30 and idx_min_vl < len(epocas) - 5:
        ax.axvspan(ep_min, epocas[-1], color="#e15759", alpha=0.08, label="Região de Overfitting")
        ax.axvline(ep_min, color="#8b1c1c", linestyle=":", lw=1.2)
    ax.set_title(f"Curva de Perda (Entropia Cruzada) — MLP v4 ({num_epocas} Épocas)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Época de Treinamento", fontsize=11, fontweight="bold")
    ax.set_ylabel("Perda Média (Cross-Entropy)", fontsize=11, fontweight="bold")
    if num_epocas <= 20:
        ax.set_xticks(epocas)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(caminho_perda, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Gráfico de Acurácia (%)
    caminho_acc = pasta_destino / f"{prefixo}_acuracia.png"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(epocas, acc_tr_pct, label="Treino", color="#2b5c8f", lw=2, marker=marker_tr, markersize=4)
    ax.plot(epocas, acc_vl_pct, label="Validação", color="#59a14f", lw=2, marker=marker_vl, markersize=4)
    idx_max_vl = int(torch.tensor(acc_vl_pct).argmax().item())
    ep_max = epocas[idx_max_vl]
    acc_max = acc_vl_pct[idx_max_vl]
    ax.scatter([ep_max], [acc_max], color="#59a14f", s=60, zorder=5)
    ax.annotate(
        f"Pico Validação: {acc_max:.2f}%\n(Época {ep_max})",
        xy=(ep_max, acc_max),
        xytext=(-45, -25 if idx_max_vl > num_epocas * 0.7 else 15),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#276421",
        arrowprops=dict(arrowstyle="->", color="#276421", lw=1),
    )
    ax.set_title(f"Curva de Acurácia (%) — MLP v4 com Softmax ({num_epocas} Épocas)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Época de Treinamento", fontsize=11, fontweight="bold")
    ax.set_ylabel("Acurácia (%)", fontsize=11, fontweight="bold")
    if num_epocas <= 20:
        ax.set_xticks(epocas)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    plt.savefig(caminho_acc, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Gráfico de Gap de Generalização (Monitor de Overfitting)
    caminho_gap = pasta_destino / f"{prefixo}_gap_generalizacao.png"
    gap_acc = [t - v for t, v in zip(acc_tr_pct, acc_vl_pct)]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(epocas, gap_acc, label="Gap de Generalização (Treino - Validação)", color="#b07aa1", lw=2)
    ax.fill_between(epocas, 0, gap_acc, color="#b07aa1", alpha=0.25)
    ax.axhline(0, color="gray", lw=1, linestyle="--")
    gap_final = gap_acc[-1]
    ax.scatter([epocas[-1]], [gap_final], color="#b07aa1", s=60, zorder=5)
    ax.annotate(
        f"Gap Final: +{gap_final:.2f} p.p.",
        xy=(epocas[-1], gap_final),
        xytext=(-65, 12),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#5c2d66",
        arrowprops=dict(arrowstyle="->", color="#5c2d66", lw=1),
    )
    ax.set_title("Monitor de Sobreajuste (Gap de Generalização) — MLP v4", fontsize=12, fontweight="bold")
    ax.set_xlabel("Época de Treinamento", fontsize=11, fontweight="bold")
    ax.set_ylabel("Diferença de Acurácia (p.p.)", fontsize=11, fontweight="bold")
    if num_epocas <= 20:
        ax.set_xticks(epocas)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="upper left")
    plt.tight_layout()
    plt.savefig(caminho_gap, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. Gráfico de Tempo de Execução e Estabilidade
    caminho_tempo = pasta_destino / f"{prefixo}_tempo_execucao.png"
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    ax1.plot(epocas, tempos, color="#4e79a7", lw=1.5, alpha=0.75, label="Tempo por Época (s)")
    media_t = sum(tempos) / len(tempos)
    ax1.axhline(media_t, color="#e15759", linestyle="--", lw=1.5, label=f"Média: {media_t:.2f} s/época")
    ax1.set_xlabel("Época de Treinamento", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Segundos por Época", color="#4e79a7", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#4e79a7")

    tempo_acum_min = np.cumsum(tempos) / 60.0
    ax2 = ax1.twinx()
    ax2.plot(epocas, tempo_acum_min, color="#f28e2b", lw=2, linestyle="-.", label="Tempo Acumulado (min)")
    ax2.set_ylabel("Tempo Total Acumulado (min)", color="#f28e2b", fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#f28e2b")

    linhas1, labels1 = ax1.get_legend_handles_labels()
    linhas2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(linhas1 + linhas2, labels1 + labels2, frameon=True, loc="center right")

    ax1.set_title(f"Estabilidade e Desempenho — MLP v4 ({num_epocas} Épocas)", fontsize=12, fontweight="bold")
    if num_epocas <= 20:
        ax1.set_xticks(epocas)
    ax1.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(caminho_tempo, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 5. Painel Geral Integrado (2x2)
    caminho_painel = pasta_destino / f"{prefixo}_painel_geral.png"
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=300)

    # (0, 0) Perda
    ax_l = axes[0, 0]
    ax_l.plot(epocas, perda_tr, label="Treino", color="#2b5c8f", lw=2)
    ax_l.plot(epocas, perda_vl, label="Validação", color="#e15759", lw=2)
    ax_l.scatter([ep_min], [val_min], color="#e15759", s=50, zorder=5)
    ax_l.annotate(f"Mín: {val_min:.4f} (Ép. {ep_min})", xy=(ep_min, val_min), xytext=(10, 10), textcoords="offset points", fontsize=8, fontweight="bold", color="#8b1c1c")
    if num_epocas > 30 and idx_min_vl < len(epocas) - 5:
        ax_l.axvspan(ep_min, epocas[-1], color="#e15759", alpha=0.08)
    ax_l.set_title("(A) Curva de Perda (Cross-Entropy)", fontsize=11, fontweight="bold")
    ax_l.set_xlabel("Época")
    ax_l.set_ylabel("Perda Média")
    ax_l.grid(True, linestyle="--", alpha=0.5)
    ax_l.legend(frameon=True, fontsize=9)

    # (0, 1) Acurácia
    ax_a = axes[0, 1]
    ax_a.plot(epocas, acc_tr_pct, label="Treino", color="#2b5c8f", lw=2)
    ax_a.plot(epocas, acc_vl_pct, label="Validação", color="#59a14f", lw=2)
    ax_a.scatter([ep_max], [acc_max], color="#59a14f", s=50, zorder=5)
    ax_a.annotate(f"Máx: {acc_max:.2f}% (Ép. {ep_max})", xy=(ep_max, acc_max), xytext=(-50, -18 if idx_max_vl > num_epocas * 0.7 else 10), textcoords="offset points", fontsize=8, fontweight="bold", color="#276421")
    ax_a.set_title("(B) Curva de Acurácia (%)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Época")
    ax_a.set_ylabel("Acurácia (%)")
    ax_a.grid(True, linestyle="--", alpha=0.5)
    ax_a.legend(frameon=True, fontsize=9, loc="lower right")

    # (1, 0) Gap
    ax_g = axes[1, 0]
    ax_g.plot(epocas, gap_acc, label="Gap (Treino - Val)", color="#b07aa1", lw=2)
    ax_g.fill_between(epocas, 0, gap_acc, color="#b07aa1", alpha=0.25)
    ax_g.axhline(0, color="gray", lw=1, linestyle="--")
    ax_g.scatter([epocas[-1]], [gap_final], color="#b07aa1", s=50, zorder=5)
    ax_g.annotate(f"+{gap_final:.2f} p.p.", xy=(epocas[-1], gap_final), xytext=(-50, 10), textcoords="offset points", fontsize=8, fontweight="bold", color="#5c2d66")
    ax_g.set_title("(C) Gap de Generalização (Monitor de Overfitting)", fontsize=11, fontweight="bold")
    ax_g.set_xlabel("Época")
    ax_g.set_ylabel("Diferença de Acurácia (p.p.)")
    ax_g.grid(True, linestyle="--", alpha=0.5)
    ax_g.legend(frameon=True, fontsize=9)

    # (1, 1) Tempo
    ax_t = axes[1, 1]
    ax_t.plot(epocas, tempos, color="#4e79a7", lw=1.5, alpha=0.75, label="Tempo/época (s)")
    ax_t.axhline(media_t, color="#e15759", linestyle="--", lw=1.5, label=f"Média: {media_t:.2f}s")
    ax_t.set_title("(D) Desempenho e Estabilidade Temporal", fontsize=11, fontweight="bold")
    ax_t.set_xlabel("Época")
    ax_t.set_ylabel("Segundos / Época", color="#4e79a7")
    ax_t.grid(True, linestyle="--", alpha=0.5)
    ax_t.legend(frameon=True, fontsize=9)

    fig.suptitle(f"Painel Integrado de Treinamento — MLP v4 com Softmax ({num_epocas} Épocas)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(caminho_painel, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[GRÁFICOS] 5 arquivos gerados com sucesso na pasta 'figuras/{subpasta}/':")
    print(f"  • {caminho_perda.name} (Curva de Perda - Cross-Entropy)")
    print(f"  • {caminho_acc.name} (Curva de Acurácia)")
    print(f"  • {caminho_gap.name} (Gap de Generalização)")
    print(f"  • {caminho_tempo.name} (Estabilidade Temporal)")
    print(f"  • {caminho_painel.name} (Painel Integrado 2x2)\n")

    return {
        "perda": caminho_perda,
        "acuracia": caminho_acc,
        "gap": caminho_gap,
        "tempo": caminho_tempo,
        "painel": caminho_painel,
    }


def executar_treinamento(num_epocas: int = 50) -> Tuple[Path, Dict[str, Path]]:
    """Executa o ciclo de treino da MLP v4 (Softmax + Cross-Entropy), persiste pesos e gráficos."""
    hidden_dims = [300]
    taxa_aprendizado = 0.1
    batch_size = 64

    caminho_checkpoint = PASTA_MODELO / f"mlp_treinado_v4_e{num_epocas}.pth"
    pasta_destino = PASTA_FIGURAS / f"e{num_epocas}"

    print("=" * 70)
    print(f"EXPERIMENTO MLP v4: Softmax & Entropia Cruzada ({num_epocas} Épocas)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo:           {device}")
    print(f"Arquitetura:           784 -> 300 -> 10 (Linear -> ReLU -> Logits)")
    print(f"Função de Perda:       CrossEntropyLoss (Softmax Integrado)")
    print(f"Formato dos Rótulos:   Escalares Inteiros diretos (0 a 9, sem One-Hot)")
    print(f"Épocas Solicitadas:    {num_epocas}")
    print(f"Taxa de Aprendizado:   {taxa_aprendizado}")
    print(f"Tamanho do Batch:      {batch_size}")
    print(f"Pasta de Persistência: {PASTA_MODELO}")
    print(f"Pasta de Gráficos:     {pasta_destino}")
    print(f"Arquivo de Pesos:      {caminho_checkpoint.name}")
    print(f"Padrão de Gráficos:    mlp_treinado_v4_e{num_epocas}_*.png (em figuras/e{num_epocas}/)")
    print("-" * 70)

    # 1. Pipeline de dados: Treino e Validação em RAM
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

    loader_treino = DataLoader(mem_treino, batch_size=batch_size, shuffle=True, num_workers=0)
    loader_val = DataLoader(mem_val, batch_size=batch_size, shuffle=False, num_workers=0)

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
        perda_tr, acc_tr = treinar_epoca_ce(modelo, loader_treino, criterio, otimizador, device)
        perda_vl, acc_vl = avaliar_val_ce(modelo, loader_val, criterio, device)
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
    print("TREINAMENTO MLP v4 (SOFTMAX + CROSS-ENTROPY) CONCLUÍDO COM SUCESSO!")
    print(f"• Arquivo de Pesos: {caminho_checkpoint.name} ({PASTA_MODELO})")
    print(f"• Gráficos Gerados: 5 arquivos em {pasta_destino}")
    print("-" * 70)
    print("Para avaliar este modelo no conjunto de teste oficial, execute:")
    print(f"    python mlp_v4_testar.py -c {caminho_checkpoint.name}")
    print("=" * 70)

    return caminho_checkpoint, graficos


def main():
    parser = argparse.ArgumentParser(
        description="mlp_v4_treinar.py: Treinamento da MLP com Softmax e Entropia Cruzada (Seção II)."
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
