"""Experimento MLP v3 — Módulo de Avaliação e Testes Padronizado.

Carrega um checkpoint salvo em disco (.pth) e avalia o modelo exclusivamente
sobre as 10.000 imagens do conjunto oficial de teste do MNIST.

Princípios Metodológicos e de Engenharia:
1. Prova de Persistência Real: Demonstra que o arquivo salvo em disco é
   autossuficiente, instanciando o modelo e carregando os pesos sem depender
   de variáveis de treinamento prévias em memória.
2. Isolamento de Avaliação: Carrega unicamente a partição de Teste (10.000 amostras),
   respeitando a ética do "cofre lacrado".
3. Inferência de Alta Velocidade: Mede o tempo de inferência puro da CPU
   (geralmente abaixo de 0.2s para todas as 10.000 imagens).

Uso:
    python mlp_v3_testar.py
    python mlp_v3_testar.py -c mlp_treinado_v3_e5.pth
    python mlp_v3_testar.py --checkpoint modelo/mlp_treinado_v3_e50.pth
    python mlp_v3_testar.py -c mlp_treinado_v3_e120.pth
"""

import argparse
from pathlib import Path
import re
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


def carregar_checkpoint(caminho: Path, input_dim: int = 784, hidden_dims: List[int] = [300], num_classes: int = 10) -> Tuple[MLP, Path]:
    """Instancia a arquitetura vazia e injeta os pesos salvos do arquivo .pth."""
    if not caminho.exists():
        # Se foi passado apenas o nome do arquivo, tenta buscar dentro de PASTA_MODELO
        caminho_alternativo = PASTA_MODELO / caminho.name
        if caminho_alternativo.exists():
            caminho = caminho_alternativo
        else:
            arquivos_disponiveis = list(PASTA_MODELO.glob("*.pth"))
            msg_erro = f"Checkpoint não encontrado: {caminho}\n"
            if arquivos_disponiveis:
                msg_erro += "Modelos disponíveis em 'modelo/':\n"
                for arq in arquivos_disponiveis:
                    msg_erro += f"  • {arq.name}\n"
            raise FileNotFoundError(msg_erro)

    modelo = MLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)
    estado = torch.load(caminho, weights_only=True)
    modelo.load_state_dict(estado)
    modelo.eval()
    return modelo, caminho


def avaliar_teste_mse(
    modelo: nn.Module,
    dataloader: DataLoader,
    criterio: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Avalia o modelo sobre o conjunto de teste coletando predições e rótulos reais."""
    modelo.eval()
    perda_total = 0.0
    acertos = 0
    total = 0
    todas_reais: List[int] = []
    todas_preditas: List[int] = []

    with torch.no_grad():
        for imagens, rotulos in dataloader:
            imagens = imagens.to(device)
            rotulos = rotulos.to(device)
            alvos_one_hot = torch.nn.functional.one_hot(rotulos, num_classes=10).float()

            saidas = modelo(imagens)
            perda = criterio(saidas, alvos_one_hot)

            perda_total += perda.item() * imagens.size(0)
            predicoes = saidas.argmax(dim=-1)
            acertos += (predicoes == rotulos).sum().item()
            total += imagens.size(0)

            todas_reais.extend(rotulos.cpu().tolist())
            todas_preditas.extend(predicoes.cpu().tolist())

    return perda_total / total, acertos / total, todas_reais, todas_preditas


def calcular_metricas(reais: List[int], preditas: List[int]) -> dict:
    """Calcula matriz de confusão, acurácia, precisão, recall e F1 por classe e macro."""
    num_classes = 10
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    for r, p in zip(reais, preditas):
        matriz[r, p] += 1

    total = len(reais)
    acuracia = float(np.trace(matriz) / total)

    precisao = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for c in range(num_classes):
        tp = matriz[c, c]
        fp = matriz[:, c].sum() - tp
        fn = matriz[c, :].sum() - tp
        precisao[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = 2 * (precisao[c] * recall[c]) / (precisao[c] + recall[c]) if (precisao[c] + recall[c]) > 0 else 0.0

    pares_confundidos = []
    for r in range(num_classes):
        for p in range(num_classes):
            if r != p and matriz[r, p] > 0:
                pares_confundidos.append((int(matriz[r, p]), r, p))
    pares_confundidos.sort(key=lambda x: x[0], reverse=True)

    return {
        "matriz": matriz,
        "acuracia": acuracia,
        "precisao": precisao,
        "recall": recall,
        "f1": f1,
        "macro_precisao": float(np.mean(precisao)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "top_confusoes": pares_confundidos[:5],
    }


def salvar_graficos_teste(
    prefixo_base: str,
    metricas: dict,
    imagens_teste: torch.Tensor,
    reais: List[int],
    preditas: List[int],
    pasta_figuras: Path,
) -> Dict[str, Path]:
    """Gera e salva os relatórios visuais da avaliação no conjunto de teste."""
    pasta_figuras.mkdir(parents=True, exist_ok=True)
    matriz = metricas["matriz"]
    classes = np.arange(10)

    # 1. Matriz de Confusão 10x10
    caminho_matriz = pasta_figuras / f"{prefixo_base}_matriz_confusao.png"
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    cax = ax.imshow(matriz, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(classes)
    ax.set_yticks(classes)
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_xlabel("Dígito Predito pela Rede", fontsize=11, fontweight="bold")
    ax.set_ylabel("Dígito Real (Gabarito)", fontsize=11, fontweight="bold")

    limiar = matriz.max() / 2.0
    for i in range(10):
        for j in range(10):
            val = matriz[i, j]
            cor = "white" if val > limiar else "black"
            peso = "bold" if i == j else "normal"
            ax.text(j, i, f"{val}", ha="center", va="center", color=cor, fontsize=9, fontweight=peso)

    ax.set_title(
        f"Matriz de Confusão no Teste Oficial MNIST (10.000 Imagens)\nModelo: {prefixo_base} | Acurácia: {metricas['acuracia']*100:.2f}%",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(caminho_matriz, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Métricas por Classe (Precisão, Recall, F1)
    caminho_metricas = pasta_figuras / f"{prefixo_base}_metricas_classes.png"
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    largura = 0.25
    p = metricas["precisao"] * 100
    r = metricas["recall"] * 100
    f = metricas["f1"] * 100

    ax.bar(classes - largura, p, largura, label="Precisão", color="#2b5c8f")
    ax.bar(classes, r, largura, label="Recall", color="#f28e2b")
    ax.bar(classes + largura, f, largura, label="F1-Score", color="#59a14f")

    macro_f1 = metricas["macro_f1"] * 100
    ax.axhline(macro_f1, color="#e15759", linestyle="--", lw=1.5, label=f"Macro-F1 ({macro_f1:.2f}%)")

    ax.set_xlabel("Classe (Dígito)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Desempenho (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Desempenho por Dígito no Teste Oficial — {prefixo_base}\nMacro-F1: {macro_f1:.2f}% | Acurácia: {metricas['acuracia']*100:.2f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(classes)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_ylim(85, 102)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    plt.savefig(caminho_metricas, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Galeria de Exemplos Reais de Confusão
    caminho_erros = pasta_figuras / f"{prefixo_base}_amostras_erros.png"
    indices_erros = [i for i, (real, pred) in enumerate(zip(reais, preditas)) if real != pred]

    if indices_erros:
        num_exemplos = min(16, len(indices_erros))
        indices_selecionados = indices_erros[:num_exemplos]

        fig, axes = plt.subplots(4, 4, figsize=(8, 8), dpi=300)
        for idx, ax in zip(indices_selecionados, axes.flat):
            img = imagens_teste[idx].squeeze().cpu().numpy()
            real = reais[idx]
            pred = preditas[idx]
            ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.set_title(f"Real: {real} | Pred: {pred}", color="#d95f02", fontsize=10, fontweight="bold")
            ax.axis("off")

        fig.suptitle(
            f"Exemplos Reais de Erros de Classificação no Teste MNIST\nModelo: {prefixo_base} ({len(indices_erros)} erros totais)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(caminho_erros, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 4. Painel Geral Integrado de Teste
    caminho_painel = pasta_figuras / f"{prefixo_base}_painel_teste.png"
    fig = plt.figure(figsize=(15, 6), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.3])

    ax1 = fig.add_subplot(gs[0])
    cax = ax1.imshow(matriz, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_xticks(classes)
    ax1.set_yticks(classes)
    ax1.set_xticklabels(classes, fontsize=9, fontweight="bold")
    ax1.set_yticklabels(classes, fontsize=9, fontweight="bold")
    ax1.set_xlabel("Predito", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Real", fontsize=10, fontweight="bold")
    ax1.set_title("Matriz de Confusão (10.000 amostras)", fontsize=11, fontweight="bold")

    for i in range(10):
        for j in range(10):
            val = matriz[i, j]
            ax1.text(j, i, f"{val}", ha="center", va="center", color="white" if val > limiar else "black", fontsize=8)

    ax2 = fig.add_subplot(gs[1])
    ax2.bar(classes - largura, p, largura, label="Precisão", color="#2b5c8f")
    ax2.bar(classes, r, largura, label="Recall", color="#f28e2b")
    ax2.bar(classes + largura, f, largura, label="F1-Score", color="#59a14f")
    ax2.axhline(macro_f1, color="#e15759", linestyle="--", lw=1.5, label=f"Macro-F1 ({macro_f1:.2f}%)")
    ax2.set_xlabel("Classe (Dígito)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Desempenho (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Precisão, Recall e F1 por Dígito", fontsize=11, fontweight="bold")
    ax2.set_xticks(classes)
    ax2.set_xticklabels(classes, fontweight="bold")
    ax2.set_ylim(85, 102)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, fontsize=9, loc="lower right")

    fig.suptitle(
        f"Relatório Integrado de Avaliação no Teste — {prefixo_base} (Acc: {metricas['acuracia']*100:.2f}%)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(caminho_painel, dpi=300, bbox_inches="tight")
    plt.close(fig)

    try:
        caminho_rel = pasta_figuras.relative_to(DIRETORIO_PROJETO)
    except ValueError:
        caminho_rel = pasta_figuras

    print(f"\n[GRÁFICOS DE TESTE] 4 relatórios gerados na pasta '{caminho_rel}/':")
    print(f"  • {caminho_matriz.name} (Matriz de Confusão 10x10)")
    print(f"  • {caminho_metricas.name} (Precisão, Recall e F1 por Dígito)")
    print(f"  • {caminho_erros.name} (Galeria de Amostras Reais com Erro)")
    print(f"  • {caminho_painel.name} (Dashboard Integrado de Teste)\n")

    return {
        "matriz": caminho_matriz,
        "metricas": caminho_metricas,
        "erros": caminho_erros,
        "painel": caminho_painel,
    }


def executar_testes(caminho_arquivo: str = None) -> None:
    """Executa a rotina de avaliação de um modelo salvo contra as 10.000 imagens de teste."""
    print("=" * 70)
    print("EXPERIMENTO MLP v3: Avaliação Independente no Conjunto de Teste")
    print("=" * 70)

    # 1. Resolução do checkpoint
    if caminho_arquivo is None:
        caminho_arquivo = PASTA_MODELO / "mlp_treinado_v3_e3.pth"
        if not caminho_arquivo.exists():
            checkpoints = sorted(PASTA_MODELO.glob("mlp_treinado_v3*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
            if checkpoints:
                caminho_arquivo = checkpoints[0]
            else:
                caminho_arquivo = PASTA_MODELO / "mlp_treinado.pth"
    else:
        caminho_arquivo = Path(caminho_arquivo)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo:           {device}")
    
    # 2. Carregamento do modelo a partir do arquivo em disco
    print(f"Carregando checkpoint do disco...")
    modelo, caminho_resolvido = carregar_checkpoint(caminho_arquivo)
    modelo = modelo.to(device)
    tamanho_kb = caminho_resolvido.stat().st_size / 1024
    print(f"-> Arquivo de Pesos:   {caminho_resolvido.name} ({tamanho_kb:.2f} KB)")
    print(f"-> Arquitetura:        784 -> 300 -> 10")
    print("-" * 70)

    # 3. Carregamento estrito do conjunto de Teste (10.000 imagens)
    print("Carregando conjunto de TESTE oficial (10.000 imagens)...")
    t0_data = time.time()
    _, _, ds_teste_cru = get_mnist_datasets()
    mem_teste = dataset_para_ram(ds_teste_cru)
    loader_teste = DataLoader(mem_teste, batch_size=64, shuffle=False, num_workers=0)
    print(f"-> Teste pronto em RAM em {time.time() - t0_data:.2f}s")
    print("-" * 70)

    # 4. Avaliação e inferência
    criterio = nn.MSELoss()
    t0_inferencia = time.time()
    perda_teste, acuracia_teste, reais, preditas = avaliar_teste_mse(
        modelo, loader_teste, criterio, device
    )
    tempo_inferencia = time.time() - t0_inferencia

    total_testes = len(reais)
    acertos = sum(1 for r, p in zip(reais, preditas) if r == p)
    erros = total_testes - acertos
    velocidade = total_testes / tempo_inferencia if tempo_inferencia > 0 else 0

    print("RESULTADO OFICIAL NO CONJUNTO DE TESTE:")
    print(f"• Total de Amostras Avaliadas: {total_testes:,}")
    print(f"• Perda Média (MSELoss):       {perda_teste:.5f}")
    print(f"• Acertos:                     {acertos:,} ({acuracia_teste * 100:.2f}%)")
    print(f"• Erros:                       {erros:,} ({(erros / total_testes) * 100:.2f}%)")
    print(f"• Tempo Total de Inferência:   {tempo_inferencia:.3f} s ({velocidade:,.1f} imagens/s)")
    print("-" * 70)

    # 5. Cálculo das métricas completas e análise de confusões
    metricas = calcular_metricas(reais, preditas)
    print("MÉTRICAS MACRO NO TESTE:")
    print(f"• Precisão Macro:              {metricas['macro_precisao'] * 100:.2f}%")
    print(f"• Revocação (Recall) Macro:    {metricas['macro_recall'] * 100:.2f}%")
    print(f"• F1-Score Macro:              {metricas['macro_f1'] * 100:.2f}%")
    print("-" * 70)
    print("TOP 5 CONFUSÕES MAIS FREQUENTES:")
    for rank, (qtd, real, pred) in enumerate(metricas["top_confusoes"], 1):
        print(f"  {rank}. Dígito {real} classificado como {pred}: {qtd} vezes")

    # 6. Geração dos gráficos de teste organizados por subpasta de épocas
    match_epoca = re.search(r"e\d+", caminho_resolvido.stem)
    subpasta = match_epoca.group(0) if match_epoca else caminho_resolvido.stem
    pasta_destino = PASTA_FIGURAS / subpasta

    salvar_graficos_teste(
        prefixo_base=caminho_resolvido.stem,
        metricas=metricas,
        imagens_teste=mem_teste.tensors[0],
        reais=reais,
        preditas=preditas,
        pasta_figuras=pasta_destino,
    )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="mlp_v3_testar.py: Avaliação independente de checkpoints da MLP no conjunto de teste MNIST."
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default=None,
        help="Caminho ou nome do arquivo de pesos .pth em 'modelo/' (ex: mlp_treinado_v3_e50.pth)",
    )

    args = parser.parse_args()
    executar_testes(caminho_arquivo=args.checkpoint)


if __name__ == "__main__":
    main()
