"""Confronto Empírico e Morfológico: MLP Campeão (v5) vs. CNN LeNet (cnn_v1).

Compara os dois modelos eleitos pela validação sobre o conjunto de teste oficial (10.000 imagens),
calculando:
1. Tabela comparativa (Acurácia, Macro-F1, Perda, Parâmetros, Tempo de Treino e Inferência);
2. Matriz de discordância amostra por amostra (casos onde a CNN corrige o MLP e vice-versa);
3. Comparativo de confusões crônicas (4<->9, 5<->3, 7<->2/1);
4. Painéis gráficos em 300 DPI (`figuras/confronto_mlp_vs_cnn/`).

Execute este script SOMENTE após eleger ambos os modelos pela validação e avaliá-los
no teste via `mlp_v5_testar.py` e `cnn_v1_testar.py`.

Uso:
    python confronto_mlp_vs_cnn.py \
        --mlp-checkpoint mlp_treinado_v5_e50_b256_lr0.1_m0.9_h512_s1_melhor.pth \
        --cnn-checkpoint <checkpoint_cnn_melhor.pth>
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cnn_v1_testar import carregar_checkpoint_cnn
from dataset import get_mnist_datasets
from mlp_v4_testar import calcular_metricas, carregar_checkpoint, dataset_para_ram

DIRETORIO_PROJETO = Path(__file__).resolve().parent
PASTA_MODELO = DIRETORIO_PROJETO / "modelo"
PASTA_FIGURAS = DIRETORIO_PROJETO / "figuras"
PASTA_RESULTADOS = DIRETORIO_PROJETO / "resultados"

MLP_PADRAO = "mlp_treinado_v5_e50_b256_lr0.1_m0.9_h512_s1_melhor.pth"


def inferir_detalhado(
    modelo: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    """Executa inferência coletando perda, tempo, predições e probabilidades."""
    modelo.eval()
    criterio = nn.CrossEntropyLoss()
    perda_total = 0.0
    total = 0
    reais: List[int] = []
    preditas: List[int] = []
    probs_max: List[float] = []

    t0 = time.time()
    with torch.no_grad():
        for imagens, rotulos in dataloader:
            imagens = imagens.to(device)
            rotulos = rotulos.to(device)
            logits = modelo(imagens)
            perda = criterio(logits, rotulos)
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)

            perda_total += perda.item() * imagens.size(0)
            total += imagens.size(0)
            reais.extend(rotulos.cpu().tolist())
            preditas.extend(pred.cpu().tolist())
            probs_max.extend(conf.cpu().tolist())
    tempo = time.time() - t0

    metricas = calcular_metricas(reais, preditas)
    erros = total - sum(1 for r, p in zip(reais, preditas) if r == p)
    return {
        "perda": perda_total / total,
        "acuracia": metricas["acuracia"],
        "erros": erros,
        "total": total,
        "tempo_inferencia_s": tempo,
        "reais": reais,
        "preditas": preditas,
        "probs_max": probs_max,
        "metricas": metricas,
    }


def analisar_discordancia(
    reais: List[int],
    pred_mlp: List[int],
    pred_cnn: List[int],
) -> Dict[str, object]:
    """Classifica cada amostra em: ambos acertam, CNN corrige MLP, MLP acerta CNN erra, ambos erram."""
    ambos_acertam: List[int] = []
    cnn_corrige_mlp: List[int] = []
    mlp_acerta_cnn_erra: List[int] = []
    ambos_erram: List[int] = []

    for i, (r, pm, pc) in enumerate(zip(reais, pred_mlp, pred_cnn)):
        m_ok = pm == r
        c_ok = pc == r
        if m_ok and c_ok:
            ambos_acertam.append(i)
        elif (not m_ok) and c_ok:
            cnn_corrige_mlp.append(i)
        elif m_ok and (not c_ok):
            mlp_acerta_cnn_erra.append(i)
        else:
            ambos_erram.append(i)

    return {
        "ambos_acertam": ambos_acertam,
        "cnn_corrige_mlp": cnn_corrige_mlp,
        "mlp_acerta_cnn_erra": mlp_acerta_cnn_erra,
        "ambos_erram": ambos_erram,
    }


def extrair_pares_cronicos(
    matriz_mlp: np.ndarray, matriz_cnn: np.ndarray
) -> List[Dict[str, object]]:
    """Compara os pares de dígitos mais confundidos entre MLP e CNN."""
    pares_foco = [(9, 4), (4, 9), (5, 3), (3, 5), (7, 1), (7, 2), (7, 9), (6, 4), (2, 8), (8, 3)]
    linhas = []
    for real, pred in pares_foco:
        m = int(matriz_mlp[real, pred])
        c = int(matriz_cnn[real, pred])
        linhas.append({"real": real, "pred": pred, "mlp": m, "cnn": c, "delta": c - m})
    return linhas


def _carregar_resumo_treino(caminho_ckpt: Path, pasta_resultados: Path) -> Dict[str, object]:
    """Lê o JSON de treino correspondente ao checkpoint `_melhor.pth` se existir."""
    stem = caminho_ckpt.stem
    if stem.endswith("_melhor"):
        stem = stem[: -len("_melhor")]
    caminho_json = pasta_resultados / f"{stem}.json"
    if caminho_json.exists():
        with open(caminho_json, encoding="utf-8") as f:
            return json.load(f).get("resumo", {})
    return {}


def _salvar_graficos_confronto(
    pasta_destino: Path,
    res_mlp: Dict[str, object],
    res_cnn: Dict[str, object],
    params_mlp: int,
    params_cnn: int,
    tempo_tr_mlp: Optional[float],
    tempo_tr_cnn: Optional[float],
    imagens_teste: torch.Tensor,
    discordancia: Dict[str, object],
    pares_cronicos: List[Dict[str, object]],
) -> Dict[str, Path]:
    """Gera os gráficos comparativos em 300 DPI."""
    pasta_destino.mkdir(parents=True, exist_ok=True)
    caminhos: Dict[str, Path] = {}

    # 1. Eficiência Paramétrica & Desempenho
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
    modelos = ["MLP Campeão\n(784->512->10)", "CNN LeNet\n(Conv 5x5 + Pool)"]
    cores = ["#4c72b0", "#55a868"]

    # Painel A: Erros no teste (menor é melhor)
    erros = [res_mlp["erros"], res_cnn["erros"]]
    barras = axs[0].bar(modelos, erros, color=cores, width=0.5)
    axs[0].set_title("Erros em 10.000 Imagens de Teste", fontweight="bold")
    axs[0].set_ylabel("Número de Erros")
    axs[0].grid(axis="y", linestyle=":", alpha=0.6)
    for b, v, acc in zip(barras, erros, [res_mlp["acuracia"] * 100, res_cnn["acuracia"] * 100]):
        axs[0].text(b.get_x() + b.get_width() / 2, v + 3, f"{v}\n({acc:.2f}%)", ha="center", fontweight="bold")
    axs[0].set_ylim(0, max(erros) * 1.25)

    # Painel B: Parâmetros treináveis
    params = [params_mlp / 1000.0, params_cnn / 1000.0]
    barras = axs[1].bar(modelos, params, color=cores, width=0.5)
    axs[1].set_title("Parâmetros Treináveis (milhares)", fontweight="bold")
    axs[1].set_ylabel("Milhares de Parâmetros (k)")
    axs[1].grid(axis="y", linestyle=":", alpha=0.6)
    for b, v_raw in zip(barras, [params_mlp, params_cnn]):
        axs[1].text(b.get_x() + b.get_width() / 2, (v_raw / 1000.0) + 10, f"{v_raw:,}", ha="center", fontweight="bold")
    axs[1].set_ylim(0, max(params) * 1.2)

    # Painel C: Tempo total de treino (CPU)
    t_m = tempo_tr_mlp or 0.0
    t_c = tempo_tr_cnn or 0.0
    barras = axs[2].bar(modelos, [t_m, t_c], color=cores, width=0.5)
    axs[2].set_title("Tempo Total de Treino na CPU (s)", fontweight="bold")
    axs[2].set_ylabel("Segundos (s)")
    axs[2].grid(axis="y", linestyle=":", alpha=0.6)
    for b, v in zip(barras, [t_m, t_c]):
        axs[2].text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f} s", ha="center", fontweight="bold")
    axs[2].set_ylim(0, max(t_m, t_c, 1.0) * 1.25)

    fig.suptitle("Confronto MLP Campeão (v5) vs. CNN LeNet (cnn_v1)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p_efic = pasta_destino / "confronto_eficiencia.png"
    fig.savefig(p_efic, dpi=300, bbox_inches="tight")
    plt.close(fig)
    caminhos["eficiencia"] = p_efic

    # 2. Comparação das Confusões Crônicas (Barras Pareadas)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    labels = [f"{p['real']}->{p['pred']}" for p in pares_cronicos]
    x = np.arange(len(labels))
    w = 0.36
    vals_mlp = [p["mlp"] for p in pares_cronicos]
    vals_cnn = [p["cnn"] for p in pares_cronicos]
    ax.bar(x - w / 2, vals_mlp, w, label="MLP Campeão", color="#4c72b0")
    ax.bar(x + w / 2, vals_cnn, w, label="CNN LeNet", color="#55a868")
    for xi, vm, vc in zip(x, vals_mlp, vals_cnn):
        ax.text(xi - w / 2, vm + 0.25, str(vm), ha="center", fontsize=8, fontweight="bold")
        ax.text(xi + w / 2, vc + 0.25, str(vc), ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_xlabel("Par de Confusão (Real -> Predito)", fontweight="bold")
    ax.set_ylabel("Ocorrências em 10.000 Imagens", fontweight="bold")
    ax.set_title("Confronto nos Pares de Confusão Crônicos do MNIST", fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    p_pares = pasta_destino / "confronto_pares_cronicos.png"
    fig.savefig(p_pares, dpi=300, bbox_inches="tight")
    plt.close(fig)
    caminhos["pares"] = p_pares

    # 3. Galeria Visual: Imagens onde a CNN corrigiu o MLP
    idxs_corrigidos = discordancia["cnn_corrige_mlp"]
    if idxs_corrigidos:
        n_mostrar = min(16, len(idxs_corrigidos))
        fig, axes = plt.subplots(4, 4, figsize=(9, 9.5), dpi=300)
        for ax, idx in zip(axes.flat, idxs_corrigidos[:n_mostrar]):
            img = imagens_teste[idx].squeeze().cpu().numpy()
            real = res_mlp["reais"][idx]
            pm = res_mlp["preditas"][idx]
            pc = res_cnn["preditas"][idx]
            ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.set_title(
                f"Real: {real}\nMLP: {pm} (erro) | CNN: {pc}",
                fontsize=9,
                fontweight="bold",
                color="#1b7837",
            )
            ax.axis("off")
        for ax in axes.flat[n_mostrar:]:
            ax.axis("off")
        fig.suptitle(
            f"Amostras Reais Onde a CNN Corrigiu o MLP ({len(idxs_corrigidos)} casos totais)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        p_gal = pasta_destino / "confronto_cnn_corrige_mlp.png"
        fig.savefig(p_gal, dpi=300, bbox_inches="tight")
        plt.close(fig)
        caminhos["galeria_corrigidos"] = p_gal

    # 4. Diferença da Matriz de Confusão (Erros CNN - Erros MLP)
    diff = res_cnn["metricas"]["matriz"].astype(int) - res_mlp["metricas"]["matriz"].astype(int)
    np.fill_diagonal(diff, 0)
    lim = max(abs(int(diff.min())), abs(int(diff.max())), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Variação de Erros (CNN - MLP)")
    classes = np.arange(10)
    ax.set_xticks(classes)
    ax.set_yticks(classes)
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_yticklabels(classes, fontweight="bold")
    ax.set_xlabel("Dígito Predito", fontweight="bold")
    ax.set_ylabel("Dígito Real", fontweight="bold")
    ax.set_title("Redução de Confusões: Erros(CNN) - Erros(MLP)\n(Azul = CNN errou menos | Vermelho = CNN errou mais)", fontweight="bold")
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            v = int(diff[i, j])
            if v != 0:
                ax.text(j, i, f"{v:+d}", ha="center", va="center", fontsize=8.5, fontweight="bold",
                        color="white" if abs(v) > lim * 0.55 else "black")
    plt.tight_layout()
    p_diff = pasta_destino / "confronto_diferenca_confusao.png"
    fig.savefig(p_diff, dpi=300, bbox_inches="tight")
    plt.close(fig)
    caminhos["diferenca_confusao"] = p_diff

    return caminhos


def executar_confronto(
    mlp_checkpoint: str,
    cnn_checkpoint: str,
    gerar_graficos: bool = True,
    pasta_figuras: Optional[Path] = None,
    pasta_resultados: Optional[Path] = None,
) -> Dict[str, object]:
    """Executa o confronto completo MLP vs CNN no conjunto de teste."""
    pasta_figuras = pasta_figuras or PASTA_FIGURAS
    pasta_resultados = pasta_resultados or PASTA_RESULTADOS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo_mlp, path_mlp = carregar_checkpoint(Path(mlp_checkpoint))
    modelo_cnn, path_cnn = carregar_checkpoint_cnn(Path(cnn_checkpoint))
    modelo_mlp = modelo_mlp.to(device)
    modelo_cnn = modelo_cnn.to(device)

    params_mlp = sum(p.numel() for p in modelo_mlp.parameters())
    params_cnn = sum(p.numel() for p in modelo_cnn.parameters())
    resumo_tr_mlp = _carregar_resumo_treino(path_mlp, pasta_resultados)
    resumo_tr_cnn = _carregar_resumo_treino(path_cnn, pasta_resultados)

    _, _, ds_teste_cru = get_mnist_datasets()
    mem_teste = dataset_para_ram(ds_teste_cru)
    loader_teste = DataLoader(mem_teste, batch_size=64, shuffle=False, num_workers=0)

    res_mlp = inferir_detalhado(modelo_mlp, loader_teste, device)
    res_cnn = inferir_detalhado(modelo_cnn, loader_teste, device)

    disc = analisar_discordancia(res_mlp["reais"], res_mlp["preditas"], res_cnn["preditas"])
    pares = extrair_pares_cronicos(res_mlp["metricas"]["matriz"], res_cnn["metricas"]["matriz"])

    relatorio = {
        "mlp": {
            "checkpoint": path_mlp.name,
            "parametros": params_mlp,
            "acuracia_teste_pct": round(res_mlp["acuracia"] * 100, 4),
            "erros_teste": res_mlp["erros"],
            "perda_teste": round(res_mlp["perda"], 6),
            "macro_precisao_pct": round(res_mlp["metricas"]["macro_precisao"] * 100, 4),
            "macro_recall_pct": round(res_mlp["metricas"]["macro_recall"] * 100, 4),
            "macro_f1_pct": round(res_mlp["metricas"]["macro_f1"] * 100, 4),
            "tempo_treino_s": resumo_tr_mlp.get("tempo_treino_s"),
            "epoca_salva": resumo_tr_mlp.get("epoca_salva"),
            "tempo_inferencia_s": round(res_mlp["tempo_inferencia_s"], 4),
            "top_confusoes": res_mlp["metricas"]["top_confusoes"],
        },
        "cnn": {
            "checkpoint": path_cnn.name,
            "parametros": params_cnn,
            "acuracia_teste_pct": round(res_cnn["acuracia"] * 100, 4),
            "erros_teste": res_cnn["erros"],
            "perda_teste": round(res_cnn["perda"], 6),
            "macro_precisao_pct": round(res_cnn["metricas"]["macro_precisao"] * 100, 4),
            "macro_recall_pct": round(res_cnn["metricas"]["macro_recall"] * 100, 4),
            "macro_f1_pct": round(res_cnn["metricas"]["macro_f1"] * 100, 4),
            "tempo_treino_s": resumo_tr_cnn.get("tempo_treino_s"),
            "epoca_salva": resumo_tr_cnn.get("epoca_salva"),
            "tempo_inferencia_s": round(res_cnn["tempo_inferencia_s"], 4),
            "top_confusoes": res_cnn["metricas"]["top_confusoes"],
        },
        "discordancia": {
            "ambos_acertam": len(disc["ambos_acertam"]),
            "cnn_corrige_mlp": len(disc["cnn_corrige_mlp"]),
            "mlp_acerta_cnn_erra": len(disc["mlp_acerta_cnn_erra"]),
            "ambos_erram": len(disc["ambos_erram"]),
            "indices_cnn_corrige_mlp": disc["cnn_corrige_mlp"][:24],
            "indices_ambos_erram": disc["ambos_erram"][:24],
        },
        "pares_cronicos": pares,
    }

    pasta_resultados.mkdir(parents=True, exist_ok=True)
    with open(pasta_resultados / "confronto_mlp_vs_cnn.json", "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    if gerar_graficos:
        _salvar_graficos_confronto(
            pasta_destino=pasta_figuras / "confronto_mlp_vs_cnn",
            res_mlp=res_mlp,
            res_cnn=res_cnn,
            params_mlp=params_mlp,
            params_cnn=params_cnn,
            tempo_tr_mlp=resumo_tr_mlp.get("tempo_treino_s"),
            tempo_tr_cnn=resumo_tr_cnn.get("tempo_treino_s"),
            imagens_teste=mem_teste.tensors[0],
            discordancia=disc,
            pares_cronicos=pares,
        )

    print("=" * 80)
    print("CONFRONTO DEFINITIVO: MLP CAMPEÃO (v5) vs. CNN LeNet (cnn_v1)")
    print("=" * 80)
    print(f"{'Métrica':<28} | {'MLP (784->512->10)':>22} | {'CNN LeNet':>22}")
    print("-" * 80)
    print(f"{'Parâmetros':<28} | {params_mlp:>22,} | {params_cnn:>22,}")
    print(f"{'Acurácia no Teste (%)':<28} | {relatorio['mlp']['acuracia_teste_pct']:>21.2f}% | {relatorio['cnn']['acuracia_teste_pct']:>21.2f}%")
    print(f"{'Erros (/10.000)':<28} | {relatorio['mlp']['erros_teste']:>22,} | {relatorio['cnn']['erros_teste']:>22,}")
    print(f"{'Macro-F1 (%)':<28} | {relatorio['mlp']['macro_f1_pct']:>21.2f}% | {relatorio['cnn']['macro_f1_pct']:>21.2f}%")
    print(f"{'Perda Média no Teste':<28} | {relatorio['mlp']['perda_teste']:>22.5f} | {relatorio['cnn']['perda_teste']:>22.5f}")
    print(f"{'Tempo de Treino (s)':<28} | {str(relatorio['mlp']['tempo_treino_s']):>22} | {str(relatorio['cnn']['tempo_treino_s']):>22}")
    print("-" * 80)
    print(
        f"Discordância: ambos acertam={relatorio['discordancia']['ambos_acertam']} | "
        f"CNN corrige MLP={relatorio['discordancia']['cnn_corrige_mlp']} | "
        f"MLP acerta CNN erra={relatorio['discordancia']['mlp_acerta_cnn_erra']} | "
        f"ambos erram={relatorio['discordancia']['ambos_erram']}"
    )
    print("=" * 80)
    return relatorio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confronto direto entre o MLP campeão (v5) e a CNN LeNet (cnn_v1) no teste."
    )
    parser.add_argument("--mlp-checkpoint", type=str, default=MLP_PADRAO, help="Checkpoint do MLP")
    parser.add_argument("--cnn-checkpoint", type=str, required=True, help="Checkpoint da CNN LeNet")
    parser.add_argument("--sem-graficos", action="store_true", help="Não gera as figuras comparativas")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    executar_confronto(
        mlp_checkpoint=args.mlp_checkpoint,
        cnn_checkpoint=args.cnn_checkpoint,
        gerar_graficos=not args.sem_graficos,
    )


if __name__ == "__main__":
    main()
