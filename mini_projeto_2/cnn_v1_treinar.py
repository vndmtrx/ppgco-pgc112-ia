"""Experimento CNN v1 — Treinamento da LeNet no padrão v5 (validação soberana e semente fixa).

Reaproveita o pipeline em RAM e as funções de treino/validação do v5:
1. O checkpoint salvo é o da época de MENOR perda de validação (`*_melhor.pth`);
2. `--seed` fixa os pesos iniciais e a ordem dos lotes;
3. O conjunto de teste nunca é carregado aqui;
4. Grava `resultados/resumo_cnn_v1.csv` e `resultados/<id>.json`.

Uso:
    python cnn_v1_treinar.py -e 30 -b 256 -l 0.05 -m 0.9 --seed 1
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_mnist_datasets
from graficos_mlp import salvar_graficos_treino
from mlp_v4_2_treinar import criar_otimizador, dataset_para_ram
from mlp_v5_treinar import definir_semente, treinar_guardando_melhor
from models.cnn import LeNet

DIRETORIO_PROJETO = Path(__file__).resolve().parent
PASTA_MODELO = DIRETORIO_PROJETO / "modelo"
PASTA_FIGURAS = DIRETORIO_PROJETO / "figuras"
PASTA_RESULTADOS = DIRETORIO_PROJETO / "resultados"

ARQUITETURA_LENET = "1->6->16_256->120->84->10"

COLUNAS_RESUMO_CNN = [
    "id",
    "seed",
    "epocas",
    "batch",
    "lr",
    "momentum",
    "otimizador",
    "arquitetura",
    "parametros",
    "epoca_salva",
    "acc_val_salva",
    "perda_val_salva",
    "perda_treino_salva",
    "acc_val_ultima",
    "perda_val_ultima",
    "acc_val_max",
    "epoca_acc_max",
    "tempo_treino_s",
    "dispositivo",
    "torch",
]


def montar_id_cnn(
    num_epocas: int,
    batch_size: int,
    lr: float,
    momentum: float,
    otimizador_nome: str,
    seed: int,
) -> str:
    """Nome único da execução CNN v1 com hiperparâmetros e semente."""
    mom_str = f"_m{momentum}" if momentum > 0 else ""
    opt_str = f"_{otimizador_nome}" if otimizador_nome != "sgd" else ""
    return f"cnn_treinado_v1_e{num_epocas}_b{batch_size}_lr{lr}{mom_str}{opt_str}_s{seed}"


def registrar_resumo_cnn(caminho_csv: Path, linha: Dict[str, object]) -> None:
    """Grava a linha no CSV de resumo da CNN. Se o id já existir, substitui."""
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    linhas: List[Dict[str, object]] = []
    if caminho_csv.exists():
        with open(caminho_csv, newline="", encoding="utf-8") as f:
            linhas = [r for r in csv.DictReader(f) if r["id"] != linha["id"]]
    linhas.append(linha)
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUNAS_RESUMO_CNN)
        escritor.writeheader()
        escritor.writerows(linhas)


def executar_treinamento_cnn(
    num_epocas: int,
    batch_size: int,
    lr: float,
    momentum: float,
    otimizador_nome: str,
    seed: int,
    gerar_graficos: bool = True,
    pasta_modelo: Optional[Path] = None,
    pasta_figuras: Optional[Path] = None,
    pasta_resultados: Optional[Path] = None,
) -> Dict[str, object]:
    """Fluxo principal do cnn_v1: treina LeNet, salva melhor checkpoint e registra resumo."""
    pasta_modelo = pasta_modelo or PASTA_MODELO
    pasta_figuras = pasta_figuras or PASTA_FIGURAS
    pasta_resultados = pasta_resultados or PASTA_RESULTADOS

    gerador = definir_semente(seed)
    id_exp = montar_id_cnn(num_epocas, batch_size, lr, momentum, otimizador_nome, seed)
    caminho_checkpoint = pasta_modelo / f"{id_exp}_melhor.pth"
    pasta_destino = pasta_figuras / id_exp.replace("cnn_treinado_", "cnn_")

    titulo_exp = f"E={num_epocas}, B={batch_size}, LR={lr}, Opt={otimizador_nome.upper()}"
    if momentum > 0:
        titulo_exp += f", Mom={momentum}"
    titulo_exp += f", Seed={seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"EXPERIMENTO CNN v1 (LeNet): melhor modelo pela validação ({titulo_exp})")
    print("=" * 70)
    print(f"Dispositivo:           {device}")
    print(f"Arquitetura:           LeNet ({ARQUITETURA_LENET}, Conv 5x5 + ReLU + MaxPool 2x2)")
    print("Função de Perda:       CrossEntropyLoss (Softmax integrado)")
    print(f"Semente:               {seed}")
    print("Critério do checkpoint: menor perda de validação")
    print(f"Arquivo de Pesos:      {caminho_checkpoint.name}")
    print("-" * 70)

    t_dados = time.time()
    ds_treino_cru, ds_val_cru, _ = get_mnist_datasets()
    mem_treino = dataset_para_ram(ds_treino_cru)
    mem_val = dataset_para_ram(ds_val_cru)
    print(f"-> Dados em RAM em {time.time() - t_dados:.2f}s")
    print("-> Conjunto de Teste: PERMANECE LACRADO (não carregado)")
    print("-" * 70)

    pin_mem = device.type == "cuda"
    loader_treino = DataLoader(
        mem_treino,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_mem,
        generator=gerador,
    )
    loader_val = DataLoader(
        mem_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_mem,
    )

    modelo = LeNet(in_channels=1, num_classes=10).to(device)
    res_params = modelo.resumo_parametros()
    criterio = nn.CrossEntropyLoss()
    otimizador = criar_otimizador(otimizador_nome, modelo, lr, momentum)
    print(
        f"Parâmetros treináveis: {res_params['total']:,} "
        f"(Conv: {res_params['convolucionais']:,} | Lineares: {res_params['lineares']:,})"
    )
    print("-" * 70)

    t_treino = time.time()
    historico, melhor = treinar_guardando_melhor(
        modelo, loader_treino, loader_val, criterio, otimizador, device, num_epocas
    )
    tempo_treino = time.time() - t_treino

    pasta_modelo.mkdir(parents=True, exist_ok=True)
    torch.save(melhor["estado"], caminho_checkpoint)

    acc_val_max = max(historico["acc_val"])
    epoca_acc_max = historico["acc_val"].index(acc_val_max) + 1

    print("-" * 70)
    print(f"Tempo Total de Treinamento: {tempo_treino:.2f} s ({tempo_treino / num_epocas:.3f} s/época)")
    print(
        f"Checkpoint salvo: época {melhor['epoca']} de {num_epocas} | "
        f"perda de validação {melhor['perda_val']:.4f} | acurácia de validação {melhor['acc_val'] * 100:.2f}%"
    )
    print(
        f"(Pico de acurácia na validação: {acc_val_max * 100:.2f}% na época {epoca_acc_max}. "
        f"Última época: {historico['acc_val'][-1] * 100:.2f}%.)"
    )

    resumo = {
        "id": id_exp,
        "seed": seed,
        "epocas": num_epocas,
        "batch": batch_size,
        "lr": lr,
        "momentum": momentum,
        "otimizador": otimizador_nome,
        "arquitetura": ARQUITETURA_LENET,
        "parametros": res_params["total"],
        "epoca_salva": melhor["epoca"],
        "acc_val_salva": round(melhor["acc_val"] * 100, 4),
        "perda_val_salva": round(melhor["perda_val"], 6),
        "perda_treino_salva": round(melhor["perda_treino"], 6),
        "acc_val_ultima": round(historico["acc_val"][-1] * 100, 4),
        "perda_val_ultima": round(historico["perda_val"][-1], 6),
        "acc_val_max": round(acc_val_max * 100, 4),
        "epoca_acc_max": epoca_acc_max,
        "tempo_treino_s": round(tempo_treino, 2),
        "dispositivo": str(device),
        "torch": torch.__version__,
    }

    pasta_resultados.mkdir(parents=True, exist_ok=True)
    registrar_resumo_cnn(pasta_resultados / "resumo_cnn_v1.csv", resumo)
    with open(pasta_resultados / f"{id_exp}.json", "w", encoding="utf-8") as f:
        json.dump({"resumo": resumo, "historico": historico}, f, ensure_ascii=False)

    if gerar_graficos:
        salvar_graficos_treino(
            historico,
            pasta_destino,
            id_exp,
            titulo_exp,
            "Cross-Entropy",
            prefixo_modelo="CNN LeNet",
        )

    print("=" * 70)
    print("Para avaliar este modelo no teste (só o escolhido pela validação), execute:")
    print(f"    python cnn_v1_testar.py -c {caminho_checkpoint.name}")
    print("=" * 70)
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cnn_v1_treinar.py",
        description="Experimento CNN v1 (LeNet): salva o melhor modelo pela validação, com semente fixa.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python cnn_v1_treinar.py -e 30 -b 256 -l 0.05 -m 0.9 --seed 1
  python cnn_v1_treinar.py -e 30 -b 256 -l 0.1 -m 0.9 --seed 2 --sem-graficos
        """,
    )
    parser.add_argument("-e", "--epocas", type=int, required=True, help="Número de épocas")
    parser.add_argument("-b", "--batch-size", type=int, default=256, help="Tamanho do lote (padrão: 256)")
    parser.add_argument("-l", "--lr", type=float, default=0.05, help="Taxa de aprendizado (padrão: 0.05)")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momento do SGD (padrão: 0.9)")
    parser.add_argument(
        "--otimizador",
        type=str,
        default="sgd",
        choices=["sgd"],
        help="Otimizador (enunciado fixa sgd)",
    )
    parser.add_argument("--seed", type=int, default=1, help="Semente (padrão: 1)")
    parser.add_argument(
        "--sem-graficos",
        action="store_true",
        help="Não gera os 5 gráficos (útil nas repetições por semente)",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    executar_treinamento_cnn(
        num_epocas=args.epocas,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        otimizador_nome=args.otimizador,
        seed=args.seed,
        gerar_graficos=not args.sem_graficos,
    )


if __name__ == "__main__":
    main()
