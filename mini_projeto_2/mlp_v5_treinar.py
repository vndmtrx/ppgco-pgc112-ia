"""Experimento MLP v5 — Seleção do melhor modelo pela validação, com semente fixa.

Evolui o v4.2 (mesma rede, mesmo otimizador, mesmos argumentos) com duas mudanças:
1. O checkpoint salvo é o da época de MENOR perda de validação, não o da última época;
2. `--seed` controla os pesos iniciais e a ordem dos lotes, para repetir o experimento.

O conjunto de teste continua fora do treinamento. Cada execução também grava:
- `resultados/<id>.json`: configuração, histórico por época e a época salva;
- `resultados/resumo_v5.csv`: uma linha por execução (usada por `mlp_v5_selecionar.py`).

Uso:
    python mlp_v5_treinar.py -e 50 -b 256 -l 0.1 --seed 1
    python mlp_v5_treinar.py -e 100 -b 256 -l 0.1 -m 0.9 --hidden-dims 300 100 --seed 2

Sem argumentos, o script apenas mostra a ajuda e não cria arquivos.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_mnist_datasets
from models.mlp import MLP
from mlp_v4_2_treinar import (
    avaliar_validacao,
    criar_otimizador,
    dataset_para_ram,
    salvar_graficos,
    treinar_epoca,
)

DIRETORIO_PROJETO = Path(__file__).resolve().parent
PASTA_MODELO = DIRETORIO_PROJETO / "modelo"
PASTA_FIGURAS = DIRETORIO_PROJETO / "figuras"
PASTA_RESULTADOS = DIRETORIO_PROJETO / "resultados"

COLUNAS_RESUMO = [
    "id",
    "seed",
    "epocas",
    "batch",
    "lr",
    "momentum",
    "otimizador",
    "hidden_dims",
    "parametros",
    "epoca_salva",
    "acc_val_salva",
    "perda_val_salva",
    "perda_treino_salva",
    "acc_val_ultima",
    "perda_val_ultima",
    "tempo_treino_s",
    "dispositivo",
    "torch",
]


def definir_semente(seed: int) -> torch.Generator:
    """Fixa as fontes de aleatoriedade e devolve o gerador usado no embaralhamento dos lotes.

    Deve ser chamada ANTES de criar o modelo (a inicialização dos pesos usa o gerador global).
    Em CPU o resultado se repete. Em GPU pode haver diferenças mínimas de arredondamento.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gerador = torch.Generator()
    gerador.manual_seed(seed)
    return gerador


def montar_id(
    num_epocas: int,
    batch_size: int,
    lr: float,
    momentum: float,
    otimizador_nome: str,
    hidden_dims: List[int],
    seed: int,
) -> str:
    """Nome único da execução, no mesmo padrão do v4.2, mais a semente."""
    dims_str = "-".join(str(d) for d in hidden_dims)
    arch_str = f"_h{dims_str}" if hidden_dims != [300] else ""
    mom_str = f"_m{momentum}" if momentum > 0 else ""
    opt_str = f"_{otimizador_nome}" if otimizador_nome != "sgd" else ""
    return f"mlp_treinado_v5_e{num_epocas}_b{batch_size}_lr{lr}{mom_str}{opt_str}{arch_str}_s{seed}"


def treinar_guardando_melhor(
    modelo: nn.Module,
    loader_treino: DataLoader,
    loader_val: DataLoader,
    criterio: nn.Module,
    otimizador: torch.optim.Optimizer,
    device: torch.device,
    num_epocas: int,
    verbose: bool = True,
) -> Tuple[Dict[str, list], Dict[str, object]]:
    """Treina por `num_epocas` e guarda uma cópia dos pesos da época de menor perda de validação.

    Retorna (historico, melhor). `melhor` tem: epoca, perda_val, acc_val, perda_treino e estado
    (state_dict em CPU). O teste nunca é usado aqui.
    """
    historico: Dict[str, list] = {
        "epocas": [],
        "perda_treino": [],
        "perda_val": [],
        "acc_treino": [],
        "acc_val": [],
        "tempo_epoca": [],
    }
    melhor: Dict[str, object] = {
        "epoca": 0,
        "perda_val": float("inf"),
        "acc_val": 0.0,
        "perda_treino": 0.0,
        "estado": None,
    }

    for epoca in range(1, num_epocas + 1):
        t_inicio = time.time()
        perda_tr, acc_tr = treinar_epoca(modelo, loader_treino, criterio, otimizador, device)
        perda_vl, acc_vl = avaliar_validacao(modelo, loader_val, criterio, device)
        t_ep = time.time() - t_inicio

        historico["epocas"].append(epoca)
        historico["perda_treino"].append(perda_tr)
        historico["perda_val"].append(perda_vl)
        historico["acc_treino"].append(acc_tr)
        historico["acc_val"].append(acc_vl)
        historico["tempo_epoca"].append(t_ep)

        eh_melhor = perda_vl < melhor["perda_val"]
        if eh_melhor:
            melhor = {
                "epoca": epoca,
                "perda_val": perda_vl,
                "acc_val": acc_vl,
                "perda_treino": perda_tr,
                "estado": {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()},
            }

        if verbose:
            marca = " *melhor*" if eh_melhor else ""
            print(
                f"Época [{epoca:02d}/{num_epocas:02d}] ({t_ep:.2f}s) | "
                f"Treino: {perda_tr:.4f} (Acc: {acc_tr * 100:.2f}%) | "
                f"Val: {perda_vl:.4f} (Acc: {acc_vl * 100:.2f}%){marca}"
            )

    if melhor["estado"] is None:
        raise RuntimeError("Nenhuma época teve perda de validação finita (treino divergiu).")
    return historico, melhor


def registrar_resumo(caminho_csv: Path, linha: Dict[str, object]) -> None:
    """Grava a linha no CSV de resumo. Se o id já existir, substitui a linha antiga."""
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    linhas: List[Dict[str, object]] = []
    if caminho_csv.exists():
        with open(caminho_csv, newline="", encoding="utf-8") as f:
            linhas = [r for r in csv.DictReader(f) if r["id"] != linha["id"]]
    linhas.append(linha)
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUNAS_RESUMO)
        escritor.writeheader()
        escritor.writerows(linhas)


def executar_treinamento_v5(
    num_epocas: int,
    batch_size: int,
    lr: float,
    momentum: float,
    otimizador_nome: str,
    hidden_dims: List[int],
    seed: int,
    gerar_graficos: bool = True,
    pasta_modelo: Optional[Path] = None,
    pasta_figuras: Optional[Path] = None,
    pasta_resultados: Optional[Path] = None,
) -> Dict[str, object]:
    """Fluxo principal do v5: treina, salva o melhor checkpoint e registra o resumo."""
    pasta_modelo = pasta_modelo or PASTA_MODELO
    pasta_figuras = pasta_figuras or PASTA_FIGURAS
    pasta_resultados = pasta_resultados or PASTA_RESULTADOS

    gerador = definir_semente(seed)

    id_exp = montar_id(num_epocas, batch_size, lr, momentum, otimizador_nome, hidden_dims, seed)
    caminho_checkpoint = pasta_modelo / f"{id_exp}_melhor.pth"
    pasta_destino = pasta_figuras / id_exp.replace("mlp_treinado_", "")
    dims_str = "-".join(str(d) for d in hidden_dims)

    titulo_exp = f"E={num_epocas}, B={batch_size}, LR={lr}, Opt={otimizador_nome.upper()}"
    if momentum > 0:
        titulo_exp += f", Mom={momentum}"
    titulo_exp += f", Seed={seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"EXPERIMENTO MLP v5: melhor modelo pela validação ({titulo_exp})")
    print("=" * 70)
    print(f"Dispositivo:           {device}")
    arquitetura = " -> ".join(["784", *[str(d) for d in hidden_dims], "10"])
    print(f"Arquitetura:           {arquitetura} (Linear -> ReLU -> Logits)")
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
        mem_treino, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=pin_mem, generator=gerador,
    )
    loader_val = DataLoader(
        mem_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_mem,
    )

    modelo = MLP(input_dim=784, hidden_dims=hidden_dims, num_classes=10).to(device)
    parametros = sum(p.numel() for p in modelo.parameters())
    criterio = nn.CrossEntropyLoss()
    otimizador = criar_otimizador(otimizador_nome, modelo, lr, momentum)
    print(f"Parâmetros treináveis: {parametros:,}")
    print("-" * 70)

    t_treino = time.time()
    historico, melhor = treinar_guardando_melhor(
        modelo, loader_treino, loader_val, criterio, otimizador, device, num_epocas
    )
    tempo_treino = time.time() - t_treino

    pasta_modelo.mkdir(parents=True, exist_ok=True)
    torch.save(melhor["estado"], caminho_checkpoint)

    print("-" * 70)
    print(f"Tempo Total de Treinamento: {tempo_treino:.2f} s ({tempo_treino / num_epocas:.3f} s/época)")
    print(
        f"Checkpoint salvo: época {melhor['epoca']} de {num_epocas} | "
        f"perda de validação {melhor['perda_val']:.4f} | acurácia de validação {melhor['acc_val'] * 100:.2f}%"
    )
    print(
        f"(A última época teve perda de validação {historico['perda_val'][-1]:.4f} "
        f"e acurácia {historico['acc_val'][-1] * 100:.2f}%.)"
    )

    resumo = {
        "id": id_exp,
        "seed": seed,
        "epocas": num_epocas,
        "batch": batch_size,
        "lr": lr,
        "momentum": momentum,
        "otimizador": otimizador_nome,
        "hidden_dims": dims_str,
        "parametros": parametros,
        "epoca_salva": melhor["epoca"],
        "acc_val_salva": round(melhor["acc_val"] * 100, 4),
        "perda_val_salva": round(melhor["perda_val"], 6),
        "perda_treino_salva": round(melhor["perda_treino"], 6),
        "acc_val_ultima": round(historico["acc_val"][-1] * 100, 4),
        "perda_val_ultima": round(historico["perda_val"][-1], 6),
        "tempo_treino_s": round(tempo_treino, 2),
        "dispositivo": str(device),
        "torch": torch.__version__,
    }
    registrar_resumo(pasta_resultados / "resumo_v5.csv", resumo)
    with open(pasta_resultados / f"{id_exp}.json", "w", encoding="utf-8") as f:
        json.dump({"resumo": resumo, "historico": historico}, f, ensure_ascii=False)

    if gerar_graficos:
        salvar_graficos(historico, pasta_destino, id_exp, titulo_exp)

    print("=" * 70)
    print("Para avaliar este modelo no teste (só o escolhido pela validação), execute:")
    print(f"    python mlp_v5_testar.py -c {caminho_checkpoint.name}")
    print("=" * 70)
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlp_v5_treinar.py",
        description="Experimento MLP v5: salva o melhor modelo pela validação, com semente fixa.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python mlp_v5_treinar.py -e 50 -b 256 -l 0.1 --seed 1
  python mlp_v5_treinar.py -e 100 -b 256 -l 0.1 -m 0.9 --hidden-dims 300 100 --seed 2
        """,
    )
    parser.add_argument("-e", "--epocas", type=int, required=True, help="Número de épocas")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Tamanho do lote (padrão: 64)")
    parser.add_argument("-l", "--lr", type=float, default=0.1, help="Taxa de aprendizado (padrão: 0.1)")
    parser.add_argument("-m", "--momentum", type=float, default=0.0, help="Momento do SGD (padrão: 0.0)")
    parser.add_argument(
        "--otimizador", type=str, default="sgd", choices=["sgd", "adam", "adamw"],
        help="Otimizador (padrão: sgd)",
    )
    parser.add_argument(
        "--hidden-dims", nargs="+", type=int, default=[300],
        help="Neurônios por camada oculta (padrão: 300; ex.: --hidden-dims 300 100)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semente (padrão: 42)")
    parser.add_argument(
        "--sem-graficos", action="store_true",
        help="Não gera os 5 gráficos (útil nas repetições por semente)",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    executar_treinamento_v5(
        num_epocas=args.epocas,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        otimizador_nome=args.otimizador,
        hidden_dims=args.hidden_dims,
        seed=args.seed,
        gerar_graficos=not args.sem_graficos,
    )


if __name__ == "__main__":
    main()
