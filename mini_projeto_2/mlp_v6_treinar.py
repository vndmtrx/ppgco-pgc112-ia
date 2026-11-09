"""Experimento MLP v6 — Treino unificado: MSE ou Entropia Cruzada, no padrão do v5.

Reúne num script só o que ficou espalhado em v3 (MSE + One-Hot), v4 (Softmax + Entropia Cruzada),
v4.2 (hiperparâmetros) e v5 (melhor checkpoint pela validação e semente fixa):

1. `--perda ce|mse` escolhe a função de perda (padrão: ce);
2. `--seed` fixa pesos iniciais e ordem dos lotes;
3. o checkpoint salvo é o da época de MENOR perda de validação;
4. o conjunto de teste NÃO é carregado aqui;
5. cada execução grava `resultados/<id>.json` e uma linha em `resultados/resumo_v6.csv`.

O v5 continua no repositório, sem mudanças, porque os resultados da seção 4.15 vêm dele.

Os valores padrão (lote 64, taxa 0,1, sem momento, uma camada de 300) são os do v3 e do v4, para a
comparação MSE contra Entropia Cruzada repetir as condições originais.

Uso:
    python mlp_v6_treinar.py -e 50 --perda mse --seed 1
    python mlp_v6_treinar.py -e 50 --perda ce --seed 1
    python mlp_v6_treinar.py -e 100 -b 256 -l 0.1 -m 0.9 --hidden-dims 300 100 --seed 2

Sem argumentos, o script apenas mostra a ajuda e não cria arquivos.
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import get_mnist_datasets
from graficos_mlp import salvar_graficos_treino
from models.mlp import MLP
from mlp_v4_2_treinar import criar_otimizador, dataset_para_ram
from mlp_v5_treinar import definir_semente, treinar_guardando_melhor

DIRETORIO_PROJETO = Path(__file__).resolve().parent
PASTA_MODELO = DIRETORIO_PROJETO / "modelo"
PASTA_FIGURAS = DIRETORIO_PROJETO / "figuras"
PASTA_RESULTADOS = DIRETORIO_PROJETO / "resultados"

NOMES_PERDA = {"ce": "Entropia Cruzada", "mse": "MSE com One-Hot"}

COLUNAS_RESUMO = [
    "id",
    "perda",
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
    "acc_val_max",
    "epoca_acc_max",
    "tempo_treino_s",
    "dispositivo",
    "torch",
]


class PerdaMSEOneHot(nn.Module):
    """MSE entre os 10 logits e o alvo One-Hot, com a mesma média do `nn.MSELoss` do v3.

    Recebe os rótulos como inteiros (como a entropia cruzada), então o laço de treino é o mesmo.
    """

    def forward(self, saidas: torch.Tensor, rotulos: torch.Tensor) -> torch.Tensor:
        alvo = F.one_hot(rotulos, num_classes=saidas.shape[-1]).to(saidas.dtype)
        return F.mse_loss(saidas, alvo)


def criar_criterio(perda: str) -> nn.Module:
    if perda == "ce":
        return nn.CrossEntropyLoss()
    if perda == "mse":
        return PerdaMSEOneHot()
    raise ValueError(f"Perda desconhecida: {perda!r} (use 'ce' ou 'mse').")


def montar_id(
    perda: str,
    num_epocas: int,
    batch_size: int,
    lr: float,
    momentum: float,
    otimizador_nome: str,
    hidden_dims: List[int],
    seed: int,
) -> str:
    """Nome único da execução. A perda vem logo depois da versão, para o nome mostrar o que mudou."""
    dims_str = "-".join(str(d) for d in hidden_dims)
    arch_str = f"_h{dims_str}" if hidden_dims != [300] else ""
    mom_str = f"_m{momentum}" if momentum > 0 else ""
    opt_str = f"_{otimizador_nome}" if otimizador_nome != "sgd" else ""
    return (
        f"mlp_treinado_v6_{perda}_e{num_epocas}_b{batch_size}_lr{lr}"
        f"{mom_str}{opt_str}{arch_str}_s{seed}"
    )


def perda_do_nome(nome: str) -> Optional[str]:
    """'mlp_treinado_v6_mse_e50_..._melhor.pth' -> 'mse'. Devolve None se o nome não tiver a perda."""
    achado = re.search(r"_v6_(ce|mse)_", nome)
    return achado.group(1) if achado else None


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


def executar_treinamento_v6(
    perda: str,
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
    """Treina, salva o melhor checkpoint pela validação e registra o resumo. Não toca no teste."""
    pasta_modelo = pasta_modelo or PASTA_MODELO
    pasta_figuras = pasta_figuras or PASTA_FIGURAS
    pasta_resultados = pasta_resultados or PASTA_RESULTADOS

    gerador = definir_semente(seed)
    criterio = criar_criterio(perda)

    id_exp = montar_id(perda, num_epocas, batch_size, lr, momentum, otimizador_nome, hidden_dims, seed)
    caminho_checkpoint = pasta_modelo / f"{id_exp}_melhor.pth"
    pasta_destino = pasta_figuras / id_exp.replace("mlp_treinado_", "")
    dims_str = "-".join(str(d) for d in hidden_dims)

    titulo_exp = f"{perda.upper()}, E={num_epocas}, B={batch_size}, LR={lr}, Opt={otimizador_nome.upper()}"
    if momentum > 0:
        titulo_exp += f", Mom={momentum}"
    titulo_exp += f", Seed={seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"EXPERIMENTO MLP v6: {NOMES_PERDA[perda]} ({titulo_exp})")
    print("=" * 70)
    print(f"Dispositivo:           {device}")
    arquitetura = " -> ".join(["784", *[str(d) for d in hidden_dims], "10"])
    print(f"Arquitetura:           {arquitetura} (Linear -> ReLU -> Logits)")
    print(f"Função de Perda:       {NOMES_PERDA[perda]}")
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

    acc_val = historico["acc_val"]
    i_max = max(range(len(acc_val)), key=lambda i: acc_val[i])

    print("-" * 70)
    print(f"Tempo Total de Treinamento: {tempo_treino:.2f} s ({tempo_treino / num_epocas:.3f} s/época)")
    print(
        f"Checkpoint salvo: época {melhor['epoca']} de {num_epocas} | "
        f"perda de validação {melhor['perda_val']:.4f} | acurácia de validação {melhor['acc_val'] * 100:.2f}%"
    )
    print(
        f"(Última época: perda de validação {historico['perda_val'][-1]:.4f}, "
        f"acurácia {acc_val[-1] * 100:.2f}%. Maior acurácia de validação: "
        f"{acc_val[i_max] * 100:.2f}% na época {i_max + 1}.)"
    )

    resumo = {
        "id": id_exp,
        "perda": perda,
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
        "acc_val_ultima": round(acc_val[-1] * 100, 4),
        "perda_val_ultima": round(historico["perda_val"][-1], 6),
        "acc_val_max": round(acc_val[i_max] * 100, 4),
        "epoca_acc_max": i_max + 1,
        "tempo_treino_s": round(tempo_treino, 2),
        "dispositivo": str(device),
        "torch": torch.__version__,
    }
    registrar_resumo(pasta_resultados / "resumo_v6.csv", resumo)
    with open(pasta_resultados / f"{id_exp}.json", "w", encoding="utf-8") as f:
        json.dump({"resumo": resumo, "historico": historico}, f, ensure_ascii=False)

    if gerar_graficos:
        salvar_graficos_treino(historico, pasta_destino, id_exp, titulo_exp, NOMES_PERDA[perda])

    print("=" * 70)
    print("Para avaliar este modelo no teste (só o escolhido pela validação), execute:")
    print(f"    python mlp_v6_testar.py -c {caminho_checkpoint.name}")
    print("=" * 70)
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlp_v6_treinar.py",
        description="Experimento MLP v6: MSE ou Entropia Cruzada, melhor modelo pela validação, semente fixa.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python mlp_v6_treinar.py -e 50 --perda mse --seed 1
  python mlp_v6_treinar.py -e 50 --perda ce --seed 1
  python mlp_v6_treinar.py -e 100 -b 256 -l 0.1 -m 0.9 --hidden-dims 300 100 --seed 2
        """,
    )
    parser.add_argument("-e", "--epocas", type=int, required=True, help="Número de épocas")
    parser.add_argument(
        "--perda", type=str, default="ce", choices=sorted(NOMES_PERDA),
        help="Função de perda: ce (Entropia Cruzada) ou mse (MSE com One-Hot). Padrão: ce",
    )
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
    executar_treinamento_v6(
        perda=args.perda,
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
