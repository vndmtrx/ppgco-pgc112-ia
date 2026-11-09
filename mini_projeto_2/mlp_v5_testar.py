"""Experimento MLP v5 — Avaliação no conjunto de teste, com registro de cada abertura.

Carrega um checkpoint salvo pelo `mlp_v5_treinar.py` e avalia nas 10.000 imagens do teste oficial.
Rode este script SÓ no modelo escolhido pela validação (veja `mlp_v5_selecionar.py`).

Para manter essa disciplina auditável, cada execução acrescenta uma linha em
`resultados/aberturas_teste.csv` (data, checkpoint, acurácia, erros) e avisa quantas vezes
o teste já foi aberto antes.

Uso:
    python mlp_v5_testar.py -c mlp_treinado_v5_e100_b256_lr0.1_m0.9_h300-100_s1_melhor.pth
"""

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_mnist_datasets
from mlp_v4_testar import (
    avaliar_teste_ce,
    calcular_metricas,
    carregar_checkpoint,
    dataset_para_ram,
    salvar_graficos_teste,
)

DIRETORIO_PROJETO = Path(__file__).resolve().parent
PASTA_MODELO = DIRETORIO_PROJETO / "modelo"
PASTA_FIGURAS = DIRETORIO_PROJETO / "figuras"
PASTA_RESULTADOS = DIRETORIO_PROJETO / "resultados"

COLUNAS_ABERTURAS = ["data_hora", "checkpoint", "acuracia_teste", "erros", "perda_teste"]


def ler_aberturas(caminho_csv: Path) -> List[Dict[str, str]]:
    """Lê as aberturas anteriores do teste (lista vazia se o arquivo não existir)."""
    if not caminho_csv.exists():
        return []
    with open(caminho_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def registrar_abertura(caminho_csv: Path, linha: Dict[str, object]) -> None:
    """Acrescenta uma linha ao registro de aberturas do teste."""
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    novo = not caminho_csv.exists()
    with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUNAS_ABERTURAS)
        if novo:
            escritor.writeheader()
        escritor.writerow(linha)


def descrever_arquitetura(modelo: nn.Module) -> str:
    """Texto '784 -> 300 -> 100 -> 10' a partir do modelo carregado."""
    dims = [modelo.input_dim, *modelo.hidden_dims, modelo.num_classes]
    return " -> ".join(str(d) for d in dims)


def nome_da_pasta_de_figuras(caminho_checkpoint: Path) -> str:
    """'mlp_treinado_v5_e50_b256_lr0.1_s1_melhor' -> 'v5_e50_b256_lr0.1_s1' (mesma pasta do treino)."""
    stem = caminho_checkpoint.stem
    stem = re.sub(r"^mlp_treinado_", "", stem)
    return re.sub(r"_melhor$", "", stem)


def executar_teste_v5(
    caminho_arquivo: str,
    gerar_graficos: bool = True,
    pasta_figuras: Optional[Path] = None,
    pasta_resultados: Optional[Path] = None,
) -> Dict[str, object]:
    pasta_figuras = pasta_figuras or PASTA_FIGURAS
    pasta_resultados = pasta_resultados or PASTA_RESULTADOS
    caminho_aberturas = pasta_resultados / "aberturas_teste.csv"

    print("=" * 70)
    print("EXPERIMENTO MLP v5: Avaliação no Conjunto de Teste")
    print("=" * 70)

    anteriores = ler_aberturas(caminho_aberturas)
    modelos_distintos = {a["checkpoint"] for a in anteriores}
    print(
        f"Aberturas anteriores do teste: {len(anteriores)} "
        f"({len(modelos_distintos)} checkpoint(s) distinto(s))"
    )
    if anteriores:
        print("ATENÇÃO: avalie no teste apenas o modelo escolhido pela validação.")
    print("-" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, caminho = carregar_checkpoint(Path(caminho_arquivo))
    modelo = modelo.to(device)
    print(f"Dispositivo:           {device}")
    print(f"-> Arquivo de Pesos:   {caminho.name}")
    print(f"-> Arquitetura:        {descrever_arquitetura(modelo)} (Softmax / CrossEntropy)")
    print("-" * 70)

    _, _, ds_teste_cru = get_mnist_datasets()
    mem_teste = dataset_para_ram(ds_teste_cru)
    loader_teste = DataLoader(mem_teste, batch_size=64, shuffle=False, num_workers=0)

    t_inicio = time.time()
    perda, acuracia, reais, preditas = avaliar_teste_ce(modelo, loader_teste, nn.CrossEntropyLoss(), device)
    tempo = time.time() - t_inicio

    total = len(reais)
    erros = total - sum(1 for r, p in zip(reais, preditas) if r == p)
    metricas = calcular_metricas(reais, preditas)

    print("RESULTADO NO CONJUNTO DE TESTE:")
    print(f"• Amostras:                    {total:,}")
    print(f"• Perda Média (CrossEntropy):  {perda:.5f}")
    print(f"• Acertos:                     {total - erros:,} ({acuracia * 100:.2f}%)")
    print(f"• Erros:                       {erros:,} ({erros / total * 100:.2f}%)")
    print(f"• Tempo de Inferência:         {tempo:.3f} s")
    print("-" * 70)
    print("MÉTRICAS MACRO NO TESTE:")
    print(f"• Precisão Macro:              {metricas['macro_precisao'] * 100:.2f}%")
    print(f"• Revocação (Recall) Macro:    {metricas['macro_recall'] * 100:.2f}%")
    print(f"• F1-Score Macro:              {metricas['macro_f1'] * 100:.2f}%")
    print("-" * 70)
    print("TOP 5 CONFUSÕES MAIS FREQUENTES:")
    for posicao, (qtd, real, pred) in enumerate(metricas["top_confusoes"], 1):
        print(f"  {posicao}. Dígito {real} classificado como {pred}: {qtd} vezes")

    registrar_abertura(
        caminho_aberturas,
        {
            "data_hora": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": caminho.name,
            "acuracia_teste": round(acuracia * 100, 4),
            "erros": erros,
            "perda_teste": round(perda, 6),
        },
    )

    if gerar_graficos:
        salvar_graficos_teste(
            prefixo_base=caminho.stem,
            metricas=metricas,
            imagens_teste=mem_teste.tensors[0],
            reais=reais,
            preditas=preditas,
            pasta_figuras=pasta_figuras / nome_da_pasta_de_figuras(caminho),
        )
    print("=" * 70)
    return {"acuracia": acuracia, "erros": erros, "perda": perda, "metricas": metricas}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mlp_v5_testar.py: avalia no teste o checkpoint escolhido pela validação."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, required=True,
        help="Nome ou caminho do .pth (procurado em 'modelo/' se for só o nome)",
    )
    parser.add_argument("--sem-graficos", action="store_true", help="Não gera os 4 gráficos de teste")
    args = parser.parse_args()
    executar_teste_v5(args.checkpoint, gerar_graficos=not args.sem_graficos)


if __name__ == "__main__":
    main()
