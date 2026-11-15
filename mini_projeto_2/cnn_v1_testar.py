"""Experimento CNN v1 — Avaliação da LeNet no conjunto de teste com registro auditável.

Carrega um checkpoint salvo por `cnn_v1_treinar.py` e avalia nas 10.000 imagens do teste oficial.
Cada execução registra uma linha em `resultados/aberturas_teste.csv`.

Uso:
    python cnn_v1_testar.py -c cnn_treinado_v1_e30_b256_lr0.05_m0.9_s1_melhor.pth
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import get_mnist_datasets
from mlp_v4_testar import (
    avaliar_teste_ce,
    calcular_metricas,
    dataset_para_ram,
    salvar_graficos_teste,
)
from mlp_v5_testar import ler_aberturas, registrar_abertura
from models.cnn import LeNet

DIRETORIO_PROJETO = Path(__file__).resolve().parent
PASTA_MODELO = DIRETORIO_PROJETO / "modelo"
PASTA_FIGURAS = DIRETORIO_PROJETO / "figuras"
PASTA_RESULTADOS = DIRETORIO_PROJETO / "resultados"


def carregar_checkpoint_cnn(caminho: Path) -> Tuple[LeNet, Path]:
    """Carrega um checkpoint de LeNet do disco (buscando em `modelo/` se necessário)."""
    if not caminho.exists():
        alternativo = PASTA_MODELO / caminho.name
        if alternativo.exists():
            caminho = alternativo
        else:
            raise FileNotFoundError(f"Checkpoint CNN não encontrado: {caminho}")

    estado = torch.load(caminho, weights_only=True)
    modelo = LeNet(in_channels=1, num_classes=10)
    modelo.load_state_dict(estado)
    modelo.eval()
    return modelo, caminho


def nome_da_pasta_de_figuras_cnn(caminho_checkpoint: Path) -> str:
    """'cnn_treinado_v1_e30_b256_lr0.05_m0.9_s1_melhor' -> 'cnn_v1_e30_b256_lr0.05_m0.9_s1'."""
    stem = caminho_checkpoint.stem
    stem = re.sub(r"^cnn_treinado_", "cnn_", stem)
    return re.sub(r"_melhor$", "", stem)


def executar_teste_cnn(
    caminho_arquivo: str,
    gerar_graficos: bool = True,
    pasta_figuras: Optional[Path] = None,
    pasta_resultados: Optional[Path] = None,
) -> Dict[str, object]:
    """Avalia o checkpoint da LeNet nas 10.000 amostras de teste e registra em aberturas_teste.csv."""
    pasta_figuras = pasta_figuras or PASTA_FIGURAS
    pasta_resultados = pasta_resultados or PASTA_RESULTADOS
    caminho_aberturas = pasta_resultados / "aberturas_teste.csv"

    print("=" * 70)
    print("EXPERIMENTO CNN v1 (LeNet): Avaliação no Conjunto de Teste")
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
    modelo, caminho = carregar_checkpoint_cnn(Path(caminho_arquivo))
    modelo = modelo.to(device)
    res_params = modelo.resumo_parametros()
    print(f"Dispositivo:           {device}")
    print(f"-> Arquivo de Pesos:   {caminho.name}")
    print(f"-> Arquitetura:        LeNet ({res_params['total']:,} parâmetros)")
    print("-" * 70)

    _, _, ds_teste_cru = get_mnist_datasets()
    mem_teste = dataset_para_ram(ds_teste_cru)
    loader_teste = DataLoader(mem_teste, batch_size=64, shuffle=False, num_workers=0)

    t_inicio = time.time()
    perda, acuracia, reais, preditas = avaliar_teste_ce(
        modelo, loader_teste, nn.CrossEntropyLoss(), device
    )
    tempo = time.time() - t_inicio

    total = len(reais)
    erros = total - sum(1 for r, p in zip(reais, preditas) if r == p)
    metricas = calcular_metricas(reais, preditas)

    print("RESULTADO NO CONJUNTO DE TESTE (LeNet):")
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
            pasta_figuras=pasta_figuras / nome_da_pasta_de_figuras_cnn(caminho),
        )
    print("=" * 70)
    return {
        "acuracia": acuracia,
        "erros": erros,
        "perda": perda,
        "tempo_inferencia_s": tempo,
        "metricas": metricas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="cnn_v1_testar.py: avalia no teste o checkpoint da LeNet escolhido pela validação."
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Nome ou caminho do .pth (procurado em 'modelo/' se for só o nome)",
    )
    parser.add_argument("--sem-graficos", action="store_true", help="Não gera os 4 gráficos de teste")
    args = parser.parse_args()
    executar_teste_cnn(args.checkpoint, gerar_graficos=not args.sem_graficos)


if __name__ == "__main__":
    main()
