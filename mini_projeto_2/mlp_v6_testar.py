"""Experimento MLP v6 — Avaliação no conjunto de teste, com registro de cada abertura.

Carrega um checkpoint salvo pelo `mlp_v6_treinar.py` e avalia nas 10.000 imagens do teste oficial.
A perda (ce ou mse) vem do nome do arquivo; use `--perda` só se o nome não a trouxer.
Rode este script SÓ nos modelos escolhidos pela validação (veja `mlp_v6_selecionar.py`).

Cada execução acrescenta uma linha em `resultados/aberturas_teste.csv`, o mesmo registro do v5,
então o contador de aberturas do teste vale para o projeto inteiro.

Uso:
    python mlp_v6_testar.py -c mlp_treinado_v6_ce_e50_b64_lr0.1_s1_melhor.pth
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from dataset import get_mnist_datasets
from mlp_v4_testar import (
    avaliar_teste_ce,
    calcular_metricas,
    carregar_checkpoint,
    dataset_para_ram,
    salvar_graficos_teste,
)
from mlp_v5_testar import (
    descrever_arquitetura,
    ler_aberturas,
    nome_da_pasta_de_figuras,
    registrar_abertura,
)
from mlp_v6_treinar import NOMES_PERDA, PASTA_FIGURAS, PASTA_RESULTADOS, criar_criterio, perda_do_nome


def executar_teste_v6(
    caminho_arquivo: str,
    perda: Optional[str] = None,
    gerar_graficos: bool = True,
    pasta_figuras: Optional[Path] = None,
    pasta_resultados: Optional[Path] = None,
) -> Dict[str, object]:
    pasta_figuras = pasta_figuras or PASTA_FIGURAS
    pasta_resultados = pasta_resultados or PASTA_RESULTADOS
    caminho_aberturas = pasta_resultados / "aberturas_teste.csv"

    perda = perda or perda_do_nome(Path(caminho_arquivo).name)
    if perda is None:
        raise ValueError("O nome do arquivo não traz a perda (_v6_ce_ ou _v6_mse_). Informe --perda ce|mse.")

    print("=" * 70)
    print(f"EXPERIMENTO MLP v6: Avaliação no Conjunto de Teste ({NOMES_PERDA[perda]})")
    print("=" * 70)

    anteriores = ler_aberturas(caminho_aberturas)
    modelos_distintos = {a["checkpoint"] for a in anteriores}
    print(
        f"Aberturas anteriores do teste: {len(anteriores)} "
        f"({len(modelos_distintos)} checkpoint(s) distinto(s))"
    )
    if anteriores:
        print("ATENÇÃO: avalie no teste apenas modelos escolhidos pela validação.")
    print("-" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo, caminho = carregar_checkpoint(Path(caminho_arquivo))
    modelo = modelo.to(device)
    print(f"Dispositivo:           {device}")
    print(f"-> Arquivo de Pesos:   {caminho.name}")
    print(f"-> Arquitetura:        {descrever_arquitetura(modelo)}")
    print("-" * 70)

    _, _, ds_teste_cru = get_mnist_datasets()
    mem_teste = dataset_para_ram(ds_teste_cru)
    loader_teste = DataLoader(mem_teste, batch_size=64, shuffle=False, num_workers=0)

    t_inicio = time.time()
    perda_teste, acuracia, reais, preditas = avaliar_teste_ce(
        modelo, loader_teste, criar_criterio(perda), device
    )
    tempo = time.time() - t_inicio

    total = len(reais)
    erros = total - sum(1 for r, p in zip(reais, preditas) if r == p)
    metricas = calcular_metricas(reais, preditas)

    print("RESULTADO NO CONJUNTO DE TESTE:")
    print(f"- Amostras:                    {total:,}")
    print(f"- Perda média ({NOMES_PERDA[perda]}): {perda_teste:.5f}")
    print(f"- Acertos:                     {total - erros:,} ({acuracia * 100:.2f}%)")
    print(f"- Erros:                       {erros:,} ({erros / total * 100:.2f}%)")
    print(f"- Tempo de Inferência:         {tempo:.3f} s")
    print("-" * 70)
    print("MÉTRICAS MACRO NO TESTE:")
    print(f"- Precisão Macro:              {metricas['macro_precisao'] * 100:.2f}%")
    print(f"- Revocação (Recall) Macro:    {metricas['macro_recall'] * 100:.2f}%")
    print(f"- F1-Score Macro:              {metricas['macro_f1'] * 100:.2f}%")
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
            "perda_teste": round(perda_teste, 6),
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
    return {"acuracia": acuracia, "erros": erros, "perda": perda_teste, "metricas": metricas}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mlp_v6_testar.py: avalia no teste um checkpoint escolhido pela validação."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, required=True,
        help="Nome ou caminho do .pth (procurado em 'modelo/' se for só o nome)",
    )
    parser.add_argument(
        "--perda", type=str, default=None, choices=sorted(NOMES_PERDA),
        help="Só se o nome do arquivo não trouxer a perda",
    )
    parser.add_argument("--sem-graficos", action="store_true", help="Não gera os 4 gráficos de teste")
    args = parser.parse_args()
    executar_teste_v6(args.checkpoint, perda=args.perda, gerar_graficos=not args.sem_graficos)


if __name__ == "__main__":
    main()
