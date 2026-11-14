"""Experimento CNN v1 — Escolha da configuração da LeNet usando só a validação.

Lê `resultados/resumo_cnn_v1.csv`, agrupa por hiperparâmetros (entre sementes) e elege
a configuração de maior acurácia média de validação (usando a mesma regra de tolerância
do `mlp_v5_selecionar.py`).

Uso:
    python cnn_v1_selecionar.py
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from mlp_v5_selecionar import escolher, estatisticas, ler_resumo

DIRETORIO_PROJETO = Path(__file__).resolve().parent
CSV_PADRAO = DIRETORIO_PROJETO / "resultados" / "resumo_cnn_v1.csv"

CHAVE_CONFIG_CNN = ("epocas", "batch", "lr", "momentum", "otimizador", "arquitetura")


def agrupar_cnn(linhas: List[Dict[str, str]]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    """Agrupa linhas de resumo por configuração de hiperparâmetros da CNN."""
    grupos: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for linha in linhas:
        chave = tuple(linha[c] for c in CHAVE_CONFIG_CNN)
        grupos.setdefault(chave, []).append(linha)
    return grupos


def descrever_cnn(e: Dict[str, object]) -> str:
    """Formata descrição compacta de uma configuração da CNN."""
    c = e["config"]
    return (
        f"e={c['epocas']} b={c['batch']} lr={c['lr']} m={c['momentum']} {c['otimizador']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Escolhe a configuração final da CNN pela validação.")
    parser.add_argument("--csv", type=Path, default=CSV_PADRAO, help="CSV de resumo do cnn_v1")
    parser.add_argument("--n-validacao", type=int, default=10000, help="Imagens de validação (padrão: 10000)")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Arquivo não encontrado: {args.csv}\nRode antes o cnn_v1_treinar.py.")
        sys.exit(1)

    grupos = agrupar_cnn(ler_resumo(args.csv))
    todas = []
    for chave, grupo in grupos.items():
        est = estatisticas(chave, grupo)
        est["config"] = dict(zip(CHAVE_CONFIG_CNN, chave))
        est["melhor_seed_id"] = max(grupo, key=lambda g: float(g["acc_val_salva"]))["id"]
        todas.append(est)

    todas.sort(key=lambda e: (-e["acc_media"], e["perda_media"]))
    escolhida, tol, empatadas = escolher(todas, args.n_validacao)

    print("=" * 100)
    print("CNN LeNet — CONFIGURAÇÕES ORDENADAS PELA ACURÁCIA DE VALIDAÇÃO (média ± desvio)")
    print("=" * 100)
    print(f"{'#':>2}  {'configuração':<42} {'n':>2} {'acc val %':>14} {'perda val':>9} {'ép.':>5} {'parâm.':>8} {'s':>6}")
    for i, e in enumerate(todas, 1):
        print(
            f"{i:>2}  {descrever_cnn(e):<42} {e['n']:>2} "
            f"{e['acc_media']:>8.2f}±{e['acc_desvio']:<4.2f} {e['perda_media']:>9.4f} "
            f"{e['epoca_media']:>5.1f} {e['parametros']:>8,} {e['tempo_medio']:>6.1f}"
        )
    print("-" * 100)
    print(f"Tolerância de empate: {tol:.2f} p.p. | empatadas: {len(empatadas)}")
    print(f"ESCOLHIDA: {descrever_cnn(escolhida)}")
    print(
        f"  acurácia de validação {escolhida['acc_media']:.2f} ± {escolhida['acc_desvio']:.2f} "
        f"| perda média {escolhida['perda_media']:.4f} | {escolhida['parametros']:,} parâmetros"
    )
    print(f"  melhor semente: {escolhida['melhor_seed_id']}_melhor.pth")
    print("=" * 100)


if __name__ == "__main__":
    main()
