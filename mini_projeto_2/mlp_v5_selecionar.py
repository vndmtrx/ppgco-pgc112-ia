"""Experimento MLP v5 — Escolha da configuração final usando só a validação.

Lê `resultados/resumo_v5.csv` (uma linha por execução do `mlp_v5_treinar.py`), agrupa as
execuções que diferem só na semente e compara as configurações pela acurácia de validação
na época salva (média e desvio entre sementes).

Regra de escolha (justificativa para o relatório):
1. Acha a configuração de maior acurácia média de validação.
2. Considera "empatadas" as configurações dentro de uma tolerância do melhor. A tolerância é o
   maior entre (a) o desvio entre sementes do melhor e (b) o erro-padrão de uma acurácia medida
   em N imagens de validação, sqrt(p*(1-p)/N). Com N = 10.000 e p ~ 98%, é ~0,14 ponto percentual.
3. Entre as empatadas, escolhe a de MENOR número de parâmetros (mais simples).

O teste não é lido aqui. Depois de escolher, rode `mlp_v5_testar.py` uma vez no melhor checkpoint
da configuração escolhida.

Uso:
    python mlp_v5_selecionar.py
    python mlp_v5_selecionar.py --csv resultados/resumo_v5.csv --n-validacao 10000
"""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DIRETORIO_PROJETO = Path(__file__).resolve().parent
CSV_PADRAO = DIRETORIO_PROJETO / "resultados" / "resumo_v5.csv"

CHAVE_CONFIG = ("epocas", "batch", "lr", "momentum", "otimizador", "hidden_dims")


def ler_resumo(caminho_csv: Path) -> List[Dict[str, str]]:
    with open(caminho_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def agrupar_por_configuracao(linhas: List[Dict[str, str]]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    grupos: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for linha in linhas:
        chave = tuple(linha[c] for c in CHAVE_CONFIG)
        grupos.setdefault(chave, []).append(linha)
    return grupos


def estatisticas(chave: Tuple[str, ...], grupo: List[Dict[str, str]]) -> Dict[str, object]:
    acc = [float(g["acc_val_salva"]) for g in grupo]
    perda = [float(g["perda_val_salva"]) for g in grupo]
    epocas = [int(g["epoca_salva"]) for g in grupo]
    tempo = [float(g["tempo_treino_s"]) for g in grupo]
    return {
        "config": dict(zip(CHAVE_CONFIG, chave)),
        "n": len(grupo),
        "acc_media": statistics.mean(acc),
        "acc_desvio": statistics.stdev(acc) if len(acc) > 1 else 0.0,
        "perda_media": statistics.mean(perda),
        "epoca_media": statistics.mean(epocas),
        "tempo_medio": statistics.mean(tempo),
        "parametros": int(grupo[0]["parametros"]),
        "sementes": sorted(int(g["seed"]) for g in grupo),
    }


def tolerancia_pp(melhor: Dict[str, object], n_validacao: int) -> float:
    """Tolerância (em pontos percentuais) para considerar duas configurações empatadas."""
    p = float(melhor["acc_media"]) / 100.0
    erro_padrao = 100.0 * math.sqrt(max(p * (1.0 - p), 0.0) / n_validacao)
    return max(float(melhor["acc_desvio"]), erro_padrao)


def escolher(
    todas: List[Dict[str, object]], n_validacao: int = 10000
) -> Tuple[Dict[str, object], float, List[Dict[str, object]]]:
    """Devolve (escolhida, tolerancia_pp, empatadas). `todas` deve ter ao menos uma configuração."""
    if not todas:
        raise ValueError("Nenhuma configuração para escolher.")
    melhor = max(todas, key=lambda e: e["acc_media"])
    tol = tolerancia_pp(melhor, n_validacao)
    empatadas = [e for e in todas if e["acc_media"] >= melhor["acc_media"] - tol]
    escolhida = min(empatadas, key=lambda e: (e["parametros"], -e["acc_media"]))
    return escolhida, tol, empatadas


def descrever(e: Dict[str, object]) -> str:
    c = e["config"]
    return (
        f"e={c['epocas']} b={c['batch']} lr={c['lr']} m={c['momentum']} "
        f"{c['otimizador']} h={c['hidden_dims']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Escolhe a configuração final pela validação (v5).")
    parser.add_argument("--csv", type=Path, default=CSV_PADRAO, help="CSV de resumo do v5")
    parser.add_argument("--n-validacao", type=int, default=10000, help="Imagens de validação (padrão: 10000)")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Arquivo não encontrado: {args.csv}\nRode antes o mlp_v5_treinar.py (ou executar_bateria_v5.sh).")
        sys.exit(1)

    grupos = agrupar_por_configuracao(ler_resumo(args.csv))
    todas = [estatisticas(chave, grupo) for chave, grupo in grupos.items()]
    todas.sort(key=lambda e: -e["acc_media"])
    escolhida, tol, empatadas = escolher(todas, args.n_validacao)

    print("=" * 100)
    print("CONFIGURAÇÕES ORDENADAS PELA ACURÁCIA DE VALIDAÇÃO (média ± desvio entre sementes)")
    print("=" * 100)
    print(f"{'#':>2}  {'configuração':<52} {'n':>2} {'acc val %':>14} {'perda val':>9} {'ép.':>5} {'parâm.':>8} {'s':>6}")
    for i, e in enumerate(todas, 1):
        print(
            f"{i:>2}  {descrever(e):<52} {e['n']:>2} "
            f"{e['acc_media']:>8.2f}±{e['acc_desvio']:<4.2f} {e['perda_media']:>9.4f} "
            f"{e['epoca_media']:>5.1f} {e['parametros']:>8,} {e['tempo_medio']:>6.1f}"
        )
    print("-" * 100)
    print(f"Tolerância de empate: {tol:.2f} p.p. | configurações empatadas com a melhor: {len(empatadas)}")
    print(f"ESCOLHIDA (menos parâmetros entre as empatadas): {descrever(escolhida)}")
    print(
        f"  acurácia de validação {escolhida['acc_media']:.2f} ± {escolhida['acc_desvio']:.2f} "
        f"| {escolhida['parametros']:,} parâmetros | sementes {escolhida['sementes']}"
    )
    print("Próximo passo: rodar mlp_v5_testar.py UMA vez no checkpoint '_melhor.pth' dessa configuração.")
    if escolhida["n"] < 3:
        print("AVISO: menos de 3 sementes nesta configuração; o desvio entre sementes não é confiável.")


if __name__ == "__main__":
    main()
