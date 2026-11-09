"""Experimento MLP v6 — Escolha pela validação e comparação MSE contra Entropia Cruzada.

Lê `resultados/resumo_v6.csv` (uma linha por execução) e:
1. agrupa as execuções de mesma configuração (inclui a perda) e calcula média e desvio entre sementes;
2. escolhe a configuração como no v5: entre as empatadas com a melhor (tolerância = o maior entre o
   desvio entre sementes e o erro-padrão da acurácia), fica a de menos parâmetros;
3. para cada configuração que rodou com as duas perdas, mostra MSE e CE lado a lado, com a diferença
   e se ela passa da tolerância.

Não abre o teste.

Uso:
    python mlp_v6_selecionar.py
    python mlp_v6_selecionar.py --csv outro_resumo.csv
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from mlp_v5_selecionar import escolher, tolerancia_pp

DIRETORIO_PROJETO = Path(__file__).resolve().parent
CSV_PADRAO = DIRETORIO_PROJETO / "resultados" / "resumo_v6.csv"

CHAVE_CONFIG = ("perda", "epocas", "batch", "lr", "momentum", "otimizador", "hidden_dims")
# A mesma configuração sem a perda: serve para parear MSE e CE.
CHAVE_PAREAMENTO = tuple(c for c in CHAVE_CONFIG if c != "perda")


def ler_resumo(caminho_csv: Path) -> List[Dict[str, str]]:
    with open(caminho_csv, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def agrupar_por_configuracao(linhas: List[Dict[str, str]]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    grupos: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for linha in linhas:
        grupos.setdefault(tuple(linha[c] for c in CHAVE_CONFIG), []).append(linha)
    return grupos


def estatisticas(chave: Tuple[str, ...], grupo: List[Dict[str, str]]) -> Dict[str, object]:
    def media_desvio(campo: str) -> Tuple[float, float]:
        v = [float(g[campo]) for g in grupo]
        return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)

    acc_m, acc_d = media_desvio("acc_val_salva")
    ult_m, _ = media_desvio("acc_val_ultima")
    return {
        "config": dict(zip(CHAVE_CONFIG, chave)),
        "n": len(grupo),
        "acc_media": acc_m,
        "acc_desvio": acc_d,
        "acc_ultima_media": ult_m,
        "perda_media": statistics.mean(float(g["perda_val_salva"]) for g in grupo),
        "epoca_media": statistics.mean(int(g["epoca_salva"]) for g in grupo),
        "tempo_medio": statistics.mean(float(g["tempo_treino_s"]) for g in grupo),
        "parametros": int(grupo[0]["parametros"]),
        "sementes": sorted(int(g["seed"]) for g in grupo),
    }


def descrever(e: Dict[str, object]) -> str:
    c = e["config"]
    return (
        f"{c['perda']} e={c['epocas']} b={c['batch']} lr={c['lr']} m={c['momentum']} "
        f"{c['otimizador']} h={c['hidden_dims']}"
    )


def parear_perdas(todas: List[Dict[str, object]]) -> List[Tuple[Dict[str, object], Dict[str, object]]]:
    """Pares (mse, ce) de configurações idênticas, exceto pela perda."""
    por_chave: Dict[Tuple[str, ...], Dict[str, Dict[str, object]]] = {}
    for e in todas:
        chave = tuple(e["config"][c] for c in CHAVE_PAREAMENTO)
        por_chave.setdefault(chave, {})[e["config"]["perda"]] = e
    return [(d["mse"], d["ce"]) for d in por_chave.values() if "mse" in d and "ce" in d]


def main() -> None:
    parser = argparse.ArgumentParser(description="Escolhe pela validação e compara MSE com CE (v6).")
    parser.add_argument("--csv", type=Path, default=CSV_PADRAO, help="CSV de resumo do v6")
    parser.add_argument("--n-validacao", type=int, default=10000, help="Imagens de validação (padrão: 10000)")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Arquivo não encontrado: {args.csv}\nRode antes o mlp_v6_treinar.py (ou executar_bateria_v6.sh).")
        sys.exit(1)

    todas = [estatisticas(k, g) for k, g in agrupar_por_configuracao(ler_resumo(args.csv)).items()]
    todas.sort(key=lambda e: -e["acc_media"])
    escolhida, tol, empatadas = escolher(todas, args.n_validacao)

    print("=" * 108)
    print("CONFIGURAÇÕES ORDENADAS PELA ACURÁCIA DE VALIDAÇÃO NO CHECKPOINT SALVO (média ± desvio entre sementes)")
    print("=" * 108)
    print(f"{'#':>2}  {'configuração':<50} {'n':>2} {'acc val %':>12} {'acc última':>10} {'perda val':>9} {'ép.':>5} {'parâm.':>8}")
    for i, e in enumerate(todas, 1):
        print(
            f"{i:>2}  {descrever(e):<50} {e['n']:>2} "
            f"{e['acc_media']:>6.2f}±{e['acc_desvio']:<4.2f} {e['acc_ultima_media']:>10.2f} "
            f"{e['perda_media']:>9.4f} {e['epoca_media']:>5.1f} {e['parametros']:>8,}"
        )
    print("-" * 108)
    print(f"Tolerância de empate: {tol:.2f} p.p. | configurações empatadas com a melhor: {len(empatadas)}")
    print(f"ESCOLHIDA (menos parâmetros entre as empatadas): {descrever(escolhida)}")
    print(
        f"  acurácia de validação {escolhida['acc_media']:.2f} ± {escolhida['acc_desvio']:.2f} "
        f"| {escolhida['parametros']:,} parâmetros | sementes {escolhida['sementes']}"
    )
    print("OBS.: a escolha usa o checkpoint de menor perda. A coluna 'acc última' mostra a outra leitura.")

    pares = parear_perdas(todas)
    if pares:
        print()
        print("=" * 108)
        print("MSE CONTRA ENTROPIA CRUZADA (mesma configuração, acurácia de validação no checkpoint salvo)")
        print("=" * 108)
        print(f"{'configuração (sem a perda)':<44} {'MSE':>12} {'CE':>12} {'CE - MSE':>9}  veredito")
        for mse, ce in sorted(pares, key=lambda p: int(p[0]["config"]["epocas"])):
            c = mse["config"]
            nome = f"e={c['epocas']} b={c['batch']} lr={c['lr']} m={c['momentum']} h={c['hidden_dims']}"
            dif = ce["acc_media"] - mse["acc_media"]
            limite = max(tolerancia_pp(mse, args.n_validacao), tolerancia_pp(ce, args.n_validacao))
            veredito = "fora do ruído" if abs(dif) > limite else "dentro do ruído"
            print(
                f"{nome:<44} {mse['acc_media']:>6.2f}±{mse['acc_desvio']:<4.2f} "
                f"{ce['acc_media']:>6.2f}±{ce['acc_desvio']:<4.2f} {dif:>+9.2f}  {veredito} (tol. {limite:.2f})"
            )
        print("-" * 108)
        if any(e["n"] < 3 for par in pares for e in par):
            print("AVISO: há configuração com menos de 3 sementes; o desvio entre sementes não é confiável.")


if __name__ == "__main__":
    main()
