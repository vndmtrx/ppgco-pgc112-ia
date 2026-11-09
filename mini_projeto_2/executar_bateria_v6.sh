#!/usr/bin/env bash
# ==============================================================================
# BATERIA v6: refaz a comparação MSE (v3) contra Entropia Cruzada (v4) no padrão do v5.
# Arquitetura 784 -> 300 -> 10, SGD, lote 64, taxa 0,1, sem momento (as condições originais).
# SÓ treina e valida. O teste NÃO é aberto aqui.
#
#   - Épocas 50 e 200: 3 sementes, para a comparação principal (média ± desvio).
#   - Época 500: 1 semente, para a curva em "U" da perda de validação (seção 4.9).
#     Em CPU a mesma semente repete as primeiras épocas, então as épocas 3, 50, 120 e 200 de
#     uma semente também podem ser lidas no histórico (`resultados/<id>.json`) desta execução.
#   - Gráficos só para a primeira semente de cada configuração.
#   - No fim, mlp_v6_selecionar.py compara MSE e CE pela validação.
#
# Estimativa: cerca de 1 hora na CPU (lote 64 leva ~1,5 s por época).
#
# Uso:  ./executar_bateria_v6.sh
#       SEEDS="1 2 3 4 5" ./executar_bateria_v6.sh
# ==============================================================================

set -euo pipefail

SEEDS="${SEEDS:-1 2 3}"
ARQ_LOG="bateria_v6.log"
PRIMEIRA_SEMENTE="${SEEDS%% *}"

echo "=== INÍCIO DA BATERIA v6: $(date) | sementes: ${SEEDS} ===" | tee "$ARQ_LOG"

for perda in mse ce; do
    for epocas in 50 200; do
        for seed in $SEEDS; do
            extra=""
            if [ "$seed" != "$PRIMEIRA_SEMENTE" ]; then
                extra="--sem-graficos"
            fi
            echo "" | tee -a "$ARQ_LOG"
            echo ">>> python mlp_v6_treinar.py -e $epocas --perda $perda --seed $seed $extra" | tee -a "$ARQ_LOG"
            # shellcheck disable=SC2086
            python mlp_v6_treinar.py -e "$epocas" --perda "$perda" --seed "$seed" $extra 2>&1 | tee -a "$ARQ_LOG"
        done
    done
    echo "" | tee -a "$ARQ_LOG"
    echo ">>> python mlp_v6_treinar.py -e 500 --perda $perda --seed $PRIMEIRA_SEMENTE" | tee -a "$ARQ_LOG"
    python mlp_v6_treinar.py -e 500 --perda "$perda" --seed "$PRIMEIRA_SEMENTE" 2>&1 | tee -a "$ARQ_LOG"
done

echo "" | tee -a "$ARQ_LOG"
echo "=== SELEÇÃO E COMPARAÇÃO PELA VALIDAÇÃO ===" | tee -a "$ARQ_LOG"
python mlp_v6_selecionar.py 2>&1 | tee -a "$ARQ_LOG"

echo "" | tee -a "$ARQ_LOG"
echo "=== BATERIA v6 CONCLUÍDA: $(date) ===" | tee -a "$ARQ_LOG"
