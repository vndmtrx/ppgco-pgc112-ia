#!/usr/bin/env bash
# ==============================================================================
# BATERIA CNN v1 (LeNet): 4 configurações em blocos, repetidas com 3 sementes.
# Só SGD com momento e ReLU, como pede o enunciado.
# SÓ treina e valida. O teste NÃO é aberto aqui.
#   - Cada execução salva o checkpoint de menor perda de validação (_melhor.pth)
#     e uma linha em resultados/resumo_cnn_v1.csv.
#   - Gráficos só para a primeira semente (as outras usam --sem-graficos).
#   - No fim, cnn_v1_selecionar.py escolhe a configuração pela validação.
#
# Uso:  ./executar_bateria_cnn.sh
#       SEEDS="1 2 3" ./executar_bateria_cnn.sh
# ==============================================================================

set -euo pipefail

SEEDS="${SEEDS:-1 2 3}"
ARQ_LOG="bateria_cnn_v1.log"
PRIMEIRA_SEMENTE="${SEEDS%% *}"

echo "=== INÍCIO DA BATERIA CNN v1: $(date) | sementes: ${SEEDS} ===" | tee "$ARQ_LOG"

CONFIGS=(
    # Ceteris paribus com o MLP campeão do v5 (b=256, lr=0.1, m=0.9)
    "-e 30 -b 256 -l 0.1 -m 0.9"
    # Redução de taxa para compensar o compartilhamento de pesos nas convoluções
    "-e 30 -b 256 -l 0.05 -m 0.9"
    "-e 30 -b 256 -l 0.02 -m 0.9"
    # Lote menor com taxa intermediária
    "-e 30 -b 128 -l 0.05 -m 0.9"
)

for cfg in "${CONFIGS[@]}"; do
    for seed in $SEEDS; do
        extra=""
        if [ "$seed" != "$PRIMEIRA_SEMENTE" ]; then
            extra="--sem-graficos"
        fi
        echo "" | tee -a "$ARQ_LOG"
        echo ">>> python cnn_v1_treinar.py $cfg --seed $seed $extra" | tee -a "$ARQ_LOG"
        # shellcheck disable=SC2086
        python cnn_v1_treinar.py $cfg --seed "$seed" $extra 2>&1 | tee -a "$ARQ_LOG"
    done
done

echo "" | tee -a "$ARQ_LOG"
echo "=== SELEÇÃO PELA VALIDAÇÃO (CNN v1) ===" | tee -a "$ARQ_LOG"
python cnn_v1_selecionar.py 2>&1 | tee -a "$ARQ_LOG"

echo "" | tee -a "$ARQ_LOG"
echo "=== BATERIA CNN v1 CONCLUÍDA: $(date) ===" | tee -a "$ARQ_LOG"
