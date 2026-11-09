#!/usr/bin/env bash
# ==============================================================================
# BATERIA v5: 15 configurações em blocos, repetidas com várias sementes.
# Só SGD (com ou sem momento) e ReLU, como pede o enunciado.
# Base: 1 camada de 300, lote 256, lr 0,1, momento 0,9, 50 épocas.
# Cada bloco muda uma coisa só: largura, profundidade, lote, taxa, momento.
# SÓ treina e valida. O teste NÃO é aberto aqui.
#   - Cada execução salva o checkpoint de menor perda de validação (_melhor.pth)
#     e uma linha em resultados/resumo_v5.csv.
#   - Gráficos só para a primeira semente (as outras usam --sem-graficos).
#   - No fim, mlp_v5_selecionar.py escolhe a configuração pela validação.
#
# Uso:  ./executar_bateria_v5.sh              # sementes 1 2 3
#       SEEDS="1 2 3 4 5" ./executar_bateria_v5.sh
# ==============================================================================

set -euo pipefail

SEEDS="${SEEDS:-1 2 3}"
ARQ_LOG="bateria_v5.log"
PRIMEIRA_SEMENTE="${SEEDS%% *}"

echo "=== INÍCIO DA BATERIA v5: $(date) | sementes: ${SEEDS} ===" | tee "$ARQ_LOG"

CONFIGS=(
    # Base
    "-e 50 -b 256 -l 0.1 -m 0.9"
    # Bloco A: largura (1 camada)
    "-e 50 -b 256 -l 0.1 -m 0.9 --hidden-dims 64"
    "-e 50 -b 256 -l 0.1 -m 0.9 --hidden-dims 128"
    "-e 50 -b 256 -l 0.1 -m 0.9 --hidden-dims 512"
    "-e 50 -b 256 -l 0.1 -m 0.9 --hidden-dims 1024"
    # Bloco B: profundidade
    "-e 50 -b 256 -l 0.1 -m 0.9 --hidden-dims 300 100"
    "-e 50 -b 256 -l 0.1 -m 0.9 --hidden-dims 512 256"
    "-e 50 -b 256 -l 0.1 -m 0.9 --hidden-dims 256 128 64"
    "-e 50 -b 256 -l 0.1 -m 0.9 --hidden-dims 512 256 128"
    # Bloco C: lote
    "-e 50 -b 64 -l 0.1 -m 0.9"
    "-e 50 -b 128 -l 0.1 -m 0.9"
    "-e 50 -b 512 -l 0.1 -m 0.9"
    # Bloco D: taxa de aprendizado
    "-e 50 -b 256 -l 0.05 -m 0.9"
    "-e 50 -b 256 -l 0.2 -m 0.9"
    # Bloco E: sem momento
    "-e 50 -b 256 -l 0.1"
)

for cfg in "${CONFIGS[@]}"; do
    for seed in $SEEDS; do
        extra=""
        if [ "$seed" != "$PRIMEIRA_SEMENTE" ]; then
            extra="--sem-graficos"
        fi
        echo "" | tee -a "$ARQ_LOG"
        echo ">>> python mlp_v5_treinar.py $cfg --seed $seed $extra" | tee -a "$ARQ_LOG"
        # shellcheck disable=SC2086
        python mlp_v5_treinar.py $cfg --seed "$seed" $extra 2>&1 | tee -a "$ARQ_LOG"
    done
done

echo "" | tee -a "$ARQ_LOG"
echo "=== SELEÇÃO PELA VALIDAÇÃO ===" | tee -a "$ARQ_LOG"
python mlp_v5_selecionar.py 2>&1 | tee -a "$ARQ_LOG"

echo "" | tee -a "$ARQ_LOG"
echo "=== BATERIA v5 CONCLUÍDA: $(date) ===" | tee -a "$ARQ_LOG"
