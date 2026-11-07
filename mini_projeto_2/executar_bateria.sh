#!/usr/bin/env bash
# ==============================================================================
# BATERIA DE EXPERIMENTOS DE HIPERPARÂMETROS (MINI-PROJETO 2 - SEÇÃO II)
# Executa treinamento e avaliação oficial no teste para todas as configurações,
# salvando logs, curvas e matrizes de confusão.
# ==============================================================================

set -e

ARQ_LOG="bateria_experimentos.log"
echo "=== INÍCIO DA BATERIA DE EXPERIMENTOS: $(date) ===" | tee "$ARQ_LOG"

rodar_exp() {
    local cmd_treino="$1"
    local checkpoint="$2"
    
    echo "" | tee -a "$ARQ_LOG"
    echo "======================================================================" | tee -a "$ARQ_LOG"
    echo "EXECUTANDO: $cmd_treino" | tee -a "$ARQ_LOG"
    echo "======================================================================" | tee -a "$ARQ_LOG"
    
    python $cmd_treino 2>&1 | tee -a "$ARQ_LOG"
    
    echo "" | tee -a "$ARQ_LOG"
    echo "--- AVALIAÇÃO NO TESTE: $checkpoint ---" | tee -a "$ARQ_LOG"
    python mlp_v4_testar.py -c "$checkpoint" 2>&1 | tee -a "$ARQ_LOG"
}

# ------------------------------------------------------------------------------
# BLOCO 1: Exploração de Arquitetura (Profundidade e Largura) - Seção II.B
# ------------------------------------------------------------------------------
echo ">>> BLOCO 1: Variação de Arquitetura (Neurônios e Camadas Ocultas)" | tee -a "$ARQ_LOG"

# 1.1 Estreita (128 neurônios)
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 256 -l 0.1 --hidden-dims 128" "mlp_treinado_v4_2_e50_b256_lr0.1_h128.pth"

# 1.2 Baseline LeCun (300 neurônios)
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 256 -l 0.1" "mlp_treinado_v4_2_e50_b256_lr0.1.pth"

# 1.3 Larga (512 neurônios)
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 256 -l 0.1 --hidden-dims 512" "mlp_treinado_v4_2_e50_b256_lr0.1_h512.pth"

# 1.4 Duas Camadas Ocultas (300 -> 100)
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 256 -l 0.1 --hidden-dims 300 100" "mlp_treinado_v4_2_e50_b256_lr0.1_h300-100.pth"

# 1.5 Duas Camadas Ocultas Largas (512 -> 256)
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 256 -l 0.1 --hidden-dims 512 256" "mlp_treinado_v4_2_e50_b256_lr0.1_h512-256.pth"

# 1.6 Três Camadas Ocultas Profunda (512 -> 256 -> 128)
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 256 -l 0.1 --hidden-dims 512 256 128" "mlp_treinado_v4_2_e50_b256_lr0.1_h512-256-128.pth"

# ------------------------------------------------------------------------------
# BLOCO 2: Exploração de Batch Size e Dinâmica SGD (Momento e Taxa de Aprendizado)
# ------------------------------------------------------------------------------
echo ">>> BLOCO 2: Batch Size e Dinâmica SGD" | tee -a "$ARQ_LOG"

# 2.1 Batch 64 (Original LeCun)
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 64 -l 0.1" "mlp_treinado_v4_2_e50_b64_lr0.1.pth"

# 2.2 Batch 128
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 128 -l 0.1" "mlp_treinado_v4_2_e50_b128_lr0.1.pth"

# 2.3 Batch 256 com Momentum 0.9
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 256 -l 0.1 -m 0.9" "mlp_treinado_v4_2_e50_b256_lr0.1_m0.9.pth"

# 2.4 Batch 512 com LR Escalonada e Momentum 0.9
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 512 -l 0.2 -m 0.9" "mlp_treinado_v4_2_e50_b512_lr0.2_m0.9.pth"

# ------------------------------------------------------------------------------
# BLOCO 3: Otimizadores Adaptativos Modernos (Adam e AdamW)
# ------------------------------------------------------------------------------
echo ">>> BLOCO 3: Otimizadores Adaptativos (Adam e AdamW)" | tee -a "$ARQ_LOG"

# 3.1 Adam (LR 0.001)
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 256 -l 0.001 --otimizador adam" "mlp_treinado_v4_2_e50_b256_lr0.001_adam.pth"

# 3.2 AdamW (LR 0.001)
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 256 -l 0.001 --otimizador adamw" "mlp_treinado_v4_2_e50_b256_lr0.001_adamw.pth"

# 3.3 AdamW na Rede Profunda (512 -> 256)
rodar_exp "mlp_v4_2_treinar.py -e 50 -b 256 -l 0.001 --otimizador adamw --hidden-dims 512 256" "mlp_treinado_v4_2_e50_b256_lr0.001_adamw_h512-256.pth"

# ------------------------------------------------------------------------------
# BLOCO 4: Treinamento Estendido (100 Épocas) para as Melhores Configurações
# ------------------------------------------------------------------------------
echo ">>> BLOCO 4: Treinamento Estendido (100 Épocas)" | tee -a "$ARQ_LOG"

# 4.1 1 Camada com Momentum 0.9 (100 épocas)
rodar_exp "mlp_v4_2_treinar.py -e 100 -b 256 -l 0.1 -m 0.9" "mlp_treinado_v4_2_e100_b256_lr0.1_m0.9.pth"

# 4.2 2 Camadas (300 -> 100) com Momentum 0.9 (100 épocas)
rodar_exp "mlp_v4_2_treinar.py -e 100 -b 256 -l 0.1 -m 0.9 --hidden-dims 300 100" "mlp_treinado_v4_2_e100_b256_lr0.1_m0.9_h300-100.pth"

echo "" | tee -a "$ARQ_LOG"
echo "=== BATERIA CONCLUÍDA COM SUCESSO: $(date) ===" | tee -a "$ARQ_LOG"
