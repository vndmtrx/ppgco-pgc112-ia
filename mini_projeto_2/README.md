# Mini-Projeto 2: Visão Computacional com Redes Neurais (MLP vs. CNN no MNIST)

**Disciplina:** Inteligência Artificial (PGC112 / PPGCO — UFU)  
**Professor:** Gilberto Medeiros Nakamura  
**Repositório Git:** [https://github.com/vndmtrx/ppgco-pgc112-ia](https://github.com/vndmtrx/ppgco-pgc112-ia)  

---

## 1. Visão Geral do Projeto

Este mini-projeto investiga a classificação de dígitos manuscritos sobre o clássico conjunto **MNIST** ($70.000$ imagens monocromáticas $28 \times 28$, divididas em 10 classes), contrastando duas abordagens fundamentais de aprendizado profundo:

1. **Multilayer Perceptron (MLP) com Softmax:** Rede densa alimentada pelo vetor achatado ($28 \times 28 \to 784$ pixels), ativações ReLU e otimizador SGD.
2. **Rede Neural Convolucional (CNN estilo LeNet):** Arquitetura com convoluções 2D, pooling e compartilhamento de pesos, preservando a topologia espacial da imagem.

Todo o desenvolvimento segue rigorosamente os livros-texto da disciplina: **Russell & Norvig (4ª Ed., Cap. 22)** e **Smola et al. (*Dive into Deep Learning*, Caps. 4, 5 e 7)**. A jornada completa de entendimento teórico e decisões de engenharia está documentada em [**METODOLOGIA.md**](./METODOLOGIA.md).

---

## 2. Guia de Execução Rápida

### 2.1. Ativação do Ambiente Virtual (`.venv`)

O projeto utiliza um ambiente virtual isolado com Python 3.10+ e as dependências essenciais (`torch`, `torchvision`, `matplotlib`, `numpy`, `pytest`):

```bash
# 1. Navegar até a pasta do mini-projeto
cd ppgco-pgc112-ia/mini_projeto_2

# 2. Ativar o ambiente virtual isolado
source .venv/bin/activate
```

*(Caso precise recriar o ambiente do zero: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`)*.

---

### 2.2. Comandos de Execução dos Testes e Scripts

Com o ambiente virtual ativado:

```bash
# 1. Executar a suíte completa de testes automatizados
pytest -v

# 2. Executar o pipeline de dados MNIST (baixa o dataset, calcula média/desvio e gera gráficos exploratórios)
python dataset.py
```

---

### 2.3. Execução Alternativa via Docker

Caso prefira executar sem o ambiente virtual local, utilize a imagem oficial Jupyter/PyTorch já configurada:

```bash
# Executar a suíte de testes via contêiner
docker run --rm \
  -v $HOME/du/dev/pytorch/ppgco-pgc112-ia/mini_projeto_2:/home/jovyan/work \
  -w /home/jovyan/work \
  quay.io/jupyter/pytorch-notebook:latest \
  python3 -m unittest discover -s tests -v

# Executar o pipeline de dados
docker run --rm \
  -v $HOME/du/dev/pytorch/ppgco-pgc112-ia/mini_projeto_2:/home/jovyan/work \
  -w /home/jovyan/work \
  quay.io/jupyter/pytorch-notebook:latest \
  python3 dataset.py
```

### 2.4. Linha v5 (melhor modelo pela validação)

```bash
# Treina e salva o checkpoint de menor perda de validação (usa --seed; sem o teste)
python mlp_v5_treinar.py -e 100 -b 256 -l 0.1 -m 0.9 --hidden-dims 300 100 --seed 1

# Bateria completa: 15 configurações x sementes (SEEDS="1 2 3 4 5" para mais sementes)
./executar_bateria_v5.sh

# Escolhe a configuração pela validação (média e desvio entre sementes)
python mlp_v5_selecionar.py

# Abre o teste UMA vez, no checkpoint escolhido (registra em resultados/aberturas_teste.csv)
python mlp_v5_testar.py -c <nome>_melhor.pth
```

### 2.5. Linha v6 (MSE ou Entropia Cruzada no padrão do v5)

```bash
# Treina com MSE (v3) ou entropia cruzada (v4), com semente e melhor checkpoint pela validação
python mlp_v6_treinar.py -e 50 --perda mse --seed 1
python mlp_v6_treinar.py -e 50 --perda ce  --seed 1

# Bateria que refaz a comparação MSE x CE (50 e 200 épocas com 3 sementes; 500 épocas com 1)
./executar_bateria_v6.sh

# Compara MSE e CE pela validação (não abre o teste)
python mlp_v6_selecionar.py

# Abre o teste num checkpoint escolhido (a perda vem do nome do arquivo)
python mlp_v6_testar.py -c <nome>_melhor.pth
```

### 2.6. Linha CNN v1 (LeNet no padrão do v5)

```bash
# Treina a LeNet com SGD e salva o melhor checkpoint pela perda de validação
python cnn_v1_treinar.py -e 30 -b 256 -l 0.05 -m 0.9 --seed 1

# Bateria completa: 4 configurações x 3 sementes (só treino e validação)
./executar_bateria_cnn.sh

# Escolhe a configuração pela validação (média ± desvio entre sementes)
python cnn_v1_selecionar.py

# Abre o teste UMA vez no checkpoint eleito (registra em resultados/aberturas_teste.csv)
python cnn_v1_testar.py -c <checkpoint_cnn_melhor.pth>

# Confronto direto MLP Campeão (v5) vs. CNN LeNet (cnn_v1)
python confronto_mlp_vs_cnn.py --cnn-checkpoint <checkpoint_cnn_melhor.pth>
```

---

## 3. Roteiro e Status das Etapas do Projeto

O desenvolvimento está estruturado nas 5 etapas planejadas a partir do enunciado oficial:

- [x] **Etapa 1: Pipeline de Dados & Exploração Visual (Seção I do Enunciado)**
  - [x] Download automático do MNIST via `torchvision.datasets.MNIST`.
  - [x] Cálculo empírico da média ($\mu = 0.1307$) e desvio padrão ($\sigma = 0.3081$) dos pixels.
  - [x] Partição determinística reproduzível: $50.000$ treino, $10.000$ validação e $10.000$ teste ($70.000$ total).
  - [x] Geração de gráficos: distribuição de classes ([`figuras/distribuicao_classes.png`](./figuras/distribuicao_classes.png)) e exemplos reais dos dígitos ([`figuras/exemplos_digitos.png`](./figuras/exemplos_digitos.png)).
  - [x] Suíte de 5 testes automatizados aprovada com 100% de sucesso.

- [x] **Etapa 2: Módulo de Métricas e Gráficos de Avaliação**
  - [x] Implementação matemática das métricas: Acurácia global, Precisão macro, Recall macro e F1-score macro.
  - [x] Gerador visual de Matriz de Confusão 10x10 com mapa de calor (*heatmap*) e anotações numéricas.
  - [x] Identificação automática dos 5 pares de classes mais frequentemente confundidos (análise de erros).
  - [x] Galeria visual das imagens reais do MNIST onde a rede errou a predição.
  - [x] Testes unitários para validação analítica das métricas em [`tests/test_mlp.py`](./tests/test_mlp.py).

- [x] **Etapa 3: MLP com Softmax & ReLU (Seção II do Enunciado)**
  - [x] Implementar a arquitetura base do MLP configurável em [`models/mlp.py`](./models/mlp.py).
  - [x] Replicação experimental do baseline clássico de LeCun (1998) com MSE e One-Hot em [`mlp_v1_treinar.py`](./mlp_v1_treinar.py).
  - [x] Implementação de persistência em disco em [`mlp_v2_treinar.py`](./mlp_v2_treinar.py).
  - [x] Otimização de pipeline em RAM (`TensorDataset`) e desacoplamento de treino e testes em [`mlp_v3_treinar.py`](./mlp_v3_treinar.py) e [`mlp_v3_testar.py`](./mlp_v3_testar.py).
  - [x] Estudo experimental de sobreajuste (3 a 500 épocas), validação da curva em "U" e justificativa de *Early Stopping*.
  - [x] Benchmark de aceleração em hardware (CPU Intel i7 vs. GPU NVIDIA RTX A1000 6GB), atingindo ganho de $2{,}83\times$ em treino e $\approx 90.000\text{ img/s}$ em inferência.
  - [x] Diagnóstico morfológico de confusões ($4 \leftrightarrow 9$, $7 \leftrightarrow 2$, $5 \leftrightarrow 3$, $3 \leftrightarrow 8$) e comprovação do aumento do erro $4 \to 9$ em sobreajuste.
  - [x] Implementação e confronto controlado (*ceteris paribus*) da variante com camada Softmax nativa e perda Entropia Cruzada em [`mlp_v4_treinar.py`](./mlp_v4_treinar.py) e [`mlp_v4_testar.py`](./mlp_v4_testar.py), com $98{,}30\%$ de acurácia no teste a 50 épocas (170 erros; uma execução, sem semente fixa).
  - [x] Estudo comparativo de hiperparâmetros (15 configurações: largura, profundidade, lote, momento, Adam/AdamW), com `mlp_v4_1_treinar.py`, `mlp_v4_2_treinar.py` e `executar_bateria.sh`; resultado na seção 4.14 da metodologia (uma semente, escolha feita sobre o teste).
  - [x] v5: melhor modelo pela validação, `--seed`, 3 sementes e teste aberto uma vez (`mlp_v5_*.py`, `executar_bateria_v5.sh`). Bateria de 15 configurações x 3 sementes na CPU; escolhida a camada única de 512 neurônios ($784 \to 512 \to 10$, 407.050 parâmetros), com **98,19% no teste (181 erros)**, F1 macro 98,18%.

- [x] **Etapa 4: CNN estilo LeNet (Seção III do Enunciado)**
  - [x] Implementar a arquitetura convolucional clássica em [`models/cnn.py`](./models/cnn.py) ($44.426$ parâmetros) e suíte TDD em [`tests/test_cnn.py`](./tests/test_cnn.py).
  - [x] Bateria ética com SGD e 3 sementes ([`executar_bateria_cnn.sh`](./executar_bateria_cnn.sh)), elegendo `b=256, lr=0.02, m=0.9` pela validação ($98{,}72 \pm 0{,}02\%$).
  - [x] Avaliação única e auditada no teste ([`cnn_v1_testar.py`](./cnn_v1_testar.py)): **98,98% no teste (102 erros)**, F1 macro **98,97%**, perda **0,03575**.

- [x] **Etapa 5: Comparação Sistemática (MLP vs. CNN) e Fechamento**
  - [x] Análise comparativa automatizada em [`confronto_mlp_vs_cnn.py`](./confronto_mlp_vs_cnn.py): redução de $89{,}1\%$ em parâmetros ($44.426$ vs. $407.050$), redução de $43{,}6\%$ nos erros ($102$ vs. $181$), $118$ correções sobre o MLP e painéis morfológicos em `figuras/confronto_mlp_vs_cnn/`.
  - [x] Consolidação do [**METODOLOGIA.md**](./METODOLOGIA.md) (Seções 4.18 a 4.21 e Log de Atividades).

---

## 4. Estrutura de Arquivos e Diretórios

```text
mini_projeto_2/
├── 03-projeto-visao-simples.pdf   # Especificação oficial do mini-projeto
├── README.md                      # Guia rápido de execução e status do projeto
├── METODOLOGIA.md                 # Diário metodológico e fundamentação teórica
├── requirements.txt               # Dependências do projeto (PyTorch, Torchvision, etc.)
├── pytest.ini                     # Configuração de caminhos para a suíte de testes
├── estudo_dirigido.html           # Estudo dirigido interativo: rede, matrizes e gráfico da perda editáveis, com treino e Dying ReLU
├── estudo_dirigido_motor.js       # Motor numérico da página (ida, volta com ReLU', SGD); abra o HTML na mesma pasta
│
├── dataset.py                     # Pipeline de dados: download, split 50k/10k/10k e normalização
├── mlp_v1_treinar.py              # Baseline experimental de LeCun (1998) com MSE e One-Hot
├── mlp_v2_treinar.py              # Replicação com persistência em disco (.pth)
├── mlp_v3_treinar.py              # Treinamento otimizado em RAM (TensorDataset) com épocas via CLI (-e)
├── mlp_v3_testar.py               # Avaliação independente do modelo salvo contra o teste oficial (10k)
├── mlp_v4_treinar.py              # Treinamento com Softmax e Cross-Entropy sem One-Hot (rótulos inteiros)
├── mlp_v4_testar.py               # Avaliação independente do modelo v4 contra o teste oficial (10k)
├── mlp_v4_1_treinar.py            # v4 com acumulação no dispositivo (sem sincronização por lote)
├── mlp_v4_2_treinar.py            # v4 parametrizável (lote, lr, momento, otimizador, camadas)
├── executar_bateria.sh            # Bateria v4.2 (15 configurações; treina e testa cada uma)
├── mlp_v5_treinar.py              # v5: salva o checkpoint de menor perda de validação; --seed; resumo em CSV
├── mlp_v5_testar.py               # v5: avalia no teste e registra cada abertura do teste
├── mlp_v5_selecionar.py           # v5: escolhe a configuração pela validação (média ± desvio entre sementes)
├── executar_bateria_v5.sh         # v5: 15 configurações x sementes, só treino e validação
├── mlp_v6_treinar.py              # v6: v5 com --perda ce|mse (reúne v3 e v4 no padrão do v5)
├── mlp_v6_testar.py               # v6: avalia no teste (perda lida do nome) e registra a abertura
├── mlp_v6_selecionar.py           # v6: escolhe pela validação e compara MSE x CE
├── executar_bateria_v6.sh         # v6: refaz a comparação MSE x CE (só treino e validação)
├── cnn_v1_treinar.py              # CNN v1 (LeNet): salva melhor checkpoint pela validação com --seed
├── cnn_v1_testar.py               # CNN v1: avalia no teste e registra em aberturas_teste.csv
├── cnn_v1_selecionar.py           # CNN v1: escolhe configuração pela validação entre sementes
├── executar_bateria_cnn.sh        # Bateria CNN v1: 4 configurações x 3 sementes (só treino e validação)
├── confronto_mlp_vs_cnn.py        # Confronto direto e morfológico MLP Campeão (v5) vs. LeNet (cnn_v1)
├── graficos_mlp.py                # Gráficos de treino num módulo só (usado por v6 e cnn_v1)
├── models/
│   ├── mlp.py                     # Arquitetura MLP totalmente conectada com ReLU
│   └── cnn.py                     # Arquitetura convolucional LeNet (Conv 5x5 + MaxPool + Linear)
├── data/                          # Binários baixados do MNIST (ignorado no git)
├── modelo/                        # Checkpoints (.pth), ainda não versionados
│   ├── mlp_treinado_v4_2_*.pth    # 15 modelos da bateria v4.2 (última época)
│   └── mlp_treinado_v5_*_melhor.pth # v5: época de menor perda de validação
├── resultados/                    # v5: resumo_v5.csv, <id>.json por execução, aberturas_teste.csv
├── figuras/                       # Relatórios visuais em alta resolução (300 DPI)
│   ├── distribuicao_classes.png   # Histograma de balanceamento de classes
│   ├── exemplos_digitos.png       # Grade de amostras visuais dos dígitos
│   └── <id do experimento>/       # Uma subpasta por execução (ex.: v4_2_e50_b256_lr0.1_h128/, v5_e50_b256_lr0.1_s1/)
│       ├── mlp_treinado_v[3|4]_e*_perda.png            # Curva de perda (Treino vs Validação)
│       ├── mlp_treinado_v[3|4]_e*_acuracia.png         # Curva de acurácia % (Treino vs Validação)
│       ├── mlp_treinado_v[3|4]_e*_gap_generalizacao.png # Monitor de overfitting (Gap Treino - Val)
│       ├── mlp_treinado_v[3|4]_e*_tempo_execucao.png   # Estabilidade temporal (segundos/época e acumulado)
│       ├── mlp_treinado_v[3|4]_e*_painel_geral.png     # Painel integrado de treino 2x2
│       ├── mlp_treinado_v[3|4]_e*_matriz_confusao.png  # Matriz de confusão 10x10 com heatmap
│       ├── mlp_treinado_v[3|4]_e*_metricas_classes.png # Precisão, Recall e F1 por dígito vs. Macro-F1
│       ├── mlp_treinado_v[3|4]_e*_amostras_erros.png   # Grade de amostras reais classificadas com erro
│       └── mlp_treinado_v[3|4]_e*_painel_teste.png     # Dashboard integrado de avaliação de teste
│       └── mlp_treinado_v3_e*_painel_teste.png     # Dashboard consolidado de teste oficial
└── tests/
    ├── test_dataset.py            # Testes automatizados do pipeline de dados
    ├── test_mlp.py                # Testes automatizados da arquitetura MLP e persistência
    ├── test_cnn.py                # Testes da LeNet: shapes, parâmetros 3D, logits, treino/confronto sintético
    ├── test_mlp_v6.py             # Testes do v6: perda MSE, ids, comparação MSE x CE e teste sintético
    ├── estudo_dirigido_motor.test.js   # Testes do motor da página (node; gradiente, Dying ReLU, tanh e logística)
    ├── estudo_dirigido_pagina.test.js  # Testes da página numa DOM simulada (precisa do jsdom; pula sem ele)
    ├── estudo_dirigido_medidas.js      # Medidas da seção 4.6.1 da metodologia (node; semente fixa)
    └── test_mlp_v5.py             # Testes do v5: melhor checkpoint, semente fixa e escolha pela validação
```
