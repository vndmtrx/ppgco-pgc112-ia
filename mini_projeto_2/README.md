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

---

## 3. Roteiro e Status das Etapas do Projeto

O desenvolvimento está estruturado nas 5 etapas planejadas a partir do enunciado oficial:

- [x] **Etapa 1: Pipeline de Dados & Exploração Visual (Seção I do Enunciado)**
  - [x] Download automático do MNIST via `torchvision.datasets.MNIST`.
  - [x] Cálculo empírico da média ($\mu = 0.1307$) e desvio padrão ($\sigma = 0.3081$) dos pixels.
  - [x] Partição determinística reproduzível: $50.000$ treino, $10.000$ validação e $10.000$ teste ($70.000$ total).
  - [x] Geração de gráficos: distribuição de classes ([`figuras/distribuicao_classes.png`](./figuras/distribuicao_classes.png)) e exemplos reais dos dígitos ([`figuras/exemplos_digitos.png`](./figuras/exemplos_digitos.png)).
  - [x] Suíte de 5 testes automatizados aprovada com 100% de sucesso.

- [ ] **Etapa 2: Módulo de Métricas e Gráficos de Avaliação**
  - [ ] Implementar [`metrics.py`](./metrics.py): Acurácia global, Precision macro, Recall macro e F1-score macro.
  - [ ] Implementar gerador visual de Matriz de Confusão com mapa de calor (heatmap).
  - [ ] Testes unitários para validação matemática das métricas.

- [ ] **Etapa 3: MLP com Softmax & ReLU (Seção II do Enunciado)**
  - [x] Implementar a arquitetura base do MLP configurável em [`models/mlp.py`](./models/mlp.py).
  - [x] Replicação experimental do baseline clássico de LeCun (1998) com MSE e One-Hot em [`treinar_mlp_v1.py`](./treinar_mlp_v1.py).
  - [ ] Implementar variante com camada de saída Softmax e perda Entropia Cruzada (*Cross-Entropy*).
  - [ ] Estudo comparativo de hiperparâmetros (profundidade e neurônios por camada).
  - [ ] Curvas de perda (treino vs. validação) por época e matriz de confusão no teste.
  - [ ] Diagnóstico qualitativo dos dígitos mais confundidos entre si (ex.: $4 \leftrightarrow 9$, $3 \leftrightarrow 5$).

- [ ] **Etapa 4: CNN estilo LeNet (Seção III do Enunciado)**
  - [ ] Implementar a arquitetura convolucional clássica em [`models/cnn.py`](./models/cnn.py).
  - [ ] Treinamento com SGD e monitoramento das curvas de perda.
  - [ ] Avaliação completa das métricas e matriz de confusão no teste.

- [ ] **Etapa 5: Comparação Sistemática (MLP vs. CNN) e Fechamento**
  - [ ] Análise comparativa: Desempenho (acurácia/F1) $\times$ Número de parâmetros $\times$ Tempo de treino.
  - [ ] Consolidação do [**METODOLOGIA.md**](./METODOLOGIA.md).

---

## 4. Estrutura de Arquivos e Diretórios

```text
mini_projeto_2/
├── 03-projeto-visao-simples.pdf   # Especificação oficial do mini-projeto
├── README.md                      # Guia rápido de execução e status do projeto
├── METODOLOGIA.md                 # Diário metodológico e fundamentação teórica
├── requirements.txt               # Dependências do projeto (PyTorch, Torchvision, etc.)
├── pytest.ini                     # Configuração de caminhos para a suíte de testes
├── estudo_dirigido.html           # Estudo dirigido para compreensão conceitual do modelo neural (álgebra e backprop)
│
├── dataset.py                     # Pipeline de dados: download, split 50k/10k/10k e normalização
├── treinar_mlp_v1.py              # Script experimental: baseline de LeCun (1998) com MSE e One-Hot
├── models/
│   └── mlp.py                     # Arquitetura MLP totalmente conectada com ReLU
├── data/                          # Binários baixados do MNIST (ignorado no git)
├── figuras/                       # Gráficos em alta resolução (300 DPI) gerados
│   ├── distribuicao_classes.png   # Histograma de balanceamento de classes
│   └── exemplos_digitos.png       # Grade de amostras visuais dos dígitos
└── tests/
    └── test_dataset.py            # Testes automatizados do pipeline de dados
```
