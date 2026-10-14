# Mini-Projeto 1: Buscas Clássicas e Otimização Contínua em Fitas de Bits

**Disciplina:** Inteligência Artificial (PGC112 / PPGCO — UFU)  
**Professor:** Gilberto Medeiros Nakamura  
**Repositório Git:** [https://github.com/vndmtrx/ppgco-pgc112-ia](https://github.com/vndmtrx/ppgco-pgc112-ia)  

---

## 1. Visão Geral da Estrutura do Projeto

O Mini-Projeto 1 resolve o problema de transformação de uma fita binária canônica de $L=23$ bits (`00001001000011001100111`) rumo ao vetor nulo (`00000000000000000000000`) sob duas abordagens complementares de IA:

- **[Parte 1: Buscas Clássicas em Grafos (`parte_1_buscas_classicas/`)](./parte_1_buscas_classicas/README.md):**
  - Espaço discreto $\{0, 1\}^L$, inversões de bits individuais e de blocos contíguos com custo unitário.
  - Implementação exata via **BFS (Breadth-First Search)** encontrando a solução ótima em 5 operações.
  - **Busca Gulosa (Greedy)** orientada pela Heurística de Blocos Contíguos $h_2$ com **Poda de Ciclos** para evitar laços infinitos.
  - Testes de escalabilidade de $L=2$ até $L=4096$.

- **[Parte 2: Otimização Contínua com Gradiente Descendente (`parte_2_gradiente_descendente/`)](./parte_2_gradiente_descendente/README.md):**
  - Relaxamento fuzzy no hipercubo contínuo $[0, 1]^L$, interpretando a fita como coordenadas de um ponto em 23D atraído pela origem.
  - **Gradiente Descendente Puro:** Análise da taxa $\eta$, convergência ótima em 1 passo para $\eta = 0.5$ e demonstração de *overshoot* com valores negativos para $\eta > 0.5$.
  - **Gradiente Descendente Regularizado:** Inclusão de custo de vizinhança contígua $C(\mathbf{x})$ com peso de regularização $\lambda$. Demonstração de interior de bloco uniforme vs. bordas tracionadas e dedução analítica da cota de estabilidade numérica $\eta \le \frac{1}{2 + 4\lambda}$ contra divergência oscilatória.

- **Artigo / Relatório Final:**
  - Código-fonte LaTeX (ABNT twocolumn): [`main.tex`](./main.tex)
  - Documento compilado (5 páginas): [`main.pdf`](./main.pdf)

---

## 2. Instruções de Ambiente Virtual (`.venv`) e Instalação

Para que o professor ou avaliador possa clonar o repositório e reproduzir 100% dos resultados e testes sem conflito de dependências:

### 2.1. Clonando o Repositório
```bash
git clone git@github.com:vndmtrx/ppgco-pgc112-ia.git
cd ppgco-pgc112-ia/mini_projeto_1
```

### 2.2. Criação e Ativação do Ambiente Virtual
Recomenda-se Python 3.10 ou superior:

```bash
# Criar o ambiente virtual na raiz do mini-projeto
python3 -m venv .venv

# Ativar o ambiente virtual (Linux/macOS)
source .venv/bin/activate

# Atualizar o gerenciador de pacotes pip
pip install --upgrade pip

# Instalar todas as dependências do projeto
pip install -r requirements.txt
```

*(Caso utilize Windows no PowerShell: `.venv\Scripts\Activate.ps1`)*

---

## 3. Guia de Execução Rápida: Comandos de Cada Teste e Exemplo

Com o ambiente virtual ativado (`source .venv/bin/activate`), execute os comandos abaixo diretamente do terminal.

### 3.1. Executando a Suíte Completa de Testes Automatizados (80 Testes no `pytest`)

O projeto conta com **80 testes automatizados** aprovados cobrindo todas as subrotinas:

```bash
# 1. Executar todos os 69 testes da Parte 1 (Buscas Clássicas)
pytest parte_1_buscas_classicas/tests -v

# 2. Executar todos os 11 testes da Parte 2 (Gradiente Descendente)
pytest parte_2_gradiente_descendente/tests -v

# 3. Ou rodar a suíte inteira de ambos os projetos simultaneamente:
pytest parte_1_buscas_classicas/tests parte_2_gradiente_descendente/tests
```

---

### 3.2. Reproduzindo os Experimentos da Parte 1 (Buscas Clássicas)

```bash
# Navegar até a pasta da Parte 1
cd parte_1_buscas_classicas

# 1. Testar o BFS na fita canônica de 23 bits (retorna os 5 passos ótimos)
python3 -c "from bfs_sequencial import bfs_sequencial; res = bfs_sequencial('00001001000011001100111', '00000000000000000000000'); print(f'Custo BFS: {res[\"custo_total\"]} | Passos: {len(res[\"caminho\"])-1}')"

# 2. Testar o Algoritmo Guloso com Poda de Ciclos (Heurística de Blocos h2)
python3 -c "from guloso_poda_sequencial import guloso_poda_sequencial; from heuristica import h2_blocos; res = guloso_poda_sequencial('00001001000011001100111', '00000000000000000000000', h2_blocos); print(f'Custo Guloso: {res[\"custo_total\"]} | Nós explorados: {res[\"nos_expandidos\"]}')"

# 3. Executar o benchmark comparativo e gerar os gráficos de escalabilidade (L=2 até L=4096)
python3 bench_guloso_poda.py
python3 plota_graficos_guloso_poda.py

cd ..
```

---

### 3.3. Reproduzindo os Experimentos da Parte 2 (Gradiente Descendente)

```bash
# Navegar até a pasta da Parte 2
cd parte_2_gradiente_descendente

# 1. Executar demonstração dos regimes de eta no GD Puro (eta=0.1, eta=0.5 [1 passo], eta=0.6 [overshoot])
python3 -c "from gd_puro import gradiente_descendente_puro; [print(f'eta={e}: passos={res[\"passos_executados\"]} overshoot={res[\"ocorreu_overshoot\"]}') for e in [0.1, 0.5, 0.6] for res in [gradiente_descendente_puro('00001001000011001100111', eta=e)]]"

# 2. Executar demonstração da regularização com custo contíguo (lambda=0.0 vs lambda=0.5 vs divergência em lambda=20.0)
python3 -c "from gd_regularizado import gradiente_descendente_regularizado; [print(f'lambda={l}: convergiu={res[\"convergiu\"]} divergiu={res[\"divergiu\"]}') for l in [0.0, 0.5, 20.0] for res in [gradiente_descendente_regularizado('00001001000011001100111', eta=0.1, lam=l)]]"

# 3. Gerar todos os gráficos topográficos, heatmaps e diagramas de fase (salvos em graficos/)
python3 plota_graficos_gd_puro.py
python3 plota_graficos_gd_regularizado.py

cd ..
```

---

## 4. Estrutura de Arquivos

```text
mini_projeto_1/
├── 00-projeto.pdf                     # Especificação oficial do professor
├── main.tex                           # Artigo científico em LaTeX (ABNT twocolumn, 5 páginas)
├── main.pdf                           # Artigo compilado
├── figuras/                           # Figuras utilizadas no artigo
├── requirements.txt                   # Dependências globais do ambiente Python
├── parte_1_buscas_classicas/
│   ├── README.md                      # Detalhamento e guia da Parte 1
│   ├── requirements.txt
│   ├── bfs_sequencial.py              # Algoritmo BFS exato
│   ├── guloso_poda_sequencial.py      # Algoritmo Guloso com poda de ciclos
│   ├── heuristica.py                  # Heurísticas h1 (Hamming) e h2 (Blocos contíguos)
│   ├── bench_guloso_poda.py           # Benchmark de escalabilidade (L=2..4096)
│   ├── plota_graficos_guloso_poda.py  # Geração dos gráficos da Parte 1
│   └── tests/                         # 69 testes unitários (pytest)
└── parte_2_gradiente_descendente/
    ├── README.md                      # Detalhamento e guia da Parte 2
    ├── requirements.txt
    ├── gd_puro.py                     # Gradiente Descendente puro
    ├── gd_regularizado.py             # GD regularizado com função de custo contíguo
    ├── plota_graficos_gd_puro.py      # Gráficos topográficos e heatmaps da Parte 2
    ├── plota_graficos_gd_regularizado.py # Gráficos de decomposição e diagrama de fase
    ├── graficos/                      # Figuras em 300 DPI geradas
    └── tests/                         # 11 testes unitários (pytest)
```
