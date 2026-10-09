# Parte 1: Buscas Clássicas em Grafos

Este módulo implementa e compara algoritmos clássicos de busca em espaço de estados (Busca em Largura — BFS e Busca Gulosa — Greedy) para a transformação de sequências binárias discretas.

---

## 1. Ambiente Virtual (`.venv`) e Instalação

Para executar este módulo isoladamente ou a partir da raiz:

```bash
# Na pasta parte_1_buscas_classicas/
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dependências principais:
- `torch>=2.0.0` (Implementações matriciais e paralelizáveis em tensor)
- `matplotlib>=3.7.0` (Gráficos de escalabilidade de $L=2$ a $L=4096$)
- `pytest>=7.4.0` (Suíte com 69 testes unitários)

---

## 2. Comandos dos Testes Automatizados (`pytest`)

A suíte possui **69 testes unitários** automatizados aprovados:

```bash
# Rodar todos os testes da Parte 1
pytest tests/ -v
```

Módulos testados:
- `tests/test_flips.py`: Operações de bit-flip individual e de blocos contíguos.
- `tests/test_bfs.py`: Solução ótima do BFS (5 passos exatos na fita de 23 bits).
- `tests/test_heuristica.py`: Admissibilidade e monotonicidade da heurística de blocos $h_2$ vs. Hamming $h_1$.
- `tests/test_guloso.py`: Busca gulosa direcionada.
- `tests/test_guloso_poda.py`: Eficácia da poda de ciclos para prevenir loops infinitos.
- `tests/test_utils.py`: Funções utilitárias de manipulação e métricas.

---

## 3. Comandos para Executar os Experimentos e Benchmarks

### 3.1. Execução do BFS na Fita Canônica de 23 Bits
Retorna a solução de custo mínimo garantido (5 passos):
```bash
python3 -c "
from bfs_sequencial import bfs_sequencial
res = bfs_sequencial('00001001000011001100111', '00000000000000000000000')
print('=== BFS Exato ===')
print(f'Custo Total: {res[\"custo_total\"]}')
print('Passos:')
for i, estado in enumerate(res[\"caminho\"]):
    print(f'  {i}: {estado}')
"
```

### 3.2. Execução do Algoritmo Guloso com Poda de Ciclos (Heurística de Blocos $h_2$)
Executa a busca informada convergindo em menos de 0,2 milissegundos:
```bash
python3 -c "
from guloso_poda_sequencial import guloso_poda_sequencial
from heuristica import h2_blocos
res = guloso_poda_sequencial('00001001000011001100111', '00000000000000000000000', h2_blocos)
print('=== Algoritmo Guloso com Poda ===')
print(f'Custo: {res[\"custo_total\"]} | Nós Expandidos: {res[\"nos_expandidos\"]}')
for i, (op, est) in enumerate(zip(res[\"operacoes\"], res[\"caminho\"][1:]), start=1):
    print(f'  Passo {i}: {op} -> {est}')
"
```

### 3.3. Execução do Benchmark e Geração dos Gráficos de Escalabilidade ($L=2$ a $L=4096$)
```bash
# Executa a bateria de medição de tempo e nós explorados
python3 bench_guloso_poda.py

# Plota os gráficos comparativos de desempenho
python3 plota_graficos_guloso_poda.py
```

---

## 4. Estrutura de Arquivos

```text
parte_1_buscas_classicas/
├── README.md                      # Este guia
├── METODOLOGIA.md                 # Registro de decisões e fundamentação teórica
├── requirements.txt               # Dependências Python
├── pytest.ini                     # Configuração do pytest
├── flips.py                       # Subrotinas de bit-flip
├── heuristica.py                  # Heurísticas h1 (Hamming) e h2 (Blocos contíguos)
├── bfs_sequencial.py              # Algoritmo BFS exato
├── guloso_sequencial.py           # Algoritmo Guloso clássico
├── guloso_poda_sequencial.py      # Algoritmo Guloso com poda de ciclos
├── bench_guloso_poda.py           # Benchmark de escalabilidade
├── plota_graficos_guloso_poda.py  # Geração de gráficos de desempenho
├── utils.py                       # Conversões de bits e validações
└── tests/                         # 69 testes automatizados
```
