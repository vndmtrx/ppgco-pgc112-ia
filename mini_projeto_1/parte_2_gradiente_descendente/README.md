# Parte 2: Otimização Contínua com Gradiente Descendente

Este módulo implementa a transformação contínua de sequências binárias, realizando o relaxamento *fuzzy* da fita de 23 bits para o espaço euclidiano $[0, 1]^{23}$ e aplicando o método do Gradiente Descendente puro e regularizado com custo de vizinhança contígua.

---

## 1. Ambiente Virtual (`.venv`) e Instalação

Para executar este módulo isoladamente ou a partir da raiz:

```bash
# Na pasta parte_2_gradiente_descendente/
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

As dependências principais são:
- `torch>=2.0.0` (Diferenciação automática com `autograd`)
- `matplotlib>=3.7.0` (Visualizações científicas em 300 DPI)
- `pytest>=7.4.0` (Suíte de testes automatizados)
- `numpy>=1.24.0` (Computação numérica matricial)

---

## 2. Comandos dos Testes Automatizados (`pytest`)

A suíte possui **11 testes unitários** automatizados aprovados:

```bash
# Rodar todos os testes com saída detalhada
pytest tests/ -v
```

Os testes cobrem:
1. Comparação elemento a elemento entre o gradiente analítico ($2x$) e o `torch.autograd`.
2. Verificação do passo crítico unitário ($\eta = 0.5$ atingindo a origem em 1 iteração).
3. Detecção e caracterização de *overshoot* e valores negativos para $\eta = 0.6$.
4. Dinâmica de blocos contíguos na função regularizada: comprovação de que o **interior de um bloco** ($1, 1, 1$) tem penalidade de regularização nula e gradiente efetivo $2.0$, enquanto as **bordas** sofrem força $4.0$.
5. Detecção de divergência numérica para valores de $\eta$ acima da cota crítica de estabilidade sob altos valores de $\lambda$.

---

## 3. Comandos para Executar os Experimentos e Gerar Gráficos

### 3.1. Execução do Gradiente Descendente Puro
Demonstração no terminal dos diferentes regimes de taxa de aprendizado $\eta$:
```bash
# Executa os regimes eta in {0.1, 0.5, 0.6} na fita do enunciado
python3 -c "
from gd_puro import gradiente_descendente_puro
fita = '00001001000011001100111'
for eta in [0.1, 0.5, 0.6]:
    res = gradiente_descendente_puro(fita, eta=eta, max_passos=30)
    print(f'eta={eta:.1f} | Passos={res[\"passos_executados\"]:2d} | Overshoot={res[\"ocorreu_overshoot\"]} | Perda Final={res[\"perdas\"][-1]:.6f}')
"
```

### 3.2. Execução do Gradiente Descendente Regularizado
Demonstração do acoplamento elástico $\lambda \in \{0.0, 0.5, 20.0\}$:
```bash
# Compara convergência estável (lambda=0.0 e 0.5) contra a divergência numérica (lambda=20.0)
python3 -c "
from gd_regularizado import gradiente_descendente_regularizado
fita = '00001001000011001100111'
for lam in [0.0, 0.5, 20.0]:
    res = gradiente_descendente_regularizado(fita, eta=0.1, lam=lam, max_passos=25)
    print(f'lambda={lam:4.1f} | Convergiu={res[\"convergiu\"]} | Divergiu={res[\"divergiu\"]} | Passos={res[\"passos_executados\"]}')
"
```

### 3.3. Gerar Todas as Figuras Científicas (Pasta `graficos/`)
```bash
# 1. Gráficos do GD Puro (Bacia 3D, curvas de nível e heatmaps)
python3 plota_graficos_gd_puro.py

# 2. Gráficos do GD Regularizado (Heatmaps comparativos, perdas e diagrama de fase eta x lambda)
python3 plota_graficos_gd_regularizado.py
```

Os gráficos gerados em alta resolução são:
- `graficos/01_topografia_curvas_nivel_etas_direcoes.png`
- `graficos/02_convergencia_perda_etas.png`
- `graficos/03_heatmaps_evolucao_temporal_23bits.png`
- `graficos/04_heatmaps_comparativo_lambdas_23bits.png`
- `graficos/05_decomposicao_perdas_regularizacao.png`
- `graficos/06_diagrama_fase_estabilidade_eta_lambda.png`

---

## 4. Estrutura de Arquivos

```text
parte_2_gradiente_descendente/
├── README.md                      # Este guia
├── METODOLOGIA.md                 # Registro cognitivo de engenharia e decisões
├── requirements.txt               # Dependências Python
├── pytest.ini                     # Configuração do pytest
├── gd_puro.py                     # Implementação do GD sem regularização
├── gd_regularizado.py             # Implementação do GD com custo de vizinhança
├── plota_graficos_gd_puro.py      # Script de visualização do GD puro
├── plota_graficos_gd_regularizado.py # Script de visualização do GD regularizado
├── graficos/                      # Figuras científicas salvas (300 DPI)
└── tests/                         # Suíte de testes automatizados
    ├── test_gd_puro.py
    └── test_gd_regularizado.py
```
