# Metodologia: Mini-Projeto 3 (Atenção em Séries Temporais)

Este documento define o escopo do projeto a partir do enunciado oficial e o plano de estudos necessário para a sua implementação.

---

## 1. O que o Projeto Pede

O trabalho aborda a arquitetura **Transformer** aplicada à previsão multivariada da série temporal climática **Jena Climate (2009–2016)**, com ênfase na identificação de padrões temporais por meio de atenção.

### 1.1. Dados e Pré-processamento
- **Dataset:** Jena Climate (2009–2016) contendo 14 variáveis meteorológicas registradas a cada 10 minutos.
- **Análise Estatística:** Breve análise estatística exploratória das variáveis.
- **Divisão dos Dados:** Separação estritamente cronológica entre treino e teste.
- **Janelas Deslizantes:** Organização dos dados em janelas de duração $L$, compostas por $m$ tokens (série temporal multivariada com duração $L/m$ fixa).

### 1.2. Objetivos de Previsão
A partir de uma janela de $m$ tokens de entrada:
1. **Objetivo 1:** Prever o próximo token ($t + 1$).
2. **Objetivo 2:** Prever os próximos tokens consecutivos que, em conjunto, representem um horizonte de 24 horas.

### 1.3. Arquitetura e Modelagem
- **Projeção Linear:** Mapear cada token multivariado para um vetor de dimensão $n$.
- **Codificação Posicional:** Somar a informação de posição temporal ao vetor.
- **Blocos Codificadores (Encoder):** Aplicar blocos de codificação Transformer bidirecionais.
- **Camada de Saída:** Camada linear mapeando a representação densa aos valores previstos.
- **Função de Perda:** Definir e justificar a função de perda (ex.: MSE/MAE) e plotar a curva de perda por época.
- **Hiperparâmetros a explorar:** 
  - Tamanho da janela ($m$);
  - Dimensão do vetor latente ($n$);
  - Número de cabeças de atenção ($h$);
  - Número de blocos codificadores ($N$).

### 1.4. Mapas de Atenção e Interpretabilidade
- Extrair os pesos de atenção de uma cabeça para algumas amostras.
- Calcular a média desses pesos e gerar um mapa de calor (*heatmap*).
- Identificar e discutir os padrões temporais e físicos encontrados.

### 1.5. Entrega
- Relatório em LaTeX com aproximadamente 5 páginas.
- Repositório Git com a implementação.

---

## 2. O que Preciso Estudar para Resolver o Trabalho

Para implementar e entender a solução de ponta a ponta, os tópicos de estudo são:

1. **Embeddings e Projeção Linear para Dados Contínuos:**
   - Entender como transformar um vetor de 14 variáveis meteorológicas contínuas em um vetor de dimensão $n$ (projeção linear via camada densa, em contraste com tabelas de vocabulário de NLP).

2. **Codificação Posicional (*Positional Encoding*):**
   - Compreender por que o mecanismo de atenção pura é invariante a permutações (não sabe a ordem dos passos temporais).
   - Estudar como a codificação posicional (senoidal canônica de Vaswani et al. ou vetores aprendidos) injeta a seta do tempo na sequência.

3. **Mecanismo de Atenção (*Self-Attention* e Multi-Head):**
   - Entender o papel de Queries ($Q$), Keys ($K$) e Values ($V$).
   - A matemática da atenção escalonada: $\text{softmax}(QK^\top / \sqrt{d_k}) V$.
   - Por que dividir em múltiplas cabeças (*Multi-Head*) e como elas capturam diferentes relações temporais (ex.: dinâmicas rápidas vs. ciclos diurnos).

4. **O Bloco Codificador (*Transformer Encoder*):**
   - Estrutura de conexões residuais (*skip connections*) e normalização de camada (*LayerNorm*).
   - Rede *Feed-Forward* (FFN) ponto a ponto.
   - Como agregar a sequência ao final (pooling da representação densa) para a camada linear de predição.

5. **Extração dos Pesos de Atenção e Visualização:**
   - Como interceptar a matriz de atenção no PyTorch durante o *forward pass*.
   - Como calcular a média das matrizes e plotar o mapa de calor com Matplotlib/Seaborn para discutir os ciclos físicos da série.
