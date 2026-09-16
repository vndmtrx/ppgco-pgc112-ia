# Parte 1: Buscas Clássicas

Este exercício aborda a implementação e comparação de algoritmos clássicos de busca em grafos (Busca em Largura - BFS e Algoritmo Guloso - Greedy) para solucionar o problema de transformação de sequências de bits.

---

## Enunciado

### A. Descrição do Problema

Dada uma sequência inicial de $L$ bits, procuramos a sequência de transformações do tipo *bit-flip* que a transforma em uma sequência final. O custo de cada operação é definido como:

1. Flip de 1 bit tem **custo 1**;
2. Flip de $n$ bits consecutivos tem **custo 1**.

**Instância de teste:**
- Sequência inicial: `00001001000011001100111`
- Sequência desejada: `00000000000000000000000`

---

### B. Subrotinas para flips

Implementar funções que realizem as operações de *bit-flip* descritas acima:
- Flip de um único bit na posição especificada.
- Flip de um bloco de $n$ bits consecutivos a partir de um índice inicial.

---

### C. Algoritmo BFS

Implementar o algoritmo **BFS (Breadth-First Search)** para encontrar a sequência de operações de menor custo que transforma a sequência inicial na final.

**Requisitos da resposta:**
- Retornar a sequência exata de operações realizadas.
- Retornar o custo total acumulado.

---

### D. Algoritmo Guloso (Greedy)

Defina uma heurística que contenha alguma informação sobre o estado alvo.

**Requisitos:**
- Justificar formalmente a escolha da função heurística.
- Implementar o algoritmo guloso utilizando a heurística definida.
- Retornar a sequência de transformações obtida e o custo total.

---

### E. Desempenho

1. Gere um conjunto de sequências aleatórias de bits.
2. Selecione $N$ pares `(inicial, final)` desse conjunto.
3. Execute os algoritmos BFS e Greedy em cada par.
4. Faça um levantamento das estatísticas de desempenho (e.g., número médio de operações, custo médio, tempo de execução, nós explorados).
5. Gere um gráfico para comparar o desempenho dos algoritmos em função do tamanho da sequência:
   $$L \in \{2, 4, 8, \dots, 4096\}$$
