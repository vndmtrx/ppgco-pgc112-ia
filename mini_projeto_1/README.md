# Mini-projeto: Buscas Clássicas e Otimização

**Data:** 25 de Agosto de 2026  
**Disciplina:** Inteligência Artificial (PPGCO / UFU)  

---

## I. Introdução

Este mini-projeto é dividido em duas partes:
1. **Parte 1 (Buscas Clássicas):** Aborda a implementação de algoritmos de busca clássicos (BFS e Greedy) para resolver o problema de transformação de sequências de bits.
2. **Parte 2 (Otimização com Gradiente Descendente):** Explora o problema sob a perspectiva de otimização contínua, utilizando o método do Gradiente Descendente.

---

## Estrutura dos Exercícios

- [**Parte 1: Buscas Clássicas**](./parte_1_buscas_classicas/README.md)
  - Subrotinas de flips de bits (individual e consecutivos).
  - Implementação e execução com BFS (menor custo).
  - Heurística e implementação com Algoritmo Guloso (Greedy).
  - Análise comparativa de desempenho para $L = 2, 4, \dots, 4096$.

- [**Parte 2: Otimização com Gradiente Descendente**](./parte_2_gradiente_descendente/README.md)
  - Mapeamento discreto para contínuo inspirado na lógica fuzzy ($x_k \in [0, 1]$).
  - Formulação e minimização da função erro quadrática $\mathcal{L}(x)$.
  - Implementação do Gradiente Descendente e análise de estabilidade ($\eta > 1/2$).
  - Inclusão da função de custo $C(z)$ para penalizar transformações não-consecutivas.

---

## Informações de Entrega

- O projeto pode ser entregue como um relatório, preferencialmente escrito em LaTeX para uniformizar tabelas, equações e figuras.
- **Período de entrega:** A partir de 14/09/2026 e até 2 semanas antes do encerramento das aulas.
- **Número ideal de páginas:** 3 páginas.
- **Repositório:** Incluir link para o repositório git para clonagem e verificação das implementações.
