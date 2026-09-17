# Parte 2: Otimização com Gradiente Descendente

Este exercício explora a transformação de sequências de bits sob a ótica de otimização contínua, mapeando o problema discreto para um espaço contínuo e aplicando o método do Gradiente Descendente.

---

## Enunciado

### A. Descrição do Problema

Problemas de otimização geralmente assumem que o espaço de soluções forma um plano contínuo por pedaços. No presente problema, sequências de bits são elementos intrinsecamente discretos e por isso exigem algum tipo de mapeamento entre o discreto e o contínuo.

Uma possibilidade seria mapear as sequências em números inteiros. Contudo, esse mapeamento produz transformações que se assemelham a divergências, o que não é adequado para algoritmos de otimização.

Ao invés disso, empresta-se a inspiração da lógica *fuzzy* com o mapa que transforma o $k$-ésimo bit da sequência na variável contínua:
$$x_k \in [0, 1]$$

A sequência, portanto, descreve um vetor $\mathbf{x}$. O objetivo é buscar uma sequência de mudanças contínuas que alterem os valores $x_k$, penalizando mudanças para transformações não-consecutivas.

---

### B. Função de Erro $\mathcal{L}(\mathbf{x})$

A função erro é definida como:
$$\mathcal{L}(\mathbf{x}) = \sum_{k=1}^L (x_k - y_k)^2$$

onde $\mathbf{y}$ é o vetor contínuo correspondente à sequência alvo.

**Questão teórica:**
- Explique como a busca de sequências de transformações se relaciona com a minimização de $\mathcal{L}(\mathbf{x})$.

---

### C. Implementação do Gradiente Descendente (Sem Custo)

Implementar o algoritmo do gradiente descendente para minimizar $\mathcal{L}(\mathbf{x})$ (consulte Smola). Os vetores diferença entre passos consecutivos são as representações das transformações.

**Requisitos:**
- Retornar a sequência de transformações.
- Atualizar $\mathbf{x}$ usando a regra:
  $$x_i \leftarrow x_i - \eta \cdot \frac{\partial \mathcal{L}}{\partial x_i}$$
  onde $\eta$ é o parâmetro de incremento (ou taxa de aprendizado).
- **Análise de estabilidade:** Note que os valores de $x_k$ podem sair do intervalo de validade em determinadas situações, criando comportamentos espúrios ou oscilantes. Ilustre um caso no qual $x_k$ se torna negativo para $\eta > 1/2$.

---

### D. Inclusão da Função Custo

Incluir a função de custo $C(\mathbf{z})$ que penaliza mudanças não consecutivas. Aqui, o vetor $\mathbf{z}$ descreve a probabilidade ou peso das transformações geradas pelo gradiente.

1. **Modelagem de $\mathbf{z}$:** Modele $\mathbf{z}$ tal que seja uma função suave ou não do gradiente $\nabla\mathcal{L}(\mathbf{x})$.
2. **Definição de $C(\mathbf{z})$:** Defina uma função custo que favoreça mudanças de elementos consecutivos de $\mathbf{x}$. Justifique sua resposta.
3. **Regra da Cadeia:** Considere a função objetivo:
   $$J(\mathbf{x}) = \mathcal{L}(\mathbf{x}) + \lambda C(\mathbf{z})$$
   Por meio da regra da cadeia, mostre que variações em $x_i$ também recebem contribuições devido ao custo de transformações adjacentes.
4. **Minimização:** Minimize $J(\mathbf{x})$ para diferentes valores do parâmetro $\lambda$. Discuta os resultados obtidos.
