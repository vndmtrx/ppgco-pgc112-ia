# Metodologia: Parte 2 (Otimização com Gradiente Descendente)

## 1. Entendimento do Problema e Mudança de Paradigma

### 1.1. O que realmente mudou da Parte 1 para a Parte 2?

A transição da Parte 1 para a Parte 2 representa uma mudança fundamental de paradigma computacional, idêntica à evolução histórica que viabilizou as redes neurais artificiais modernas:

1. **O paralelo histórico (Perceptron $\to$ MLP / Sigmoide):**
   - Nos primórdios da IA (Perceptron de Rosenblatt), a decisão de ativação era binária e rígida, usando a função degrau de Heaviside: ou o neurônio disparava $1$, ou disparava $0$. 
   - O grande gargalo dessa abordagem discreta é que a função degrau tem derivada nula em quase todo o domínio e indefinida no ponto de transição. Sem derivadas utilizáveis, era impossível aplicar métodos de cálculo diferencial para ajustar múltiplos pesos em camadas profundas.
   - A virada histórica que consolidou os Perceptrons Multicamadas (MLPs) a partir dos anos 1980 e que perdurou fortemente nos anos 2000/2010 foi o **relaxamento contínuo**: substituir o degrau rígido $\{0, 1\}$ por uma função suave com imagem no intervalo contínuo $(0, 1)$, tipicamente a função **sigmoide (logística)**:
     $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
   - Ao "amaciar" o degrau em uma curva diferenciável em todos os pontos, tornou-se viável calcular gradientes e treinar redes via retropropagação (*backpropagation*) e gradiente descendente.

2. **A transição no nosso mini-projeto:**
   - **Na Parte 1 (Buscas Clássicas):** Estávamos operando no mundo do "degrau de Heaviside". A fita de bits continha valores estritamente discretos $\{0, 1\}$. Cada decisão de inversão (*flip*) era um salto atômico tudo-ou-nada. O único caminho viável para navegar nesse espaço combinatório era através de buscas discretas em grafos (BFS ou Algoritmo Guloso), testando transições de estados e verificando igualdades exatas.
   - **Na Parte 2 (Gradiente Descendente):** Fazemos exatamente o mesmo relaxamento da sigmoide / lógica fuzzy. Em vez de estados discretos, transformamos cada posição $k$ da fita em uma variável real contínua:
     $$x_k \in [0.0, 1.0]$$
   - A fita inicial rígida `00001001000011001100111` deixa de ser um texto e passa a ser simplesmente uma lista de 23 números. O meu grande insight aqui foi entender que **cada bit virou uma dimensão**:
     - Se a fita tivesse 2 bits, seriam 2 eixos ($x_1, x_2$), como uma folha de papel 2D.
     - Se tivesse 3 bits, seriam 3 eixos ($x_1, x_2, x_3$), como o espaço 3D da sala onde estou.
     - Como a fita do professor tem 23 bits, estamos em um espaço de **23 dimensões (23 eixos)**!
     - A mente humana não desenha 23 dimensões no ar, mas para o algoritmo é só uma lista de 23 coordenadas numéricas:
       $$\mathbf{x}^{(0)} = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, \dots, 1.0]$$
     - E o que o Gradiente Descendente faz na prática? Ele pega esse ponto no espaço de 23 eixos e vai puxando cada um desses eixos até todos eles chegarem no ponto $(0, 0, \dots, 0)$, que é o centro de tudo (a origem).
   - Um valor `1.0` indica ativação plena (o antigo bit 1), `0.0` indica inativação plena (o alvo desejado), e qualquer número intermediário (como `0.6` ou `0.2`) representa um estado de transição contínua.
   - Com essa representação, o problema deixa de exigir buscas cegas por vizinhança combinatória e passa a permitir o uso do cálculo diferencial: calculamos a inclinação exata da superfície de erro e deslizamos gradualmente em direção ao ponto de mínimo.

### 1.2. Por que o objetivo é "zerar o array"?

No enunciado, a sequência alvo foi fixada como $\mathbf{y} = [0, 0, \dots, 0]$. 

Ao definir a função de erro como a distância quadrática euclidiana:
$$\mathcal{L}(\mathbf{x}) = \sum_{k=1}^L (x_k - y_k)^2$$

Como cada $y_k = 0$, a função se reduz a:
$$\mathcal{L}(\mathbf{x}) = \sum_{k=1}^L x_k^2 = \|\mathbf{x}\|_2^2$$

**O insight fundamental:**
- Minimizar $\mathcal{L}(\mathbf{x})$ significa empurrar todos os elementos do vetor simultaneamente em direção à origem (ao vetor nulo).
- O processo de "resolver o problema" deixa de ser uma busca em árvore e se torna o ato de drenar a energia ou magnitude do vetor $\mathbf{x}$ até que todos os seus componentes cheguem a zero.
- Cada iteração do gradiente descendente calcula a inclinação (derivada) dessa parábola para cada coordenada e subtrai uma fração desse gradiente multiplicada pela taxa de aprendizado $\eta$.

---

### 1.3. A Intuição do Erro e a Dedução do Passo Crítico ($\eta > 1/2$)

Para mim, a função de erro deixou de ser uma equação abstrata e passou a fazer sentido como uma pergunta simples: **"quão longe cada número da fita está do valor desejado (zero)?"**.

Quando a fita começa cheia de valores ligados (em 1.0), a distância até o objetivo é grande, logo o erro total é alto. Conforme o algoritmo avança e empurra esses números para 0.0, a distância de cada um até o zero vai sumindo, até que o erro total finalmente zera. É como drenar uma caixa d'água até esvaziar completamente.

A regra para atualizar a fita usa a derivada desse erro. A derivada nada mais é do que a taxa de variação (como a velocidade que aprendemos na física do ensino médio), indicando a inclinação da ladeira e a força com que o erro precisa ser reduzido. Pela regra de derivação polinomial (a famosa "regra do tombo"), a inclinação de uma função quadrática dobra o valor da posição atual.

Foi a partir dessa inclinação que finalmente entendi por que o enunciado alerta que uma taxa de aprendizado ($\eta$) maior que 0.5 é um perigo:
- Se um número começa em 1.0 e precisa chegar a 0.0, a distância real que ele precisa percorrer é de exatamente 1 unidade.
- Como a inclinação da ladeira no ponto inicial vale 2.0 (o dobro de 1.0), o tamanho do passo dado pelo algoritmo é a taxa de aprendizado multiplicada por essa inclinação de 2.0.
- Se escolhermos uma taxa de 0.1, damos um passo de 0.2 e chegamos em 0.8 (vamos nos aproximando com calma).
- Se escolhermos uma taxa de 0.5, o passo dá exatamente 1.0, e acertamos o zero em cheio logo na primeira tentativa.
- Mas se escolhermos uma taxa maior que 0.5 (como 0.6), o tamanho do salto vira 1.2. Como a distância até o zero era de apenas 1.0, o algoritmo dá um salto grande demais, passa direto do ponto (o chamado *overshoot*) e aterrissa em -0.2.

Esse resultado negativo quebra a lógica do problema: em uma fita inspirada em lógica fuzzy e sigmoides, os valores precisam obrigatoriamente ficar entre 0.0 e 1.0. Um valor negativo não faz sentido físico aqui e causa instabilidade e oscilações no algoritmo.

---

## 2. Decisões de Design e Engenharia

### Decisão 1: Mapeamento Discreto-Contínuo

Seguindo o padrão de decisões do projeto:

1. **Alternativas avaliadas:**
   - **Mapear a fita para um único número inteiro na reta real:** Interpretar a sequência binária como a representação binária de um único número grande $N \in \mathbb{N}$ e tentar otimizar $N \to 0$.
   - **Mapear posição a posição no intervalo fuzzy $[0, 1]^L$:** Cada bit vira uma dimensão independente em um espaço vetorial $L$-dimensional.

2. **Por que descartamos a primeira alternativa:**
   - Interpretar a sequência como um único inteiro cria descontinuidades caóticas. Uma pequena alteração no bit mais significativo (ex.: de `01111111` para `10000000`) gera um salto gigantesco no valor inteiro (de 127 para 128), enquanto os bits individuais mudaram completamente de estado. Isso destrói a métrica de distância do problema e impede qualquer convergência estável por gradiente.

3. **Decisão final:**
   - Adotamos o relaxamento fuzzy vetorizado em $[0, 1]^L$. Cada posição $k$ tem seu próprio eixo contínuo $x_k \in \mathbb{R}$, preservando a localidade e permitindo calcular gradientes individuais e custos de vizinhança entre posições adjacentes.

---

## 3. Log de Atividades

| Data | Atividade | Decisões / Observações |
| :--- | :--- | :--- |
| 16/09/2026 | Inicialização | Estrutura criada e enunciados mapeados. |
| 04/10/2026 | Revisão Teórica e Referências | Mapeamento das referências bibliográficas canônicas (Russell & Norvig, 4ª ed.) nos algoritmos e na metodologia da Parte 1. |
| 05/10/2026 | Consolidação Conceitual | Registro da transição discreto $\to$ contínuo (paralelo Perceptron/Sigmoide), interpretação da fita como 23 dimensões e objetivo de puxar as coordenadas para a origem. |
| 06/10/2026 | Dedução do Passo Crítico | Dedução da perda quadrática, da regra do tombo ($2x$), do perigo de overshoot para $\eta > 0.5$ e da analogia do coelho de Zenão. |
