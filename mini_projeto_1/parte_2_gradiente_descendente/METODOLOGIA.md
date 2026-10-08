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

### 1.4. O Problema do "Gradiente Egoísta" e a Independência dos Bits

Outro ponto que ficou claro ao analisar o cálculo é que, no gradiente descendente básico, cada número da fita opera de forma completamente isolada. O bit da posição 10 calcula o seu próprio ajuste e não tem a menor ideia do que os bits vizinhos (nas posições 9 e 11) estão fazendo. Eles são totalmente independentes.

Isso entra em conflito direto com o que aprendemos na Parte 1 do projeto:
- Na Parte 1, inverter um bloco contíguo de vários bits seguidos custava exatamente o mesmo (custo 1) que inverter um único bit solitário. O algoritmo guloso se aproveitava justamente disso para zerar blocos inteiros de uma vez só.
- No gradiente descendente puro, essa vantagem se perde: cada posição tenta descer para zero por conta própria, como se estivesse sozinha no mundo. É um comportamento "egoísta" que não reproduz a economia de escala de mexer em blocos.

---

### 1.5. A Intuição do Custo de Bloco: O Exemplo Prático de Desmanchar `01110`

Para resolver essa falta de coordenação entre vizinhos, o enunciado propõe adicionar uma função de custo que penaliza mudanças não consecutivas. O entendimento dessa penalidade ficou muito claro quando pensei em como desmanchar um bloco de 3 bits ativos: `01110`.

Existem dois caminhos bem diferentes para levar essa sequência até `00000`:

1. **Caminho 1: Desmanchar bit a bit aos poucos (Sem coordenação / Custo alto)**
   - O algoritmo primeiro desliga um bit (`01100`), depois o outro (`01000`) e por fim o último (`00000`).
   - Na Parte 1, isso exigiria 3 operações separadas, acumulando custo 3. No modelo contínuo, a cada rodada quebramos a harmonia entre os vizinhos, criando degraus bruscos na fita e pagando multa máxima por agir de forma fragmentada.

2. **Caminho 2: Descer o bloco inteiro junto (Com coordenação / Custo baixo)**
   - Os três bits do meio decaem para zero simultaneamente, em perfeita sincronia (`01110` direto para `00000`).
   - Na Parte 1, isso é uma única operação de bloco com custo 1. No modelo contínuo, como os três vizinhos estão mudando na mesma intensidade, a diferença entre eles no miolo do bloco é zero. Não há atrito entre vizinhos, pagando a menor penalidade possível.

**A Queda de Braço e o Ponto de Equilíbrio:**
A partir desse exemplo, compreendi o papel dos dois botões de ajuste que temos nas mãos:
- O peso da penalidade ($\lambda$): se colocarmos zero, o algoritmo vira uma bagunça e cada bit desce sozinho. Mas se colocarmos uma penalidade pesada demais, o algoritmo fica "com medo" de mexer em qualquer bit isolado para não pagar multa de vizinho, travando a solução.
- A taxa de aprendizado ($\eta$): se for pequena demais, precisamos de passos infinitos para chegar a zero; se for grande demais, passamos do ponto e caímos em valores negativos. O objetivo prático é encontrar o equilíbrio ideal entre esses dois parâmetros.

---

### 1.6. A Metáfora dos Paraquedistas e a "Inércia Espacial" de Bordas

Para conseguir enxergar como essa penalidade matemática realmente atua durante as iterações do algoritmo, elaborei uma metáfora física:

Imagine a fita de números como uma fila de paraquedistas que precisam descer do céu (altura 1.0) até o chão (altura 0.0):
- **No Gradiente Puro (Sem Custo):** Não existe ligação entre eles. Cada paraquedista que está no ar calcula sua distância até o chão e abre seu paraquedas individual. Todos que estão no ar descem simultaneamente, mas cada um cuida da sua própria vida sem olhar para os lados.
- **No Gradiente com Custo de Bloco:** O professor introduz uma regra física: todos os paraquedistas vizinhos estão amarrados entre si por uma corda elástica. 

Quando analisamos essa corda elástica na sequência `01001110110`, um insight fundamental emergiu ao acompanhar o primeiro passo numérico:
1. **O miolo flui livre:** Na posição 5 (o miolo do bloco de 3 uns), os vizinhos da esquerda e da direita também estão no alto e descem juntos. As cordas internas continuam frouxas, portanto o miolo desce com velocidade máxima, sem sofrer atrito de penalidade.
2. **A tensão das bordas e os zeros puxados para cima:** O elástico sempre busca relaxar a sua tensão. Por isso, nas fronteiras do bloco, acontece uma troca dupla:
   - Os números em 1.0 sentem a corda dos vizinhos puxando-os para baixo com mais força.
   - Os números em 0.0 vizinhos ao bloco **são ligeiramente puxados para cima** (subindo, por exemplo, de 0.00 para 0.04 ou 0.08).
   - O algoritmo amacia o degrau: em vez de uma quebra brusca de parede (`0.0` colado em `1.0`), o elástico deforma a fita como uma rampa suave (`0.04 -> 0.56 -> 0.60`), fazendo a tensão da corda despencar rapidamente logo nas primeiras iterações.
3. **O ganho de escala:** Em blocos contíguos longos, a penalidade só existe nas duas extremidades (as pontas), enquanto todos os elementos internos descem sem resistência. Se tentássemos desmanchar a sequência em bits picados e solitários, teríamos cordas esticadas em todos os nós, gerando atrito e custo proibitivos.

Essa penalidade funciona como uma **inércia espacial (ou tensão superficial)**: o sistema resiste a criar rugas ou descontinuidades pontudas, favorecendo o amortecimento de blocos inteiros em sincronia.

4. **A analogia do filtro de desfoque (Blur) e a dissipação suave:**
   - Compreendi que a penalidade de vizinhança atua de forma análoga a um filtro de desfoque (*blur* no processamento de imagens): ela espalha o contraste das bordas e homogeniza a vizinhança.
   - Se o peso da penalidade ($\lambda$) for excessivo, o sistema sofre de suavização excessiva (*over-smoothing*), diluindo a informação dos bits e travando a descida em direção a zero.
   - Com uma taxa de aprendizado ($\eta$) equilibrada e comedida, o algoritmo permite que a tensão do elástico atue dissipando os picos isolados com calma e suavidade, sem dar saltos bruscos que causem *overshoot*.

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
| 07/10/2026 | Modelagem de Custo e Inércia | Formulação da analogia dos paraquedistas, tensão de elásticos, miolo livre vs. bordas e o efeito de desfoque/dissipação. |
