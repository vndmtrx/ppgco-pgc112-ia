# Metodologia: Mini-Projeto 2 (Visão Computacional)

> **Sobre este documento e o nosso fluxo de trabalho:**  
> Assim como fizemos no Mini-Projeto 1, este documento não é um artigo formal ou teórico abstrato. Ele é o registro do **nosso entendimento do problema**, da nossa jornada de compreensão da literatura canônica (Russell & Norvig, 4ª Edição, Cap. 22; e Smola et al., *Dive into Deep Learning*, Caps. 4, 5 e 7) e das **escolhas conceituais e de engenharia** que decidimos adotar para implementar o projeto.

---

## 1. Como Construímos o Entendimento dos Fundamentos

Antes de escrever qualquer código ou tentar rodar o MNIST, decidimos dar um passo atrás para consolidar os conceitos físicos e matemáticos básicos. Percebemos que, sem uma intuição clara do cálculo, o treinamento de uma rede neural pareceria uma "caixa preta mágica".

### 1.1. O que Entendemos por Derivada (A Analogia do Sistema de Freio)

A derivada costuma ser ensinada como limites complicados, mas o que nos fez entender o conceito na prática foi a **física e o movimento**:

- **Aceleração e Velocidade:** Já tínhamos claro que a aceleração é a derivada da velocidade no tempo ($a = \frac{\Delta v}{\Delta t}$). Ou seja: mede a rapidez com que a velocidade muda a cada instante do contínuo.
- **O Teste do Freio:** Para visualizar o ajuste de um peso em IA, imaginamos um teste de frenagem onde um carro a $100\text{ km/h}$ deve parar exatamente na marca de $15\text{ metros}$. Se pisamos a meio curso e o carro para em $54\text{ metros}$, temos um erro de $+39\text{ metros}$.  
  A pergunta prática de engenharia é: *"Se eu apertar o parafuso da regulagem do freio em $1\text{ mm}$, quantos metros a distância de parada vai encolher?"*. Essa relação de troca é a derivada $\frac{\Delta \text{Distância}}{\Delta \text{Regulagem}}$.

### 1.2. Derivada como Proporção Viva (e não uma Regrinha de 3 Estática)

Discutimos a diferença entre uma proporção de colégio e a derivada:
- Numa **regra de três comum**, a taxa é fixa e estática (ex.: $\frac{\Delta A}{\Delta B} = 2$ para sempre).
- Mas numa curva real (como o erro de uma rede neural), a proporção **muda dependendo de onde estamos pisando**. A derivada é uma nova função $f'(x)$ que nos dá a "marcha atual" daquele ponto exato.
- **O porquê do $\Delta$ (Delta):** Entendemos que para saber a velocidade ou o ajuste, não importa a coordenada absoluta acumulada (como o total de $150.000\text{ km}$ no odômetro do carro), mas apenas o **tamanho do empurrão** ($\Delta = \text{novo} - \text{velho}$) que ocorreu naquele intervalo minúsculo.

### 1.3. A Visão Geométrica: Por que a Derivada é a Tangente?

Entendemos a ponte entre a geometria e a trigonometria:
- Ao encostar uma régua por fora de uma curva suave, ela se apoia em um único ponto (reta tangente).
- A inclinação de qualquer rampa é a divisão da subida horizontal pela vertical: $\frac{\Delta y}{\Delta x}$, que na trigonometria é rigorosamente a **tangente do ângulo** ($\tan \theta$).
- É por isso que, quando chegamos no fundo do vale (onde o erro é mínimo), a régua fica perfeitamente deitada (horizontal). O ângulo é $0^\circ$, a tangente é zero e a derivada é zero ($\frac{d\mathcal{L}}{dw} = 0$).

---

## 2. A Camada Neural, a Matriz de Pesos e o Truque do Dummy `+1`

Ao analisar a Seção 22.1 e a **Figura 22.3** do livro de Russell & Norvig, nos deparamos com a estrutura de um Perceptron Multicamadas (MLP) e entendemos como ela funciona por baixo dos panos na máquina:

### 2.1. O que são os Dummies (`+1`) na Figura 22.3?

A figura do livro mostra bloquinhos azuis com `+1` multiplicando pesos $w_{0,3}$, $w_{0,4}$, etc. Esclarecemos o motivo dessa escolha de representação:

1. **A necessidade física do Bias ($b$):**  
   Um neurônio calcula $z = \sum w_i x_i + b$. Se não houvesse o termo $b$, a reta seria forçada a passar pela origem $(0,0)$. Em imagens, se todos os pixels fossem pretos ($x = 0$), o neurônio ficaria paralisado em zero sem poder tomar nenhuma decisão. O bias é a propensão natural ou limiar de disparo do neurônio.
2. **O truque da entrada dummy $+1$:**  
   Em vez de fazer uma multiplicação de matrizes e depois ter que somar o bias em uma instrução separada ($z = \mathbf{W}\mathbf{x} + \mathbf{b}$), os autores aumentam o vetor de entrada em 1 dimensão colocando um `+1` fixo na primeira posição:
   $$\mathbf{x}_{\text{ext}} = \begin{bmatrix} \mathbf{+1} \\ x_1 \\ x_2 \end{bmatrix}, \quad \mathbf{w}_{\text{ext}} = \begin{bmatrix} \mathbf{w_0} \\ w_1 \\ w_2 \end{bmatrix}$$
   Ao fazer isso, o produto escalar $\mathbf{w}_{\text{ext}}^\top \mathbf{x}_{\text{ext}}$ já calcula $(w_0 \cdot 1) + w_1 x_1 + w_2 x_2$. O ato de somar o bias acontece de carona dentro da própria multiplicação, economizando instruções de máquina (transformação afim em multiplicação puramente linear).

### 2.2. A Matriz Pertence à Camada (e o Fluxo de Redução)

Um ponto fundamental que consolidamos na leitura:
- **Para 1 neurônio:** Temos a multiplicação elemento a elemento de dois vetores (`vetor * vetor`), seguida de uma **soma de redução** (*reduction sum*), resultando em um **único valor escalar** ($z$).  
  *(Na computação e na programação funcional, isso é rigorosamente o conceito de um `reduce(soma)`: uma operação que dobra os elementos do vetor sobre si mesmos até produzir um único resultado agregado).*
- **Para a camada inteira:** A matriz de pesos pertence à **camada**, não a um neurônio isolado.
  - Se temos $m$ neurônios recebendo $n+1$ valores de entrada, montamos um retângulo (matriz) de tamanho $m \times (n+1)$, onde cada linha transposta é a fita de pesos de um neurônio.
  - Na multiplicação da matriz pela coluna de entrada, a **soma de redução de cada linha** gera um dos elementos da coluna de saída $\mathbf{z} = [z_1, z_2, \dots, z_m]^\top$.

### 2.3. Discussão de Paradigma: Por que não usar Classes `Perceptron` e Threads?

Refletimos sobre uma intuição comum nos anos 2000 (no auge da Programação Orientada a Objetos): criar uma `class Perceptron`, instanciar nós de um grafo com ponteiros para conexões e talvez atribuir uma `Thread` do Sistema Operacional para cada neurônio simular disparos biológicos concorrentes.

Discutimos por que esse modelo clássico de POO colapsa em desempenho e por que o Deep Learning moderno adotou a **álgebra tensorial contígua**:
1. **Localidade de Memória e Cache Misses:** Instanciar milhares de objetos soltos no Heap fragmenta a memória RAM. A CPU gastaria a maior parte dos ciclos apenas navegando por ponteiros desordenados.
2. **Custo de Chaveamento de Contexto (*Context Switching*):** Milhares de threads concorrentes disputando núcleos da CPU geram sobrecarga massiva no escalonador do SO.
3. **A Solução por Tensores Contíguos:** Ao empilhar os pesos em uma única matriz $\mathbf{W}$ contígua na memória, a GPU não cria threads do sistema operacional; ela utiliza milhares de pequenos núcleos dedicados de silício (ALUs/Tensor Cores) que executam a instrução vetorial em bloco num único ciclo de clock.  
   *(Essa mesma decisão de engenharia foi adotada no Mini-Projeto 1, onde descartamos instâncias de classes de fita em favor de inteiros nativos na ULA).*

### 2.4. Por que Matrizes Separadas por Camada?

Entendemos que não colocamos todos os neurônios da rede em uma única matriz porque a rede funciona em linha de montagem:
1. A primeira camada recebe as entradas brutas da imagem e gera o vetor $\mathbf{z}^{(1)}$;
2. Esse vetor passa pela função de ativação não linear (a **ReLU**): $\mathbf{a}^{(1)} = \max(0, \mathbf{z}^{(1)})$;
3. Esse resultado $\mathbf{a}^{(1)}$ ganha um novo `+1` e **vira a entrada de dados da camada seguinte**;
4. Se não houvesse camadas separadas com a ativação não linear no meio, o produto de duas matrizes lineares colapsaria para uma única matriz linear simples, e a rede perderia todo o poder de aprender padrões visuais complexos.

Essa é a fundamentação da **Equação (22.3)** do Russell:
$$h_{\mathbf{w}}(\mathbf{x}) = \mathbf{g}^{(2)}\Big(\mathbf{W}^{(2)} \cdot \mathbf{g}^{(1)}(\mathbf{W}^{(1)} \mathbf{x})\Big)$$

---

## 3. O Pipeline de Dados: MNIST, Normalização e Divisão dos Dados (Seção I)

Implementamos o pipeline de carregamento e exploração dos dados no módulo [`dataset.py`](./dataset.py). Esta etapa responde integralmente às exigências da Seção I do enunciado da disciplina.

### 3.1. Carregamento Automático e Estrutura dos Dados
- O conjunto de dados MNIST foi obtido diretamente via `torchvision.datasets.MNIST(root='./data', download=True)`.
- As amostras consistem em imagens em escala de cinza de $28 \times 28$ pixels (canal único $C=1$), com rótulos inteiros de $0$ a $9$.
- O total de dados disponibilizado é de $70.000$ imagens: $60.000$ disponibilizadas no pacote original de treino e $10.000$ no pacote oficial de teste.

### 3.2. Cálculo Empírico e Significado Físico da Média ($\mu$) e Desvio Padrão ($\sigma$)
Para justificar rigorosamente a normalização afim Z-score $x_{\text{norm}} = \frac{x - \mu}{\sigma}$, implementamos a função `compute_mean_std()` que processa todas as 60.000 imagens de treino em sua escala contínua original $[0.0, 1.0]$:
- **Média empírica obtida:** $\mu_{\text{emp}} = 0.1307$ (idêntica ao valor canônico da literatura: $0.1307$).
- **Desvio padrão empírico obtido:** $\sigma_{\text{emp}} = 0.3081$ (idêntico ao valor canônico da literatura: $0.3081$).

#### O que nos diz o valor $\mu \approx 0.1307$ (Por que não é 0.5)?
Se os pixels variam de $0.0$ (preto) a $1.0$ (branco), uma média de $0.5$ indicaria uma imagem com $50\%$ de branco e $50\%$ de preto (como um tabuleiro de xadrez) ou completamente cinza intermediária.  
O fato de a média ser $0.1307$ revela a física dos dados: as imagens do MNIST são **esparsas**, com quase **$87\%$ de fundo escuro puro** e apenas cerca de $13\%$ de intensidade luminosa concentrada nos traços manuscritos centrais.

#### Por que Normalizar? (A Analogia com o Ponto de Gravidade do Projeto 1)
No Mini-Projeto 1 (Parte 2: Gradiente Descendente), o vetor nulo $\mathbf{0}$ funcionava como o centro de atração gravitacional do hipercubo. A normalização Z-score realiza exatamente essa centralização no espaço das imagens:
1. **Sem Normalização (Pixels Crus em $[0, 1]$):** Como todas as entradas $x_i$ são estritamente não-negativas ($x_i \ge 0$), a derivada do erro em relação a cada peso $\frac{\partial \mathcal{L}}{\partial w_i} = \delta \cdot x_i$ terá sempre o mesmo sinal para todos os pesos de um mesmo neurônio. Isso força todos os pesos a subirem juntos ou descerem juntos, gerando oscilações em "zigue-zague" e atrasando brutalmente a convergência do SGD.
2. **Com a Normalização (Centro de Gravidade no Zero):** Subtraindo a média $\mu = 0.1307$, o fundo preto vira negativo ($\frac{0 - 0.1307}{0.3081} \approx -0.42$) e o traço branco vira positivo ($\frac{1 - 0.1307}{0.3081} \approx +2.82$). A nuvem de dados passa a ter média zero e variância unitária, tornando o gradiente simétrico e permitindo que os pesos subam ou desçam de forma independente em direção ao mínimo global.

### 3.3. As Três Caixas de Dados: Treino (50.000), Validação (10.000) e Teste (10.000)
Para evitar o vício do modelo e vazamento de dados (*data leakage*), organizamos as $70.000$ imagens em três partições funcionais estritas:

1. **Caixa 1 Original ($60.000$ imagens de treino do MNIST):** Nós a dividimos em duas pilhas:
   - **Pilha de Treino ($50.000$ imagens):** O conjunto exclusivo onde o algoritmo calcula os gradientes e atualiza os pesos sinápticos $w$.
   - **Pilha de Validação ($10.000$ imagens):** Imagens que **nunca** entram no cálculo de gradientes. São avaliadas apenas ao final de cada época para monitorar a curva de perda, escolher hiperparâmetros (como profundidade e taxa de aprendizado) e diagnosticar precocemente o *overfitting* ou *underfitting*.
2. **Caixa 2 Original ($10.000$ imagens de teste do MNIST):**
   - **Pilha de Teste ($10.000$ imagens):** Funciona como um "cofre lacrado" (a prova final). Permanece intocada durante todo o desenvolvimento e só é aberta uma única vez para aferir a generalização definitiva dos modelos.

#### O Papel da Semente Determinística (`seed=42`)
A divisão das $60.000$ imagens foi fixada com `torch.Generator().manual_seed(42)`. Sem essa semente, cada execução sortearia imagens diferentes para a validação, gerando comportamentos estocásticos que impediriam a comparação justa entre diferentes arquiteturas de rede. A semente garante 100% de repetibilidade científica.

### 3.4. Gráficos Exploratórios Gerados
Executando o script `dataset.py`, geramos automaticamente duas figuras de alta resolução salvas na pasta [`figuras/`](./figuras/):
- **`figuras/distribuicao_classes.png`:** Gráfico de barras comparativo demonstrando que os três conjuntos (treino, validação e teste) mantêm uma distribuição praticamente uniforme entre as classes $0$ a $9$ (aproximadamente $5.000$ imagens por classe no treino e $1.000$ por classe na validação e teste), comprovando que o problema está bem balanceado.
- **`figuras/exemplos_digitos.png`:** Grade visual com 5 amostras reais de cada um dos dígitos de $0$ a $9$, ilustrando a diversidade caligráfica (espessura do traço, inclinação e variações de estilo manuscrito).

### 3.5. Validação Automatizada (Suíte de Testes)
A integridade de todo o pipeline de dados foi blindada por 5 testes unitários automatizados em [`tests/test_dataset.py`](./tests/test_dataset.py), validando:
1. Contagem exata das dimensões dos splits ($50.000 + 10.000 + 10.000 = 70.000$);
2. Formato e tipo tensorial das imagens (`torch.float32`, dimensão `(1, 28, 28)`);
3. Agrupamento em lotes (*batching*) pelos `DataLoaders`;
4. Reprodutibilidade estrita da partição pela semente determinística;
5. Integridade do histograma de contagem de classes.

---

## 4. Redes Neurais sob a Ótica da Física: Molas, Mínima Entropia e Emergência

Para consolidar em definitivo o entendimento do *Backpropagation* e do treinamento neural antes de avançar para a implementação das arquiteturas completas, formulamos uma ponte epistemológica com a mecânica clássica e a teoria da informação.

### 4.1. A Analogia da Mola e a Lei de Hooke (Erro como Energia Potencial)
Tradicionalmente, a função de perda quadrática $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$ é apresentada como uma fórmula matemática fria. No entanto, sua estrutura é idêntica à **Energia Potencial Elástica** de uma mola na física clássica:
$$E = \frac{1}{2} k (\Delta x)^2$$
- O alvo real $y$ representa o **ponto de repouso** (equilíbrio mecânico de tensão nula).
- A saída atual da rede $\hat{y}$ representa o deslocamento deformado do sistema.
- Se a rede prevê um valor superior ao alvo ($\hat{y} > y$), a mola está sob **pressão/compressão positiva**.
- Se a rede prevê um valor inferior ($\hat{y} < y$), a mola está sob **tensão/estiramento negativo**.

A força restauradora que busca puxar o sistema de volta ao repouso é governada pela **Lei de Hooke**:
$$F = -\frac{dE}{dx} = -k \cdot (\Delta x)$$
Isso explica fisicamente por que o Gradiente Descendente sempre carrega o sinal de menos: a derivada mede a deformação da mola e a força restauradora puxa os pesos no sentido oposto para dissipar o erro acumulado. O treinamento é, rigorosamente, a busca pelo **estado de menor energia potencial**.

### 4.2. O Backpropagation como Onda de Tensão Elástica e a Matriz Transposta ($\mathbf{W}^\top$)
Visualizamos uma rede neural de múltiplas camadas como uma **treliça elástica interligada**:
1. **Passagem para Frente (*Forward Pass*):** O estímulo de entrada percorre a primeira fileira de molas, propaga-se pelas camadas intermediárias via multiplicação matricial ($\mathbf{W}$) e alcança a ponta de saída.
2. **Deformação na Ponta:** Ao confrontar o gabarito, a ponta bate fora da marca, gerando uma tensão mecânica de erro $\delta_{\text{saída}} = (\hat{y} - y)$.
3. **A Onda de Retorno (*Backward Pass*):** Pela 3ª Lei de Newton (Ação e Reação) e pela Regra da Cadeia, essa tensão não se dissipa na ponta: ela **viaja de volta por toda a treliça**.  
   Para propagar a culpa para trás, a matemática utiliza **as mesmas matrizes de pesos da ida, porém transpostas ($\mathbf{W}^\top$)**:
   $$\boldsymbol{\delta}^{(l)} = (\mathbf{W}^{(l+1)})^\top \times \boldsymbol{\delta}^{(l+1)}$$
   Cada parafuso (peso $w$) sente a vibração da tensão que o atravessou e cede ligeiramente ($\Delta w = -\eta \cdot \text{tensão} \cdot \text{estímulo}$). A cada iteração, a treliça como um todo relaxa até acomodar-se na configuração de mínima tensão global.

### 4.3. Emergência e Especialização de Padrões (A Analogia com *O Problema dos Três Corpos*)
Discutimos como uma rede descobre propriedades visuais (linhas, bordas, arcos) sem que nenhum ser humano a instrua sobre geometria:
- **A Ilusão do Todo a partir de Partes Simples:** Na célebre cena do romance *O Problema dos Três Corpos* (Liu Cixin), 30 milhões de soldados com placas pretas e brancas formam portas lógicas e barramentos que calculam trajetórias gravitacionais complexas, embora nenhum soldado individual compreenda mecânica celeste. Em redes neurais, cada neurônio executa apenas somas ponderadas locais com ReLU, mas a interação global faz emergir o reconhecimento visual.
- **A Quebra de Simetria:** No início do treinamento, os pesos são inicializados com pequenos números aleatórios (ruído gaussiano). Isso quebra a simetria: por mero acaso estatístico, determinados neurônios nascem com leve afinidade por pixels verticais, outros por cantos ou arcos.
- **A Pressão de Seleção do Gradiente:** Ao tentar minimizar a energia da mola, o gradiente amplifica os neurônios que já possuíam correlação favorável com cada componente do traço. Conforme um neurônio se especializa em linhas verticais, o erro residual força os demais a se especializarem no que sobrou (curvas e diagonais), gerando uma divisão espontânea de trabalho ("detetives de propriedades") que desembola o novelo de dados e torna as classes linearmente separáveis.

### 4.4. A Busca pela Mínima Entropia (Teoria da Informação de Shannon)
Estabelecemos a correspondência formal da função de perda com a Teoria da Informação (Smola et al., Cap. 4.1.3):
- No início do treino, com pesos aleatórios, a rede distribui probabilidades quase uniformes entre as 10 classes ($\approx 10\%$ para cada dígito). O sistema encontra-se no estado de **máxima ignorância e máxima entropia**.
- A função de perda **Entropia Cruzada (*Cross-Entropy Loss*)** mede a Divergência de Kullback-Leibler entre a distribuição prevista pela rede e a distribuição real (gabarito categórico).
- O processo de aprendizagem atua como um resfriamento termodinâmico: o SGD atua dissipando a incerteza e a surpresa estatística, conduzindo os parâmetros para um estado estável e cristalino de **mínima entropia**, onde o modelo atribui probabilidade próxima de $100\%$ à classe correta.

### 4.5. O Experimento Empírico 2x2: Anatomia do Neurônio Detector de Diagonais e a Solução do *Dying ReLU*

Para validar no menor microcosmo possível os conceitos de inicialização, ativação, gradiente e emergência, realizamos um experimento no Jupyter Notebook com uma imagem binária mínima de $2 \times 2$ pixels (4 entradas: $p_1, p_2, p_3, p_4$):
$$\mathbf{X} = \begin{bmatrix} p_1 & p_2 \\ p_3 & p_4 \end{bmatrix}$$

O objetivo pedagógico foi treinar um único neurônio artificial para atuar como um **detector específico de diagonal principal** ($p_1=1, p_4=1$), rejeitando linhas horizontais, imagens pretas e anti-diagonais.

#### 1. A Armadilha do *Dying ReLU* (Neurônio Morto)
Inicializamos manualmente os pesos e o viés com:
$$\mathbf{W} = \begin{bmatrix} 0.1 & 0.1 & 0.1 & 0.1 \end{bmatrix}, \quad b = -0.5$$

Ao passar a imagem diagonal $\mathbf{x}_{\text{diag}} = \begin{bmatrix} 1 & 0 & 0 & 1 \end{bmatrix}$:
$$z_{\text{diag}} = (1 \times 0.1) + (0 \times 0.1) + (0 \times 0.1) + (1 \times 0.1) + (-0.5) = 0.2 - 0.5 = -0.3$$
Como a função de ativação era a $\text{ReLU}(z) = \max(0, z)$, a saída foi $\text{ReLU}(-0.3) = 0.0$ em vez do esperado $1.0$.

Ao inspecionar o gradiente com `loss.backward()`, constatamos empiricamente o fenômeno do **Dying ReLU**:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = 2(\hat{y} - y) \cdot \text{ReLU}'(z) \cdot \mathbf{x} = 2(0 - 1) \cdot \mathbf{0} \cdot \mathbf{x} = \mathbf{0}$$
Como a derivada da ReLU para valores estritamente negativos é identicamente nula, a tensão do erro foi bloqueada na saída, impedindo qualquer ajuste nos pesos (o neurônio adormeceu).

#### 2. Quebra de Simetria e Emergência do Peso Inibitório
Ao corrigir a inicialização do bias para $b = 0.0$ (permitindo a passagem inicial do sinal) e submeter o neurônio a 100 épocas de Gradiente Descendente Estocástico (SGD com $\eta = 0.1$) em um lote com 4 imagens canônicas (Diagonal, Horizontal, Preta e Anti-diagonal), observamos uma notável emergência geométrica nos pesos convergidos:
$$\mathbf{W}_{\text{aprendido}} \approx \begin{bmatrix} +0.315 & -0.320 & +0.038 & +0.674 \end{bmatrix}, \quad b_{\text{aprendido}} \approx +0.006$$

- **Pesos Excitadores ($w_1 \approx +0.315$, $w_4 \approx +0.674$):** O gradiente amplificou os canais correspondentes aos pixels da diagonal principal. A soma de ambos ($0.315 + 0.674 \approx 0.99$) atinge exatamente o alvo unitário desejado.
- **O Peso Inibitório como Filtro Anti-Confusão ($w_2 \approx -0.320$):** O neurônio descobriu espontaneamente que uma linha horizontal também acende o pixel superior esquerdo ($p_1=1$). Para não confundir uma reta horizontal com uma diagonal, ele atribuiu um **peso negativo e de magnitude equivalente** ao pixel vizinho ($p_2$). Assim, na imagem horizontal ($[1, 1, 0, 0]$), a ativação do segundo pixel cancela o primeiro ($+0.315 - 0.320 \approx -0.005$), anulando a saída na ReLU!
- **Convergência das Predições Finais:**
  - Imagem Diagonal: **$0.996$** (Alvo: $1.0$ — Detecção confirmada);
  - Imagem Horizontal: **$0.001$** (Alvo: $0.0$);
  - Imagem Preta: **$0.006$** (Alvo: $0.0$);
  - Imagem Anti-diagonal: **$0.000$** (Alvo: $0.0$).

Este microcosmo comprova que os neurônios aprendem "antídotos" (pesos inibitórios) para diferenciar traços conflitantes, princípio idêntico ao que o MLP e a CNN utilizarão nas 784 entradas do MNIST para discriminar dígitos morfologicamente semelhantes como o $4$ e o $9$, ou o $3$ e o $5$.

### 4.6. Concepção e Arquitetura do Estudo Dirigido Interativo (`estudo_dirigido.html`)

Para aprofundar e consolidar a compreensão conceitual e mecânica do modelo neural antes de iniciar a codificação em larga escala no PyTorch, o discente solicitou ao assistente de IA em *pair programming* (**Antigravity**) a concepção e implementação de uma ferramenta web interativa de estudo dirigido ([`estudo_dirigido.html`](./estudo_dirigido.html)).

A motivação do discente foi dispor de um ambiente de experimentação visual onde as especificações matemáticas extraídas diretamente dos livros de referência da disciplina — em especial a formulação do Perceptron e coordenadas homogêneas de **Russell & Norvig (Cap. 22)** e a dinâmica de retropropagação, funções de custo e camadas densas de **Smola et al. (*D2L*, Cap. 4)** — pudessem ser manipuladas e testadas na prática. O objetivo foi permitir ao aluno alterar valores de entrada e pesos e observar simultaneamente:

1. **A Dupla Representação Pareada (Biológica vs. Tensorial):** Visualizar de um lado o diagrama arquitetural com nós e conexões sinápticas coloridas e, do outro, a formulação matricial rigorosa em estilo LaTeX com colchetes sólidos ($\begin{bmatrix} \dots \end{bmatrix}$), compreendendo como cada perninha sináptica se mapeia diretamente em um elemento de vetor ou linha de matriz contígua em memória.
2. **Execução Passo a Passo do Fluxo de Informação (*Stepper* Dinâmico):** Habilitar um controle de fases com animação por onde o usuário pode avançar e retroceder para acompanhar:
   - Os pulsos elétricos de dados ($x_i \cdot w_i$) trafegando pelas conexões sinápticas;
   - A soma afim de redução no núcleo celular ($\sum$);
   - O teste de disparo da comporta não-linear (ReLU);
   - A tensão mecânica da mola elástica contra o alvo gabarito ($\delta = \hat{y} - y$);
   - A onda reversa de retropropagação (*backpropagation*) propagando a culpa proporcional através da matriz transposta ($\mathbf{W}^\top$);
   - O alívio de tensão e recalibração dos pesos via Gradiente Descendente Estocástico (SGD).
3. **Escalaridade em Três Níveis de Abstração:**
   - **Nível 1 (O Perceptron Simples):** O neurônio isolado, a entrada dummy $x_0 = +1.0$ e o produto escalar afim $\mathbf{w}^\top \mathbf{x}$;
   - **Nível 2 (A Rede Multicamadas):** Duas camadas completas, demonstrando o empilhamento das linhas na matriz $\mathbf{W}_1$ ($2 \times 2$), o vetor de ativações intermediárias $\mathbf{a}_1$ e o retorno do erro pela transposta $\mathbf{W}_2^\top$;
   - **Nível 3 (O Estudo de Caso 2×2):** O microcosmo empírico testado no Jupyter, com grade de pixels interativa clicável, demonstração em tempo real da trava do *Dying ReLU* e carregamento dos pesos aprendidos pelo SGD ($w_1=+0.315, w_2=-0.320, w_3=+0.038, w_4=+0.674, b=+0.006$).

Essa ferramenta serviu como um elo epistemológico fundamental, permitindo ao aluno validar hipóteses, visualizar o comportamento dos gradientes e eliminar quaisquer dúvidas conceituais antes de prosseguir com os scripts definitivos do mini-projeto.

---

## 5. Log de Atividades

| Data | Atividade / Marco | Decisões e Entendimentos Registrados |
| :--- | :--- | :--- |
| **11/10/2026** | Inicialização e Análise do Problema | Leitura do enunciado oficial (`03-projeto-visao-simples.pdf`), mapeamento das demandas (MNIST, normalização, MLP vs CNN, hiperparâmetros e análise morfológica de confusões) e correlações bibliográficas com Russell & Norvig (Cap. 22) e Smola (*D2L*, Caps. 4, 5 e 7). |
| **14/10/2026** | Fundamentação Física da Derivada | Resgate dos fundamentos do cálculo diferencial: analogia mecânica da calibragem do sistema de freio de um carro (parar de 100 km/h a 0 km/h na marca de 15m), conceito de proporção dinâmica instantânea e geometria da reta tangente ($\tan \theta$). |
| **18/10/2026** | A Camada Neural e o Dummy `+1` | Análise da Figura 22.3 do Russell. Compreensão do dummy input `+1` (coordenadas homogêneas) para embutir o bias $w_0$ na multiplicação matricial limpa, a matriz de pesos pertencendo à camada e a operação de soma de redução. |
| **21/10/2026** | Discussão de Paradigma: POO vs. Tensores | Reflexão histórica de 2001: por que descartamos classes `Perceptron` com nós de grafo e threads concorrentes (evitando fragmentação de memória e *cache misses*) em favor de matrizes contíguas de alta performance na GPU. |
| **26/10/2026** | Alinhamento de Escopo e Ferramental | Decisão metodológica de focar estritamente no ecossistema PyTorch, descartando implementações redundantes do zero e priorizando a geração rica de gráficos analíticos e diagnósticos visuais. |
| **29/10/2026** | Pipeline de Dados e Divisão (Split) | Implementação de `dataset.py`. Download oficial do MNIST e cálculo empírico da média ($\mu = 0.1307$) e desvio padrão ($\sigma = 0.3081$). Divisão rigorosa em Treino ($50.000$), Validação ($10.000$) e Teste ($10.000$). |
| **29/10/2026** | Exploração Visual e Blindagem TDD | Geração dos gráficos em 300 DPI em `figuras/` (balanceamento das classes e amostras caligráficas). Criação de `tests/test_dataset.py` e configuração de `pytest.ini`, com 5 testes automatizados aprovados (100% de sucesso). |
| **29/10/2026** | Consolidação Teórica dos Dados | Discussão e registro do porquê de $\mu \approx 0.1307$ (dados esparsos com 87% de fundo preto), a física da normalização (analogia com o centro de gravidade do Projeto 1 para eliminar oscilações em zigue-zague no gradiente) e a ética do teste lacrado. |
| **29/10/2026 (Noite)** | Epistemologia Física: Molas e Entropia | Descoberta da equivalência entre a função de perda e a energia elástica de uma mola (Lei de Hooke e alívio de tensão com matrizes transpostas), emergência de propriedades por quebra de simetria (*Three Body Problem*) e convergência para o estado de mínima entropia de Shannon. Criação de `laboratorio_backprop.py`. |
| **30/10/2026** | Concepção do Estudo Dirigido Interativo | Solicitação do aluno ao Antigravity para criar um laboratório interativo (`estudo_dirigido.html`) baseado no Russell & Norvig e Smola, unificando a visão biológica do perceptron com a álgebra matricial em LaTeX para experimentação de valores. |
| **04/11/2026 (Manhã)** | O Toy Model 2x2 e a Solução do Dying ReLU | Implementação empírica no Jupyter Notebook do detector de diagonais. Diagnóstico do gradiente nulo quando $z < 0$ com ReLU. Treinamento com SGD demonstrando o surgimento espontâneo do peso inibitório ($w_2 \approx -0.320$) para anular horizontais e convergência a 0.996. |
| **04/11/2026 (Tarde)** | Estudo Dirigido com Matrizes LaTeX e Backprop | Evolução do `estudo_dirigido.html` solicitada pelo aluno com animação completa das 3 fases, notação formal de matrizes em LaTeX com colchetes, onda reversa via $\mathbf{W}^\top$ e integração com os pesos treinados no notebook. |





