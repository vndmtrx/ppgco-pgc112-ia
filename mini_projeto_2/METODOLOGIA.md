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
1. **Sem Normalização (Pixels Crus em $[0, 1]$):** Como todas as entradas $x_i$ são estritamente não-negativas ($x_i \ge 0$), a derivada do erro em relação a cada peso $\frac{\partial \mathcal{L}}{\partial w_i} = \delta \cdot x_i$ terá sempre o mesmo sinal para todos os pesos de um mesmo neurônio. Isso força todos os pesos a subirem juntos ou descerem juntos, gerando oscilações em "zigue-zague" e atrasando a convergência do SGD.
2. **Com a Normalização (Centro de Gravidade no Zero):** Subtraindo a média $\mu = 0.1307$, o fundo preto vira negativo ($\frac{0 - 0.1307}{0.3081} \approx -0.42$) e o traço branco vira positivo ($\frac{1 - 0.1307}{0.3081} \approx +2.82$). A nuvem de dados passa a ter média zero e variância unitária, tornando o gradiente simétrico e permitindo que os pesos subam ou desçam de forma independente em direção ao mínimo global.

### 3.3. As Três Caixas de Dados: Treino (50.000), Validação (10.000) e Teste (10.000)
Para evitar o vício do modelo e vazamento de dados (*data leakage*), organizamos as $70.000$ imagens em três partições funcionais estritas:

1. **Caixa 1 Original ($60.000$ imagens de treino do MNIST):** Nós a dividimos em duas pilhas:
   - **Pilha de Treino ($50.000$ imagens):** O conjunto exclusivo onde o algoritmo calcula os gradientes e atualiza os pesos sinápticos $w$.
   - **Pilha de Validação ($10.000$ imagens):** Imagens que **nunca** entram no cálculo de gradientes. São avaliadas apenas ao final de cada época para monitorar a curva de perda, escolher hiperparâmetros (como profundidade e taxa de aprendizado) e diagnosticar precocemente o *overfitting* ou *underfitting*.
2. **Caixa 2 Original ($10.000$ imagens de teste do MNIST):**
   - **Pilha de Teste ($10.000$ imagens):** Funciona como um "cofre lacrado" (a prova final). Permanece intocada durante todo o desenvolvimento e deve ser aberta uma única vez, no modelo escolhido pela validação, para aferir a generalização.

#### O Papel da Semente Determinística (`seed=42`)
A divisão das $60.000$ imagens foi fixada com `torch.Generator().manual_seed(42)`. Sem essa semente, cada execução sortearia imagens diferentes para a validação, gerando comportamentos estocásticos que impediriam a comparação justa entre diferentes arquiteturas de rede. A semente garante 100% de repetibilidade científica.

### 3.4. Gráficos Exploratórios Gerados
Executando o script `dataset.py`, geramos automaticamente duas figuras de alta resolução salvas na pasta [`figuras/`](./figuras/):
- **`figuras/distribuicao_classes.png`:** Gráfico de barras comparativo demonstrando que os três conjuntos (treino, validação e teste) mantêm uma distribuição praticamente uniforme entre as classes $0$ a $9$ (aproximadamente $5.000$ imagens por classe no treino e $1.000$ por classe na validação e teste), indicando que o problema está bem balanceado.
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

Para consolidar o entendimento do *Backpropagation* e do treinamento neural antes de avançar para a implementação das arquiteturas completas, formulamos uma ponte epistemológica com a mecânica clássica e a teoria da informação.

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
- O processo de aprendizagem atua como um resfriamento termodinâmico: o SGD atua dissipando a incerteza e a surpresa estatística, conduzindo os parâmetros para um estado estável de **mínima entropia**, onde o modelo atribui probabilidade próxima de $100\%$ à classe correta.

### 4.5. O Experimento Empírico 2x2: Anatomia do Neurônio Detector de Diagonais e a Solução do *Dying ReLU*

Para validar no menor microcosmo possível os conceitos de inicialização, ativação, gradiente e emergência, montei e executei um experimento no Jupyter Notebook com uma imagem binária mínima de $2 \times 2$ pixels (4 entradas: $p_1, p_2, p_3, p_4$):
$$\mathbf{X} = \begin{bmatrix} p_1 & p_2 \\ p_3 & p_4 \end{bmatrix}$$

O meu objetivo prático foi treinar um único neurônio artificial para atuar como um **detector específico de diagonal principal** ($p_1=1, p_4=1$), rejeitando linhas horizontais, imagens pretas e anti-diagonais.

#### 1. A Armadilha do *Dying ReLU* (Neurônio Morto)
Inicializei manualmente os pesos e o viés com:
$$\mathbf{W} = \begin{bmatrix} 0.1 & 0.1 & 0.1 & 0.1 \end{bmatrix}, \quad b = -0.5$$

Ao passar a imagem diagonal $\mathbf{x}_{\text{diag}} = \begin{bmatrix} 1 & 0 & 0 & 1 \end{bmatrix}$:
$$z_{\text{diag}} = (1 \times 0.1) + (0 \times 0.1) + (0 \times 0.1) + (1 \times 0.1) + (-0.5) = 0.2 - 0.5 = -0.3$$
Como a função de ativação era a $\text{ReLU}(z) = \max(0, z)$, a saída foi $\text{ReLU}(-0.3) = 0.0$ em vez do esperado $1.0$.

Ao inspecionar o gradiente com `loss.backward()`, comprovei na prática o fenômeno do **Dying ReLU**:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = 2(\hat{y} - y) \cdot \text{ReLU}'(z) \cdot \mathbf{x} = 2(0 - 1) \cdot \mathbf{0} \cdot \mathbf{x} = \mathbf{0}$$
Como a derivada da ReLU para valores estritamente negativos é zero, a tensão do erro foi bloqueada na saída, impedindo qualquer ajuste nos pesos (o neurônio adormeceu).

#### 2. Quebra de Simetria e Emergência do Peso Inibitório
Ao corrigir a inicialização do bias para $b = 0.0$ (permitindo a passagem inicial do sinal) e treinar o neurônio por 100 épocas de Gradiente Descendente Estocástico (SGD com $\eta = 0.1$) em um lote com 4 imagens canônicas (Diagonal, Horizontal, Preta e Anti-diagonal), observei uma notável emergência geométrica nos pesos convergidos:
$$\mathbf{W}_{\text{aprendido}} \approx \begin{bmatrix} +0.315 & -0.320 & +0.038 & +0.674 \end{bmatrix}, \quad b_{\text{aprendido}} \approx +0.006$$

- **Pesos Excitadores ($w_1 \approx +0.315$, $w_4 \approx +0.674$):** O gradiente amplificou os canais correspondentes aos pixels da diagonal principal. A soma de ambos ($0.315 + 0.674 \approx 0.99$) atinge exatamente o alvo unitário desejado.
- **O Peso Inibitório como Filtro Anti-Confusão ($w_2 \approx -0.320$):** O neurônio descobriu sozinho que uma linha horizontal também acende o pixel superior esquerdo ($p_1=1$). Para não confundir uma reta horizontal com uma diagonal, ele atribuiu um **peso negativo e de magnitude equivalente** ao pixel vizinho ($p_2$). Assim, na imagem horizontal ($[1, 1, 0, 0]$), a ativação do segundo pixel cancela o primeiro ($+0.315 - 0.320 \approx -0.005$), anulando a saída na ReLU!
- **Convergência das Predições Finais:**
  - Imagem Diagonal: **$0.996$** (Alvo: $1.0$ — Detecção confirmada);
  - Imagem Horizontal: **$0.001$** (Alvo: $0.0$);
  - Imagem Preta: **$0.006$** (Alvo: $0.0$);
  - Imagem Anti-diagonal: **$0.000$** (Alvo: $0.0$).

Este microcosmo me comprovou que os neurônios aprendem "antídotos" (pesos inibitórios) para diferenciar traços conflitantes, princípio idêntico ao que o MLP e a CNN utilizarão nas 784 entradas do MNIST para discriminar dígitos morfologicamente semelhantes como o $4$ e o $9$, ou o $3$ e o $5$.

### 4.6. Concepção e Arquitetura do Estudo Dirigido Interativo (`estudo_dirigido.html`)

Para aprofundar e consolidar o meu entendimento conceitual e mecânico do modelo neural antes de iniciar a codificação em larga escala no PyTorch, eu pedi ao Antigravity para criar comigo uma ferramenta web interativa de estudo dirigido ([`estudo_dirigido.html`](./estudo_dirigido.html)).

A minha motivação foi dispor de um ambiente de experimentação visual onde eu pudesse testar na prática as especificações matemáticas extraídas diretamente dos livros de referência da matéria — em especial a formulação do Perceptron e coordenadas homogêneas de **Russell & Norvig (Cap. 22)** e a dinâmica de retropropagação, funções de custo e camadas densas de **Smola et al. (*D2L*, Cap. 4)**. Meu objetivo foi poder alterar valores de entrada e pesos e ver, simultaneamente e em tempo real:

1. **A Dupla Representação Pareada (Biológica vs. Tensorial):** Visualizar de um lado o diagrama arquitetural com nós e conexões sinápticas coloridas e, do outro, a formulação matricial rigorosa em estilo LaTeX com colchetes sólidos ($\begin{bmatrix} \dots \end{bmatrix}$), compreendendo como cada perninha sináptica se mapeia diretamente em um elemento de vetor ou linha de matriz contígua em memória.
2. **Execução Passo a Passo do Fluxo de Informação (*Stepper* Dinâmico):** Ter um controle de fases com animação por onde eu pudesse avançar e retroceder para acompanhar:
   - Os pulsos elétricos de dados ($x_i \cdot w_i$) trafegando pelas conexões sinápticas;
   - A soma afim de redução no núcleo celular ($\sum$);
   - O teste de disparo da comporta não-linear (ReLU);
   - A tensão mecânica da mola elástica contra o alvo gabarito ($\delta = \hat{y} - y$);
   - A onda reversa de retropropagação (*backpropagation*) propagando a culpa proporcional através da matriz transposta ($\mathbf{W}^\top$);
   - O alívio de tensão e recalibração dos pesos via Gradiente Descendente Estocástico (SGD).
3. **Escalaridade em Três Níveis de Abstração:**
   - **Nível 1 (O Perceptron Simples):** O neurônio isolado, a entrada dummy $x_0 = +1.0$ e o produto escalar afim $\mathbf{w}^\top \mathbf{x}$;
   - **Nível 2 (A Rede Multicamadas):** Duas camadas completas, demonstrando o empilhamento das linhas na matriz $\mathbf{W}_1$ ($2 \times 2$), o vetor de ativações intermediárias $\mathbf{a}_1$ e o retorno do erro pela transposta $\mathbf{W}_2^\top$;
   - **Nível 3 (O Estudo de Caso 2×2):** O microcosmo empírico que testei no Jupyter, com grade de pixels interativa clicável, demonstração em tempo real da trava do *Dying ReLU* e carregamento dos pesos aprendidos pelo SGD ($w_1=+0.315, w_2=-0.320, w_3=+0.038, w_4=+0.674, b=+0.006$).

Essa ferramenta foi fundamental para o meu aprendizado, permitindo que eu validasse hipóteses, visse o comportamento dos gradientes e tirasse minhas dúvidas conceituais antes de partir para a implementação definitiva.

**Nota posterior (09/11/2026): a página foi reimplementada.** Ao revisar o código, vi que a primeira versão tinha dois problemas:

1. **O gradiente ignorava a derivada da ReLU.** Ele usava $\delta \cdot x_i$ nos três níveis. Com $w = 0{,}1$ e $b = -0{,}5$, a página mostrava o neurônio "acordando" ($w$ de $0{,}1$ para $0{,}2$), o oposto da seção 4.5, onde o gradiente é exatamente zero. Por isso a frase "demonstração em tempo real da trava do *Dying ReLU*" acima não era verdadeira nessa versão.
2. **Os pesos nunca eram atualizados.** O passo 6 mostrava o peso novo, mas só como conta. Não existia botão que gravasse a atualização, então a página não treinava.

A nova versão usa um motor numérico separado (`estudo_dirigido_motor.js`, na mesma pasta) com $\delta_L = (a - y) \odot f'(z)$ e $\delta_l = (\mathbf{W}_{l+1}^\top \delta_{l+1}) \odot \text{ReLU}'(z_l)$. O que mudou para quem usa:

- **Tudo é editável e recalcula na hora.** Os valores de $\mathbf{W}$, $b$ e das entradas, o alvo e a taxa $\eta$ podem ser mudados, e o diagrama, as matrizes e o gráfico se atualizam.
- **"Aplicar 1 passo", "Treinar 20 passos" e "Auto"** gravam a atualização de verdade. O gráfico mostra a perda por passo.
- **Neurônio morto fica visível.** Com $z \le 0$ o neurônio fica tracejado e cinza, o fio mostra "$\nabla$ 0", e um aviso explica que o erro existe mas não chega aos pesos. Com os pesos "morto" ($w = 0{,}1$, $b = -0{,}5$) o treino não muda nenhum peso, e a curva da perda fica reta. Com $b = 0$ o mesmo neurônio aprende e o peso do pixel 2 fica negativo, como na seção 4.5.
- **As fases** (ida, perda, volta, atualização) destacam cada etapa nas matrizes e nos fios, sem esconder os números das outras.
- **Escolha da função de ativação.** Cada experimento tem um seletor (ReLU, tanh ou logística; a saída também aceita linear), e no experimento de duas camadas as ocultas e a saída são independentes. A diferença que a página torna visível: na ReLU, $z \le 0$ dá derivada exatamente $0$ (neurônio morto). Na tanh e na logística a derivada nunca chega a $0$, mas fica minúscula longe de $z = 0$ (neurônio saturado, com $\sigma'(z) \le 0{,}25$). Com os pesos "Morto" ($w = 0{,}1$, $b = -0{,}5$, $z = -0{,}3$) a logística não morre e o treino aprende, ao contrário da ReLU. A saída da logística fica em $(0, 1)$ e a da tanh em $(-1, 1)$, e a página avisa quando o alvo $y$ está fora da faixa.
- **Valores maiores nas barras, com a faixa seguindo a ativação.** Com ReLU ou saída linear, o alvo $y$ vai de $1$ a $100$ e as entradas $x$ de $0$ a $20$ (experimentos 1 e 2), para ver taxas altas divergirem. Medi, com pesos sorteados, que essa faixa não serve para tanh e logística: com $x$ até $20$, $83\%$ dos neurônios tanh e $68\%$ dos logísticos saturam, porque a tanh só tem derivada útil para $|z| < 1{,}8$ e a logística para $|z| < 3{,}6$. Por isso, com essas funções as entradas passam a ir de $-3$ a $3$, e o alvo fica na faixa da saída ($0$ a $1$ na logística, $-1$ a $1$ na tanh). É o mesmo motivo da normalização do MNIST (seção 3.2). A ReLU, com pesos sorteados, nasce morta em cerca de $50\%$ dos sorteios, e isso é o Dying ReLU, não um problema de escala. Uma opção marcável faz o sorteio repetir até todos os neurônios ficarem vivos.
- **O viés como entrada fixa.** Um botão mostra ou oculta o nó "+1" no diagrama e a coluna $b$ de $[b \mid W]$ nas matrizes ($z = [b \mid W] \cdot [1 ; a]$ ou $z = W a + b$).
- A página não depende mais de nenhum arquivo externo (a primeira versão carregava o Tailwind de uma URL do Antigravity).

**Como verifiquei.** Os testes do motor (`node tests/estudo_dirigido_motor.test.js`, 18 testes) conferem o gradiente por diferenças finitas, o caso do Dying ReLU, o bloqueio da culpa em neurônio oculto morto e as predições com os pesos do Jupyter ($0{,}995$; $0{,}001$; $0{,}006$; $0$). Um segundo teste (`tests/estudo_dirigido_pagina.test.js`, 41 testes, precisa do `jsdom`) executa a página numa DOM simulada e confere os controles e os avisos. Esses testes não substituem abrir a página no navegador para avaliar o visual. As medidas da seção 4.6.1 saem de `node tests/estudo_dirigido_medidas.js`.

#### 4.6.1. Situações Identificadas nos Experimentos com a Ferramenta de Estudo Dirigido

Depois de reimplementar a página (nota acima), usei os experimentos para medir três coisas que antes eu só supunha. Os números vêm de `node tests/estudo_dirigido_medidas.js`, com semente fixa, que roda o mesmo motor da página. São medidas de redes pequenas, não do MNIST.

##### 4.6.1.1. Neurônio morto (ReLU) e neurônio saturado (tanh, logística) não são a mesma coisa

| | ReLU | tanh | logística |
| :--- | :---: | :---: | :---: |
| Derivada $f'(z)$ | $1$ se $z > 0$, senão **exatamente $0$** | $1 - \tanh^2(z)$, máx. $1$ | $\sigma(z)(1 - \sigma(z))$, máx. $0{,}25$ |
| Estado ruim | Morto | Saturado | Saturado |
| Gradiente no estado ruim | $0$ | minúsculo, **não zero** | minúsculo, **não zero** |
| Consegue sair sozinho? | Não, com a entrada e os pesos atuais | Sim, devagar | Sim, devagar |

Chamo de "saturado" o neurônio cuja derivada caiu abaixo de $10\%$ do máximo. Esse limiar é uma convenção minha para a página, não um valor da literatura. Com ele, a tanh fica viva para $|z| < 1{,}81$ e a logística para $|z| < 3{,}63$. Com os pesos "Morto" do experimento 3 ($w = 0{,}1$, $b = -0{,}5$, $z = -0{,}3$), a ReLU não aprende nada em $100$ passos e a logística aprende, pois a derivada dela em $z = -0{,}3$ não é zero.

##### 4.6.1.2. A escala da entrada decide se tanh e logística funcionam

Experimento 1 com pesos e viés sorteados em $[-1, 1]$, $20.000$ sorteios por célula:

| $x$ máximo | ReLU morta | tanh saturada | logística saturada |
| :---: | :---: | :---: | :---: |
| $1$ | $50{,}7\%$ | $1{,}2\%$ | $0{,}0\%$ |
| $4$ | $49{,}8\%$ | $34{,}7\%$ | $7{,}1\%$ |
| $20$ | $50{,}3\%$ | $83{,}4\%$ | $68{,}4\%$ |

- A ReLU nasce morta em cerca de $50\%$ dos sorteios, qualquer que seja a escala. A razão é a simetria do sorteio: o sinal de $z$ é aleatório. No detector $2 \times 2$ deu $50\%$ também. Isso é o Dying ReLU, não um defeito de escala.
- Tanh e logística dependem da escala. Com $x$ até $20$, mais de dois terços saturam. É o mesmo motivo da normalização do MNIST (seção 3.2): média $0$ e desvio $1$ mantêm o $z$ na região em que a derivada é útil.
- Por isso, quando a ativação é tanh ou logística, a página limita as entradas a $[-3, 3]$ e o alvo à faixa da saída (logística: $0$ a $1$; tanh: $-1$ a $1$).

##### 4.6.1.3. Um alvo no limite da faixa deixa o treino lento

A tanh e a logística nunca chegam ao limite da faixa: só se aproximam quando $z$ cresce sem parar. Perto do limite a derivada vai a zero, e o gradiente $\delta = (a - y) \cdot f'(z)$ encolhe duas vezes, pelo erro e pela derivada. Medi na rede $2 \to 2 \to 1$ (tanh nas duas camadas, $x = [1, -1]$, $\eta = 0{,}2$) que montei a partir de uma captura de tela da página:

- Estado da captura, alvo $y = 1$: saída $0{,}962$, $z = 1{,}97$, $\tanh'(z) = 0{,}075$. O erro é $-0{,}04$ e $\delta_{\text{saída}} = -0{,}04 \times 0{,}075 = -2{,}9 \cdot 10^{-3}$. Ao passar pelas camadas, $\delta_A = 3{,}5 \cdot 10^{-5}$ e $\delta_B = -1{,}0 \cdot 10^{-3}$.
- Com esses pesos, para a perda ficar abaixo de $10^{-4}$:

| Alvo $y$ | Perda inicial | Passos |
| :---: | :---: | :---: |
| $0{,}50$ | $0{,}1066$ | $40$ |
| $0{,}80$ | $0{,}0131$ | $103$ |
| $0{,}90$ | $0{,}0019$ | $181$ |
| $0{,}99$ | $0{,}0004$ | $370$ |
| $1{,}00$ | $0{,}0007$ | $999$ |

  (O alvo $0{,}95$ já começa abaixo de $10^{-4}$ e não entra na comparação.) Quanto mais perto do limite, mais passos, mesmo com perda inicial menor.
- Com os mesmos pesos e o alvo no lado oposto ($y = -1$, saturado do lado errado), a perda cai abaixo de $0{,}01$ em $23$ passos ($\eta = 0{,}2$) ou $7$ passos ($\eta = 0{,}5$). Com ReLU e saída linear, $12$ passos. Ou seja, o neurônio saturado do lado errado ainda aprende rápido, porque o erro é grande. O lento é o alvo colado ao limite.

##### 4.6.1.4. Limites das medidas e relação com a seção 4.12

**Limites do que medi.** São uma rede, uma taxa e um estado inicial (determinístico, sem sorteio no treino). Não testei isso no MNIST. A página, por enquanto, ainda deixa o alvo ir até o limite da faixa da função.

**Relação com a seção 4.12.** A saída do v3 era linear (sem ativação), então $f'(z) = 1$ e o mecanismo acima não explica a lentidão do v3 em relação ao v4. Ele explicaria uma rede com MSE depois de uma tanh ou logística com alvos $0$ e $1$. É o caso que a entropia cruzada com Softmax evita: o gradiente dela na saída é $p - y$, sem o fator $f'(z)$.

### 4.7. O Salto Conceitual do One-Hot: Tratando Rótulos como Classes Categóricas e Não Grandezas Numéricas

Durante os primeiros testes práticos com a arquitetura MLP sobre o MNIST, me surgiu uma dúvida fundamental sobre a mecânica de cálculo da função de perda:
- Ao analisar a discrepância de erro $|y_{\text{pred}} - y|$, comecei a suspeitar: **o que estaria distorcendo a convergência do aprendizado? Seria a função de ativação ReLU? Seria a formulação da função de perda quadrática?**
- Foi aí que me veio o estalo: **percebi que estávamos tratando os dígitos manuscritos como grandezas numéricas escalares contínuas (números com magnitude e ordem), e não como elementos abstratos de um conjunto de classes equivalentes (onde todos os dígitos possuem exatamente o mesmo peso e importância).**

#### A Falácia Métrica da Saída Escalar
Se a rede possuísse apenas 1 neurônio de saída treinado para prever um escalar contínuo $y \in \{0, 1, \dots, 9\}$, uma métrica de distância ordinária $|y_{\text{pred}} - y|$ introduziria uma topologia artificial e desastrosa no espaço de busca:
1. **Distorção de Penalidade:** Se a imagem apresentada for um $0$ e a rede predizer $7$, o erro é $|7 - 0| = 7$; mas se predizer $1$, o erro é $|1 - 0| = 1$. O algoritmo de gradiente concluiria erroneamente que errar um zero dizendo que é sete é **7 vezes mais grave** do que dizer que é um, embora do ponto de vista morfológico e de classificação ambas sejam confusões categóricas com o mesmo custo prático.
2. **Falsa Média Geométrica:** Um número intermediário como o $4$ seria interpretado como o "ponto médio" entre o $1$ e o $7$, o que não possui nenhum fundamento semântico ou caligráfico.

#### A Formalização em $\mathbb{R}^{10}$ e a Conexão com LeCun (1998)
Para que todas as 10 classes possuam rigorosamente o mesmo peso e a mesma distância geométrica mútua, compreendi que as classes precisam ser **ortogonais no espaço**:
- Cada classe $c \in \{0, \dots, 9\}$ é projetada em um vetor unitário da base canônica em $\mathbb{R}^{10}$ (**codificação One-Hot** ou *"1-of-K coding"*):
  $$\mathbf{y}_{\text{one-hot}} = \mathbf{e}_c = \begin{bmatrix} 0 & \dots & 1.0 & \dots & 0 \end{bmatrix}$$
- Essa representação está no cerne do trabalho seminal de **LeCun et al. (1998)**:
  - Na época de LeCun, a ativação utilizada na saída era a tangente hiperbólica ($\tanh \in [-1, +1]$), o que levava a um alvo bipolar ($+1.0$ para a classe correta e $-1.0$ para as demais);
  - Na formulação moderna que adotei aqui, com ReLU, utilizamos $1.0$ para o canal excitado e $0.0$ para os canais inibidos (`F.one_hot`);
- **Resultado:** A camada de saída passa a ter **10 neurônios competidores**, cada um responsável por estimar a pertinência da imagem a um dígito específico. A distância euclidiana entre quaisquer duas classes distintas passa a ser estritamente constante ($\|\mathbf{e}_i - \mathbf{e}_j\| = \sqrt{2}$ para todo $i \neq j$), eliminando qualquer hierarquia numérica arbitrária e permitindo que o modelo aprenda filtros independentes para cada classe.

#### 4.7.1. Validação Empírica do Benchmark de LeCun (1998)
Com o One-Hot devidamente formulado e utilizando a perda quadrática MSE, implementei o script [`mlp_v1_treinar.py`](./mlp_v1_treinar.py) e executei no meu terminal os dois testes clássicos de rede densa documentados na Tabela 1 de LeCun et al. (1998):

1. **MLP de 1 Camada Oculta ($784 \to 300 \to 10$):**
   - Total de $238.510$ parâmetros treináveis;
   - Treinado com SGD puro ($\eta = 0.1$) por 3 épocas (tempo: $\approx 25$ segundos);
   - **Resultado no Teste Oficial (10.000 imagens):** $9.541$ acertos (**$95{,}41\%$ de acurácia**, erro de **$4{,}59\%$**);
   - **Conexão com a literatura:** Bateu quase exatamente com o benchmark histórico publicado por LeCun em 1998 (**$4{,}7\%$ de erro**).
2. **MLP de 2 Camadas Ocultas ($784 \to 300 \to 100 \to 10$):**
   - Total de $266.610$ parâmetros treináveis;
   - **Resultado no Teste Oficial:** $9.478$ acertos (**$94{,}78\%$ de acurácia**, erro de **$5{,}22\%$**);
   - **Diagnóstico:** Em poucas épocas (apenas 3) e sem decaimento de taxa de aprendizado, a rede mais profunda demora ligeiramente mais para coordenar as duas camadas intermediárias. LeCun precisou de mais épocas e programação de $\eta$ para convergir a $3{,}05\%$.

Este experimento serviu como a linha de base (*baseline*) inicial para o nosso projeto, indicando que o pipeline está íntegro e reproduz o resultado da literatura canônica.

### 4.8. A Investigação de Desempenho na CPU: A Descoberta do Gargalo Oculto no Pipeline de Dados

Após validar o modelo baseline de LeCun e adicionar o suporte a persistência em disco em [`mlp_v2_treinar.py`](./mlp_v2_treinar.py), executei o script no meu terminal monitorando o tempo pelo utilitário `time`. O script atingiu **$95{,}58\%$ de acurácia** no teste oficial (10.000 imagens), porém os tempos de execução reportados acenderam um sinal de alerta:

```text
Tempo de Treinamento (3 épocas): 19.78 s
real    0m23,645s
user    1m49,853s
sys     0m3,392s
```

#### A Inquietação: O Paradoxo da CPU Moderna
A máquina utilizada para o experimento possui um processador **Intel® Core™ i7-11390H @ 3.40GHz** (4 núcleos físicos, 8 threads e instruções vetoriais avançadas AVX-512) sem placa de vídeo NVIDIA dedicada. 

Ao fazer uma conta de ordem de grandeza da física do problema:
- O modelo possui apenas $238.510$ parâmetros escalares ($784 \to 300 \to 10$);
- Para cada imagem, o *forward* e o *backward* realizam apenas multiplicações de matrizes pequenas com batches de 64;
- O tempo de CPU `user` acusou **$1\text{m}49\text{s}$** em apenas $23\text{s}$ de relógio (`real`), evidenciando que todos os núcleos físicos da CPU estavam a plena carga ($109\text{s} / 23{,}6\text{s} \approx 4{,}6\times$ paralelismo de OpenMP).

A pergunta que me fiz foi: *"Por que um processador com AVX-512 capaz de bilhões de operações por segundo precisa de quase 20 segundos de treinamento para calcular três épocas de uma rede tão leve?"*.

#### O Profiling Empírico e o Isolamento do Gargalo
Para entender onde a CPU estava gastando tempo, decidi decompor o script e medir cada parte isoladamente:

1. **Teste 1 — Puro Forward + Backward da MLP:** Alimentamos a rede diretamente na memória sem o `DataLoader`, apenas medindo a matemática do gradiente para as 3 épocas:
   $$\text{Tempo gasto:} \quad \mathbf{2{,}37\text{ segundos}}$$
   **O Estalo:** O cálculo real dos pesos da rede neural levou apenas $2{,}37$ segundos! Os outros **$17{,}41$ segundos** (mais de $88\%$ do tempo total de treino) não tinham relação com a rede neural nem com o SGD.
2. **Teste 2 — A Esteira de Dados (`dataset.py` e `DataLoader`):** Cronometramos exclusivamente a leitura dos batches no dataset padrão do `torchvision`:
   - `num_workers = 0`: $6{,}14\text{ s}$ por época;
   - `num_workers = 2`: $4{,}47\text{ s}$ por época;
   - `num_workers = 4`: $3{,}62\text{ s}$ por época.
   
Multiplicando $\approx 4{,}5\text{ s}$ por época pelas 3 épocas de treino ($\approx 13{,}5\text{ s}$) somadas às passagens de validação ($\approx 3{,}5\text{ s}$), encontramos de onde vinha o tempo perdido.

#### A Causa Raiz
No código herdado do `torchvision`, o `DataLoader` busca cada uma das 50.000 imagens individualmente: a cada época, executa em espaço Python as chamadas `transforms.ToTensor()` e `transforms.Normalize()` imagem por imagem, transferindo tensores entre processos via IPC. O processador gastava mais de $80\%$ de seus ciclos convertendo imagens, e não treinando a inteligência da rede.

Além disso, identifiquei dois atritos menores no código:
- A alocação manual de matrizes de zeros com `torch.zeros().scatter_()` a cada mini-batch para o One-Hot;
- A conversão desnecessária de 10.000 tensores para listas Python (`.cpu().tolist()`) a cada época dentro de `avaliar_mse()`, cujos resultados eram descartados no treino com `_, _`.

#### A Solução: Adoção do `TensorDataset` em Memória RAM
A pergunta de engenharia foi imediata:

> *"O conjunto completo do MNIST tem 60.000 imagens de $28 \times 28$ pixels. Em ponto flutuante de 32 bits, isso ocupa meros **$150\text{ MB}$ de memória RAM**. Por que normalizar cada imagem repetidamente no disco e na CPU a cada época, se podemos manter o dataset inteiro pré-normalizado na memória principal?"*

Implementamos uma função de pré-carregamento contíguo em RAM:
```python
def dataset_para_ram(dataset: Dataset) -> TensorDataset:
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    imagens, rotulos = next(iter(loader))
    return TensorDataset(imagens, rotulos)
```

E otimizamos o One-Hot com a primitiva nativa em C++ `torch.nn.functional.one_hot(rotulos, 10).float()`.

#### Resultados Comparativos
Validamos a técnica diretamente no mesmo hardware:

| Métrica | Baseline (`mlp_v2_treinar.py`) | Otimizado (`TensorDataset`) | Ganho Obtido |
| :--- | :---: | :---: | :---: |
| **Tempo de Treinamento (3 épocas)** | **$19{,}78\text{ s}$** | **$4{,}76\text{ s}$** | **$4{,}15\times$ mais rápido ($-75{,}9\%$)** |
| **Tempo Médio por Época** | $\approx 6{,}59\text{ s}$ | **$\approx 1{,}58\text{ s}$** | **$-5{,}01\text{ s}$ / época** |
| **Iteração dos Dados (3 épocas)** | $\approx 17{,}41\text{ s}$ | **$0{,}84\text{ s}$** | **$20{,}7\times$ mais rápido** |
| **Pegada de Memória Adicional** | Desprezível | **$\approx 150\text{ MB}$** | Irrisório para sistemas modernos |
| **Acurácia no Teste Oficial** | $95{,}58\%$ | **$95{,}38\%$** | Idêntica (variação estocástica normal) |

Essa descoberta reforçou na prática um dos princípios vitais da computação de alto desempenho em Aprendizado Profundo: **a CPU ou GPU só é tão rápida quanto o canal que a alimenta**. Tirando o gargalo de I/O em nível de amostra, o treinamento de uma MLP clássica atinge tempos da ordem de fração de segundo por época mesmo rodando exclusivamente em CPU.

#### 4.8.1. A Dinâmica da Convergência e o Experimento de Escala (5 vs. 50 vs. 120 Épocas)

Com a alta velocidade de processamento viabilizada pelo `TensorDataset`, criei junto com o Antigravity o script [`mlp_v3_treinar.py`](./mlp_v3_treinar.py), permitindo passar o número de épocas via argumento de linha de comando (`-e` ou `--epocas`).

Executei três rodadas experimentais completas no terminal para investigar a dinâmica temporal e de convergência da rede MLP ($784 \to 300 \to 10$ com MSE e SGD $\eta = 0.1$):

| Épocas ($E$) | Tempo de Treino | Taxa Temporal | Acurácia Treino | Acurácia Validação | Acurácia Teste (10k) | Erro no Teste | Checkpoint Salvo |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5** | $6{,}19\text{ s}$ | $1{,}24\text{ s}$ / época | $96{,}21\%$ | $95{,}70\%$ | $95{,}95\%$ | $4{,}05\%$ | `mlp_treinado_v3_e5.pth` |
| **50** | $61{,}42\text{ s}$ | $1{,}23\text{ s}$ / época | $98{,}93\%$ | $97{,}46\%$ | $97{,}76\%$ | $2{,}24\%$ | `mlp_treinado_v3_e50.pth` |
| **120** | $155{,}30\text{ s}$ | $1{,}29\text{ s}$ / época | $99{,}60\%$ | $97{,}75\%$ | **$97{,}93\%$** | **$2{,}07\%$** | `mlp_treinado_v3_e120.pth` |

#### Descobertas e Diagnósticos Teóricos:
1. **Linearidade Temporal Perfeita:** A taxa média manteve-se cravada entre $1{,}23\text{ s}$ e $1{,}29\text{ s}$ por época de ponta a ponta. Treinar 120 épocas completas (processando $6.000.000$ de amostras) levou apenas **$2\text{m}35\text{s}$ na CPU**.
2. **Corte do Erro Pela Metade:** Enquanto nas 3 épocas iniciais de LeCun o erro de teste era de $4{,}59\%$, o treinamento estendido por 120 épocas fez o erro cair para **$2{,}07\%$** (apenas 207 falhas em 10.000 imagens).
3. **Identificação do Platô de Validação (Zona de Parada):**
   - Entre as épocas 1 e 50, a perda de validação despenca rapidamente de $0{,}022$ para $0{,}0080$.
   - A partir da época 60, o conjunto de treino continua aprendendo detalhes finos (subindo até $99{,}60\%$), mas a validação entra em um platô assintótico ($\approx 97{,}75\%$). Isso delimita claramente que, para esta topologia de 300 neurônios sem regularização adicional, entre **50 e 80 épocas** situa-se a zona de eficiência máxima de aprendizado antes do início do sobreajuste (*overfitting*).

#### Reflexão Epistemológica: O Fenômeno de *Grokking* se Aplicaria Aqui?
A partir desse experimento de treinamento prolongado, me surgiu uma indagação científica natural (inspirada no artigo seminal de Power et al., 2022 — *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, arXiv:2201.02177):  
*"Se deixássemos esse modelo treinando por milhares ou dezenas de milhares de épocas, muito além do platô de overfitting, a rede passaria por um salto tardio de generalização (Grokking) ou apenas sofreria a saturação clássica do overfitting?"*

#### A Conexão com o Mini-Projeto 1: O Sistema de Duas Molas e a Força Gravitacional do Weight Decay
Essa reflexão conecta-se diretamente com o modelo de física mecânica que construímos no **Mini-Projeto 1** (onde tratamos o vetor nulo $\mathbf{0}$ como um centro de atração gravitacional) e na **Seção 4.1 deste documento** (onde definimos o erro quadrático como a energia elástica de uma mola pela Lei de Hooke):

1. **A Primeira Mola (A Mola dos Dados):**  
   A função de perda do erro $\mathcal{L}_{\text{dados}} = \frac{1}{2}(\hat{y} - y)^2$ exerce uma força restauradora $F_{\text{dados}} = -(\hat{y} - y)$ que deforma e traciona os pesos para que a saída encoste no gabarito das imagens.
2. **A Segunda Mola (A Mola Gravitacional do Weight Decay):**  
   Quando adicionamos a regularização $L_2$ (Weight Decay), a função de perda total passa a ser:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{dados}} + \frac{1}{2} \lambda \|\mathbf{W}\|_2^2$$
   Ao calcular o gradiente para o passo do SGD, a derivada do termo de regularização em relação ao peso é simplesmente:
   $$\nabla_\mathbf{W} \left(\frac{1}{2} \lambda \|\mathbf{W}\|_2^2\right) = \lambda \mathbf{W} \quad \implies \quad \mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla \mathcal{L}_{\text{dados}} - \underbrace{\eta \lambda \mathbf{W}_t}_{\text{Força Restauradora da Origem}}$$
   O termo $-\lambda \mathbf{W}$ é **rigorosamente a Lei de Hooke ($F = -k x$) centrada na origem $\mathbf{0}$!** Trata-se de uma mola gravitacional elástica ancorada no ponto zero do espaço vetorial, que puxa continuamente todos os pesos contíguos de volta para o centro de massa, punindo a magnitude desmedida das conexões.

#### Como as Duas Molas Explicam o Grokking
Essa leitura mecânica desvenda com precisão a física do artigo de Grokking da OpenAI:
- **Fase 1 (Memorização Desordenada):** Em problemas algorítmicos com poucos dados, a *Mola dos Dados* é muito forte e estica os pesos em direções caóticas até zerar o erro de treino. Nesse ponto, a rede decorou tudo, mas os pesos estão distantes da origem, formando uma treliça elástica hiperestendida e desordenada (acurácia de validação nula).
- **Fase 2 (O Trabalho Silencioso da Mola Gravitacional):** Quando o erro de treino chega a zero ($\nabla \mathcal{L}_{\text{dados}} \to 0$), a *Mola dos Dados* relaxa e para de puxar! É aí que a *Mola Gravitacional do Weight Decay* ($-\lambda \mathbf{W}$) assume o comando absoluto da física do sistema.
- **A Transição de Fase (O Grokking):** Durante dezenas de milhares de passos em que o erro de treino permanece em zero, essa mola gravitacional continua puxando os pesos silenciosamente em direção à origem, "espremendo" os parâmetros e encolhendo os vetores supérfluos. Ela obriga a rede a encontrar a configuração geométrica de **mínima energia potencial global** — que é exatamente o circuito simétrico puro (ex.: representação circular de Fourier) capaz de generalizar perfeitamente para amostras nunca vistas. Sem essa mola centrípeta na origem, o Grokking **jamais acontece**.

#### O Contraste com o Nosso Experimento MNIST
Compreendida essa mecânica, fica evidente por que no nosso experimento com o MNIST ocorre o platô clássico e não o Grokking:
1. **SGD Puro sem Mola Gravitacional ($\lambda = 0$):** No nosso script `mlp_v3_treinar.py`, treinamos com SGD padrão sem regularização $L_2$ (`weight_decay = 0`). Assim que a *Mola dos Dados* relaxa ao atingir o fundo do vale (loss $\approx 0{,}0037$), não existe nenhuma força elástica na origem puxando os pesos de volta para comprimir a rede em um circuito mínimo; os pesos ficam livres para passear em deriva pelas regiões planas do vale.
2. **Volume de Amostras e Regime Clássico:** Além disso, com 50.000 imagens ricas e contínuas, a *Mola dos Dados* é alimentada por tantas restrições simultâneas que os pesos são guiados para uma bacia generalizável desde a época 1 (batendo $92\%$ de validação logo de saída), sem precisar passar pela fase de memorização cega.
3. **Teto de Ruído (Limite de Bayes):** Ao contrário da aritmética modular (que é uma verdade lógica exata $100\%$ pura), o MNIST tem limites humanos na caligrafia de dígitos truncados. Estender para 10.000 épocas sem regularização apenas faria a rede usar sua capacidade livre para memorizar micro-ruídos estocásticos de fundo.

Portanto, a nossa analogia física das molas e da atração gravitacional do Mini-Projeto 1 não apenas se aplica perfeitamente, como é **o próprio motor físico fundamental** que explica por que o Grokking existe na presença de regularização e por que o nosso modelo converge para o platô assintótico clássico.

### 4.9. Evidência Empírica de Overfitting e Justificativa do Early Stopping (Teste de 500 Épocas)

Para testar no limite a nossa hipótese teórica formulada na seção anterior — de que estender o treinamento indefinidamente não traria saltos tardios de generalização e apenas provocaria o sobreajuste clássico —, decidi executar um teste de estresse extremo: **500 épocas ininterruptas** utilizando o script [`mlp_v3_treinar.py`](./mlp_v3_treinar.py):

```bash
python mlp_v3_treinar.py -e 500
```

O treinamento processou 25 milhões de passagens de imagem na CPU em **$766{,}37\text{ segundos}$** ($\approx 12\text{ minutos e }46\text{ segundos}$), mantendo taxa estável de $1{,}53\text{ s}$ por época. Em seguida, avaliei o arquivo gerado ([`mlp_treinado_v3_e500.pth`](./modelo/mlp_treinado_v3_e500.pth)) no conjunto de teste oficial lacrado utilizando [`mlp_v3_testar.py`](./mlp_v3_testar.py):

```bash
python mlp_v3_testar.py -c mlp_treinado_v3_e500.pth
```

#### 4.9.1. O Confronto dos Dados: 120 Épocas (Ótimo Global) vs. 500 Épocas (Overfitting)

A tabela comparativa de toda a série experimental ilustra com precisão a curva de saturação e a posterior degradação da rede:

| Épocas ($E$) | Tempo Total | Loss Treino (MSE) | Acurácia Treino | Loss Validação (MSE) | Acurácia Validação | Loss Teste (MSE) | Acurácia Teste (10k) | Erros no Teste | Diagnóstico Metodológico |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **5** | $6{,}19\text{ s}$ | $0{,}01303$ | $96{,}21\%$ | $0{,}01402$ | $95{,}70\%$ | $0{,}01362$ | $95{,}95\%$ | $405$ | Subajuste inicial (*Underfitting*) |
| **50** | $61{,}42\text{ s}$ | $0{,}00557$ | $98{,}93\%$ | $0{,}00818$ | $97{,}46\%$ | $0{,}00778$ | $97{,}76\%$ | $224$ | Zona de aprendizado acelerado |
| **120** | $155{,}30\text{ s}$ | $0{,}00371$ | $99{,}60\%$ | $0{,}00754$ | $97{,}75\%$ | **$0{,}00721$** | **$97{,}93\%$** | **$207$** | **Ponto Ótimo de Generalização** |
| **500** | $766{,}37\text{ s}$ | **$0{,}00178$** | **$99{,}93\%$** | $0{,}00813$ | $97{,}56\%$ | $0{,}00789$ | $97{,}72\%$ | $228$ | **Sobreajuste (*Overfitting*)** |

Ao isolar o salto de **120 para 500 épocas**, o diagnóstico é claro:
- **No Treino (Memorização Extrema):** A perda de treino despencou para menos da metade ($0{,}00371 \to 0{,}00178$) e a acurácia subiu para **$99{,}93\%$** (a rede acertou praticamente todas as 50.000 imagens de treino).
- **Na Validação (Deterioração):** A perda de validação subiu de $0{,}00754$ para $0{,}00813$ (piora de $+7{,}8\%$) e a acurácia recuou de $97{,}75\%$ para $97{,}56\%$.
- **No Teste Oficial Lacrado (Piora da Generalização Real):** A acurácia caiu de **$97{,}93\%$ para $97{,}72\%$**, resultando em **$+21$ imagens adicionais erradas** (o total de falhas saltou de $207$ para $228$).

#### 4.9.2. A Curva em "U" da Perda de Validação: O Livro-Texto em Ação

Ao inspecionar o log das 500 épocas gerado no terminal, comprovei na prática a formação da **curva em formato de "U"** preconizada pela literatura clássica de Aprendizado de Máquina (Russell & Norvig, Cap. 19; Smola et al., Cap. 4.4):

1. **Fase Descendente (Épocas 1 a 120):** A perda de validação despencou rapidamente de $0{,}02161$ (época 1) para $0{,}00818$ (época 50) e $0{,}00754$ (época 120), acompanhando a descida contínua da perda de treino.
2. **O Fundo do Vale (Mínimo Global):** A perda de validação atingiu seu ponto mais baixo da história do modelo exatamente na **época 172**:
   $$\mathcal{L}_{\text{val}}^{\min} = \mathbf{0{,}00745} \quad (\text{Época 172})$$
3. **Fase Ascendente de Degradação (Épocas 173 a 500):** A partir da época 173, a perda de validação inverteu a inclinação da reta tangente e passou a **subir gradualmente**, fechando a época 500 em **$0{,}00813$**.

Enquanto o otimizador continuava acreditando que o modelo estava melhorando (porque o erro de treino caía rumo a zero), o modelo estava, na realidade, piorando a sua capacidade preditiva em relação ao mundo exterior. Trata-se da demonstração empírica mais límpida do sobreajuste (*overfitting*).

#### 4.9.3. Por que Treinar 2.000 ou 10.000 Épocas Seria Desperdício Computacional Puro?

A partir desses dados, me fiz a seguinte pergunta crítica: *"Nesse modelo, treinar 2.000 ou 10.000 épocas não faria diferença nenhuma, correto?"*.

A resposta empírica e teórica é um categórico **sim**. Insistir em milhares de épocas seria mero desperdício de energia e ciclos de CPU, por três fundamentos matemáticos:

1. **O Teto da Capacidade Representacional ($784 \to 300 \to 10$):**  
   Uma camada oculta com 300 neurônios e ativações ReLU projeta 300 hiperplanos lineares de corte no espaço de 784 dimensões. Em 120 a 170 épocas de SGD, todos esses hiperplanos já se acomodaram nos melhores vales possíveis para separar os 10 dígitos. O modelo esgotou sua capacidade de representação linear-por-partes (*representational ceiling*). Rodar mais 5.000 épocas não cria neurônios novos nem introduz novas dimensões; apenas retorce as margens existentes.
2. **O Limite de Erro de Bayes e o Ruído Intrínseco do MNIST:**  
   O MNIST não é aritmética booleana exata; é caligrafia humana real. O conjunto contém ambiguidades visuais e traços truncados que mesmo seres humanos têm dificuldade em classificar com 100% de certeza (ex.: dígitos $4$ e $9$ com o topo semiaberto, ou $3$ e $5$ com curvas imperfeitas). Não tenho um limite teórico conhecido para o erro desta MLP (o Erro de Bayes do MNIST não foi medido aqui). A seção 4.14 chegou a $1{,}60\%$ de erro com a mesma família de rede, então os $2{,}07\%$ desta série (v3, MSE, 120 épocas) não são um teto: indicam apenas que, nesta configuração, o ganho com mais épocas se esgota. Qualquer época adicional serve unicamente para a rede "decorar o ruído" dessas amostras anômalas.
3. **A Ausência de Regularização Gravitacional ($L_2$ / Weight Decay):**  
   Como formalizamos na analogia física da Seção 4.8.1, sem o termo de *Weight Decay* ($\lambda = 0$), não existe nenhuma mola elástica centrada na origem $\mathbf{0}$ para recolher os parâmetros e induzir *Grokking*. Na ausência dessa força centrípeta, os pesos vagam em deriva (*drift*) pelas regiões planas do espaço de perda, aumentando desnecessariamente de magnitude e tornando a rede hiper-reativa a ruídos microscópicos de fundo.

#### 4.9.4. A Justificativa Científica do Early Stopping (Critério de Parada)

No escopo da disciplina (conforme solicitado na Seção II do enunciado para a análise de hiperparâmetros e critérios de parada), este experimento justifica a adoção do **Early Stopping** (Parada Precoce):

- **Política Ótima de Parada:** Monitorar a perda de validação ($\mathcal{L}_{\text{val}}$) e salvar o checkpoint dos pesos no ponto de mínimo global ($\approx$ época 120 a 170). Se após um intervalo de tolerância (*patience*, tipicamente 10 a 20 épocas) a perda de validação não renovar sua mínima histórica, interrompe-se o treinamento. *(Nota posterior: a janela de $120$ a $170$ épocas vale para o v3, com MSE, lote $64$ e sem momento. O v5, da seção 4.15, implementa só o primeiro passo desta política: salva o checkpoint de menor perda de validação. Ele não interrompe o treino. Com lote $256$ e momento $0{,}9$, o melhor ponto ficou entre as épocas $2$ e $14$. Ou seja, a janela depende de lote, taxa e momento.)*
- **Eficiência Computacional e Desempenho:** Treinar até a época 120 consumiu **$155\text{ s}$**; estender até a época 500 consumiu **$766\text{ s}$** ($5\times$ mais tempo). O Early Stopping poupa **$80\%$ dos recursos computacionais** e entrega um modelo com desempenho de generalização **melhor** ($97{,}93\%$ vs. $97{,}72\%$, uma execução de cada).

---

### 4.10. Organização Estruturada de Artefatos Gráficos por Subpastas de Épocas (`figuras/e<épocas>/`)

À medida que os experimentos proliferam (com baterias executadas para 3, 50, 120 e 500 épocas), a persistência simultânea de 5 gráficos de treino e 4 relatórios de teste por rodada ameaçava poluir a pasta `figuras/` com dezenas de imagens dispersas.

Adotei, portanto, uma reestruturação arquitetural no subsistema de plotagem:
1. **Pasta Raiz Preservada (`figuras/`):** Mantém exclusivamente os gráficos descritivos universais do conjunto de dados ([`distribuicao_classes.png`](./figuras/distribuicao_classes.png) e [`exemplos_digitos.png`](./figuras/exemplos_digitos.png)).
2. **Subdiretórios Dedicados por Tamanho/Regime (`figuras/e<épocas>/`):**
   - [`figuras/e3/`](./figuras/e3/): Conjunto completo dos 9 gráficos para 3 épocas (treino inicial / baseline rápido).
   - [`figuras/e50/`](./figuras/e50/): Conjunto completo dos 9 gráficos para 50 épocas (zona de convergência ativa).
   - [`figuras/e120/`](./figuras/e120/): Conjunto completo dos 9 gráficos para 120 épocas (ponto ótimo global de generalização).
   - [`figuras/e500/`](./figuras/e500/): Conjunto completo dos 9 gráficos para 500 épocas (estudo extremo de overfitting).
3. **Casamento Automático nos Scripts:**
   - Em [`mlp_v3_treinar.py`](./mlp_v3_treinar.py), a função `salvar_graficos_treinamento` recebe `num_epocas` e direciona a saída para `figuras/e{num_epocas}/`, criando a subpasta sob demanda (`mkdir(parents=True, exist_ok=True)`).
   - Em [`mlp_v3_testar.py`](./mlp_v3_testar.py), o módulo extrai via expressão regular (`r"e\d+"`) o identificador do checkpoint carregado (ex.: `e50` a partir de `mlp_treinado_v3_e50.pth`) e grava os 4 relatórios de teste dentro da mesma subpasta, unificando os 9 artefatos de cada experimento em um diretório isolado, limpo e autoexplicativo.

---

### 4.11. Benchmark de Aceleração em Hardware: CPU (Intel i7) vs. GPU (NVIDIA RTX A1000)

A título de descoberta empírica e validação de infraestrutura computacional, realizei um ensaio comparativo cruzado entre duas máquinas físicas distintas executando o mesmo código-fonte do mini-projeto.

#### 4.11.1. Contexto da Descoberta: Dois Ambientes Físicos e Perfis de Carga

Muitas vezes presume-se que redes neurais pequenas do tipo MLP ($784 \to 300 \to 10$) não se beneficiem expressivamente de placas de vídeo dedicadas, devido ao gargalo de transferência de tensores entre a memória RAM principal do computador e a memória VRAM da GPU via barramento PCI Express (*host-to-device PCIe latency*). 

Para testar essa hipótese na prática, configurei o mesmo pipeline em dois perfis de máquina:

1. **Bancada 1 — CPU Pura (`eu@fantasma`):**
   - **Notebook:** Dell Inspiron 15 5510
   - **Processador:** 11th Gen Intel Core i7-11390H (4 núcleos / 8 threads @ 5.0 GHz)
   - **GPU:** Intel TigerLake-LP GT2 (*Iris Xe Graphics* integrada)
   - **Memória RAM:** 32 GB DDR4
   - **Backend PyTorch:** CPU multicore com otimizações de vetorização AVX-512
2. **Bancada 2 — Aceleração Dedicada em GPU (`rolim@aroeira`):**
   - **Estação de Trabalho Móvel:** Dell Precision 3581
   - **Processador:** 13th Gen Intel Core i7-13800H (14 núcleos / 20 threads @ 5.0 GHz)
   - **GPU Dedicada:** **NVIDIA RTX A1000 Laptop GPU** (Arquitetura Ampere, 6 GB GDDR6 VRAM dedicada, 2.048 núcleos CUDA, 64 Tensor Cores, Compute Capability 8.6)
   - **Driver e CUDA:** NVIDIA Driver `550.163.01` com `PyTorch 2.6.0+cu124`
   - **Memória RAM:** 32 GB DDR5

#### 4.11.2. O Confronto de Desempenho Computacional (CPU vs. GPU)

O ganho de vazão com o processamento paralelo em GPU e a execução assíncrona de kernels CUDA foi substancial:

| Dimensão de Desempenho | CPU (Intel Core i7-11390H) | GPU (NVIDIA RTX A1000 6GB) | Ganho Real (*Speedup*) | Impacto Prático na Engenharia |
| :--- | :---: | :---: | :---: | :--- |
| **Tempo por Época de Treino** | $1{,}53\text{ s}$ | **$0{,}54\text{ s}$** | **$2{,}83\times$ mais rápido** | Quase 3 épocas na GPU no tempo de 1 na CPU |
| **Bateria de 50 Épocas** | $83{,}15\text{ s}$ | **$27{,}68\text{ s}$** | **$3{,}00\times$ mais rápido** | Treinamento interativo em menos de meio minuto |
| **Bateria de 100 Épocas** | $\approx 150\text{ s}$ | **$54{,}21\text{ s}$** | **$2{,}77\times$ mais rápido** | Concluído em menos de 1 minuto |
| **Experimento de 500 Épocas** | $766{,}37\text{ s}$ ($12\text{m }46\text{s}$) | **$274{,}50\text{ s}$ ($4\text{m }34\text{s}$)** | **$2{,}79\times$ mais rápido** | **Economia direta de mais de 8 minutos de execução!** |
| **Vazão de Inferência (Teste)** | $76{,}673\text{ img/s}$ | **$89{,}744\text{ img/s}$** | **$+17{,}0\%$ de taxa de transferência** | Quase 90.000 imagens avaliadas por segundo |
| **Tempo de Teste (10.000 imagens)** | $0{,}130\text{ s}$ | **$0{,}111\text{ s}$** | **$-14{,}6\%$ de latência** | Avaliação global em pouco mais de 100 milissegundos |

#### 4.11.3. Por que a GPU Venceu com Folga Mesmo em uma Rede Pequena?

Na literatura de computação de alto desempenho, modelos densos pequenos com batches de 64 imagens frequentemente sofrem com a sobrecarga de lançamento de *kernels* e envio de dados pela interface PCIe. No entanto, nesta experiência, a GPU superou a CPU por uma margem de quase $3\times$ devido a três fatores decisivos da nossa arquitetura:

1. **Pré-Carregamento Contíguo em RAM (`TensorDataset`):**  
   Como todas as 50.000 amostras já foram convertidas em um único bloco de tensores em memória RAM contígua, o Python não realiza conversões dinâmicas de I/O em disco. A transferência do lote para a GPU torna-se uma cópia rápida de memória direta (`cudaMemcpyAsync`), deixando os núcleos da RTX A1000 livres para operar em saturação.
2. **One-Hot Nativo e Aritmética de Matrizes em C++:**  
   O cálculo dos alvos One-Hot e a perda quadrática MSE ocorrem integralmente dentro da VRAM da GPU via kernels otimizados da biblioteca cuBLAS, sem nenhum vai-e-volta (*ping-pong*) de tensores intermediários para a CPU durante o forward e backward.
3. **Paralelismo Maciço dos 2.048 Núcleos CUDA:**  
   Enquanto a CPU distribui a multiplicação de matrizes ($64 \times 784$ por $784 \times 300$) entre 8 a 20 threads vetoriais, a GPU despacha os blocos de tensores simultaneamente através de seus multiprocessadores streaming (SMs), liquidando os 238.510 parâmetros da rede com latência insignificante.

#### 4.11.4. A Série Experimental na GPU: 50, 100, 200 e 500 Épocas

A execução completa dos regimes de treinamento na GPU confirmou o comportamento de saturação e overfitting identificado no notebook CPU:

| Épocas ($E$) | Tempo de Treino | Loss Treino (MSE) | Acc Treino | Loss Val (MSE) | Acc Val | Loss Teste (MSE) | Acurácia Teste (10k) | Erros no Teste | Diagnóstico Metodológico |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **50** | $27{,}68\text{ s}$ | $0{,}00571$ | $98{,}96\%$ | $0{,}00833$ | $97{,}63\%$ | $0{,}00810$ | $97{,}86\%$ | $214$ | Convergência rápida |
| **100** | $54{,}21\text{ s}$ | $0{,}00414$ | $99{,}44\%$ | $0{,}00769$ | $97{,}77\%$ | $0{,}00747$ | $97{,}87\%$ | $213$ | Aproximação do platô |
| **200** | $109{,}80\text{ s}$ | $0{,}00296$ | $99{,}76\%$ | $0{,}00755$ | $97{,}51\%$ | **$0{,}00729$** | **$97{,}89\%$** | **$211$** | **Ponto Ótimo Global de Generalização** |
| **500** | $274{,}50\text{ s}$ | **$0{,}00182$** | **$99{,}93\%$** | $0{,}00829$ | $97{,}31\%$ | $0{,}00797$ | $97{,}68\%$ | $232$ | **Degradação (+21 erros adicionais)** |

#### 4.11.5. Análise Morfológica dos Erros: A Dinâmica das Confusões no Overfitting

A análise das confusões mais frequentes registradas pelo módulo [`mlp_v3_testar.py`](./mlp_v3_testar.py) atende diretamente ao requisito da Seção II do enunciado oficial (diagnóstico qualitativo dos dígitos mais confundidos):

1. **A Confusão Crônica $4 \to 9$:**  
   Em 50 e 100 épocas, o modelo confundiu o dígito 4 com 9 entre 10 e 12 vezes. No entanto, ao forçar 500 épocas de sobreajuste, o erro **disparou para 18 ocorrências** (+50% de aumento). A rede supercalibrou pesos para traços fechados e passou a "alucinar" fechamento no topo aberto do 4.
2. **Ambiguidades Caligráficas Naturais do MNIST:**  
   - $7 \leftrightarrow 2$: traço horizontal na base ou curvatura do topo;
   - $5 \leftrightarrow 3$: curvas inferiores idênticas na escrita cursiva;
   - $3 \leftrightarrow 8$: fechamento dos laços à esquerda devido a borrões de tinta;
   - $9 \leftrightarrow 4$: haste vertical com curvatura dúbia no topo.

---

### 4.12. Concepção da Linha v4 e Transição Conceitual: MSE + One-Hot (v3) vs. Softmax + Entropia Cruzada (v4)

Com a infraestrutura computacional estabelecida e validada tanto em CPU quanto em GPU, dei início à preparação para a **Seção II do enunciado oficial** (*MLP com Softmax*).

Para comparar de forma justa, adotei uma decisão metodológica fundamental: **o experimento com controle estrito de variáveis (*ceteris paribus*)**. Antes de introduzir variações de hiperparâmetros de arquitetura (como número de camadas ou quantidade de neurônios), o primeiro passo da linha **v4** consiste em isolar uma única variável: **a substituição do algoritmo de perda e da formulação probabilística de saída**.

#### 4.12.1. O Contraste Matemático: Puxão Vetorial no MSE vs. Competição Probabilística no Softmax

A diferença conceitual entre as duas abordagens sintetiza décadas de evolução no aprendizado de máquina:

1. **A Abordagem Histórica (v3 — LeCun 1998): Regressão com MSE e One-Hot Manual:**
   - A rede projeta saídas lineares contínuas $\hat{\mathbf{y}} \in \mathbb{R}^{10}$.
   - O alvo é um vetor binário esparso $\mathbf{y} \in \{0, 1\}^{10}$.
   - A perda é a distância euclidiana quadrática elemento a elemento:
     $$\mathcal{L}_{\text{MSE}} = \frac{1}{10} \sum_{c=0}^{9} (\hat{y}_c - y_c)^2$$
   - **Mecânica:** O otimizador precisa de 10 "molas" independentes. O neurônio da classe correta é puxado para $1{,}0$ e os outros 9 neurônios são puxados ativamente para $0{,}0$.
   - **Limitação (hipótese, não testada):** Quando um neurônio está muito distante do alvo, a derivada quadrática simples pode não penalizar a incerteza com severidade proporcional, desacelerando o aprendizado inicial. Duas causas possíveis, que não separei: (1) o `MSELoss` tira a média sobre as 10 saídas, então o gradiente por saída é cerca de $5$ vezes menor que o da entropia cruzada para o mesmo erro, com a mesma taxa de aprendizado; (2) no MSE cada saída corre sozinha para $0$ ou $1$, enquanto no Softmax as classes disputam os mesmos $100\%$. Um teste direto seria rodar o v3 com taxa cerca de $5$ vezes maior.

2. **A Abordagem Moderna (v4 — Seção II do Enunciado): Classificação Probabilística com Softmax e Entropia Cruzada:**
   - Os 10 logits brutos $z_c$ são normalizados pela função Softmax em probabilidades verdadeiras que somam $1{,}0$ ($100\%$):
     $$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=0}^{9} e^{z_j}}$$
   - A perda é calculada pela Entropia Cruzada Categórica (*Negative Log-Likelihood*):
     $$\mathcal{L}_{\text{CE}} = - \sum_{c=0}^{9} y_c \ln(P_c)$$

#### 4.12.2. Por que o One-Hot Torna-se Desnecessário na Entropia Cruzada?

Na formulação da Entropia Cruzada, como o vetor One-Hot possui $y_c = 0$ para as 9 classes incorretas e $y_k = 1$ exclusivamente para a classe verdadeira $k$, a somatória colapsa analiticamente em um único termo:
$$\mathcal{L}_{\text{CE}} = - \ln(P_k)$$

Multiplicar por zero 9 vezes a cada amostra seria mero desperdício de ciclos e de memória. O PyTorch explora essa propriedade em `nn.CrossEntropyLoss()`:
- O tensor de alvos recebe diretamente o índice escalar inteiro da classe verdadeira ($y \in \{0, \dots, 9\}$), economizando alocação de tensores esparsos na RAM/VRAM.
- O Softmax e o cálculo logarítmico são combinados em um único *kernel* fundido em C++/CUDA (*log-sum-exp trick*), garantindo estabilidade numérica contra *underflow* e *overflow*.
- Graças à normalização intrínseca do Softmax ($\sum P_j = 1$), ao maximizar a probabilidade da classe correta $P_k$, o modelo esmaga matematicamente e por consequência todas as 9 probabilidades concorrentes, dispensando alvos manuais para cada neurônio.

#### 4.12.3. Formulação das Quatro Hipóteses Científicas (*Ceteris Paribus*)

Com a arquitetura ($784 \to 300 \to 10$), o otimizador (SGD, $\text{lr} = 0.1$), o tamanho de lote (64) e as partições em RAM idênticos aos do experimento v3 (a semente dos pesos iniciais não foi fixada nessas execuções; ver 4.15), formulei quatro hipóteses para guiar o confronto empírico:

1. **Hipótese 1 (Aceleração do Arranque Inicial):**  
   O gradiente da Entropia Cruzada em relação ao logit linear $z_k$ possui a forma elegante $\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z_k} = P_k - y_k$. Quando a rede erra completamente ($P_k \to 0$ para a classe verdadeira $y_k = 1$), a força propulsora atua em magnitude máxima unitária ($-1{,}0$), sem a atenuação suave característica do MSE. Espera-se que a acurácia de validação salte acima de $95\%$ logo na primeira época.
2. **Hipótese 2 (Superação do Teto Histórico de Generalização):**  
   Ao tratar a classificação como modelagem probabilística categórica e não como regressão euclidiana, o modelo deve superar o melhor resultado anterior, de $97{,}89\%$ (211 erros), alcançado pelo MSE no conjunto de teste lacrado de 10.000 imagens.
3. **Hipótese 3 (Assinatura em "U" e Degradação de Calibração):**  
   Em regimes prolongados (como 200 épocas), a perda de validação da Entropia Cruzada sofrerá elevação precoce devido à superconfiança dos logits sem decaimento de pesos (*weight decay*), demonstrando na prática a necessidade do *Early Stopping*.
4. **Hipótese 4 (Supressão Morfológica das Confusões Patológicas):**  
   Como a função Softmax impõe restrição estocástica ($\sum_{j=0}^9 P_j = 1$), uma classe falsa só pode vencer se suprimir a probabilidade conjunta de todas as outras classes. Isso deve reduzir expressivamente assimetrias patológicas observadas no MSE, tais como $5 \to 3$ e $7 \to 2$.

---

### 4.13. Implementação e Avaliação Empírica: O Confronto Direto v3 (MSE) vs. v4 (Cross-Entropy)

Para testar as quatro hipóteses, implementei a família v4 através de dois módulos autônomos: [`mlp_v4_treinar.py`](./mlp_v4_treinar.py) e [`mlp_v4_testar.py`](./mlp_v4_testar.py). Ambos respeitam a arquitetura de persistência padronizada, salvando seus checkpoints em `modelo/` e exportando suítes completas de 9 gráficos analíticos em alta resolução (300 DPI) para cada diretório de épocas em `figuras/e<épocas>/`.

#### 4.13.1. Arquitetura dos Módulos e Eliminação do Overhead de One-Hot

O código da v4 difere cirurgicamente da v3 em apenas dois aspectos fundamentais do loop de treino:
1. **Rótulos Escalares Inteiros:** Os alvos deixam de passar pela expansão esparsa `torch.nn.functional.one_hot(rotulos, num_classes=10)`. O tensor de rótulos permanece com dimensão `(batch_size,)` e tipo `torch.long`. Como demonstrado na Seção 4.12.2, eliminar os 9 zeros a cada amostra reduz $90\%$ de matrizes esparsas redundantes na RAM e VRAM.
2. **Critério Nativo:** O modelo projeta seus 10 logits lineares puros diretamente para `nn.CrossEntropyLoss()`. O cálculo do Softmax e do logaritmo negativo ocorre de forma acoplada em kernel C++/CUDA via função estável *LogSumExp* ($\ln \sum e^{z_j} = m + \ln \sum e^{z_j - m}$, com $m = \max z_j$), blindando o treinamento contra *underflow* e *overflow*.

No módulo de teste [`mlp_v4_testar.py`](./mlp_v4_testar.py), o modelo salvo em disco é carregado em isolamento e avaliado exclusivamente sobre as 10.000 imagens de teste oficial com `nn.CrossEntropyLoss()`.

---

#### 4.13.2. O Confronto Experimental Completo: 50 Épocas vs. 200 Épocas

Para garantir comparabilidade direta, executei ambas as arquiteturas nas mesmas condições controladas em dois horizontes de tempo: **50 épocas** (regime padrão de convergência) e **200 épocas** (regime prolongado de investigação de sobreajuste).

| Dimensão Experimental Avaliada | v3 (MSE) — 50 Épocas | v4 (Softmax) — 50 Épocas | v3 (MSE) — 200 Épocas | v4 (Softmax) — 200 Épocas |
| :--- | :---: | :---: | :---: | :---: |
| **Tempo Total de Execução** | $61{,}49\text{ s}$ ($1{,}23\text{ s/ép}$) | **$59{,}70\text{ s}$ ($1{,}19\text{ s/ép}$)** | $297{,}42\text{ s}$ ($1{,}49\text{ s/ép}$) | $308{,}09\text{ s}$ ($1{,}54\text{ s/ép}$) |
| **Acurácia Treino (Época 1)** | $89{,}24\%$ | **$92{,}09\%$ ($+2{,}85$ p.p.)** | $88{,}39\%$ | **$92{,}03\%$ ($+3{,}64$ p.p.)** |
| **Acurácia Validação (Época 1)** | $93{,}22\%$ | **$95{,}20\%$ ($+1{,}98$ p.p.)** | $92{,}80\%$ | **$94{,}94\%$ ($+2{,}14$ p.p.)** |
| **Acurácia Treino Final** | $98{,}96\%$ | **$100{,}00\%$ (na Época 18)** | $99{,}78\%$ | **$100{,}00\%$ (na Época 17)** |
| **Época do Mínimo na Validação** | Época 48 ($\mathcal{L}_{\text{val}} = 0{,}00823$) | **Época 11 ($\mathcal{L}_{\text{val}} = 0{,}0755$)** | Época 171 ($\mathcal{L}_{\text{val}} = 0{,}00744$) | **Época 10 ($\mathcal{L}_{\text{val}} = 0{,}0731$)** |
| **Acurácia no Teste Oficial (10k)** | $97{,}59\%$ | **$98{,}30\%$ (melhor até então)** | $97{,}84\%$ | **$98{,}26\%$** |
| **Erros de Teste (Cofre Lacrado)** | $241$ erros | **$170$ erros ($-71$ erros / $-29{,}5\%$)** | $216$ erros | **$174$ erros ($-42$ erros / $-19{,}4\%$)** |
| **Perda Média no Teste Oficial** | $0{,}00806$ (MSE) | $0{,}07042$ (Cross-Entropy) | $0{,}00730$ (MSE) | $0{,}08173$ (Cross-Entropy) |
| **F1-Score Macro no Teste** | $97{,}57\%$ | **$98{,}29\%$** | $97{,}82\%$ | **$98{,}25\%$** |
| **Pior Par de Confusão** | $5 \to 3$ e $7 \to 2$ ($13$ vezes cada) | $4 \to 9$ (**apenas $8$ vezes**) | $4 \to 9$ e $5 \to 3$ ($11$ vezes cada) | $2 \to 7$ e $9 \to 4$ (**apenas $8$ vezes**) |

---

#### 4.13.3. Análise Físico-Matemática dos Gráficos de Treinamento

Inspecionando os gráficos comparativos em [`figuras/e50/`](./figuras/e50/) e [`figuras/e200/`](./figuras/e200/):

1. **A Confirmação da Hipótese 1 (Aceleração do Gradiente):**  
   Comparando [`mlp_treinado_v3_e50_acuracia.png`](./figuras/e50/mlp_treinado_v3_e50_acuracia.png) com [`mlp_treinado_v4_e50_acuracia.png`](./figuras/e50/mlp_treinado_v4_e50_acuracia.png):
   - A v3 levou 4 épocas para ultrapassar $95\%$ de validação e nunca atingiu $100\%$ de acurácia no treino, mesmo a 200 épocas ($99{,}78\%$).
   - A v4 ultrapassou $95{,}2\%$ de validação logo na primeira época e **memorizou as 50.000 amostras de treino ($100{,}00\%$) na Época 18 (e Época 17 no ensaio de 200)**.
   - Isso decorre diretamente da força de gradiente $\frac{\partial \mathcal{L}}{\partial z} = P_k - y_k$: enquanto a probabilidade da classe correta não for próxima de $1{,}0$, o otimizador recebe um vetor de gradiente vigoroso, que não sofre desaceleração prematura.

2. **A Confirmação da Hipótese 3: A Curva em "U" e o Paradoxo da Calibração:**  
   No gráfico [`mlp_treinado_v4_e200_perda.png`](./figuras/e200/mlp_treinado_v4_e200_perda.png) e [`mlp_treinado_v4_e200_gap_generalizacao.png`](./figuras/e200/mlp_treinado_v4_e200_gap_generalizacao.png), evidencia-se um fenômeno de livro-texto de aprendizado profundo:
   - A perda de validação atinge seu vale mínimo **muito cedo, na Época 10 ($\mathcal{L}_{\text{val}} = 0{,}0731$)**.
   - Da época 10 à época 200, a perda de treino desce assintoticamente a zero ($0{,}0001$), enquanto a perda de validação sobe monotonicamente de $0{,}0731$ para $0{,}0965$ (uma elevação de **$+32{,}0\%$**).
   - **O Paradoxo:** Por que a perda de validação sobe $+32\%$ enquanto a acurácia de validação permanece alta e estável em $\sim 97{,}88\%$?  
     A resposta reside na **Degradação de Calibração de Probabilidades** (*probability calibration degradation*, Guo et al., 2017; Goodfellow et al., 2016):
     À medida que o SGD continua atualizando a rede sem regularização $L_2$ (*weight decay*), os pesos $\|\mathbf{W}\|$ continuam crescendo para aproximar a perda de treino de zero absoluto. Isso faz os logits $z_k$ atingirem amplitudes gigantescas ($\pm 15$ ou $\pm 20$). Quando a rede acerta, a predição é hiperconfiante ($P_k = 0{,}9999$). No entanto, nas poucas amostras ambíguas de validação onde o modelo erra por pouco (ex.: $P_{\text{correta}} = 0{,}02$), a penalização logarítmica $-\ln(0{,}02) \approx 3{,}91$ é implacável! Pouquíssimas amostras superconfiantes e erradas puxam a média da perda para cima, embora a contagem discreta de acertos permaneça quase inalterada.

---

#### 4.13.4. Análise dos Gráficos de Teste Oficial (10.000 Imagens)

Comparando [`mlp_treinado_v4_e50_painel_teste.png`](./figuras/e50/mlp_treinado_v4_e50_painel_teste.png) e [`mlp_treinado_v3_e50_painel_teste.png`](./figuras/e50/mlp_treinado_v3_e50_painel_teste.png):

1. **Hipótese 2 compatível com os dados: menor erro de teste até então ($98{,}30\%$):**
   - O modelo **`mlp_v4_treinado_e50.pth`** atingiu $98{,}30\%$ de acurácia no teste (170 erros em 10.000 amostras), o melhor resultado até a seção 4.13. Foi uma execução, sem semente fixa. A bateria da 4.14 e o v5 (4.15) trazem outros números.
   - Isso representa uma **redução de 71 erros** em relação ao modelo v3 de 50 épocas ($241 \to 170$, queda de $-29{,}5\%$ nas falhas) e **46 erros a menos** que o ponto ótimo histórico do MSE a 200 épocas ($216 \to 170$).

2. **O Efeito do Treinamento Prolongado (50 vs. 200 Épocas na v4):**
   - A 50 épocas: **$170$ erros** ($98{,}30\%$ de acurácia, perda de teste $0{,}07042$).
   - A 200 épocas: **$174$ erros** ($98{,}26\%$ de acurácia, perda de teste $0{,}08173$).
   - A perda média no teste lacrado aumentou em **$+16{,}1\%$** (de $0{,}07042$ para $0{,}08173$) e surgiram **4 erros adicionais**, provando empiricamente que forçar o treinamento após a saturação degrada sutilmente a capacidade de generalização e piora a calibração de incerteza da rede.

3. **A Confirmação da Hipótese 4: Supressão das Concentrações Patológicas de Erro:**
   Examinando as matrizes de confusão [`mlp_treinado_v3_e50_matriz_confusao.png`](./figuras/e50/mlp_treinado_v3_e50_matriz_confusao.png) vs. [`mlp_treinado_v4_e50_matriz_confusao.png`](./figuras/e50/mlp_treinado_v4_e50_matriz_confusao.png):
   - **Na v3 (MSE):** A perda euclidiana independente permitiu o acúmulo de erros crônicos graves em dígitos com morfologias parecidas: as confusões $5 \to 3$ e $7 \to 2$ acumularam **13 ocorrências cada**, e $4 \to 9$ acumulou de 9 a 11 ocorrências.
   - **Na v4 (Softmax):** Nenhum par de confusão em todo o teste excedeu **8 ocorrências**. O erro patológico $5 \to 3$ caiu de 13 para apenas 6 ocorrências (uma queda drástica de **$-53{,}8\%$**).
   - **Fundamentação:** No Softmax, as 10 probabilidades competem por uma soma fixa de $1{,}0$. Para que o dígito 3 vença uma imagem do dígito 5, seu logit precisa superar a soma combinada de todas as outras classes no denominador. A penalização cruzada mútua impede que pequenas correlações espúrias nos pixels resultem em erros repetitivos.

---

#### 4.13.5. Conclusão Metodológica e Definição da Janela Ótima de *Early Stopping*

A síntese dos experimentos v3 vs. v4 estabelece os seguintes parâmetros metodológicos consolidados:
1. **Vantagem da Entropia Cruzada neste experimento:** A transição de MSE para Cross-Entropy com Softmax nativo elevou o patamar de acurácia de $97{,}8\%$ para **$98{,}3\%$**, eliminando quase um terço de todos os erros da rede sem adicionar parâmetros à arquitetura. Cada configuração rodou uma vez e sem semente fixa: a diferença de 71 erros é grande, mas diferenças menores precisam de repetição (ver 4.15).
2. **Janela Ótima de Parada (*Early Stopping*):** Para esta arquitetura sob SGD ($\text{lr} = 0.1$), o intervalo ideal de interrupção situa-se entre **30 e 50 épocas** (lote $64$, sem momento; com lote $256$ e momento $0{,}9$ o v5 achou o melhor ponto entre as épocas $2$ e $14$, seção 4.15). Continuar o treinamento até 200 épocas consome $5\times$ mais tempo e energia de máquina e produz uma perda de teste $16\%$ pior devido ao sobreajuste de confiança.

#### 4.13.6. Acervo Completo de Gráficos Pareados em Alta Resolução (300 DPI)

Todo o acervo de 36 relatórios visuais gerados para os regimes de 50 e 200 épocas está organizado e acessível:

- **Regime de 50 Épocas (`figuras/e50/`):**
  - Curva de Perda: [`mlp_treinado_v3_e50_perda.png`](./figuras/e50/mlp_treinado_v3_e50_perda.png) vs. [`mlp_treinado_v4_e50_perda.png`](./figuras/e50/mlp_treinado_v4_e50_perda.png)
  - Curva de Acurácia: [`mlp_treinado_v3_e50_acuracia.png`](./figuras/e50/mlp_treinado_v3_e50_acuracia.png) vs. [`mlp_treinado_v4_e50_acuracia.png`](./figuras/e50/mlp_treinado_v4_e50_acuracia.png)
  - Gap de Generalização: [`mlp_treinado_v3_e50_gap_generalizacao.png`](./figuras/e50/mlp_treinado_v3_e50_gap_generalizacao.png) vs. [`mlp_treinado_v4_e50_gap_generalizacao.png`](./figuras/e50/mlp_treinado_v4_e50_gap_generalizacao.png)
  - Dashboard de Treino: [`mlp_treinado_v3_e50_painel_geral.png`](./figuras/e50/mlp_treinado_v3_e50_painel_geral.png) vs. [`mlp_treinado_v4_e50_painel_geral.png`](./figuras/e50/mlp_treinado_v4_e50_painel_geral.png)
  - Matriz de Confusão 10x10: [`mlp_treinado_v3_e50_matriz_confusao.png`](./figuras/e50/mlp_treinado_v3_e50_matriz_confusao.png) vs. [`mlp_treinado_v4_e50_matriz_confusao.png`](./figuras/e50/mlp_treinado_v4_e50_matriz_confusao.png)
  - Dashboard de Teste Integrado: [`mlp_treinado_v3_e50_painel_teste.png`](./figuras/e50/mlp_treinado_v3_e50_painel_teste.png) vs. [`mlp_treinado_v4_e50_painel_teste.png`](./figuras/e50/mlp_treinado_v4_e50_painel_teste.png)
  - Amostras com Erro: [`mlp_treinado_v3_e50_amostras_erros.png`](./figuras/e50/mlp_treinado_v3_e50_amostras_erros.png) vs. [`mlp_treinado_v4_e50_amostras_erros.png`](./figuras/e50/mlp_treinado_v4_e50_amostras_erros.png)

- **Regime de 200 Épocas (`figuras/e200/`):**
  - Curva de Perda: [`mlp_treinado_v3_e200_perda.png`](./figuras/e200/mlp_treinado_v3_e200_perda.png) vs. [`mlp_treinado_v4_e200_perda.png`](./figuras/e200/mlp_treinado_v4_e200_perda.png)
  - Curva de Acurácia: [`mlp_treinado_v3_e200_acuracia.png`](./figuras/e200/mlp_treinado_v3_e200_acuracia.png) vs. [`mlp_treinado_v4_e200_acuracia.png`](./figuras/e200/mlp_treinado_v4_e200_acuracia.png)
  - Gap de Generalização: [`mlp_treinado_v3_e200_gap_generalizacao.png`](./figuras/e200/mlp_treinado_v3_e200_gap_generalizacao.png) vs. [`mlp_treinado_v4_e200_gap_generalizacao.png`](./figuras/e200/mlp_treinado_v4_e200_gap_generalizacao.png)
  - Dashboard de Treino: [`mlp_treinado_v3_e200_painel_geral.png`](./figuras/e200/mlp_treinado_v3_e200_painel_geral.png) vs. [`mlp_treinado_v4_e200_painel_geral.png`](./figuras/e200/mlp_treinado_v4_e200_painel_geral.png)
  - Matriz de Confusão 10x10: [`mlp_treinado_v3_e200_matriz_confusao.png`](./figuras/e200/mlp_treinado_v3_e200_matriz_confusao.png) vs. [`mlp_treinado_v4_e200_matriz_confusao.png`](./figuras/e200/mlp_treinado_v4_e200_matriz_confusao.png)
  - Dashboard de Teste Integrado: [`mlp_treinado_v3_e200_painel_teste.png`](./figuras/e200/mlp_treinado_v3_e200_painel_teste.png) vs. [`mlp_treinado_v4_e200_painel_teste.png`](./figuras/e200/mlp_treinado_v4_e200_painel_teste.png)
  - Amostras com Erro: [`mlp_treinado_v3_e200_amostras_erros.png`](./figuras/e200/mlp_treinado_v3_e200_amostras_erros.png) vs. [`mlp_treinado_v4_e200_amostras_erros.png`](./figuras/e200/mlp_treinado_v4_e200_amostras_erros.png)

#### 4.13.7. Discrepância Encontrada: Figuras e Checkpoints das Seções 4.8 a 4.13 Não Estão no Repositório

Ao revisar o repositório antes de começar a CNN, vi que as figuras citadas nas seções 4.8 a 4.13 (`figuras/e3/`, `e50/`, `e120/`, `e200/` e `e500/`, com as versões v3 e v4) e os checkpoints `.pth` dessas execuções não existem na cópia atual do projeto. Dos $37$ links para `figuras/` desta metodologia, só $2$ funcionam (`distribuicao_classes.png` e `exemplos_digitos.png`). Antes da bateria de 07/11 eu limpei as pastas `figuras/` e `modelo/` da máquina da GPU, e essa é a causa provável. Não consegui confirmar se existem cópias nas outras máquinas.

Isso cria três problemas para a leitura das seções 4.8 a 4.13:

1. **Não são verificáveis.** As tabelas dessas seções vêm de uma única execução cada. Quem clonar o repositório não encontra as figuras nem os pesos.
2. **Não são reproduzíveis.** O `mlp_v3_treinar.py` e o `mlp_v4_treinar.py` não fixam a semente dos pesos iniciais nem da ordem dos lotes. Um treino novo dá números diferentes. Já vi $170$ e $186$ erros para a mesma configuração (seção 4.15).
3. **Padrões diferentes.** As seções antigas, a bateria 4.14 e o v5 usam critérios diferentes de semente, de checkpoint (última época contra menor perda) e de escolha do modelo.

**Decisão.** Vou refazer essas execuções no padrão do v5, pelo novo script v6 (seção 4.17), para a metodologia ficar toda sob as mesmas regras: semente declarada, checkpoint registrado e figuras guardadas no repositório. Quando a bateria v6 rodar, registro aqui os números novos e aviso onde eles diferirem das tabelas originais. Até lá, os números das seções 4.8 a 4.13 valem como registro exploratório de uma execução, sem semente fixa e sem figuras versionadas.

---

### 4.14. Exploração Sistemática de Hiperparâmetros (Bateria v4.2 na NVIDIA GeForce RTX 3070)

Para atender de forma exaustiva aos requisitos da **Seção II do enunciado oficial** (*"Avaliação da quantidade de camadas e quantidade de neurônios nas camadas como hiperparâmetros; teste de diferentes arquiteturas; variação de otimizadores e parâmetros de treinamento"*), implementamos o módulo parametrizável `mlp_v4_2_treinar.py` e executamos uma bateria automatizada com **15 experimentos controlados** na máquina secundária com GPU dedicada (RTX 3070 8GB via WSL2).

Todos os 15 modelos foram treinados sobre as 50.000 amostras de treino (com validação em 10.000 amostras) e, em seguida, avaliados por `mlp_v4_testar.py` sobre as **10.000 imagens do conjunto oficial de teste do MNIST**. Atenção: isso abriu o teste $15$ vezes, uma por modelo, e o campeão foi escolhido olhando esses resultados. A seção 4.15 trata esse problema.

#### 4.14.1. Tabela Comparativa Geral dos 15 Experimentos

| # | Bloco / Objetivo | Hiperparâmetros & Arquitetura | Épocas | Loss Teste | Acurácia Teste | Erros (/10k) | Tempo Total | Tempo/Época | Top Confusões Mais Frequentes |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 1. Arquitetura | `-b 256 -l 0.1 --hidden-dims 128` | 50 | 0.0742 | **97.92%** | **208** | 29.9s | 0.598s | 9->4 (11x), 4->9 (10x), 9->7 (9x) |
| 2 | 1. Arquitetura | `-b 256 -l 0.1` | 50 | 0.0661 | **98.08%** | **192** | 31.2s | 0.625s | 4->9 (11x), 5->3 (10x), 2->7 (8x) |
| 3 | 1. Arquitetura | `-b 256 -l 0.1 --hidden-dims 512` | 50 | 0.0625 | **98.27%** | **173** | 33.2s | 0.663s | 9->4 (9x), 4->9 (8x), 7->9 (7x) |
| 4 | 1. Arquitetura | `-b 256 -l 0.1 --hidden-dims 300 100` | 50 | 0.0765 | **98.14%** | **186** | 36.3s | 0.726s | 7->9 (10x), 9->4 (8x), 7->2 (7x) |
| 5 | 1. Arquitetura | `-b 256 -l 0.1 --hidden-dims 512 256` | 50 | 0.0749 | **98.21%** | **179** | 33.6s | 0.672s | 7->9 (10x), 4->9 (8x), 9->4 (7x) |
| 6 | 1. Arquitetura | `-b 256 -l 0.1 --hidden-dims 512 256 128` | 50 | 0.0880 | **98.18%** | **182** | 36.3s | 0.726s | 4->9 (7x), 5->3 (7x), 7->2 (6x) |
| 7 | 2. Batch / SGD | `-b 64 -l 0.1` | 50 | 0.0712 | **98.14%** | **186** | 75.4s | 1.509s | 4->9 (10x), 2->7 (8x), 9->4 (8x) |
| 8 | 2. Batch / SGD | `-b 128 -l 0.1` | 50 | 0.0701 | **98.08%** | **192** | 45.1s | 0.901s | 4->9 (10x), 5->3 (9x), 7->9 (9x) |
| 9 | 2. Batch / SGD | `-b 256 -l 0.1 -m 0.9` | 50 | 0.0759 | **98.25%** | **175** | 35.9s | 0.718s | 9->4 (10x), 4->9 (9x), 5->3 (7x) |
| 10 | 2. Batch / SGD | `-b 512 -l 0.2 -m 0.9` | 50 | 0.0903 | **98.08%** | **192** | 25.3s | 0.507s | 4->9 (9x), 5->3 (8x), 9->4 (8x) |
| 11 | 3. Otimizadores | `-b 256 -l 0.001 --otimizador adam` | 50 | 0.0918 | **98.20%** | **180** | 34.1s | 0.682s | 4->9 (10x), 2->7 (8x), 5->3 (8x) |
| 12 | 3. Otimizadores | `-b 256 -l 0.001 --otimizador adamw` | 50 | 0.0873 | **98.24%** | **176** | 34.4s | 0.688s | 4->9 (10x), 9->4 (10x), 2->7 (7x) |
| 13 | 3. Otimizadores | `-b 256 -l 0.001 --otimizador adamw --hidden-dims 512 256` | 50 | 0.1113 | **98.17%** | **183** | 36.6s | 0.732s | 9->4 (10x), 3->5 (9x), 6->5 (7x) |
| 14 | 4. Campeões (100e) | `-b 256 -l 0.1 -m 0.9` | 100 | 0.0805 | **98.26%** | **174** | 62.6s | 0.626s | 4->9 (9x), 9->4 (9x), 5->3 (7x) |
| 15 | 4. Campeões (100e) | `-b 256 -l 0.1 -m 0.9 --hidden-dims 300 100` | 100 | 0.1048 | **98.40%** | **160** | 68.6s | 0.686s | 4->9 (10x), 7->9 (8x), 9->4 (7x) |

---

#### 4.14.2. Análise dos 4 Blocos Investigativos

1. **Bloco 1: Variação de Arquitetura (Largura vs. Profundidade - Seção II.B)**
   - **Largura (1 Camada Oculta):**
     - $128$ neurônios: $97{,}92\%$ ($208$ erros). Capacidade expressiva limitada para capturar variações caligráficas mais sutis.
     - $300$ neurônios (Baseline LeCun): $98{,}08\%$ ($192$ erros). Equilíbrio canônico.
     - $512$ neurônios: **$98{,}27\%$ ($173$ erros)** com a menor perda de teste de toda a bateria de 50 épocas ($\mathcal{L}_{\text{teste}} = 0{,}0625$). A expansão da largura da camada oculta proporciona uma separabilidade linear superior antes do colapso nos 10 logits.
   - **Profundidade (Múltiplas Camadas Ocultas):**
     - $300 \to 100$: $98{,}14\%$ ($186$ erros) em 50 épocas. Apresenta convergência suave e excelente comportamento de generalização.
     - $512 \to 256$: $98{,}21\%$ ($179$ erros). Elevada capacidade de abstração intermediária.
     - $512 \to 256 \to 128$ (3 camadas): $98{,}18\%$ ($182$ erros). Apresenta indícios leves de sobreajuste em 50 épocas (treino bate $100\%$ na época 24, e perda de validação sobe de $0{,}081$ para $0{,}101$). Nesta bateria, com uma semente e escolha pelo teste, as diferenças entre $1$, $2$ e $3$ camadas ($173$ a $186$ erros) ficam dentro do ruído de uma execução. A seção 4.15.4 mostra que o ranking entre profundidades depende do critério de escolha do checkpoint, então não concluo que menos camadas sejam melhores.

2. **Bloco 2: Tamanho do Batch & Dinâmica SGD (Hardware e Otimização)**
   - **Throughput e Despacho de Hardware:**
     - Batch $64$: $75{,}4\text{s}$ ($1{,}509\text{s}$/época) — $98{,}14\%$.
     - Batch $128$: $45{,}1\text{s}$ ($0{,}901\text{s}$/época) — $98{,}08\%$.
     - Batch $256$: $31{,}2\text{s}$ ($0{,}625\text{s}$/época) — $98{,}08\%$.
     - Batch $512$: **$25{,}3\text{s}$ ($0{,}507\text{s}$/época)** — $98{,}08\%$.
     - *Conclusão de Engenharia:* Aumentar o lote de $64$ para $512$ acelerou o treinamento em **$3{,}0\times$** na GPU sem perda mensurável de acurácia, confirmando que o gargalo em batches pequenos não é o poder aritmético dos CUDA cores, mas a latência de despacho de kernels no loop Python sobre a camada WSL2/Hyper-V.
   - **Impacto do Momento:**
     - SGD puro (Batch 256, lr 0.1): $98{,}08\%$ ($192$ erros).
     - SGD + Momentum $0.9$ (Batch 256, lr 0.1): **$98{,}25\%$ ($175$ erros)**. O momento atua como um acumulador inercial que amortece oscilações ortogonais e acelera o avanço em direções de gradiente persistente. *(Nota posterior: no v5, com $3$ sementes e escolha pela validação, o momento chegou ao melhor ponto cerca de $4$ vezes mais rápido, mas a diferença de acurácia ($97{,}78$ contra $97{,}62$) ficou na ordem da tolerância de $0{,}14$ ponto. Leio o ganho de $0{,}17$ ponto desta linha como compatível com ruído; ver seção 4.15.3.)*

3. **Bloco 3: Otimizadores Modernos (Adam e AdamW)**
   - **Ajuste de Escala:** Demonstramos empiricamente que otimizadores adaptativos exigem taxa de aprendizado cerca de duas ordens de grandeza menor ($\eta = 0{,}001$) do que o SGD ($\eta = 0{,}1$) devido à divisão pelo desvio padrão móvel $\sqrt{v_t}$.
   - **Adam:** $98{,}20\%$ ($180$ erros).
   - **AdamW:** **$98{,}24\%$ ($176$ erros)**. O desacoplamento do decaimento de peso (*weight decay*) proposto por Loshchilov & Hutter (2019) conferiu ligeira vantagem de generalização em relação ao Adam padrão.

4. **Bloco 4: Treinamento Estendido de 100 Épocas e a Maior Acurácia de Teste**
   - No regime de 100 épocas com SGD ($\text{lr} = 0{,}1$, $\text{momentum} = 0{,}9$), a arquitetura de duas camadas ocultas **$784 \to 300 \to 100 \to 10$** obteve a maior acurácia de teste entre as 15 configurações (escolha feita sobre o teste; será revista por validação no v5, seção 4.15):
     $$\mathbf{98{,}40\% \text{ de Acurácia no Teste Oficial (Apenas } 160 \text{ Erros em } 10.000 \text{ Imagens)}}$$
   - A combinação de duas etapas hierárquicas de redução de dimensionalidade ($784 \to 300$ extraindo traços primitivos, e $300 \to 100$ recombinando-os em conceitos de partes de dígitos) com a inércia do Momentum permitiu refinar as superfícies de decisão até suprimir falhas que os modelos rasos persistiam cometendo.

---

#### 4.14.3. Análise Morfológica das Confusões Crônicas

Ao analisar a matriz de confusão $10 \times 10$ e as galerias de erros geradas em todos os 15 experimentos, constatou-se coerência morfológica nas falhas residuais:

1. **Dígito $4$ classificado como $9$ (e $9$ classificado como $4$):**
   - É a confusão mais frequente em todos os 15 experimentos (de 8 a 11 ocorrências).
   - *Causa Morfológica:* Na escrita manual rápida, o laço superior do $9$ muitas vezes não se fecha completamente, tornando-o idêntico a um $4$ em formato de "cadeirinha aberta". Em contrapartida, quando a haste do $4$ é ligeiramente curvada, os pixels ativam os mesmos detectores do círculo do $9$.
2. **Dígito $5$ classificado como $3$ (e $3$ classificado como $5$):**
   - *Causa Morfológica:* Ambos compartilham o bojo inferior arredondado. A distinção depende exclusivamente do traço superior (uma barra horizontal reta com canto vivo no $5$ vs. arco suave no $3$). Em imagens rasterizadas de $28 \times 28$ binarizadas ou suavizadas, pequenas inclinações apagam esse canto vivo.
3. **Dígito $7$ classificado como $2$ ou $9$:**
   - *Causa Morfológica:* O traço horizontal cortando o corpo do $7$ (muito comum em caligrafia europeia) induz a rede a interpretar a haste inferior como a base plana de um $2$.

---

### 4.15. v5: Escolha do Modelo pela Validação e Semente Fixa

**Por que fiz o v5.** Ao revisar a bateria da seção 4.14, vi que o modelo de $98{,}40\%$ foi escolhido olhando o conjunto de teste, e que o teste foi aberto 15 vezes (uma por modelo). O enunciado pede o contrário: a validação serve para ajustar hiperparâmetros e o teste serve para a avaliação final. Cruzando o log da bateria com as curvas de validação:

- Esse modelo (300 → 100, 100 épocas, momento $0{,}9$) é o 12º de 15 pela acurácia de validação final ($97{,}92\%$) e tem a maior perda de validação mínima dos 15 ($0{,}0906$).
- Pela validação, ficariam à frente, por exemplo, os modelos 13, 9 e 14 (acurácia de $98{,}03\%$, $98{,}02\%$ e $98{,}01\%$).
- Com $10.000$ imagens de validação, o erro-padrão de uma acurácia perto de $98\%$ é de cerca de $0{,}14$ ponto percentual. Com uma semente por configuração, não consigo separar os modelos 7 a 14.
- O teste foi, em média, $0{,}35$ ponto percentual mais fácil que a validação (de $0{,}14$ a $0{,}72$). Não devo comparar as duas escalas diretamente.
- A mesma configuração ($784 \to 300 \to 10$, lote 64, $\eta = 0{,}1$, 50 épocas) deu $170$ erros na seção 4.13 e $186$ erros na bateria 4.14. Esses $16$ erros de diferença são o ruído de uma execução, sem semente fixa.

**O que muda no v5** (arquivos `mlp_v5_treinar.py`, `mlp_v5_testar.py`, `mlp_v5_selecionar.py`, `executar_bateria_v5.sh`):

1. O checkpoint salvo é o da época de **menor perda de validação**, não o da última (`*_melhor.pth`). Usei a perda e não a acurácia porque a acurácia de validação dá saltos de uma imagem ($0{,}01$ ponto percentual) e a perda tem um mínimo claro.
2. `--seed` fixa os pesos iniciais e a ordem dos lotes. Em CPU, a mesma semente repete o resultado.
3. Cada execução grava uma linha em `resultados/resumo_v5.csv` (configuração, parâmetros, época salva, validação e tempo). Isso já deixa pronta a coluna de parâmetros e de tempo que o enunciado pede na comparação com a CNN.
4. O teste fica fora do treino e do `selecionar`. O `mlp_v5_testar.py` registra cada abertura em `resultados/aberturas_teste.csv` e avisa quantas vezes o teste já foi aberto.

**Como escolho a configuração final.** Repito as configurações com 3 sementes. Comparo pela acurácia média de validação na época salva. Considero empatadas as configurações dentro de uma tolerância do melhor (o maior entre o desvio entre sementes e o erro-padrão de $\approx 0{,}14$ ponto percentual) e fico com a de menos parâmetros. Só então abro o teste, uma vez, no melhor checkpoint dessa configuração.

#### 4.15.1. Desenho da Bateria v5

Não repeti as 15 configurações da seção 4.14. Reorganizei a bateria em torno de uma **base** e de blocos que mudam uma coisa por vez (*ceteris paribus*). A base é uma camada oculta de $300$ neurônios, lote $256$, $\eta = 0{,}1$, momento $0{,}9$ e $50$ épocas, com ReLU e SGD. Cada configuração roda com as sementes $1$, $2$ e $3$, o que dá $45$ treinos. Todos rodaram na CPU, só com treino e validação.

**Alternativas e decisões:**

1. **Manter Adam e AdamW (bloco 3 da seção 4.14).** Descartei. O enunciado fixa a ReLU e o SGD, e a bateria deve responder ao que ele pede. Os resultados de Adam e AdamW ficam na 4.14 como registro exploratório.
2. **Variar o número de épocas (50 e 100).** Descartei. O v5 guarda a época de menor perda de validação, então o número de épocas passa a ser só um teto. Com momento, o melhor ponto ficou entre as épocas $2$ e $14$ em todas as configurações, longe de $50$. Sem momento ficou entre $32$ e $36$, ainda abaixo do teto.
3. **Decisão.** Cinco blocos, $15$ configurações: largura ($64$, $128$, $512$, $1024$), profundidade ($300 \to 100$, $512 \to 256$, $256 \to 128 \to 64$, $512 \to 256 \to 128$), lote ($64$, $128$, $512$), taxa de aprendizado ($0{,}05$ e $0{,}2$) e SGD sem momento.

#### 4.15.2. Resultados da Bateria (Validação, 3 Sementes)

Cada linha é a média e o desvio entre as 3 sementes, no checkpoint de menor perda de validação. O tempo é de treino na CPU, e por isso não se compara com os tempos em GPU da seção 4.14.

| Bloco | Configuração (camadas ocultas, lote, $\eta$, momento) | Parâmetros | Acurácia de validação | Perda de validação | Época salva (média) | Tempo (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Base | $300$, $256$, $0{,}1$, $0{,}9$ | $238.510$ | $97{,}78 \pm 0{,}16$ | $0{,}0806$ | $7{,}3$ | $53$ |
| Largura | $64$ | $50.890$ | $96{,}82 \pm 0{,}23$ | $0{,}1139$ | $4{,}3$ | $29$ |
| Largura | $128$ | $101.770$ | $97{,}59 \pm 0{,}35$ | $0{,}0959$ | $8{,}7$ | $31$ |
| Largura | $512$ | $407.050$ | $\mathbf{97{,}95 \pm 0{,}08}$ | $0{,}0744$ | $8{,}3$ | $54$ |
| Largura | $1024$ | $814.090$ | $97{,}93 \pm 0{,}12$ | $0{,}0708$ | $7{,}0$ | $85$ |
| Profundidade | $300 \to 100$ | $266.610$ | $97{,}64 \pm 0{,}22$ | $0{,}0930$ | $6{,}3$ | $51$ |
| Profundidade | $512 \to 256$ | $535.818$ | $97{,}60 \pm 0{,}16$ | $0{,}0818$ | $3{,}7$ | $74$ |
| Profundidade | $256 \to 128 \to 64$ | $242.762$ | $97{,}26 \pm 0{,}36$ | $0{,}1039$ | $5{,}7$ | $47$ |
| Profundidade | $512 \to 256 \to 128$ | $567.434$ | $97{,}63 \pm 0{,}30$ | $0{,}0908$ | $5{,}7$ | $77$ |
| Lote | $64$ | $238.510$ | $94{,}50 \pm 1{,}12$ | $0{,}2456$ | $3{,}3$ | $75$ |
| Lote | $128$ | $238.510$ | $97{,}41 \pm 0{,}49$ | $0{,}1068$ | $7{,}3$ | $51$ |
| Lote | $512$ | $238.510$ | $97{,}74 \pm 0{,}10$ | $0{,}0771$ | $10{,}0$ | $45$ |
| Taxa | $\eta = 0{,}05$ | $238.510$ | $97{,}77 \pm 0{,}04$ | $0{,}0763$ | $10{,}0$ | $45$ |
| Taxa | $\eta = 0{,}2$ | $238.510$ | $97{,}35 \pm 0{,}44$ | $0{,}1156$ | $5{,}0$ | $46$ |
| Momento | sem momento | $238.510$ | $97{,}62 \pm 0{,}01$ | $0{,}0769$ | $33{,}7$ | $43$ |

#### 4.15.3. O que os Números Mostram

Com $10.000$ imagens de validação, o erro-padrão é de cerca de $0{,}14$ ponto percentual. Uso esse valor como tolerância de empate. Diferenças menores que ele eu não leio como diferença real.

- **Largura.** A acurácia sobe de $64$ para $512$ neurônios e para aí. Com $1024$ neurônios fica empatada com $512$ ($97{,}93$ contra $97{,}95$) e com o dobro de parâmetros. Pela regra de desempate, a menor das duas vence.
- **Lote.** O lote $64$ é muito pior ($94{,}50 \pm 1{,}12$) e é a configuração mais instável entre as sementes. Os lotes $256$ e $512$ ficam empatados.
- **Taxa.** As taxas $0{,}05$ e $0{,}1$ empatam. A taxa $0{,}2$ piora e oscila mais entre sementes.
- **Momento.** Sem momento, o melhor ponto cai na época $32$ a $36$. Com momento, cai na $5$ a $10$. A acurácia sem momento ($97{,}62$) fica $0{,}16$ abaixo da base, uma diferença da mesma ordem da tolerância. Leio isso como: o momento chega ao mesmo lugar bem mais rápido, sem diferença clara de acurácia.
- **Épocas.** Nenhuma configuração salvou o melhor ponto perto da época $50$.

**Hipóteses que não testei:**

- O lote $64$ com $\eta = 0{,}1$ e momento $0{,}9$ pode estar com um passo efetivo grande demais, pois o momento amplia o passo e o lote pequeno deixa o gradiente mais ruidoso. Na seção 4.14 o lote $64$ foi treinado sem momento e deu $98{,}14\%$ no teste, o que é coerente com essa hipótese, mas não a prova. Um teste direto seria rodar o lote $64$ com $\eta$ menor.
- A profundidade não ajudar pode ser uma propriedade do MNIST, que é simples. Veja a seção 4.15.4 antes de aceitar essa leitura.

#### 4.15.4. Perda Mínima Contra Acurácia Máxima: o Critério Muda o Ranking

Ao comparar o checkpoint de menor perda com a última época, vi um efeito que muda a leitura dos blocos. Em $44$ das $45$ execuções, a **última época tem acurácia de validação maior** que o checkpoint de menor perda (em média $+0{,}47$ ponto percentual, até $+2{,}66$). Ao mesmo tempo, a perda de validação da última época é em média $1{,}47$ vez a perda mínima. O modelo continua acertando mais enquanto a perda sobe. É o mesmo efeito de calibração da seção 4.13.3: a rede fica mais confiante, e as poucas imagens que erra pesam muito na perda.

Por isso o ranking depende do critério usado para salvar o modelo:

| Critério | 1º | 2º | 3º | Camada única de $512$ |
| :--- | :--- | :--- | :--- | :---: |
| Checkpoint de menor perda (o do v5) | $512$ ($97{,}95$) | $1024$ ($97{,}93$) | $300$ ($97{,}78$) | 1º |
| Acurácia da última época (50) | $512 \to 256$ ($98{,}28$) | $512 \to 256 \to 128$ ($98{,}24$) | $256 \to 128 \to 64$ ($98{,}13$) | 8º ($97{,}99$) |

Pelo critério da última época, as redes com $2$ e $3$ camadas ficam à frente da camada única, e a diferença ($0{,}29$ ponto percentual) passa da tolerância de $0{,}14$. A conclusão "mais camadas não ajudam" vale só para o critério da perda. O lote $64$ continua o pior nos dois critérios ($96{,}34$ pela última época).

**Decisão.** Defini a perda de validação como critério antes de rodar a bateria, e mantenho essa escolha. Trocar o critério depois de ver o resultado seria escolher o método olhando os dados. Registro aqui o outro ranking para que a diferença fique visível no relatório.

#### 4.15.5. Avaliação no Teste: Aberto Uma Vez, no Modelo Escolhido pela Validação

Mantive o critério da perda de validação, definido antes da bateria. Abri o teste uma única vez, no checkpoint `mlp_treinado_v5_e50_b256_lr0.1_m0.9_h512_s1_melhor.pth` ($784 \to 512 \to 10$, $407.050$ parâmetros, época $7$ salva). O registro está em `resultados/aberturas_teste.csv` (abertura $1$, antes dela o contador era $0$).

| Métrica no teste (10.000 imagens) | Valor |
| :--- | :---: |
| Acurácia | $98{,}19\%$ ($181$ erros) |
| Perda média (entropia cruzada) | $0{,}0604$ |
| Precisão macro | $98{,}19\%$ |
| Revocação macro | $98{,}17\%$ |
| F1 macro | $98{,}18\%$ |

**Como leio o número:**

- O erro-padrão da acurácia é de cerca de $0{,}14$ ponto percentual, o que dá um intervalo de $95\%$ de aproximadamente $97{,}9\%$ a $98{,}5\%$. É um modelo de uma semente.
- Na validação, essa semente deu $97{,}99\%$. O teste ficou $0{,}20$ ponto acima, dentro da diferença média de $0{,}35$ que vi na 4.15 (o teste é um pouco mais fácil).
- O modelo escolhido pelo teste na seção 4.14 teve $160$ erros ($98{,}40\%$). Este teve $181$. Essa diferença de $21$ erros é compatível com o que eu esperava de uma escolha feita sobre o teste, que sai otimista, mais o ruído de uma execução. Com uma semente cada, não consigo separar os dois efeitos.

**Desempenho por dígito** (calculado da matriz de confusão):

| Dígito | Revocação | Observação |
| :---: | :---: | :--- |
| $1$ | $99{,}56\%$ | Melhor. Traço simples e único. |
| $0$ | $98{,}98\%$ | |
| $3$ | $98{,}51\%$ | Precisão baixa ($\approx 97{,}5\%$): outros dígitos viram $3$. |
| $4$ | $98{,}27\%$ | |
| $2$ | $97{,}97\%$ | |
| $6$ | $97{,}91\%$ | |
| $5$ | $97{,}87\%$ | |
| $8$ | $97{,}84\%$ | |
| $7$ | $97{,}57\%$ | |
| $9$ | $97{,}22\%$ | Pior revocação e precisão baixa ($\approx 97{,}5\%$). |

**Dígitos mais confundidos (Seção II.C):**

1. **$9 \to 4$ ($10$ vezes) e $4 \to 9$ ($9$ vezes).** São $19$ dos $181$ erros, cerca de $10\%$. É o mesmo par que lidera a seção 4.14.3, agora no modelo final.
2. **$5 \to 3$ ($8$ vezes).** O inverso, $3 \to 5$, aparece só $3$ vezes.
3. **$7 \to 1$ ($7$ vezes) e $7 \to 2$ ($6$ vezes).** O $7$ perde para dois vizinhos diferentes.
4. **$6 \to 4$ ($6$ vezes).**

**Por quê.** Olhei os $16$ primeiros erros da galeria (`_amostras_erros.png`), que são os primeiros por índice e não uma amostra dos $181$. Leio nelas o seguinte:

- Boa parte é ambígua até para mim. Há um $6$ com laço fechado e só um pequeno gancho no topo, que a rede lê como $0$, e um $3$ com traço superior comprido, lido como $7$.
- O $5$ lido como $3$ e o $4$ lido como $9$ combinam com a explicação morfológica da 4.14.3: poucos pixels separam os dois dígitos.
- A confusão não é simétrica ($5 \to 3$ com $8$ e $3 \to 5$ com $3$). Isso sugere que a rede tem uma fronteira mais folgada para o $3$. Não testei essa leitura.

Essas explicações vêm de olhar exemplos. Para afirmá-las, eu precisaria olhar os erros de cada par.

#### 4.15.6. Estado

- **Feito:** scripts e testes do v5 (`tests/test_mlp_v5.py`), bateria de $45$ treinos, escolha pela validação e avaliação única no teste.
- **Modelo final do MLP:** $784 \to 512 \to 10$, lote $256$, $\eta = 0{,}1$, momento $0{,}9$, $407.050$ parâmetros, $98{,}19\%$ no teste, tempo de treino de $\approx 53$ s na CPU. Estes são os valores a comparar com a CNN.
- **Pendente:** as tabelas das seções 4.13 e 4.14 ficam como registro exploratório, de uma execução, sem semente fixa e com escolha feita sobre o teste.
- **Cuidado:** o teste já foi aberto. Não escolho outra configuração olhando este resultado.

---

### 4.16. Por que o MLP Ignora a Estrutura Espacial (Seção II.A)

O MLP recebe a imagem $28 \times 28$ achatada em um vetor de $784$ números. Imagino esse achatamento como cortar a imagem em $28$ fitas, uma por linha, e colar uma atrás da outra.

- **Vizinhança horizontal:** quase se mantém. Dois pixels lado a lado na mesma linha ficam lado a lado no vetor. A exceção é a borda: o último pixel de uma linha fica colado ao primeiro da linha seguinte, e eles não são vizinhos na imagem.
- **Vizinhança vertical:** se perde. Dois pixels um sobre o outro ficam a $28$ posições de distância no vetor. Só uma regra algébrica (o corte em linhas de $28$) diz que eles estão próximos, e a rede não recebe essa regra.
- **Diagonais:** também se perdem, pela mesma razão.

A camada linear $\mathbf{W}\mathbf{x}$ trata cada entrada como uma coluna independente de $\mathbf{W}$. Ela não usa a posição de uma entrada em relação à outra. Por isso, se eu aplicar a mesma permutação de pixels em todas as imagens, o MLP pode aprender uma função equivalente: basta que as colunas de $\mathbf{W}$ sejam permutadas da mesma forma. A rede não perde informação nessa troca, e também não usa a ordem que o desenho original tinha.

**Duas ressalvas:**

- Isso vale para a capacidade de aprender. Em uma execução real, os números mudam um pouco, porque a inicialização e a ordem dos lotes são sorteadas. Eu não rodei esse experimento. Ele seria um treino com pixels embaralhados por uma permutação fixa, comparando a acurácia de validação com a do original.
- A rede pode aprender relações de vizinhança a partir dos dados, pois imagens de $7$ sempre acendem pixels próximos. Mas essa relação não vem da arquitetura. A rede precisa descobri-la, e gasta pesos para isso.

**O que a CNN muda.** A convolução usa um filtro pequeno, de $5 \times 5$, deslizando pela imagem. Ela olha vizinhos de verdade, nas duas direções, e reutiliza os mesmos pesos em todas as posições. Isso reduz o número de parâmetros e usa a estrutura espacial que o MLP joga fora. A comparação em desempenho, parâmetros e tempo fica para a seção III do enunciado.

### 4.17. v6: Um Script Só para MSE e Entropia Cruzada, no Padrão do v5

**Por que fiz o v6.** A seção 4.13.7 mostra que as seções 4.8 a 4.13 não são verificáveis nem reproduzíveis. O v3 (MSE) e o v4 (Entropia Cruzada) eram dois scripts separados, sem semente, com os gráficos copiados em cada um. O v5 resolveu a semente e o checkpoint, mas só tinha entropia cruzada. Para refazer a comparação sob as mesmas regras, eu precisava de um script que aceitasse as duas perdas.

**Alternativas:**

1. **Adicionar `--seed` ao v3 e ao v4.** Descartei. Os dois scripts continuariam sem o melhor checkpoint, sem o resumo em CSV e com o gráfico duplicado.
2. **Alterar o v5.** Descartei. Os resultados da seção 4.15 vêm dele, e quero poder reproduzi-los com o mesmo código.
3. **Criar o v6 reaproveitando o v5.** Escolhi esta. O v6 importa as funções do v5 e do v4.2 que já foram testadas, e o v5 fica sem mudanças.

**O que o v6 faz:**

- `--perda ce|mse` escolhe a perda. A `mse` calcula o erro quadrático entre os $10$ logits e o alvo One-Hot, com a mesma média do `nn.MSELoss` do v3. Um teste confere que os dois dão o mesmo valor.
- Os padrões (lote $64$, $\eta = 0{,}1$, sem momento, uma camada de $300$) são os do v3 e do v4, para repetir as condições originais.
- Mantém `--seed`, o checkpoint de menor perda de validação, o teste fechado durante o treino e o resumo em CSV (`resultados/resumo_v6.csv`). Acrescentei duas colunas: a maior acurácia de validação e a época em que ela ocorreu, por causa do efeito da seção 4.15.4.
- O nome do arquivo traz a perda (`mlp_treinado_v6_mse_e50_...` ou `..._ce_...`). O `mlp_v6_testar.py` lê a perda do nome e usa o mesmo registro de aberturas do teste do v5.
- O `mlp_v6_selecionar.py` não mistura MSE com entropia cruzada. Ele escolhe como no v5 e mostra cada par MSE contra CE, com a diferença e se ela passa da tolerância.
- Os gráficos de treino ficam em `graficos_mlp.py`, um módulo só, com o nome da perda como parâmetro. Assim o gráfico de uma execução MSE não leva o rótulo "Cross-Entropy".

**Como uso.** A bateria `executar_bateria_v6.sh` roda MSE e CE com $50$ e $200$ épocas em $3$ sementes, e $500$ épocas em $1$ semente, para a curva em "U" da seção 4.9. Em CPU, a mesma semente repete as primeiras épocas de um treino mais longo (um teste confere isso), então os pontos de $3$, $50$ e $120$ épocas também saem do histórico gravado. O teste só abre em dois modelos, um por perda, escolhidos pela validação.

**Estado.** O código e os testes estão prontos ($12$ testes novos, $33$ no total passam). Uma execução curta de $2$ épocas com o MNIST real funcionou para as duas perdas, sem abrir o teste. Não interpreto os números dessa execução curta. Falta rodar a bateria completa.

---

### 4.18. A Convolução, o Compartilhamento de Pesos e o Pooling (Seção III.A)

Na seção 4.16, deixei clara a cegueira espacial do MLP: ao achatar a matriz $28 \times 28$ em um vetor de $784$ posições, cortei a imagem em $28$ fitas horizontais e perdi a noção de vizinhança vertical e diagonal. A camada linear trata cada entrada como coluna independente e precisa de $407.050$ parâmetros (no modelo campeão do v5) para reaprender na força bruta quais pixels costumam acender juntos.

A Rede Neural Convolucional (CNN) resolve esse problema mantendo a imagem no plano 2D e substituindo o olhar global de cada neurônio por uma **lupa deslizante**:

- **Convolução (o estêncil local):** Um pequeno núcleo (*kernel*) de tamanho $5 \times 5$ desliza passo a passo sobre a grade da imagem. Em cada parada, ele calcula o produto escalar entre seus $25$ pesos e os $25$ pixels daquela vizinhança imediata. Como a janela respeita a geometria 2D, pixels vizinhos na horizontal, vertical e diagonal interagem diretamente antes de qualquer achatamento.
- **Compartilhamento de pesos (*weight sharing*):** Em vez de aprender um detector de borda diferente para o canto superior esquerdo e outro para o centro da imagem, o mesmo filtro de $25$ pesos varre a imagem inteira. Fisicamente, isso impõe uma **simetria de conservação por translação**: um arco ou uma quina continuam sendo o mesmo traço geométrico onde quer que apareçam no quadro $28 \times 28$.
- **Pooling (subamostragem):** Após a convolução e a ativação ReLU, aplico `MaxPool2d(2, 2)` com passo $2$, dividindo a altura e a largura pela metade e retendo apenas o maior sinal de cada bloco $2 \times 2$.

Durante a concepção desta etapa, esbarrei em duas dúvidas conceituais imediatas que precisei resolver antes de escrever a arquitetura:

#### Tropeço Lógico 1: O Ceticismo do MaxPool ("Estamos jogando 75% dos dados no lixo?")

**Minha dúvida inicial.** Ao olhar a operação de `MaxPool2d` em janelas $2 \times 2$, a reação instintiva foi de desconfiança: de cada $4$ valores calculados pelo filtro, guardamos apenas $1$ e descartamos os outros $3$. Se eliminarmos $75\%$ dos dados duas vezes seguidas (indo de $24 \times 24$ para $12 \times 12$, e depois de $8 \times 8$ para $4 \times 4$), não estaríamos destruindo a nitidez necessária para distinguir um $4$ de um $9$?

**A correção física.** O pooling não descarta a presença do padrão, mas sim a **coordenada milimétrica exata** onde ele ocorreu. Depois que um filtro convolucional detectou um traço horizontal ou uma curvatura fechada (gerando um valor positivo alto naquele mapa de ativação), o que importa para a classificação é saber *que a feição existe naquela região*, e não se ela caiu no pixel $(10, 14)$ ou $(11, 15)$. Reter o máximo preserva o sinal mais forte da vizinhança e produz **invariância local a pequenas translações e inclinações caligráficas**, ao mesmo tempo em que reduz o mapa espacial para que a convolução seguinte enxergue uma fração maior do dígito inteiro (aumento do campo receptivo).

#### Tropeço Lógico 2: A Geometria 3D dos Canais ("O filtro da segunda convolução não tem só 25 pesos?")

**Minha dúvida inicial.** Na primeira camada (`1 -> 6`), temos $6$ filtros de tamanho $5 \times 5$, resultando em $6 \times (25 + 1) = 156$ parâmetros. Ao passar para a segunda convolução (`6 -> 16`), calculei intuitivamente que cada filtro teria novamente $25 + 1 = 26$ parâmetros, achando que os $16$ filtros custariam $416$ pesos.

**A correção geométrica.** Esse cálculo trata a saída da primeira camada como se fosse uma única folha plana. Na prática, o primeiro bloco entrega um **volume de $6$ mapas de características** com formato $(6, 12, 12)$. Cada um dos $16$ filtros da segunda convolução precisa combinar informação de todos os $6$ canais simultaneamente; logo, ele não é um quadrado 2D, mas um **paralelepípedo 3D de dimensões $5 \times 5 \times 6 = 150$ pesos** (somados a $1$ bias). A segunda convolução custa, portanto, $16 \times (150 + 1) = 2.416$ parâmetros.

---

### 4.19. Anatomia da LeNet (`cnn_v1`), Armadilha do Softmax Duplo e o Gargalo Denso (Seção III.B)

Seguindo a linhagem clássica de LeCun et al. (1998) e o enunciado da Seção III.B, implementei a classe `LeNet` em [`models/cnn.py`](models/cnn.py) com dois blocos convolucionais seguidos de três camadas lineares:

$$(1, 28, 28) \xrightarrow{\text{Conv1 } (5\times 5)} (6, 24, 24) \xrightarrow{\text{Pool1 } (2\times 2)} (6, 12, 12) \xrightarrow{\text{Conv2 } (5\times 5)} (16, 8, 8) \xrightarrow{\text{Pool2 } (2\times 2)} (16, 4, 4) \xrightarrow{\text{Flatten}} 256 \to 120 \to 84 \to 10$$

#### Tropeço Lógico 3: A Armadilha do Softmax Duplo

**Minha dúvida inicial.** Como o enunciado trata da classificação probabilística em $10$ classes, a tentação imediata ao escrever o `forward()` da `LeNet` foi encerrar a rede com `nn.Softmax(dim=-1)`.

**A correção numérica.** No PyTorch, `nn.CrossEntropyLoss()` já combina `LogSoftmax` e `NLLLoss` em uma única operação estável (via identidade *log-sum-exp*). Se o `forward()` aplicasse `Softmax` antes, a função de perda aplicaria um segundo `LogSoftmax` sobre probabilidades já comprimidas em $[0, 1]$, achatando as diferenças entre classes e degradando os gradientes. Por isso, a `LeNet` — assim como o `MLP` das linhas v4 e v5 — devolve **logits puros** no `forward()`.

#### Diagnóstico do Gargalo Paramétrico

Ao somar os parâmetros treináveis camada por camada em `LeNet.resumo_parametros()`, surgiu um contraste revelador em relação ao MLP campeão do v5 ($784 \to 512 \to 10$, com **$407.050$** parâmetros):

| Bloco / Camada | Formato dos Pesos | Parâmetros | Fração da LeNet |
| :--- | :--- | :---: | :---: |
| `Conv1` ($1 \to 6$, $5\times 5$) | $(6, 1, 5, 5) + 6$ | $156$ | $0{,}35\%$ |
| `Conv2` ($6 \to 16$, $5\times 5$) | $(16, 6, 5, 5) + 16$ | $2.416$ | $5{,}44\%$ |
| **Subtotal Convolucional** | — | **$2.572$** | **$5{,}79\%$** |
| `Linear 1` ($256 \to 120$) | $(120, 256) + 120$ | $30.840$ | **$69{,}42\%$** |
| `Linear 2` ($120 \to 84$) | $(84, 120) + 84$ | $10.164$ | $22{,}88\%$ |
| `Linear 3` ($84 \to 10$) | $(10, 84) + 10$ | $850$ | $1{,}91\%$ |
| **Subtotal Linear (Denso)** | — | **$41.854$** | **$94{,}21\%$** |
| **Total LeNet** | — | **$44.426$** | **$100{,}00\%$** |

Dois fatos saltam dessa tabela:
1. A `LeNet` inteira tem **$44.426$ parâmetros** — menos de $11\%$ do tamanho do MLP campeão ($407.050$ parâmetros, uma redução de $89{,}1\%$).
2. Quase todo o custo paramétrico da `LeNet` não está nas convoluções (que gastam apenas $2.572$ pesos, ou $5{,}79\%$), mas na **ponte densa de entrada do classificador** (`Linear 256 -> 120`), que sozinha concentra $69{,}42\%$ de todos os pesos da rede.

**Blindagem TDD (`tests/test_cnn.py`).** Criei $6$ testes estruturais automatizados validando: (i) os shapes intermediários exatos de cada camada; (ii) a contagem analítica de parâmetros por bloco e a razão de gargalo na `fc1`; (iii) a saída em logits puros; (iv) fluxo de gradientes não-nulos em todas as camadas; (v) sensibilidade da CNN à permutação fixa dos pixels (provando que ela de fato depende da topologia espacial).

---

## 5. Log de Atividades

| Data | Atividade / Marco | Decisões e Entendimentos Registrados |
| :--- | :--- | :--- |
| **11/10/2026** | Inicialização e Análise do Problema | Leitura do enunciado oficial (`03-projeto-visao-simples.pdf`), mapeamento das demandas (MNIST, normalização, MLP vs CNN, hiperparâmetros e análise morfológica de confusões) e correlações bibliográficas com Russell & Norvig (Cap. 22) e Smola (*D2L*, Caps. 4, 5 e 7). |
| **14/10/2026** | Fundamentação Física da Derivada | Resgate dos fundamentos do cálculo diferencial: analogia mecânica da calibragem do sistema de freio de um carro (parar de 100 km/h a 0 km/h na marca de 15m), conceito de proporção dinâmica instantânea e geometria da reta tangente ($\tan \theta$). |
| **18/10/2026** | A Camada Neural e o Dummy `+1` | Análise da Figura 22.3 do Russell. Compreensão do dummy input `+1` (coordenadas homogêneas) para embutir o bias $w_0$ na multiplicação matricial limpa, a matriz de pesos pertencendo à camada e a operação de soma de redução. |
| **21/10/2026** | Discussão de Paradigma: POO vs. Tensores | Reflexão histórica de 2001: por que descarte de classes `Perceptron` com nós de grafo e threads concorrentes (evitando fragmentação de memória e *cache misses*) em favor de matrizes contíguas de alta performance na GPU. |
| **26/10/2026** | Alinhamento de Escopo e Ferramental | Minha decisão metodológica de focar estritamente no ecossistema PyTorch, descartando implementações redundantes do zero e priorizando a geração rica de gráficos analíticos e diagnósticos visuais. |
| **29/10/2026** | Pipeline de Dados e Divisão (Split) | Implementação de `dataset.py`. Download oficial do MNIST e cálculo empírico da média ($\mu = 0.1307$) e desvio padrão ($\sigma = 0.3081$). Divisão rigorosa em Treino ($50.000$), Validação ($10.000$) e Teste ($10.000$). |
| **29/10/2026** | Exploração Visual e Blindagem TDD | Geração dos gráficos em 300 DPI em `figuras/` (balanceamento das classes e amostras caligráficas). Criação de `tests/test_dataset.py` e configuração de `pytest.ini`, com 5 testes automatizados aprovados (100% de sucesso). |
| **29/10/2026** | Consolidação Teórica dos Dados | Compreensão do porquê de $\mu \approx 0.1307$ (dados esparsos com 87% de fundo preto), a física da normalização (analogia com o centro de gravidade do Projeto 1 para eliminar oscilações em zigue-zague no gradiente) e a ética do teste lacrado. |
| **29/10/2026 (Noite)** | Epistemologia Física: Molas e Entropia | Descoberta da equivalência entre a função de perda e a energia elástica de uma mola (Lei de Hooke e alívio de tensão com matrizes transpostas), emergência de propriedades por quebra de simetria (*Three Body Problem*) e convergência para o estado de mínima entropia de Shannon. Criação de `laboratorio_backprop.py`. |
| **30/10/2026** | Concepção do Estudo Dirigido Interativo | Minha solicitação ao Antigravity para criarmos um laboratório interativo (`estudo_dirigido.html`) baseado no Russell & Norvig e Smola, unificando a visão biológica do perceptron com a álgebra matricial em LaTeX para eu poder experimentar valores. |
| **04/11/2026 (Manhã)** | O Toy Model 2x2 e a Solução do Dying ReLU | Meu experimento prático no Jupyter Notebook com o detector de diagonais. Diagnóstico do gradiente nulo quando $z < 0$ com ReLU. Treinamento com SGD demonstrando o surgimento espontâneo do peso inibitório ($w_2 \approx -0.320$) para anular horizontais e convergência a 0.996. |
| **04/11/2026 (Tarde)** | Estudo Dirigido com Matrizes LaTeX e Backprop | Evolução do `estudo_dirigido.html` que pedi ao Antigravity com animação completa das 3 fases, notação formal de matrizes em LaTeX com colchetes, onda reversa via $\mathbf{W}^\top$ e integração com os pesos que treinei no notebook. |
| **04/11/2026 (Fim de Tarde)** | O Salto Conceitual do One-Hot e Benchmark LeCun | Meu insight sobre a falácia de tratar dígitos como números escalares e adoção formal da codificação One-Hot em $\mathbb{R}^{10}$ (*1-of-K coding* de LeCun). Treinamento e replicação dos experimentos clássicos de LeCun (1998) com 1 e 2 camadas ocultas, atingindo 4.59% de erro de teste nas 10.000 amostras oficiais. |
| **05/11/2026 (Manhã)** | Diagnóstico de Gargalo de CPU e Otimização com TensorDataset | Profiling empírico de `mlp_v2_treinar.py` na CPU Intel i7-11390H. Descoberta de que o forward/backward consumia apenas 2.37s e mais de 80% do tempo era gasto em overhead do `DataLoader` e transformações repetidas no Python. Implementação de pré-carregamento contíguo em RAM via `TensorDataset` (~150 MB) e One-Hot em C++, reduzindo o tempo de treino de 19.78s para 4.76s (>4x mais rápido) com 95.4% de acurácia. |
| **05/11/2026 (Tarde)** | Experimento de Escala (5 vs. 50 vs. 120 Épocas) e Reflexão sobre Grokking | Evolução de `mlp_v3_treinar.py` com CLI dinâmico (`-e`). Execução de baterias com 5, 50 e 120 épocas no terminal. Redução da taxa de erro de teste para 2.07% (97.93% de acurácia) mantendo estabilidade linear de ~1.25s/época na CPU. Diagnóstico do platô de validação e análise teórica contrastando o regime clássico do MNIST com o fenômeno de generalização tardia (*Grokking*, Power et al., 2022). |
| **05/11/2026 (Noite)** | Desacoplamento Arquitetural: Treinamento vs. Testes Independentes | Separação padronizada do pipeline v3 em dois módulos canônicos: `mlp_v3_treinar.py` (exclusivo para treino e validação, mantendo o teste 100% lacrado sem data leakage) e `mlp_v3_testar.py` (avaliação independente de checkpoints .pth sobre as 10.000 imagens de teste em 0.13s / >76.000 imagens/s). |
| **06/11/2026 (Manhã)** | Experimento Extremo de 500 Épocas e Justificativa do Early Stopping | Execução de `mlp_v3_treinar.py -e 500` (766s na CPU) e teste com `mlp_v3_testar.py`. Evidência empírica de overfitting: perda de validação formando curva em "U" (mínimo de 0.00745 na época 172, subindo para 0.00813) e piora da acurácia de teste oficial de 97.93% (207 erros) para 97.72% (228 erros, +21 falhas). Documentação dos limites teóricos (teto de 300 neurônios, Erro de Bayes do MNIST e ausência de weight decay) e adoção do Early Stopping como critério de parada. |
| **06/11/2026 (Manhã)** | Padronização de Nomenclatura por Família (`mlp_v*_***.py`) | Reorganização e renomeação padronizada de todos os scripts da família MLP (`mlp_v1_treinar.py`, `mlp_v2_treinar.py`, `mlp_v3_treinar.py` e `mlp_v3_testar.py`) para ordenação alfabética natural por arquitetura e etapa de evolução, facilitando a navegação no repositório. |
| **06/11/2026 (Manhã)** | Geração Automática de Suíte Modular de Gráficos Pareados | Integração da função `salvar_graficos_treinamento()` em `mlp_v3_treinar.py`, gerando automaticamente na pasta `figuras/` um conjunto modular de 5 gráficos em alta resolução (300 DPI) seguindo o padrão `mlp_treinado_v3_e{E}_<tipo>.png`: 1) `_perda.png` (curva em U e zona de overfitting); 2) `_acuracia.png` (convergência e platô); 3) `_gap_generalizacao.png` (monitor de sobreajuste Treino - Validação); 4) `_tempo_execucao.png` (estabilidade temporal e taxa s/época); e 5) `_painel_geral.png` (dashboard 2x2 integrando as 4 dimensões). Geração retroativa para 3, 50, 120 e 500 épocas. |
| **06/11/2026 (Manhã)** | Avaliação Completa no Teste com Métricas e Heatmaps | Evolução de `mlp_v3_testar.py` para cálculo de Precisão, Recall e F1-score (por classe e macro), Matriz de Confusão 10x10, identificação das 5 confusões mais frequentes e geração de 4 relatórios visuais pareados: `_matriz_confusao.png`, `_metricas_classes.png`, `_amostras_erros.png` (grade de imagens reais com erro) e `_painel_teste.png`. Testes unitários atualizados em `tests/test_mlp.py` (12/12 testes aprovados). |
| **06/11/2026 (Tarde)** | Organização Estruturada dos Gráficos por Regime de Épocas | Reestruturação do salvamento de figuras em `mlp_v3_treinar.py` e `mlp_v3_testar.py` para isolamento em subdiretórios dedicados `figuras/e{num_epocas}/` (ex.: `figuras/e3/`, `figuras/e50/`, etc.). A modularização mantém a pasta raiz limpa (preservando apenas `distribuicao_classes.png` e `exemplos_digitos.png`), agrupando os 9 relatórios visuais específicos de cada experimento de forma coesa, elegante e escalável. |
| **06/11/2026 (Tarde)** | Validação e Benchmark na GPU NVIDIA RTX A1000 | Configuração do ambiente PyTorch com CUDA 12.4 na máquina secundária (Dell Precision 3581). Aceleração de treinamento medida: redução do tempo por época de 1.53s para 0.54s (ganho de 2.83x) e 500 épocas concluídas em apenas 4m34s (vs. 12m46s na CPU). Bateria experimental completa (50, 100, 200 e 500 épocas) indicando o pico de generalização em 200 épocas (97.89% / 211 erros) e a degradação no teste oficial lacrado a 500 épocas (97.68% / 232 erros, aumento de 50% nas confusões do dígito 4 com 9). |
| **06/11/2026 (Fim de Tarde)** | Concepção Metodológica da Linha v4 (Softmax & Cross-Entropy) | Planejamento da transição conceitual para a Seção II do enunciado. Decisão de controle rigoroso de variáveis (*ceteris paribus*): clonar a mesma arquitetura 784 -> 300 -> 10, batch 64 e SGD (lr=0.1) do v3, alterando unicamente a perda de MSE para `nn.CrossEntropyLoss()`. Formalização teórica de por que o One-Hot é desnecessário na Entropia Cruzada (colapso da somatória em $-\ln P_k$ e eliminação de matrizes esparsas em RAM/VRAM). |
| **06/11/2026 (Noite)** | Implementação e Confronto Empírico v3 vs. v4 (50 e 200 Épocas) | Implementação de `mlp_v4_treinar.py` e `mlp_v4_testar.py`. Execução da bateria experimental completa (*ceteris paribus*) comparando v3 (MSE) e v4 (Softmax) para 50 e 200 épocas na CPU. Resultados compatíveis com as 4 hipóteses: menor erro de teste até então, com **98.30% de acurácia no teste a 50 épocas (apenas 170 erros, redução de 29.5% em relação ao MSE de 50 épocas)**, memorização de 100% do treino na época 17-18, diagnóstico da degradação de calibração de probabilidades a 200 épocas (aumento de +32% na perda de validação com acurácia estável) e supressão expressiva dos erros crônicos ($5 \to 3$ caindo de 13 para 6). Identificação do intervalo de 30 a 50 épocas como a janela ideal de Early Stopping. Geração do acervo completo de 36 relatórios visuais pareados em `figuras/e50/` e `figuras/e200/`. |
| **07/11/2026** | Exploração Sistemática de Hiperparâmetros na RTX 3070 (Bateria v4.2) | Execução automatizada de 15 experimentos controlados investigando arquitetura (largura de 128 a 512, profundidade de 1 a 3 camadas), batch size (64 a 512), SGD com momentum (0.9), Adam e AdamW. Medição do speedup de 3.0x no batch 512 sem perda de precisão e maior acurácia de teste entre as 15 configurações: arquitetura de 2 camadas (300 -> 100) com SGD Momentum 0.9, 98.40% de acurácia (escolha feita sobre o teste; revista no v5) (apenas 160 erros em 10.000 imagens de teste). Catalogação completa de gráficos e análise morfológica das confusões crônicas (4 vs 9, 5 vs 3, 7 vs 2/9). |
| **09/11/2026** | Linha v5: melhor modelo pela validação e semente fixa | Revisão da bateria 4.14: o modelo de 98.40% foi escolhido pelo teste e fica em 12º de 15 pela validação. Criação de `mlp_v5_treinar.py` (checkpoint de menor perda de validação, `--seed`, resumo em CSV), `mlp_v5_testar.py` (registro das aberturas do teste), `mlp_v5_selecionar.py` (escolha pela validação) e `executar_bateria_v5.sh`; 10 testes novos. Ajuste de tom e de afirmações da metodologia. |
| **09/11/2026 (Tarde)** | Bateria v5: 15 configurações x 3 sementes na CPU | Redesenho da bateria em blocos (largura, profundidade, lote, taxa, momento), sem Adam/AdamW, pois o enunciado fixa o SGD. 45 treinos só com treino e validação. Configuração escolhida pela validação: 1 camada de 512 neurônios (97.95 ± 0.08%). Achado: em 44 de 45 execuções a última época tem acurácia maior que o checkpoint de menor perda, e o ranking depende do critério (seção 4.15.4). Teste aberto uma vez no modelo escolhido: 98.19% (181 erros), F1 macro 98.18% (seção 4.15.5). |
| **09/11/2026 (Tarde)** | Discrepância das figuras antigas e criação do v6 | Constatação de que as figuras e os checkpoints das seções 4.8 a 4.13 não estão no repositório (seção 4.13.7). Criação de `mlp_v6_treinar.py` (`--perda ce|mse`, semente, melhor checkpoint), `mlp_v6_testar.py`, `mlp_v6_selecionar.py`, `executar_bateria_v6.sh` e `graficos_mlp.py`; 12 testes novos (seção 4.17). Bateria ainda por rodar. |
| **09/11/2026 (Noite)** | Estudo dirigido: ReLU morta, saturação e escala | Reimplementação da página com treino de verdade, matrizes editáveis, dummy +1 e escolha de ativação (ReLU, tanh, logística). Medidas (`tests/estudo_dirigido_medidas.js`): a ReLU nasce morta em cerca de 50% dos sorteios; com $x$ até 20, 83% dos neurônios tanh e 68% dos logísticos saturam; alvo no limite da faixa da tanh leva 999 passos contra 40 com alvo 0,5 (seção 4.6.1). |
| **09/11/2026 (Tarde)** | Por que o MLP ignora a estrutura espacial | Minha analogia das fitas: cortar a imagem em linhas e colar uma atrás da outra mantém a vizinhança horizontal e desfaz a vertical (seção 4.16). |
| **13/11/2026** | Fundamentação da Convolução e LeNet (`models/cnn.py`) | Redação das Seções 4.18 e 4.19 (lupa deslizante, weight sharing e pooling) com resolução dos tropeços lógicos (ceticismo do MaxPool, geometria 3D $5\times 5\times 6$ dos filtros e armadilha do Softmax duplo). Implementação de `models/cnn.py` ($44.426$ parâmetros, $69{,}42\%$ concentrados na `fc1`) e $6$ testes estruturais aprovados em `tests/test_cnn.py`. |
