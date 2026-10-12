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

### 1.7. A Analogia da Montanha e do Rio (Alinhamento com o Professor da Disciplina)

Ao discutir essas intuições diretamente com o professor da disciplina (Prof. Gilberto Medeiros Nakamura), a metáfora que sintetizou perfeitamente a mecânica do algoritmo foi a descida da montanha em direção ao vale:
- **O gradiente como curso do rio:** *"O gradiente é como descer uma montanha. Você varia o passo e vê como altera a altura (a função). O rio sempre descreve um gradiente."* A água não toma decisões discretas; ela simplesmente segue a inclinação máxima do relevo rumo ao ponto mais baixo.
- **A anulação no ponto de mínimo:** *"Num ponto de mínimo, o gradiente se anula (como num vale entre duas montanhas, pois a inclinação é nula)"*. Quando a fita atinge o objetivo ($0.0$), a inclinação da parábola ($2x$) se extingue, cessando naturalmente o movimento.
- **Passos mais curtos perto do vale:** *"Perto do mínimo, o gradiente se aproxima de zero. Daí o motivo dos passos serem mais curtos perto dos mínimos."* Isso valida formalmente a analogia do coelho de Zenão e o produto $(\eta \times \text{Derivada})$: a própria física da superfície atua como um amortecedor suave, desacelerando a descida conforme a fita repousa no chão.

---

### 1.8. Comprovação Empírica dos Regimes de Passo no Jupyter Notebook

Para validar toda a teoria na prática antes de modularizar o código, montei um ambiente iterativo no Jupyter Notebook com uma fita reduzida `[0.0, 1.0, 0.0]`, monitorando a função de perda quadrática $\mathcal{L} = \sum x^2$ e o gradiente $\nabla \mathcal{L} = 2x$ gerado automaticamente pelo PyTorch (`autograd`).

Executei baterias de 50 iterações variando sistematicamente a taxa de aprendizado ($\eta$), comprovando 4 comportamentos empíricos fundamentais:

1. **Regime Lento e Cauteloso ($\eta = 0.1$):**
   - O valor decaiu de forma estritamente suave e monótona ($1.00 \to 0.80 \to 0.64 \to 0.51 \dots$).
   - A inclinação foi diminuindo a cada passada (o amortecimento natural), exigindo **cerca de 45 passos** para que o erro ficasse abaixo de $10^{-6}$. Demonstra com clareza a corrida do coelho de Zenão.

2. **Regime Acelerado e Estável ($\eta = 0.2$):**
   - A descida foi duas vezes mais agressiva ($1.00 \to 0.60 \to 0.36 \dots$).
   - O erro convergiu para zero na metade do tempo (em torno da iteração 20), sem nenhuma instabilidade ou oscilação.

3. **Regime Crítico Perfeito ($\eta = 0.5$):**
   - Como a distância até a origem é $1.0$ e o gradiente inicial é $2.0$, o passo $\Delta = 0.5 \times 2.0 = 1.0$.
   - **Resultado:** A fita atingiu exatamente `[0.0, 0.0, 0.0]` no **Passo 1**, zerando a função de perda imediatamente em uma única iteração. É o ponto analítico ótimo de convergência em passo único para parábolas puras.

4. **Regimes com Overshoot e Oscilação Amortecida ($\eta = 0.6$ e $\eta = 0.8$):**
    - **Para $\eta = 0.6$:** No primeiro passo o valor saltou para **$-0.20$** (passou do chão e ficou negativo!). Como o número agora é negativo, na iteração seguinte a ladeira empurrou para o lado oposto, e o valor voltou para $+0.04$, depois $-0.008$.
    - **Para $\eta = 0.8$:** O salto foi ainda mais brusco, aterrissando em **$-0.60$** no primeiro passo, depois $+0.36$, depois $-0.216$.
    - **Conclusão:** Valores de $\eta > 0.5$ comprovadamente quebram a restrição de domínio da lógica fuzzy ($[0.0, 1.0]$) e produzem uma oscilação pendular (quicando entre positivo e negativo) até a energia do sistema se dissipar.

---

### 1.9. A Geometria da Bacia: Desmistificando o "Ponto de Sela" e a Relação com Tangente Hiperbólica ($\tanh$)

Durante a criação dos gráficos de curvas de nível, surgiram dois insights conceituais de extrema relevância:

1. **Por que a superfície não é uma sela (Saddle Point)?**
   - Inicialmente, imaginei que a busca pelo melhor $\eta$ pudesse formar uma superfície em sela (como num problema de *minimax*, onde uma direção quer maximizar e a outra quer minimizar).
   - Compreendi que, no nosso problema, ambas as componentes da perda buscam a minimização simultânea: tanto a distância ao solo ($\sum x^2 \to 0$) quanto a quebra de vizinhos ($C(\mathbf{z}) \to 0$).
   - O relevo é uma **bacia parabólica convexa circular em $\mathbb{R}^3$** ($z = x_1^2 + x_2^2$), onde todas as direções inclinam para o mesmo fundo do vale. 
   - Ao desenhar as trajetórias descendo de múltiplos ângulos na bacia, ficou evidente que a física é perfeitamente simétrica: taxas $\eta \le 0.5$ convergem em linha reta para o centro, enquanto $\eta = 0.6$ atravessa a estrela da origem e perfura o lado oposto em $(-0.2, -0.2)$.

2. **Por que o overshoot viola o domínio e a conexão com a Tangente Hiperbólica ($\tanh$):**
   - O overshoot negativo só é uma falha grave aqui porque adotamos a semântica da lógica fuzzy e da função sigmoide, cuja imagem está estritamente contida em $[0.0, 1.0]$. Um valor negativo representa probabilidade/ativação inexistente.
   - Refleti sobre como modelos mais maduros de redes neurais resolveram essa limitação adotando a **Tangente Hiperbólica ($\tanh$)**: uma função suave centrada na origem, variando no intervalo $[-1.0, 1.0]$, onde valores negativos são permitidos e representam inibições simétricas.

3. **Verificação Formal e Engenharia de Software:**
   - Para assegurar que o módulo `gd_puro.py` atende a todos os requisitos matemáticos, implementei uma suíte de testes com `pytest` (`tests/test_gd_puro.py`).
   - Foram validadas com 100% de sucesso a equivalência entre o gradiente analítico ($2x$) e o `autograd`, o término em 1 passo analítico para $\eta = 0.5$, a detecção precisa de overshoot para $\eta = 0.6$ e a convergência estrita da fita de 23 bits abaixo do limiar de $10^{-6}$.

---

### 1.10. A Dinâmica da Função de Custo Regularizada ($J(\mathbf{x}) = \mathcal{L} + \lambda C$) e a "Chicotada Elástica"

Para atender ao Item III.D do enunciado, implementei a função de custo contíguo que penaliza a tensão de cisalhamento entre vizinhos:
$$C(\mathbf{x}) = \sum_{k=1}^{L-1} (x_{k+1} - x_k)^2$$

Testando no Jupyter Notebook a sequência de teste `[0, 1, 0, 0, 1, 1, 1, 0]` fixando a taxa $\eta = 0.1$ e variando o hiperparâmetro de acoplamento $\lambda \in \{0.0, 0.5, 20.0\}$, obtive uma descoberta empírica fundamental:

1. **Sem regularização ($\lambda = 0.0$):**
   - O gradiente puro atua de forma independente. O bit solitário e os bits do bloco sofrem exatamente a mesma força de descida ($2x_i = 2.0$). 
   - A convergência é asséptica e sem comunicação espacial.

2. **Acoplamento equilibrado ($\lambda = 0.5$):**
   - A dinâmica espacial opera como previsto: o miolo do bloco de 3 bits (posição 5) tem tensão interna zero e desce com força $2.0$.
   - O bit solitário (posição 1) sofre tração dupla das duas fronteiras zeradas, triplicando a força de restauração.
   - Os elementos em zero vizinhos aos blocos sobem ligeiramente para suavizar o degrau (o efeito de rampa/desfoque), e a sequência se assenta como uma onda amortecida, convergindo suavemente para zero no passo 25.

3. **Super-acoplamento e a Chicotada Elástica Divergente ($\lambda = 20.0$):**
   - O teste revelou um fenômeno espetacular: com $\lambda = 20.0$ e $\eta = 0.1$, a perda explodiu de $84.0$ no passo 1 para $1.57 \times 10^8$ no passo 4 e atingiu o infinito (`inf`, com valores em $10^{27}$) antes do passo 20!
   - **A dedução analítica:** A inclusão da penalidade de corda adiciona termos da forma $2\lambda(x_i - x_{i-1}) + 2\lambda(x_i - x_{i+1})$. Na pior configuração (alternância entre 0 e 1), o módulo do gradiente atinge até $2 + 4\lambda$. Para $\lambda = 20$, essa força salta para $82.0$.
   - Ao multiplicar por $\eta = 0.1$, o salto efetivo por iteração vira $0.1 \times 82.0 = \mathbf{8.2}$!
   - Como qualquer passo efetivo maior que $1.0$ gera divergência explosiva (conforme demonstrado por Smola na pág. 502), o sistema sofreu uma **chicotada elástica incontrolável**, ejetando os valores para fora de qualquer escala finita.
   - **Conclusão de Engenharia para o Relatório:** Os hiperparâmetros $\lambda$ e $\eta$ são intimamente acoplados. Aumentar a penalidade de bloco $\lambda$ torna a superfície de perda muito mais íngreme, exigindo obrigatoriamente a redução proporcional da taxa de aprendizado ($\eta \le \frac{1}{2 + 4\lambda}$) para garantir a estabilidade numérica.

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
| 08/10/2026 | Síntese e Alinhamento Docente | Registro da analogia da montanha e do rio (Prof. Nakamura) e validação da convergência suave no vale nulo. |
| 11/10/2026 | Comprovação Prática no Jupyter | Execução experimental dos 4 regimes de taxa de aprendizado ($\eta \in \{0.1, 0.2, 0.5, 0.6, 0.8\}$), registrando a convergência monótona, o passo crítico unitário e o overshoot oscilatório. |
| 11/10/2026 | Engenharia, Gráficos e Testes | Implementação do módulo `gd_puro.py`, geração dos gráficos multidirecionais e heatmaps em `graficos/`, suíte `pytest` aprovada (6/6) e reflexão geométrica sobre bacias convexas e $\tanh$. |
| 11/10/2026 | Estudo do Acoplamento $\lambda$ | Validação empírica no Jupyter com $\lambda \in \{0.0, 0.5, 20.0\}$, dedução da cota de força máxima $2 + 4\lambda$ e da chicotada divergente. |


