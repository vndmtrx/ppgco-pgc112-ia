# Metodologia — Parte 1: Buscas Clássicas

## 1. Definições e Modelagem do Problema

### Comparação entre Modelos de Representação

No início do projeto, avaliamos maneiras de representar a sequência de bits no Python:

1. **Strings textuais (`"00001001..."`):**
   - Muito intuitiva, mas toda vez que invertemos bits precisamos recortar a string e criar uma nova (`seq[:i] + ... + seq[i+1:]`). Em buscas com milhares de nós, isso consome muita memória e fica lento.
2. **Listas / Vetores (`[0, 1, 0, 0...]`):**
   - É fácil alterar elementos, mas listas gastam muita memória por item. Além disso, listas não entram direto no conjunto de visitados (`set`), precisando converter para tupla o tempo todo.
3. **Tensores PyTorch (`torch.uint8`):**
   - Excelente para cálculos numéricos e gradientes contínuos (que usaremos na Parte 2), mas desnecessário e pesado para buscas discretas em grafos.
4. **Bytes e Bytearray (`bytes`, `bytearray`):**
   - Chegamos a cogitar essa abordagem, mas percebemos que as operações de bit a bit não têm suporte nativo direto sobre eles (como aplicar XOR diretamente em partes da sequência). Além disso, teríamos problemas semelhantes com o conjunto de visitados (`set`): o `bytearray` é mutável e não entra direto em um `set`, exigindo conversões constantes (como ocorria entre listas e tuplas).
5. **Inteiros nativos com operações bit a bit (`int`):**
   - A sequência inteira de zeros e uns é guardada como um único número inteiro. O flip de 1 bit ou de vários bits vira uma conta direta de hardware usando XOR (`^`) e deslocamento (`<<`).
   - Foi a nossa escolha por ser muito leve e rápida. Um inteiro de 4096 bits gasta menos de 600 bytes de memória, e o cálculo de hash para saber se o estado já foi visitado é instantâneo.
   - *Convenção de Endianness:* Adotamos a ordenação da direita para a esquerda (onde a posição 0 é o bit menos significativo, $2^0$). Essa escolha simplificou enormemente as operações bitwise, permitindo gerar máscaras diretamente com `1 << posicao` e `((1 << tamanho_bloco) - 1) << posicao`, sem a necessidade de recalcular índices invertidos (`L - 1 - posicao`) a cada operação.

---

### Discussão de Design: Classes de Fita vs. Funções Puras

Durante o desenvolvimento, chegamos a experimentar modelar o problema usando classes orientadas a objetos:

1. **Primeira ideia (`FitaBits` como estado completo):**
   - Criamos uma classe imutável (`@dataclass(frozen=True)`) que guardava o número inteiro e o tamanho $L$. Cada flip gerava uma nova instância de fita.
   - *Por que descartamos:* Em buscas como o BFS, onde vamos gerar milhares de estados, criar um objeto novo no heap do Python a cada flip adiciona um custo desnecessário de memória e coleta de lixo.
2. **Segunda ideia (`FitaBits` como configurador/máscara do ambiente):**
   - Pensamos em usar a classe apenas como um configurador fixo (guardando o tamanho $L$, formatando strings e gerando as máscaras), enquanto os flips operavam sobre inteiros.
   - *Por que simplificamos:* Embora fizesse sentido, ainda trazia uma complexidade de classes que o enunciado não pedia. O enunciado pede apenas "subrotinas de flip".
3. **Decisão final (Funções puras):**
   - Adotamos funções soltas simples e diretas, sem classes. Os estados são apenas números inteiros (`int`), e o tamanho $L$ é passado como argumento. É a forma mais simples e direta de implementar.

---

### Estrutura das Funções Implementadas

Optamos por manter o código o mais simples e direto possível, apenas com funções puras:

- `carregar(texto)`: Lê a string de zeros e uns e retorna o número inteiro correspondente e o tamanho $L$.
- `formatar(estado, tamanho)`: Converte o número inteiro de volta para string binária, garantindo que os zeros à esquerda sejam preservados.
- `flip_bit(estado, posicao, tamanho)`: Inverte um único bit usando XOR com a máscara `1 << posicao`. Adotamos a convenção onde a posição 0 é o bit da direita ($2^0$).
- `flip_bloco(estado, posicao, tamanho_bloco, tamanho)`: Inverte um bloco de bits contíguos usando a máscara `((1 << tamanho_bloco) - 1) << posicao`.
- **Tratamento de overflow:** As funções verificam se a posição ou a janela do bloco tentam acessar posições além do tamanho $L$, disparando `IndexError` para evitar estados inválidos.

#### Como funcionam as máscaras utilizadas

Para simplificar o entendimento das operações e facilitar futuras consultas, as máscaras de bits foram construídas da seguinte forma:

1. **Máscara de 1 bit (`1 << posicao`):**
   - Desloca o bit 1 para a esquerda até a posição desejada.
   - Exemplo: para a posição 2, temos `1 << 2` que resulta em `0b100` (4). Aplicar XOR (`^`) com essa máscara inverte apenas o bit naquela posição.

2. **Máscara de bloco contíguo (`((1 << tamanho_bloco) - 1) << posicao`):**
   - Primeiro, `(1 << tamanho_bloco) - 1` gera uma sequência com exatamente `tamanho_bloco` bits 1 alinhados na base. Por exemplo, para um bloco de 3 bits: `(1 << 3) - 1 = 8 - 1 = 7` (`0b111`).
   - Em seguida, deslocamos essa sequência inteira para a posição inicial desejada com `<< posicao`. Por exemplo, começando na posição 1: `0b111 << 1` resulta em `0b1110`.
   - O XOR com essa máscara inverte todos os bits dentro dessa janela de uma só vez, sem necessidade de laços de repetição (*loops*).

---

## 2. Instruções de Execução dos Testes

Para rodar todos os testes (unitários e doctests) com Pytest:

```bash
pytest
```

Para rodar os testes unitários via biblioteca padrão:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Para rodar os testes embutidos nas docstrings (doctests):

```bash
python -m doctest flips.py -v
```

---

## 3. Experimentos e Observações

- **Validação das Sub-rotinas (Subitem B):**
  - Criamos 10 testes unitários cobrindo conversão de texto, flips individuais, flips de bloco, reversibilidade, proteção de limites com `IndexError` e o caso de teste inicial do enunciado.
  - Criamos também 10 testes embutidos nas docstrings (doctests).
  - Todos os testes passaram em tempo inferior a 0.001s no ambiente local e no contêiner Docker oficial.

- **Perspectivas Futuras de Validação (Testes Randômicos e Verificação Formal):**
  - Para validações futuras em larga escala, é possível utilizar testes baseados em propriedades com a biblioteca **Hypothesis**, gerando automaticamente casos de teste com sequências e tamanhos $L$ variados para checar propriedades como reversibilidade ($x \oplus m \oplus m = x$).
  - Em um cenário de verificação formal, ferramentas como **Lean 4** também poderiam ser exploradas para provar propriedades matemáticas do modelo.

---

## 4. Log de Atividades

| Data | Atividade | Decisões / Observações |
| :--- | :--- | :--- |
| 16/09/2026 | Inicialização | Estrutura de pastas criada e enunciados mapeados. |
| 20/09/2026 | Sub-rotinas de Flip | Implementação de funções puras com inteiros (`int`), tratamento de limites com `IndexError` e validação com testes unitários e doctests. |
