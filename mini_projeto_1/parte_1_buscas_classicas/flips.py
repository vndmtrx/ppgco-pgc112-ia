"""Sub-rotinas simples para operações de bit-flip.

Adotamos números inteiros para representar a sequência de bits de forma
rápida e leve na memória. A posição 0 representa o bit da direita (2^0).
"""

# Regra do enunciado: qualquer flip tem custo 1
CUSTO_FLIP = 1


def carregar(texto: str) -> tuple[int, int]:
    """Lê uma string binária (ex: '00001001...') e devolve o estado em inteiro e o tamanho L.

    >>> estado, tamanho = carregar("000101")
    >>> tamanho
    6
    >>> estado
    5
    >>> carregar("0000")
    (0, 4)
    """
    if not texto:
        raise ValueError("A sequência de bits não pode ser vazia.")

    for caractere in texto:
        if caractere not in ("0", "1"):
            raise ValueError(f"Caractere inválido: '{caractere}'. Use apenas '0' e '1'.")

    # Converte o texto binário para número inteiro e pega o comprimento
    estado = int(texto, 2)
    tamanho = len(texto)
    return estado, tamanho


def formatar(estado: int, tamanho: int) -> str:
    """Mostra o número inteiro em binário com zeros à esquerda até o tamanho L.

    >>> formatar(5, 6)
    '000101'
    >>> formatar(0, 4)
    '0000'
    """
    if estado < 0:
        raise ValueError("O estado não pode ser negativo.")

    # Verificação simples de overflow: o valor não pode ter mais bits que o tamanho da fita
    if estado >= (1 << tamanho):
        raise IndexError(f"Overflow: o estado precisa de mais do que {tamanho} bits.")

    return f"{estado:0{tamanho}b}"


def flip_bit(estado: int, posicao: int, tamanho: int) -> int:
    """Inverte um único bit na posição especificada (posição 0 é o bit à direita).

    >>> flip_bit(0, 0, 3)
    1
    >>> flip_bit(5, 1, 3)
    7
    """
    # Verificação de limites da posição
    if posicao < 0 or posicao >= tamanho:
        raise IndexError(f"Posição {posicao} fora dos limites (tamanho={tamanho}).")

    # Aplica XOR no bit escolhido e retorna o novo número
    return estado ^ (1 << posicao)


def flip_bloco(estado: int, posicao: int, tamanho_bloco: int, tamanho: int) -> int:
    """Inverte um bloco de bits consecutivos a partir da posição.

    >>> flip_bloco(0, 0, 3, 5)
    7
    >>> flip_bloco(14, 1, 3, 5)
    0
    """
    # Verificações de limites da janela e prevenção de overflow
    if posicao < 0:
        raise IndexError("A posição inicial não pode ser negativa.")
    if tamanho_bloco < 1:
        raise ValueError("O tamanho do bloco deve ser de pelo menos 1 bit.")
    if (posicao + tamanho_bloco) > tamanho:
        raise IndexError(f"Overflow: bloco [{posicao}, {posicao + tamanho_bloco}) excede o tamanho {tamanho}.")

    # Cria a máscara de 1s para o bloco e aplica XOR
    mascara = ((1 << tamanho_bloco) - 1) << posicao
    return estado ^ mascara
