"""Algoritmo de Busca em Largura Vetorizado com PyTorch (SIMD / Tensores C++).

Utiliza broadcasting matricial e indexação tensorial direta em memória RAM
para expandir níveis inteiros da fronteira simultaneamente em C++ nativo.
"""

import sys
import time
from typing import Optional

import torch

from flips import carregar
from utils import executar_cli_busca, gerar_operacoes, imprimir_resultado


def bfs_torch(
    estado_inicial: int,
    tamanho: int,
    estado_alvo: int = 0,
    retornar_estatisticas: bool = False,
) -> tuple[list[tuple], int] | tuple[tuple[list[tuple], int] | None, dict] | None:
    """Executa o BFS vetorizado utilizando tensores do PyTorch em C++.

    Args:
        estado_inicial: Inteiro representando a fita de bits inicial.
        tamanho: Tamanho L da fita.
        estado_alvo: Inteiro da fita desejada (padrão: 0, fita zerada).
        retornar_estatisticas: Se True, retorna métricas detalhadas.
    """
    inicio_tempo = time.perf_counter()

    if estado_inicial == estado_alvo:
        solucao = ([], 0)
        if retornar_estatisticas:
            stats = {
                "nos_expandidos": 0,
                "nos_visitados": 1,
                "tempo_s": time.perf_counter() - inicio_tempo,
            }
            return solucao, stats
        return solucao

    operacoes = gerar_operacoes(tamanho)
    mascaras_list = [op[0] for op in operacoes]
    acoes_list = [op[1] for op in operacoes]

    mascaras_tensor = torch.tensor(mascaras_list, dtype=torch.int64)
    M = len(acoes_list)
    tamanho_espaco = 1 << tamanho

    visitados = torch.zeros(tamanho_espaco, dtype=torch.bool)
    visitados[estado_inicial] = True

    pai_tensor = torch.full((tamanho_espaco,), -1, dtype=torch.int64)
    acao_tensor = torch.full((tamanho_espaco,), -1, dtype=torch.int16)

    fronteira = torch.tensor([estado_inicial], dtype=torch.int64)
    nos_expandidos = 0

    while len(fronteira) > 0:
        N = len(fronteira)
        novos = fronteira.unsqueeze(1) ^ mascaras_tensor.unsqueeze(0)

        # Teste de objetivo antecipado
        eh_alvo = (novos == estado_alvo)
        if eh_alvo.any():
            idx = torch.nonzero(eh_alvo, as_tuple=False)[0]
            p_idx, a_idx = idx[0].item(), idx[1].item()
            pai_final = fronteira[p_idx].item()
            acao_final = acoes_list[a_idx]

            # Reconstrução do caminho ótimo via tabela de pais
            caminho = [acao_final]
            atual = pai_final
            while atual != -1 and atual != estado_inicial:
                acao_id = acao_tensor[atual].item()
                if acao_id == -1:
                    break
                caminho.append(acoes_list[acao_id])
                atual = pai_tensor[atual].item()
            caminho.reverse()

            solucao = (caminho, len(caminho))
            if retornar_estatisticas:
                # Nós expandidos até o nó pai que encontrou o alvo
                total_expandidos = nos_expandidos + p_idx + 1

                # Estados visitados/alcançados até o momento do teste de objetivo:
                # Estados anteriores + novos gerados pelos pais 0..p_idx-1 e pai p_idx (0..a_idx)
                gerados_ate_alvo = []
                if p_idx > 0:
                    gerados_ate_alvo.append(novos[:p_idx, :].reshape(-1))
                gerados_ate_alvo.append(novos[p_idx, : a_idx + 1])
                estados_gerados = torch.cat(gerados_ate_alvo)

                visitados_total = visitados.clone()
                visitados_total[estados_gerados] = True
                total_visitados = int(visitados_total.sum().item())

                stats = {
                    "nos_expandidos": total_expandidos,
                    "nos_visitados": total_visitados,
                    "tempo_s": time.perf_counter() - inicio_tempo,
                }
                return solucao, stats
            return solucao

        # Se não achou na camada atual, contabiliza todos os pais expandidos
        nos_expandidos += N

        # Filtragem vetorial em registradores SIMD
        nao_visitados = ~visitados[novos]
        if not nao_visitados.any():
            break

        candidatos_novos = novos[nao_visitados]
        pais_matriz = fronteira.unsqueeze(1).expand(N, M)
        acoes_matriz = torch.arange(M, dtype=torch.int16).unsqueeze(0).expand(N, M)

        candidatos_pais = pais_matriz[nao_visitados]
        candidatos_acoes = acoes_matriz[nao_visitados]

        # Deduplicação com preservação estrita da ordem FIFO (menor índice de descoberta)
        unicos, inverse = torch.unique(candidatos_novos, return_inverse=True)
        K = len(candidatos_novos)
        first_idx = torch.empty(len(unicos), dtype=torch.int64).fill_(K)
        first_idx.scatter_reduce_(dim=0, index=inverse, src=torch.arange(K), reduce="amin")

        # Ordena a fronteira pela ordem cronológica exata da fila FIFO
        sort_order = torch.argsort(first_idx)
        chosen_indices = first_idx[sort_order]

        estados_fifo = candidatos_novos[chosen_indices]
        pais_fifo = candidatos_pais[chosen_indices]
        acoes_fifo = candidatos_acoes[chosen_indices]

        pai_tensor[estados_fifo] = pais_fifo
        acao_tensor[estados_fifo] = acoes_fifo
        visitados[estados_fifo] = True

        fronteira = estados_fifo

    if retornar_estatisticas:
        stats = {
            "nos_expandidos": nos_expandidos,
            "nos_visitados": int(visitados.sum().item()),
            "tempo_s": time.perf_counter() - inicio_tempo,
        }
        return None, stats
    return None


def resolver_bfs_torch(
    string_inicial: str,
    string_alvo: Optional[str] = None,
    retornar_estatisticas: bool = False,
) -> tuple[list[tuple], int] | tuple[tuple[list[tuple], int] | None, dict] | None:
    """Função de alto nível que recebe strings binárias diretamente."""
    estado_inicial, tamanho = carregar(string_inicial)
    if string_alvo is None:
        estado_alvo = 0
    else:
        estado_alvo, tamanho_alvo = carregar(string_alvo)
        if tamanho_alvo != tamanho:
            raise ValueError(f"Tamanho do alvo ({tamanho_alvo}) diferente do tamanho inicial ({tamanho}).")

    return bfs_torch(
        estado_inicial=estado_inicial,
        tamanho=tamanho,
        estado_alvo=estado_alvo,
        retornar_estatisticas=retornar_estatisticas,
    )


def executar_cli() -> None:
    """Interface CLI para o BFS Vetorizado com PyTorch."""
    executar_cli_busca(
        titulo="BFS Vetorizado (PyTorch C++ SIMD Engine)",
        funcao_resolver=resolver_bfs_torch,
        info_adicional=lambda: f"PyTorch Engine: {torch.__version__} | SIMD Tensor Acceleration",
    )


if __name__ == "__main__":
    executar_cli()
