"""Testes do v5: checkpoint do melhor modelo, semente fixa e escolha pela validação.

Usam dados sintéticos pequenos (rápidos, sem baixar o MNIST).
"""

import csv

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.mlp import MLP
from mlp_v4_2_treinar import avaliar_validacao, criar_otimizador
from mlp_v5_selecionar import agrupar_por_configuracao, escolher, estatisticas, tolerancia_pp
from mlp_v5_treinar import (
    COLUNAS_RESUMO,
    definir_semente,
    montar_id,
    registrar_resumo,
    treinar_guardando_melhor,
)
from mlp_v5_testar import descrever_arquitetura, nome_da_pasta_de_figuras

DEVICE = torch.device("cpu")
ENTRADAS = 20


def _loaders(seed: int, n_treino: int = 64, n_val: int = 200):
    """Treino pequeno com rótulos aleatórios força sobreajuste: a validação piora depois de uns passos."""
    g = definir_semente(seed)
    x_tr = torch.randn(n_treino, ENTRADAS)
    y_tr = torch.randint(0, 10, (n_treino,))
    x_vl = torch.randn(n_val, ENTRADAS)
    y_vl = torch.randint(0, 10, (n_val,))
    loader_tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=16, shuffle=True, generator=g)
    loader_vl = DataLoader(TensorDataset(x_vl, y_vl), batch_size=64, shuffle=False)
    return loader_tr, loader_vl


def _treinar(seed: int, epocas: int = 30):
    loader_tr, loader_vl = _loaders(seed)
    modelo = MLP(input_dim=ENTRADAS, hidden_dims=[64], num_classes=10)
    criterio = nn.CrossEntropyLoss()
    otimizador = criar_otimizador("sgd", modelo, 0.3, 0.0)
    hist, melhor = treinar_guardando_melhor(
        modelo, loader_tr, loader_vl, criterio, otimizador, DEVICE, epocas, verbose=False
    )
    return modelo, loader_vl, criterio, hist, melhor


def test_epoca_salva_e_a_de_menor_perda_de_validacao():
    _, _, _, hist, melhor = _treinar(seed=0)
    minimo = min(hist["perda_val"])
    assert melhor["perda_val"] == pytest.approx(minimo)
    assert melhor["epoca"] == hist["perda_val"].index(minimo) + 1


def test_melhor_nao_e_a_ultima_quando_a_validacao_piora():
    """Com rótulos aleatórios a perda de validação sobe: o melhor modelo vem antes do fim."""
    modelo, _, _, hist, melhor = _treinar(seed=0)
    assert melhor["epoca"] < len(hist["epocas"])
    ultimo = {k: v for k, v in modelo.state_dict().items()}
    assert any(not torch.equal(melhor["estado"][k], ultimo[k]) for k in ultimo)


def test_pesos_salvos_reproduzem_a_perda_de_validacao_registrada():
    _, loader_vl, criterio, _, melhor = _treinar(seed=1)
    outro = MLP(input_dim=ENTRADAS, hidden_dims=[64], num_classes=10)
    outro.load_state_dict(melhor["estado"])
    perda, acc = avaliar_validacao(outro, loader_vl, criterio, DEVICE)
    assert perda == pytest.approx(melhor["perda_val"], abs=1e-6)
    assert acc == pytest.approx(melhor["acc_val"], abs=1e-9)


def test_mesma_semente_mesmos_pesos_e_semente_diferente_muda():
    _, _, _, _, a = _treinar(seed=3, epocas=5)
    _, _, _, _, b = _treinar(seed=3, epocas=5)
    _, _, _, _, c = _treinar(seed=4, epocas=5)
    assert all(torch.equal(a["estado"][k], b["estado"][k]) for k in a["estado"])
    assert any(not torch.equal(a["estado"][k], c["estado"][k]) for k in a["estado"])


def test_montar_id_inclui_semente_e_arquitetura():
    assert montar_id(50, 256, 0.1, 0.0, "sgd", [300], 7) == "mlp_treinado_v5_e50_b256_lr0.1_s7"
    assert (
        montar_id(100, 256, 0.1, 0.9, "sgd", [300, 100], 1)
        == "mlp_treinado_v5_e100_b256_lr0.1_m0.9_h300-100_s1"
    )


def test_registrar_resumo_substitui_linha_com_mesmo_id(tmp_path):
    caminho = tmp_path / "resumo.csv"
    base = {c: "0" for c in COLUNAS_RESUMO}
    registrar_resumo(caminho, {**base, "id": "a", "acc_val_salva": "97.0"})
    registrar_resumo(caminho, {**base, "id": "b"})
    registrar_resumo(caminho, {**base, "id": "a", "acc_val_salva": "98.0"})
    with open(caminho, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    assert [l["id"] for l in linhas] == ["b", "a"]
    assert linhas[-1]["acc_val_salva"] == "98.0"


def test_descrever_arquitetura_usa_o_modelo_carregado():
    assert descrever_arquitetura(MLP(784, [300, 100], 10)) == "784 -> 300 -> 100 -> 10"


def test_nome_da_pasta_de_figuras_bate_com_o_do_treino():
    from pathlib import Path

    nome = nome_da_pasta_de_figuras(Path("mlp_treinado_v5_e50_b256_lr0.1_s1_melhor.pth"))
    assert nome == "v5_e50_b256_lr0.1_s1"
    assert nome == montar_id(50, 256, 0.1, 0.0, "sgd", [300], 1).replace("mlp_treinado_", "")


def _linha(seed, hidden, acc, params, epoca=10, perda=0.08):
    return {
        "id": f"x_{hidden}_{seed}", "seed": str(seed), "epocas": "50", "batch": "256", "lr": "0.1",
        "momentum": "0.0", "otimizador": "sgd", "hidden_dims": hidden, "parametros": str(params),
        "epoca_salva": str(epoca), "acc_val_salva": str(acc), "perda_val_salva": str(perda),
        "tempo_treino_s": "30",
    }


def test_escolher_prefere_menos_parametros_dentro_da_tolerancia():
    linhas = [
        _linha(1, "512-256", 98.10, 535818), _linha(2, "512-256", 98.00, 535818),
        _linha(1, "300", 98.00, 238510), _linha(2, "300", 97.95, 238510),
        _linha(1, "128", 96.80, 101770), _linha(2, "128", 96.90, 101770),
    ]
    todas = [estatisticas(k, g) for k, g in agrupar_por_configuracao(linhas).items()]
    escolhida, tol, empatadas = escolher(todas, n_validacao=10000)
    assert tol >= 0.1
    assert escolhida["config"]["hidden_dims"] == "300"  # empata com a melhor e tem metade dos parâmetros
    assert len(empatadas) == 2  # 128 neurônios fica fora da tolerância


def test_tolerancia_usa_o_maior_entre_desvio_e_erro_padrao():
    melhor = {"acc_media": 98.0, "acc_desvio": 0.5}
    assert tolerancia_pp(melhor, 10000) == pytest.approx(0.5)
    melhor = {"acc_media": 98.0, "acc_desvio": 0.01}
    assert tolerancia_pp(melhor, 10000) == pytest.approx(0.14, abs=0.01)
