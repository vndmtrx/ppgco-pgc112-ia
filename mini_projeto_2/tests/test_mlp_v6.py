"""Testes do v6: perda MSE com One-Hot, ids, seleção por perda e avaliação no teste.

Usam dados sintéticos pequenos (rápidos, sem baixar o MNIST).
"""

import csv

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import mlp_v6_testar
from graficos_mlp import salvar_graficos_treino
from models.mlp import MLP
from mlp_v4_2_treinar import criar_otimizador
from mlp_v5_treinar import definir_semente, treinar_guardando_melhor
from mlp_v6_selecionar import (
    CHAVE_CONFIG,
    agrupar_por_configuracao,
    estatisticas,
    parear_perdas,
)
from mlp_v6_treinar import (
    COLUNAS_RESUMO,
    PerdaMSEOneHot,
    criar_criterio,
    montar_id,
    perda_do_nome,
    registrar_resumo,
)

DEVICE = torch.device("cpu")
ENTRADAS = 20


def test_mse_one_hot_e_igual_ao_mseloss_do_v3():
    torch.manual_seed(0)
    saidas = torch.randn(32, 10)
    rotulos = torch.randint(0, 10, (32,))
    esperado = nn.MSELoss()(saidas, nn.functional.one_hot(rotulos, num_classes=10).float())
    assert PerdaMSEOneHot()(saidas, rotulos).item() == pytest.approx(esperado.item())


def test_criar_criterio_devolve_a_perda_certa_e_recusa_nome_errado():
    assert isinstance(criar_criterio("ce"), nn.CrossEntropyLoss)
    assert isinstance(criar_criterio("mse"), PerdaMSEOneHot)
    with pytest.raises(ValueError):
        criar_criterio("hinge")


def test_montar_id_traz_a_perda_e_perda_do_nome_le_de_volta():
    a = montar_id("mse", 50, 64, 0.1, 0.0, "sgd", [300], 1)
    b = montar_id("ce", 100, 256, 0.1, 0.9, "sgd", [300, 100], 2)
    assert a == "mlp_treinado_v6_mse_e50_b64_lr0.1_s1"
    assert b == "mlp_treinado_v6_ce_e100_b256_lr0.1_m0.9_h300-100_s2"
    assert perda_do_nome(a + "_melhor.pth") == "mse"
    assert perda_do_nome(b) == "ce"
    assert perda_do_nome("mlp_treinado_v5_e50_b256_lr0.1_s1_melhor.pth") is None


def _treinar(perda: str, seed: int, epocas: int):
    g = definir_semente(seed)
    x_tr, y_tr = torch.randn(64, ENTRADAS), torch.randint(0, 10, (64,))
    x_vl, y_vl = torch.randn(200, ENTRADAS), torch.randint(0, 10, (200,))
    loader_tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=16, shuffle=True, generator=g)
    loader_vl = DataLoader(TensorDataset(x_vl, y_vl), batch_size=64, shuffle=False)
    modelo = MLP(input_dim=ENTRADAS, hidden_dims=[64], num_classes=10)
    otimizador = criar_otimizador("sgd", modelo, 0.3, 0.0)
    return treinar_guardando_melhor(
        modelo, loader_tr, loader_vl, criar_criterio(perda), otimizador, DEVICE, epocas, verbose=False
    )


@pytest.mark.parametrize("perda", ["ce", "mse"])
def test_treino_com_cada_perda_guarda_a_epoca_de_menor_perda_de_validacao(perda):
    hist, melhor = _treinar(perda, seed=0, epocas=15)
    minimo = min(hist["perda_val"])
    assert melhor["perda_val"] == pytest.approx(minimo)
    assert melhor["epoca"] == hist["perda_val"].index(minimo) + 1
    assert all(p == p for p in hist["perda_treino"])  # sem NaN


def test_mesma_semente_repete_as_primeiras_epocas_de_um_treino_mais_longo():
    """Base da afirmação do executar_bateria_v6.sh: uma execução longa contém as curtas."""
    curto, _ = _treinar("mse", seed=5, epocas=5)
    longo, _ = _treinar("mse", seed=5, epocas=10)
    assert longo["perda_val"][:5] == curto["perda_val"]
    assert longo["perda_treino"][:5] == curto["perda_treino"]


def test_registrar_resumo_v6_substitui_linha_com_mesmo_id(tmp_path):
    caminho = tmp_path / "resumo.csv"
    base = {c: "0" for c in COLUNAS_RESUMO}
    registrar_resumo(caminho, {**base, "id": "a", "perda": "ce"})
    registrar_resumo(caminho, {**base, "id": "b", "perda": "mse"})
    registrar_resumo(caminho, {**base, "id": "a", "perda": "ce", "acc_val_salva": "98.0"})
    with open(caminho, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    assert [l["id"] for l in linhas] == ["b", "a"]
    assert linhas[-1]["acc_val_salva"] == "98.0"


def _linha(perda, seed, acc, epocas="50"):
    return {
        "id": f"x_{perda}_{epocas}_{seed}", "perda": perda, "seed": str(seed), "epocas": epocas,
        "batch": "64", "lr": "0.1", "momentum": "0.0", "otimizador": "sgd", "hidden_dims": "300",
        "parametros": "238510", "epoca_salva": "10", "acc_val_salva": str(acc),
        "perda_val_salva": "0.08", "acc_val_ultima": str(acc), "tempo_treino_s": "60",
    }


def test_selecao_nao_mistura_mse_com_entropia_cruzada():
    linhas = [_linha("mse", 1, 97.5), _linha("mse", 2, 97.7), _linha("ce", 1, 98.0), _linha("ce", 2, 98.2)]
    grupos = agrupar_por_configuracao(linhas)
    assert len(grupos) == 2
    assert all(len(g) == 2 for g in grupos.values())
    assert "perda" in CHAVE_CONFIG


def test_parear_perdas_so_junta_configuracoes_identicas():
    linhas = [
        _linha("mse", 1, 97.5), _linha("ce", 1, 98.0),            # par completo (50 épocas)
        _linha("mse", 1, 97.0, epocas="200"),                      # sem o par em CE
    ]
    todas = [estatisticas(k, g) for k, g in agrupar_por_configuracao(linhas).items()]
    pares = parear_perdas(todas)
    assert len(pares) == 1
    mse, ce = pares[0]
    assert mse["config"]["perda"] == "mse" and ce["config"]["perda"] == "ce"
    assert mse["config"]["epocas"] == ce["config"]["epocas"] == "50"


def test_graficos_de_treino_geram_os_cinco_arquivos(tmp_path):
    hist, _ = _treinar("mse", seed=0, epocas=4)
    caminhos = salvar_graficos_treino(hist, tmp_path / "fig", "mlp_x", "MSE, teste", "MSE com One-Hot")
    assert set(caminhos) == {"perda", "acuracia", "gap", "tempo", "painel"}
    assert all(p.exists() and p.stat().st_size > 0 for p in caminhos.values())


def test_testar_v6_le_a_perda_do_nome_e_registra_a_abertura(tmp_path, monkeypatch):
    """Roda o fluxo de teste com imagens sintéticas (não abre o MNIST real)."""
    ds = TensorDataset(torch.rand(120, 1, 28, 28), torch.randint(0, 10, (120,)))
    monkeypatch.setattr(mlp_v6_testar, "get_mnist_datasets", lambda: (None, None, ds))

    caminho = tmp_path / "mlp_treinado_v6_mse_e1_b64_lr0.1_s1_melhor.pth"
    torch.save(MLP(784, [32], 10).state_dict(), caminho)

    res = mlp_v6_testar.executar_teste_v6(
        str(caminho), gerar_graficos=False, pasta_resultados=tmp_path / "res"
    )
    assert 0.0 <= res["acuracia"] <= 1.0
    with open(tmp_path / "res" / "aberturas_teste.csv", newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    assert len(linhas) == 1 and linhas[0]["checkpoint"] == caminho.name


def test_testar_v6_exige_perda_quando_o_nome_nao_a_traz(tmp_path):
    with pytest.raises(ValueError):
        mlp_v6_testar.executar_teste_v6(str(tmp_path / "mlp_treinado_v5_e1_s1_melhor.pth"), gerar_graficos=False)
