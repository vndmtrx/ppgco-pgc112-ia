"""Testes unitários para a arquitetura LeNet (models/cnn.py).

Valida:
- Dimensões exatas dos tensores em cada camada (28x28 -> 24x24 -> 12x12 -> 8x8 -> 4x4 -> 256);
- Contagem analítica de parâmetros por bloco e diagnóstico do gargalo na primeira camada densa;
- Saída em logits puros (sem Softmax duplo antes da CrossEntropyLoss);
- Fluxo de gradientes em todas as camadas convolucionais e lineares;
- Sensibilidade à estrutura espacial (permutação de pixels altera a representação).
"""

import pytest
import torch
import torch.nn as nn

from models.cnn import LeNet
from models.mlp import MLP


def test_dimensoes_intermediarias_exatas():
    """Rastreia a redução espacial camada a camada até o achatamento (B, 256) e logits (B, 10)."""
    modelo = LeNet()
    x = torch.randn(4, 1, 28, 28)

    # Conv1 (5x5, stride 1, sem padding): 28 - 5 + 1 = 24
    z1 = modelo.features[0](x)
    assert z1.shape == (4, 6, 24, 24)
    a1 = modelo.features[1](z1)
    # MaxPool1 (2x2, stride 2): 24 / 2 = 12
    p1 = modelo.features[2](a1)
    assert p1.shape == (4, 6, 12, 12)

    # Conv2 (5x5, stride 1, sem padding): 12 - 5 + 1 = 8
    z2 = modelo.features[3](p1)
    assert z2.shape == (4, 16, 8, 8)
    a2 = modelo.features[4](z2)
    # MaxPool2 (2x2, stride 2): 8 / 2 = 4
    p2 = modelo.features[5](a2)
    assert p2.shape == (4, 16, 4, 4)

    flat = torch.flatten(p2, start_dim=1)
    assert flat.shape == (4, 256)

    fc1 = modelo.classifier[1](modelo.classifier[0](flat))
    assert fc1.shape == (4, 120)
    fc2 = modelo.classifier[3](modelo.classifier[2](fc1))
    assert fc2.shape == (4, 84)
    logits = modelo.classifier[4](fc2)
    assert logits.shape == (4, 10)
    assert torch.equal(logits, modelo(x))


def test_entrada_3d_sem_canal_e_invalida_2d():
    modelo = LeNet()
    x_3d = torch.randn(2, 28, 28)
    assert modelo(x_3d).shape == (2, 10)

    with pytest.raises(ValueError):
        modelo(torch.randn(2, 784))


def test_contagem_analitica_parametros_e_gargalo_denso():
    """Confere a matemática 3D dos filtros e localiza o gargalo paramétrico na fc1.

    Conv1: 6 * (1 * 5 * 5 + 1)    =    156
    Conv2: 16 * (6 * 5 * 5 + 1)   =  2.416
    FC1:   256 * 120 + 120        = 30.840
    FC2:   120 * 84 + 84          = 10.164
    FC3:   84 * 10 + 10           =    850
    Total:                          44.426
    """
    cnn = LeNet()
    res = cnn.resumo_parametros()

    assert res["conv1"] == 156
    assert res["conv2"] == 2416
    assert res["fc1"] == 30840
    assert res["fc2"] == 10164
    assert res["fc3"] == 850
    assert res["convolucionais"] == 2572
    assert res["lineares"] == 41854
    assert res["total"] == 44426
    assert sum(p.numel() for p in cnn.parameters()) == 44426

    # Diagnóstico do gargalo: >69% dos pesos estão na ponte Flatten -> Linear(256, 120)
    assert res["fc1"] / res["total"] > 0.69
    assert res["convolucionais"] / res["total"] < 0.06

    # Confronto direto com o MLP campeão (784 -> 512 -> 10 = 407.050 parâmetros)
    mlp_campeao = MLP(input_dim=784, hidden_dims=[512], num_classes=10)
    params_mlp = sum(p.numel() for p in mlp_campeao.parameters())
    assert params_mlp == 407050
    assert res["total"] / params_mlp < 0.11


def test_saida_em_logits_puros_nao_probabilidades():
    """Garante que o forward não aplica Softmax (evita Softmax duplo com CrossEntropyLoss)."""
    torch.manual_seed(0)
    modelo = LeNet()
    x = torch.randn(5, 1, 28, 28)
    logits = modelo(x)
    somas = logits.sum(dim=-1)
    assert not torch.allclose(somas, torch.ones_like(somas), atol=1e-3)


def test_fluxo_de_gradientes_em_todas_as_camadas():
    torch.manual_seed(1)
    modelo = LeNet()
    criterio = nn.CrossEntropyLoss()
    x = torch.randn(8, 1, 28, 28)
    y = torch.randint(0, 10, (8,))

    perda = criterio(modelo(x), y)
    perda.backward()

    for nome, param in modelo.named_parameters():
        assert param.grad is not None, f"Gradiente nulo em {nome}"
        assert torch.any(param.grad != 0), f"Gradiente zero em {nome}"


def test_sensibilidade_a_estrutura_espacial():
    """Permutar os pixels da imagem muda a resposta da CNN (ao contrário da simetria do MLP)."""
    torch.manual_seed(7)
    modelo = LeNet()
    modelo.eval()
    img = torch.randn(1, 1, 28, 28)
    perm = torch.randperm(784)
    img_perm = img.view(1, 784)[:, perm].view(1, 1, 28, 28)

    with torch.no_grad():
        out_orig = modelo(img)
        out_perm = modelo(img_perm)
    assert not torch.allclose(out_orig, out_perm, atol=1e-4)


def test_cnn_v1_treinar_e_testar_sintetico(tmp_path, monkeypatch):
    """Valida o fluxo completo de treino com semente fixa, melhor checkpoint e registro no teste."""
    import csv
    from torch.utils.data import TensorDataset
    import cnn_v1_treinar
    import cnn_v1_testar

    torch.manual_seed(99)
    ds_tr = TensorDataset(torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,)))
    ds_vl = TensorDataset(torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,)))
    ds_ts = TensorDataset(torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,)))

    monkeypatch.setattr(cnn_v1_treinar, "get_mnist_datasets", lambda: (ds_tr, ds_vl, ds_ts))
    monkeypatch.setattr(cnn_v1_testar, "get_mnist_datasets", lambda: (ds_tr, ds_vl, ds_ts))

    p_mod = tmp_path / "modelo"
    p_fig = tmp_path / "figuras"
    p_res = tmp_path / "resultados"

    res1 = cnn_v1_treinar.executar_treinamento_cnn(
        num_epocas=3,
        batch_size=16,
        lr=0.05,
        momentum=0.9,
        otimizador_nome="sgd",
        seed=1,
        gerar_graficos=True,
        pasta_modelo=p_mod,
        pasta_figuras=p_fig,
        pasta_resultados=p_res,
    )
    res2 = cnn_v1_treinar.executar_treinamento_cnn(
        num_epocas=3,
        batch_size=16,
        lr=0.05,
        momentum=0.9,
        otimizador_nome="sgd",
        seed=1,
        gerar_graficos=False,
        pasta_modelo=p_mod,
        pasta_figuras=p_fig,
        pasta_resultados=p_res,
    )

    assert res1["perda_val_salva"] == pytest.approx(res2["perda_val_salva"])
    assert res1["parametros"] == 44426

    ckpt = p_mod / f"{res1['id']}_melhor.pth"
    assert ckpt.exists()

    # Garante que os 5 gráficos de treino foram criados
    subpasta_fig = p_fig / cnn_v1_testar.nome_da_pasta_de_figuras_cnn(ckpt)
    assert len(list(subpasta_fig.glob("*.png"))) == 5

    # Testa carregamento e registro de abertura
    out_teste = cnn_v1_testar.executar_teste_cnn(
        str(ckpt),
        gerar_graficos=True,
        pasta_figuras=p_fig,
        pasta_resultados=p_res,
    )
    assert 0.0 <= out_teste["acuracia"] <= 1.0
    assert len(list(subpasta_fig.glob("*.png"))) == 9

    with open(p_res / "aberturas_teste.csv", newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    assert len(linhas) == 1
    assert linhas[0]["checkpoint"] == ckpt.name


def test_confronto_mlp_vs_cnn_sintetico(tmp_path, monkeypatch):
    """Valida o script de confronto com dados sintéticos, JSON de saída e 4 figuras."""
    import json
    from torch.utils.data import TensorDataset
    import confronto_mlp_vs_cnn

    disc = confronto_mlp_vs_cnn.analisar_discordancia(
        reais=[0, 1, 2, 3],
        pred_mlp=[0, 9, 2, 8],
        pred_cnn=[0, 1, 7, 8],
    )
    assert disc["ambos_acertam"] == [0]
    assert disc["cnn_corrige_mlp"] == [1]
    assert disc["mlp_acerta_cnn_erra"] == [2]
    assert disc["ambos_erram"] == [3]

    torch.manual_seed(5)
    ds_ts = TensorDataset(torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,)))
    monkeypatch.setattr(confronto_mlp_vs_cnn, "get_mnist_datasets", lambda: (None, None, ds_ts))

    ckpt_mlp = tmp_path / "mlp_treinado_v5_e1_b64_lr0.1_h32_s1_melhor.pth"
    ckpt_cnn = tmp_path / "cnn_treinado_v1_e1_b64_lr0.05_m0.9_s1_melhor.pth"
    torch.save(MLP(784, [32], 10).state_dict(), ckpt_mlp)
    torch.save(LeNet().state_dict(), ckpt_cnn)

    p_fig = tmp_path / "figuras"
    p_res = tmp_path / "resultados"
    rel = confronto_mlp_vs_cnn.executar_confronto(
        mlp_checkpoint=str(ckpt_mlp),
        cnn_checkpoint=str(ckpt_cnn),
        gerar_graficos=True,
        pasta_figuras=p_fig,
        pasta_resultados=p_res,
    )

    assert rel["mlp"]["parametros"] == 784 * 32 + 32 + 32 * 10 + 10
    assert rel["cnn"]["parametros"] == 44426
    assert (p_res / "confronto_mlp_vs_cnn.json").exists()
    with open(p_res / "confronto_mlp_vs_cnn.json", encoding="utf-8") as f:
        salvo = json.load(f)
    assert salvo["cnn"]["parametros"] == 44426
    pasta_confronto = p_fig / "confronto_mlp_vs_cnn"
    assert (pasta_confronto / "confronto_eficiencia.png").exists()
    assert (pasta_confronto / "confronto_pares_cronicos.png").exists()
    assert (pasta_confronto / "confronto_diferenca_confusao.png").exists()


