# Ambiente de Desenvolvimento e Execução

Este documento descreve a estratégia de isolamento de ambientes e as bibliotecas essenciais para a execução dos exercícios.

---

## 1. Isolamento por Exercício (Sem Ambiente Global)

Não haverá um ambiente Python global ou compartilhado para todo o repositório. **Cada pasta de exercício possui seu próprio ambiente isolado (`.venv`) e seu próprio `requirements.txt`**, respeitando eventuais peculiaridades e dependências específicas de cada implementação.

Exemplo de estrutura de isolamento:
```text
mini_projeto_1/
├── parte_1_buscas_classicas/
│   ├── .venv/                 <-- Ambiente virtual isolado desta parte
│   ├── requirements.txt       <-- Dependências específicas desta parte
│   ├── ...
│   └── tests/
│
└── parte_2_gradiente_descendente/
    ├── .venv/                 <-- Ambiente virtual isolado desta parte
    ├── requirements.txt       <-- Dependências específicas desta parte
    ├── ...
    └── tests/
```

---

## 2. Stack Enxuto Essencial

Adotaremos apenas o conjunto mínimo e indispensável de ferramentas:

- **PyTorch (`torch`):** Mandatório para os experimentos e exigência para diferenciação e computação tensorial.
- **Matplotlib:** Biblioteca padrão e suficiente para plotar todos os gráficos necessários para os relatórios e análises de desempenho.
- **NumPy:** Suporte numérico básico (opcional/auxiliar caso necessário junto ao PyTorch).
- **Pytest:** Mandatório para a suíte de testes unitários que valida cada algoritmo e função implementada.

*(Qualquer biblioteca adicional só será incluída se estritamente necessária para uma demanda pontual de determinado exercício).*

---

## 3. Validação via Contêiner Docker

Os experimentos e testes podem ser validados utilizando a imagem Jupyter/PyTorch oficial:

```bash
docker run -it --rm \
  -p 8888:8888 \
  -v $HOME/du/dev/pytorch:/home/jovyan/work \
  quay.io/jupyter/pytorch-notebook:latest
```

Dentro do contêiner, navega-se até a pasta do exercício correspondente para rodar os testes e scripts de forma isolada.

---

## 4. Como Criar e Rodar em Cada Pasta

Para trabalhar em um exercício específico:

```bash
# 1. Entrar na pasta do exercício
cd mini_projeto_1/parte_1_buscas_classicas  # (ou outro exercício)

# 2. Criar e ativar o ambiente virtual isolado daquele exercício
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar o requirements específico daquela pasta
pip install --upgrade pip
pip install -r requirements.txt

# 4. Executar os testes automatizados
pytest
python -m doctest flips.py -v
python -m unittest discover -s tests -p "test_*.py" -v
```
