# Bootcamp Ciência de Dados e Inteligência Artificial: Análise Preditiva de Classificação de Falhas em Chapas de Aço Inoxidável

Este projeto foi desenvolvido como parte de um bootcamp de Inteligência Artificial promovido pelo SENAI. O objetivo principal é utilizar técnicas de ciência de dados e machine learning para classificar automaticamente falhas em chapas de aço inox com base em características extraídas de imagens e dados em um dataset.

---

## 📁 Estrutura do Projeto

```
Bootcamp-Senai/
├── bootcamp_train.csv            # Dataset original fornecido pela indústria
├── dados/
│   └── dados_tratados.csv        # Dados tratados e prontos para modelagem
├── modular_projeto_ia/
│   ├── main.py                   # Execução completa do pipeline
│   ├── dashboard.py              # Interface interativa em Streamlit
│   ├── preprocessamento.py       # Limpeza e tratamento de dados
│   ├── analise_exploratoria.py   # Geração de gráficos e análises iniciais
│   ├── treinamento_modelos.py    # Modelos Random Forest e Regressão Logística
│   ├── avaliacao_modelos.py      # Avaliação com métricas e visualizações
│   ├── feature_importance.py     # Importância das variáveis por falha
│   ├── comparacao_modelos.py     # Gráficos comparativos entre modelos
│   └── README.txt                # Observações e anotações locais
|   └── requirements.txt          # Lista de dependências Python para o projeto.
```

---

## 🚀 Como Executar

### 1. Instale as dependências

Crie um ambiente virtual e instale:

```bash
pip install -r requirements.txt
```

> Obs: Você pode gerar seu `requirements.txt` com:
> ```bash
> pip freeze > requirements.txt
> ```

---

### 2. Executar pipeline completo via terminal:

```bash
python modular_projeto_ia/main.py
```

---

### 3. Executar dashboard interativo (Streamlit):

```bash
cd modular_projeto_ia
streamlit run dashboard.py
```

---

## 🎯 Funcionalidades do Dashboard

- 📊 Visualização de distribuição das falhas
- 📦 Boxplots das variáveis numéricas
- 🤖 Treinamento de modelos com multirrótulo
- 📈 Avaliação com F1, Acurácia, Precisão e Recall
- 📌 Importância das variáveis para cada tipo de falha
- 🔍 Comparação visual entre Random Forest e Regressão Logística
- 🧪 **Simulador interativo**: insira valores e veja a previsão de falhas com probabilidades

---

## 🧠 Técnicas Utilizadas

- **Tratamento de dados inconsistentes**
- **Imputação por mediana e exclusão de colunas com ruído**
- **Modelagem multirrótulo com `MultiOutputClassifier`**
- **Padronização e codificação de variáveis**
- **Avaliação com métricas por rótulo**
- **Streamlit para visualização e simulação**

---

## 📌 Observações

- As colunas de falhas são binárias e podem ocorrer em conjunto.
- A coluna `indice_de_luminosidade` foi removida por conter valores negativos fisicamente impossíveis (90%).
- A coluna 'Id' foi removida por não influenciar em nada no processamento dos dados.
- A predição é baseada em características geométricas e estatísticas extraídas das imagens.

---

## 👨‍💻 Autor

Desenvolvido por **Lucas Nunes** no Bootcamp SENAI - 2025.
