# ============================================================
# DASHBOARD INTERATIVO - PROJETO DE CLASSIFICAÇÃO DE FALHAS ||
# ============================================================

# Importando bibliotecas necessárias
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Importando funções dos módulos do projeto
from preprocessamento import carregar_e_tratar_dados
from treinamento_modelos import preparar_dados, treinar_modelos
from avaliacao_modelos import avaliar_modelos
from feature_importance import gerar_feature_importance
from comparacao_modelos import comparar_modelos

# Configurações iniciais da página do Streamlit
st.set_page_config(page_title="Dashboard - Falhas em Chapas de Aço", layout="wide")

# Função para carregar os dados tratados e colunas de falhas
@st.cache_data
def carregar_dados():
    if not os.path.exists("dados/dados_tratados.csv"):
        st.warning("Arquivo de dados tratados não encontrado. Executando pré-processamento...")
        df, falhas = carregar_e_tratar_dados("bootcamp_train.csv")
    else:
        df = pd.read_csv("dados/dados_tratados.csv")
        falhas = [col for col in df.columns if col.startswith("falha_")]
    return df, falhas

# Carregamento do DataFrame e colunas de falhas
df, falha_cols = carregar_dados()

# Abas para organizar as seções do dashboard
abas = st.tabs([
    "📊 Análise Exploratória",
    "🤖 Treinamento & Avaliação",
    "🔎 Importância & Comparação",
    "⚙️ Simulador Interativo"
])

# =======================
# Análise Exploratória ||
# =======================
with abas[0]:
    st.header("📊 Análise Exploratória")
    dist_falhas = df[falha_cols].sum().sort_values(ascending=False)

    # Gráfico de barras mostrando a distribuição das falhas
    fig1, ax1 = plt.subplots(figsize=(7, 3))  # ⬅️ Diminuído aqui
    sns.barplot(x=dist_falhas.index, y=dist_falhas.values, ax=ax1, palette="viridis")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # Boxplots para variáveis numéricas
    st.subheader("📦 Boxplots das Variáveis Numéricas")
    cols_box = [col for col in df.select_dtypes(include=['float64', 'int64']).columns if col not in falha_cols + ['num_falhas']]
    fig2, axs = plt.subplots(nrows=(len(cols_box) + 4) // 5, ncols=5, figsize=(16, 8))  # ⬅️ Reduzido um pouco
    axs = axs.flatten()
    for i, col in enumerate(cols_box):
        sns.boxplot(y=df[col], ax=axs[i], color="skyblue")
        axs[i].set_title(col, fontsize=9)
    for j in range(i+1, len(axs)):
        fig2.delaxes(axs[j])
    plt.tight_layout()
    st.pyplot(fig2)


# ======================================
# Treinamento dos Modelos e Avaliação ||
# ======================================
with abas[1]:
    st.header("🤖 Treinamento e Avaliação dos Modelos")
    X_train, X_test, y_train, y_test, feature_names = preparar_dados(df, falha_cols)
    modelos = treinar_modelos(X_train, y_train)
    avaliar_modelos(modelos, X_test, y_test, falha_cols)
    st.success("Modelos treinados e avaliados com sucesso.")

# ========================================
# Comparação e Importância de Variáveis ||
# ========================================
with abas[2]:
    st.header("🔎 Importância das Variáveis e Comparação de Modelos")
    gerar_feature_importance(modelos["Random Forest"], feature_names, falha_cols)
    st.image("modular_projeto_ia/imagens/feature_importance_rf_todos_em_um.png", caption="Importância das Variáveis - Random Forest")

    comparar_modelos()
    st.image("modular_projeto_ia/imagens/comparativo_f1.png", caption="F1 Score")
    st.image("modular_projeto_ia/imagens/comparativo_recall.png", caption="Recall")
    st.image("modular_projeto_ia/imagens/comparativo_precision.png", caption="Precisão")
    st.image("modular_projeto_ia/imagens/comparativo_accuracy.png", caption="Acurácia")

# ===================================
# Simulador Interativo de Previsão ||
# ===================================
with abas[3]:
    st.header("Simulador Interativo de Previsão de Falhas")

    df_temp = df.drop(columns=falha_cols + ['num_falhas'])
    df_temp = pd.get_dummies(df_temp, drop_first=True)
    medianas = df_temp.median()
    entrada_usuario = {}

    st.markdown("Ajuste os valores das características abaixo para simular uma inspeção e obter uma previsão de defeitos do modelo Random Forest.")
    st.info("Obs: Os valores iniciais representam as medianas do conjunto de treino para as características numéricas. As características apresentadas são as utilizadas pelo modelo treinado.")

    colunas = st.columns(4)
    for i, col in enumerate(df_temp.columns):
        if df_temp[col].nunique() == 2 and set(df_temp[col].unique()) <= {0, 1}:
            entrada_usuario[col] = colunas[i % 4].selectbox(col, ["Não (Tipo Ausente)", "Sim (Tipo Presente)"]) == "Sim (Tipo Presente)"
        else:
            entrada_usuario[col] = float(colunas[i % 4].number_input(col, value=float(medianas[col])))

    input_df = pd.DataFrame([entrada_usuario])
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df_temp)
    input_scaled = scaler.transform(input_df)

    modelo_rf = modelos["Random Forest"]
    pred = modelo_rf.predict(input_scaled)
    proba = modelo_rf.predict_proba(input_scaled)

    st.subheader("🧾 Previsão de Falhas para a Inspeção Simulada")
    for i, col in enumerate(falha_cols):
        proba_classe = proba[i][0]  # array com [prob_negativo, prob_positivo]
        probabilidade = proba_classe[1] * 100 if pred[0][i] else proba_classe[0] * 100
        st.write(f"**{col}**: {'✅ Detectado' if pred[0][i] else '❌ Ausente'} — Probabilidade: {probabilidade:.2f}%")


st.markdown("---")
st.markdown("Desenvolvido no Bootcamp de Ciência de Dados e IA – SENAI")
