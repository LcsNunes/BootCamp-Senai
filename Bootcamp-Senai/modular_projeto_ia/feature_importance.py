import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Aqui estamos criando um DataFrame para armazenar a importância das features para cada falha
def gerar_feature_importance(modelo_rf_multi, X_columns, falha_cols):
    os.makedirs("imagens", exist_ok=True)
    feature_importance_df = pd.DataFrame(index=X_columns)

    # Loop para extrair as importâncias de cada modelo (1 por tipo de falha)
    for i, col in enumerate(falha_cols):
        rf_model = modelo_rf_multi.estimators_[i]  # modelo para a falha i
        importances = rf_model.feature_importances_
        feature_importance_df[col] = importances

    # Selecionar os top 15 mais importantes para cada falha
    top_features_per_falha = {}
    for col in feature_importance_df.columns:
        top_features = feature_importance_df[col].sort_values(ascending=False).head(15)
        top_features_per_falha[col] = top_features

    # Layout dos subplots
    num_falhas = len(top_features_per_falha)
    cols = 3  # quantidade de gráficos por linha
    rows = math.ceil(num_falhas / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
    axes = axes.flatten()

    # Criar um gráfico de barras para cada falha
    for idx, (falha, series) in enumerate(top_features_per_falha.items()):
        sns.barplot(x=series.values, y=series.index, ax=axes[idx], palette="viridis")
        axes[idx].set_title(f"Top 15 Features - {falha}")
        axes[idx].set_xlabel("Importância")
        axes[idx].set_ylabel("Variável")

    # Remove gráficos em branco se sobrarem subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar layout e salvar imagem
    plt.tight_layout()
    plt.savefig("modular_projeto_ia/imagens/feature_importance_rf_todos_em_um.png")
    plt.close()
    print("✅ Importância das variáveis salva em 'modular_projeto_ia/imagens/feature_importance_rf_todos_em_um.png'")
