import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

# Função principal da análise exploratória
def analise_exploratoria(df, falha_cols):
    os.makedirs("imagens", exist_ok=True)

    # Análise da distribuição das variáveis alvo (falhas), saber qual o número de amostras com determinada falha.
    # Assim da para entender se tem um desbalanceamento, por exemplo um tipo de falha aparecer muito mais do que as outras.
    falhas_presentes_df = df[falha_cols]
    falhas_dist = falhas_presentes_df.sum().sort_values(ascending=False)
    print("Distribuição das classes de falha:")
    print(falhas_dist)

    plt.figure(figsize=(10,6))
    sns.barplot(x=falhas_dist.index, y=falhas_dist.values, palette="viridis")
    plt.title("Distribuição de Tipos de Falhas")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("modular_projeto_ia/imagens/grafico_distribuicao_falhas.png")
    plt.close()

    # Verificando desbalanceamento entre as classes
    # Mostra em porcentagem a quantidade de falhas presente em todas as amostras do dataset.
    print("\nProporção de amostras com falha vs sem falha por tipo")
    total_amostras = len(df)
    for col in falha_cols:
        qtd = df[col].sum()
        print(f"{col}: {qtd} ({(qtd / total_amostras) * 100:.2f}%)")

    df['num_falhas'] = falhas_presentes_df.sum(axis=1)

    # Contagem e porcentagem de amostras por número de falhas
    coocorrencia = df['num_falhas'].value_counts().sort_index()
    total_amostras = len(df)

    print("\nDistribuição de amostras por número de falhas:")
    for num_falhas, qtd in coocorrencia.items():
        perc = (qtd / total_amostras) * 100
        print(f"{int(num_falhas)} falha(s): {qtd} amostras ({perc:.2f}%)")

    # Gráfico
    plt.figure(figsize=(10, 6))
    sns.countplot(x='num_falhas', data=df, palette="crest")
    plt.title("Distribuição do Número de Falhas por Amostra")
    plt.tight_layout()
    plt.savefig("modular_projeto_ia/imagens/grafico_numero_falhas_por_amostra.png")
    plt.close()

    # Gerando boxplots combinados para colunas numéricas
    print("\nGerando boxplots combinados para colunas numéricas")

    # Seleciona apenas colunas numéricas, excluindo colunas de falha e num_falhas, determina quantas colunas terão na imagem(5)
    # e coloca todos os boxplots em uma única imagem, para facilitar a comparação
    colunas_boxplot = df.select_dtypes(include=['float64', 'int64']).columns
    colunas_boxplot = [col for col in colunas_boxplot if col not in falha_cols + ['num_falhas']]
    n_colunas = 5
    n_linhas = int(np.ceil(len(colunas_boxplot) / n_colunas))
    fig, axes = plt.subplots(n_linhas, n_colunas, figsize=(20, 4 * n_linhas))
    axes = axes.flatten()  # transforma em uma lista para indexação simples

    # Cria cada boxplot individual
    for i, col in enumerate(colunas_boxplot):
        sns.boxplot(y=df[col], ax=axes[i], color="skyblue")
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(axis='y', labelsize=8)
        axes[i].set_xlabel("")

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("modular_projeto_ia/imagens/boxplots_todos_em_uma_imagem.png")
    plt.close()
    print("Boxplots salvos em 'modular_projeto_ia/imagens/boxplots_todos_em_uma_imagem.png'")
