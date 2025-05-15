import os
import matplotlib.pyplot as plt
import pandas as pd

def comparar_modelos():
    os.makedirs("imagens", exist_ok=True)

    # Dados comparativos extraídos das execuções anteriores
    dados = {
        'Falha': ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros'],
        'F1_RF': [0.13, 0.81, 0.93, 0.58, 0.24, 0.43, 0.54],
        'F1_LR': [0.15, 0.82, 0.95, 0.59, 0.13, 0.44, 0.46],
        'Recall_RF': [0.07, 0.69, 0.91, 0.42, 0.16, 0.34, 0.47],
        'Recall_LR': [0.09, 0.70, 0.93, 0.44, 0.07, 0.35, 0.38],
        'Precision_RF': [0.78, 0.99, 0.94, 0.93, 0.45, 0.61, 0.65],
        'Precision_LR': [0.60, 0.98, 0.96, 0.91, 0.37, 0.61, 0.56],
        'Accuracy_RF': [0.90, 0.97, 0.97, 0.96, 0.54, 0.79, 0.73],
        'Accuracy_LR': [0.90, 0.97, 0.98, 0.96, 0.54, 0.79, 0.69]
    }

    # Aqui o dicionário é transformado em DataFrame, pra facilitar o uso com gráficos
    df_comp = pd.DataFrame(dados)

    # Mostra em porcentagem o comparativo entre os dois modelos, para ver qual teve o melhor resultado
    resumo_rf = df_comp[[col for col in df_comp.columns if col.endswith("_RF")]].mean().rename(lambda x: x.replace("_RF", ""))
    resumo_lr = df_comp[[col for col in df_comp.columns if col.endswith("_LR")]].mean().rename(lambda x: x.replace("_LR", ""))

    df_resumo = pd.DataFrame({
        'Random Forest (%)': (resumo_rf * 100).round(2),
        'Regressão Logística (%)': (resumo_lr * 100).round(2)
    })

    print("\nResumo Comparativo (%)")
    print(df_resumo)

    # Função para gerar os gráficos comparativos
    def plot_comparativo(metric, label_y):
        plt.figure(figsize=(10,6))
        bar_width = 0.35
        index = range(len(df_comp))

        plt.bar(index, df_comp[f'{metric}_RF'], bar_width, label='Random Forest', color='royalblue')
        plt.bar([i + bar_width for i in index], df_comp[f'{metric}_LR'], bar_width, label='Regressão Logística', color='mediumseagreen')

        plt.xlabel('Tipo de Falha')
        plt.ylabel(label_y)
        plt.title(f'Comparação de {label_y} por Tipo de Falha - Random Forest vs Regressão Logística')
        plt.xticks([i + bar_width / 2 for i in index], df_comp['Falha'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'modular_projeto_ia/imagens/comparativo_{metric.lower()}.png')
        plt.close()

    plot_comparativo('F1', 'F1 Score')
    plot_comparativo('Recall', 'Recall')
    plot_comparativo('Precision', 'Precisão')
    plot_comparativo('Accuracy', 'Acurácia')
