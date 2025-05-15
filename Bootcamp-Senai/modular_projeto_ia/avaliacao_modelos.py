import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

os.makedirs("imagens", exist_ok=True)

def avaliar_modelos(modelos, X_test, y_test, falha_cols):
    for nome, modelo in modelos.items():
        print(f"\nAvaliação do Modelo: {nome}\n")
        y_pred = modelo.predict(X_test)
        y_pred = pd.DataFrame(y_pred, columns=falha_cols)

        metrics = []
        for col in falha_cols:
            acc = accuracy_score(y_test[col], y_pred[col])
            f1 = f1_score(y_test[col], y_pred[col])
            precision = precision_score(y_test[col], y_pred[col])
            recall = recall_score(y_test[col], y_pred[col])
            metrics.append([col, acc, f1, precision, recall])

        resultado = pd.DataFrame(metrics, columns=["Falha", "Accuracy", "F1 Score", "Precision", "Recall"])
        print(resultado)

        for metric in ["Accuracy", "F1 Score", "Precision", "Recall"]:
            plt.figure(figsize=(8, 5))
            sns.barplot(data=resultado, x="Falha", y=metric, palette="mako")
            plt.title(f"{metric} por Tipo de Falha - {nome}")
            plt.tight_layout()
            plt.savefig(f"modular_projeto_ia/imagens/{nome.replace(' ', '_').lower()}_{metric.lower()}.png")
            plt.close()