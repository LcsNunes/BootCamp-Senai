from preprocessamento import carregar_e_tratar_dados
from treinamento_modelos import preparar_dados, treinar_modelos
from avaliacao_modelos import avaliar_modelos
from analise_exploratoria import analise_exploratoria
from feature_importance import gerar_feature_importance
from comparacao_modelos import comparar_modelos

if __name__ == "__main__":
    # Carrega e trata os dados
    df, falha_cols = carregar_e_tratar_dados("bootcamp_train.csv")
    
    # Análise exploratória
    analise_exploratoria(df, falha_cols)

    # Prepara os dados para treinamento
    X_train, X_test, y_train, y_test, feature_names = preparar_dados(df, falha_cols)

    # Treinamento dos modelos
    modelos = treinar_modelos(X_train, y_train)

    # Avaliação dos modelos
    avaliar_modelos(modelos, X_test, y_test, falha_cols)

    # Geração da importância das variáveis para o modelo Random Forest
    gerar_feature_importance(modelos['Random Forest'], feature_names, falha_cols)

    # Geração gráficos de comparação entre os modelos
    comparar_modelos()