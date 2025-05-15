import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

def preparar_dados(df, falha_cols):
    # Nessa variável X nós estamos tirando as colunas de falhas, pq são as nossas variáveis alvos,
    # precisamos tirar elas para que o modelo consiga aprender sem ver as respostas.
    X = df.drop(columns=falha_cols + ['num_falhas'])
    y = df[falha_cols]

    # Variáveis dummy para categóricas
    # Aqui eu transformo as colunas em binárias, pq o modelo nao sabe o que é A300 ou A400
    # Então transformando em binário, que seria 0 | 1, e o drop_first=True é pra evitar redundância
    X = pd.get_dummies(X, drop_first=True)

    # Armazena os nomes das colunas para depois usar nos gráficos de feature importance
    feature_names = X.columns

    # Padronização (normalização da escala das variáveis)
    # É usada para que o modelo não dê mais peso para uma coluna só porque os números são maiores
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Aqui ele vira um numpy.ndarray para treino mais eficiente

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y['falha_1']
    )

    return X_train, X_test, y_train, y_test, feature_names

def treinar_modelos(X_train, y_train):
    modelos = {
        'Regressão Logística': MultiOutputClassifier(LogisticRegression(max_iter=1000)),
        'Random Forest': MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    }
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
    return modelos
