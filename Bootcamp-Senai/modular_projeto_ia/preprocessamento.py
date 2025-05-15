import pandas as pd
import numpy as np
import os

true_false = {
    'Sim': True, 'sim': True, 'True': True, True: True,
    'Não': False, 'nao': False, 'Nao': False, 'False': False, False: False,
    ' S': True, ' N': False,
    1: True, 0: False,
    1.0: True, 0.0: False
}

def carregar_e_tratar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv)

    # Utilizando o .head e .info, deu para perceber que algumas colunas tinham valores negátivos que fisicamente, não deveriam existir números negativos
    # Ex:Areas, perimetros, comprimentros, espessuras, e indice de luminosidade.
    # precisamos verificar se houve erro de medição ou digitação.
    # Caso algum valor negativo seja encontrado, substituiremos por NaN para tratarmos depois (remoção ou imputação)
    colunas_positivas = [
        'x_minimo', 'x_maximo', 'y_minimo', 'y_maximo',
        'area_pixels', 'perimetro_x', 'perimetro_y',
        'comprimento_do_transportador', 'espessura_da_chapa_de_aço',
        'indice_de_luminosidade'
    ]

    print("\nVerificação de Valores Negativos")
    for col in colunas_positivas:
        if col in df.columns:
            total = len(df)
            negativos = df[df[col] < 0]
            qtd_negativos = len(negativos)
            perc_negativos = (qtd_negativos / total) * 100
            print(f"{col}: {qtd_negativos} valores negativos ({perc_negativos:.2f}%)")

            # Substituição dos negativos por NaN (para tratamento posterior)
            df.loc[df[col] < 0, col] = pd.NA

    # Após essa verificação, pude perceber que com exceção da coluna "indice_de_luminosidade" todas as outras colunas tinham em média 10% de valores negativos.
    # então escolhi fazer a troca por NaN e depois usar a mediana, utilizei mediana no lugar de média, para evitar problemas com outliers
    # Já na coluna indice_de_luminosidade tinha cerca de 90% de valores negativos (fisicamente impossíveis), então decidir excluir a coluna.
    # E coluna Id também foi excluída pois não traria valor para nosso treinamento
    if 'indice_de_luminosidade' in df.columns:
        df.drop(columns=['indice_de_luminosidade'], inplace=True)
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    # Padronizar colunas tipo_do_aço_A300 e tipo_do_aço_A400
    for col_aco in ['tipo_do_aço_A300', 'tipo_do_aço_A400']:
        if col_aco in df.columns:
            if df[col_aco].dtype == 'object':
                df[col_aco] = df[col_aco].astype(str).str.strip().map(true_false)
            else:
                df[col_aco] = df[col_aco].map(true_false)
            df[col_aco] = df[col_aco].fillna(False).astype(bool)

    # Precisamos padronizar as colunas de falhas para booleano (True/False)
    falha_cols = ['falha_1', 'falha_2', 'falha_3', 'falha_4', 'falha_5', 'falha_6', 'falha_outros']
    for col in falha_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].map(true_false).astype(bool)
        elif not pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].notna() & (df[col] != 0)

    # Imputação dos NaNs restantes (mediana) - somente colunas numéricas
    colunas_numericas = df.select_dtypes(include=['number']).columns
    for col in colunas_numericas:
        if df[col].isnull().sum() > 0:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)

    df['num_falhas'] = df[falha_cols].sum(axis=1)

    # Salvar o DataFrame tratado em um novo arquivo .csv
    os.makedirs("dados", exist_ok=True)
    df.to_csv("dados/dados_tratados.csv", index=False)
    print("✅ Dados tratados salvos em 'dados/dados_tratados.csv'")
    return df, falha_cols
