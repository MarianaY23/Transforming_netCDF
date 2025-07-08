# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 15:55:05 2025

@author: marit
"""
import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np    
from shapely.ops import polygonize
from shapely.geometry import MultiLineString
import matplotlib.pyplot as plt
from collections import defaultdict
from shapely.geometry import Point

#%%
# Começaremos pelo caso mais básico: qual pixel que a estação cai. 

# Primeiro passo - pontos de monitoramento

# tratamento dos pontos de monitoramento
geometriesPath = r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\inputs\Monitoramento_QAr_BR_latlon_2024.csv"
df_geometries = pd.read_csv(geometriesPath, encoding='utf-8')


# removendo redundâncias
stations =  df_geometries.drop_duplicates(subset=['Estacao'])
#selecionando apenas stações do Estado de SP
stations_SP = stations.query('ESTADO == "SP"')
stations.to_csv(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\stations.csv")
stations_SP.to_csv(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\stations_SP.csv",index=False)


# Abrindo os dados
arquivo1 = r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\inputs\dados_Formatados\dados_Formatados\SP.xlsx"
arquivo2 = r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\inputs\dados_Formatados\dados_Formatados\SP2.xlsx"
df1 = pd.read_excel(arquivo1)
df2 = pd.read_excel(arquivo2)
#df com as informações de datetime e consumos
df_pol_SP = pd.concat([df1, df2], ignore_index=True)

#fazendo um datetime 
times = pd.to_datetime(df_pol_SP[['Ano', 'Mes', 'Dia', 'Hora']]
               .astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H')
df_pol_SP['datetime'] = times
df_pol_SP.to_csv(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\df_pol_SP.csv",index=False)
#%%
#dados que serão utilizados 
df_pol_SP = pd.read_csv(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\df_pol_SP.csv")
df_column = pd.read_csv(r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\estacoes.csv")
#%%
# Criando uma subpasta dentro de outputs para cada poluente e um csv de
# cada base de dados contendo aquele poluente 

def PolFiles(df, coluna='Poluente', save_Path ='outputs'):
    """
    Essa função cria uma subpasta dentro de outputs para cada poluente e um csv
    de cada base de dados contendo cada poluente 

    Parameters
    ----------
    df : dataframe que será selecionado
        Neste caso será o df_pol_SP que é um df que contém as concentrações medidas
        no estado de SP no ano de 2023.
    coluna : coluna do DataFrame 
        DESCRIPTION. The default is 'Poluente'.
    save_Path : TYPE, optional
        Caminho para salvar dentro da função. The default is 'outputs'.

    Returns
    -------
    None.

    """
    save_Path = Path(save_Path)
    save_Path.mkdir(parents=True, exist_ok=True)
    poluentes = df[coluna].dropna().unique()
    
    for pol in poluentes:
       nome_pasta = pol.replace(" ", "_").replace("/", "-")  
       pasta_poluente = save_Path / nome_pasta
       pasta_poluente.mkdir(exist_ok=True)

       df_filtrado = df[df[coluna] == pol]
       caminho_csv = pasta_poluente / f"{nome_pasta}.csv"

       df_filtrado.to_csv(caminho_csv, index=False, encoding='utf-8')
    

PolFiles(df_pol_SP)
#%% Dentro de cada subpastas de cada poluente, separa um csv para cada estacao 
# salvando NomeEstacao_Poluente em cada supasta 
#usando o Codigo de cada estacao 
def stationsPol(pasta_base ='outputs', station='Codigo'):   
    """
    Nesta função dentro de cada subpasta de cada poluente criada anteriormente,
    separa um csv para cada estacao.

    Parameters
    ----------
    pasta_base :
        DESCRIPTION. The default is 'outputs'.
    station : 
        DESCRIPTION. The default is 'Codigo'.

    Returns
    -------
    None.

    """
    pasta_base = Path(pasta_base)

    for subpasta in pasta_base.iterdir():
        if subpasta.is_dir():
            nome_poluente = subpasta.name
            arquivo_csv = subpasta / f"{nome_poluente}.csv"

            if not arquivo_csv.exists():
                print(f"[!] CSV principal não encontrado para: {nome_poluente}")
                continue


            df = pd.read_csv(arquivo_csv)

            estacoes = df[station].dropna().unique()

            for est in estacoes:
                nome_est = str(est).replace(" ", "_").replace("/", "-")
                df_est = df[df[station] == est]
                caminho_csv_est = subpasta / f"{nome_est}_{nome_poluente}.csv"
                df_est.to_csv(caminho_csv_est, index=False, encoding='utf-8')
               
    
stationsPol('outputs')

#%% FAZENDO NOVOS CSVs 
# um csv por estacao que contem uma coluna com poluentes e todos os poluentes que esss coluna mede

geometriesPath = r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\inputs\Monitoramento_QAr_BR_latlon_2024.csv"
df_estacoes = pd.read_csv(geometriesPath, encoding='utf-8')

def estacoes(df,
             pasta_saida=r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\estacoes",
             station='Codigo',
             nome_estacao='Estacao',
             coluna_poluente='Poluente'):
    """
    Função utilizada para criar um csv por estacao que contem uma coluna com poluentes e todos os poluentes que essa coluna mede 

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    pasta_saida : TYPE, optional
        DESCRIPTION. The default is r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\estacoes".
    station : TYPE, optional
        DESCRIPTION. The default is 'Codigo'.
    nome_estacao : TYPE, optional
        DESCRIPTION. The default is 'Estacao'.
    coluna_poluente : TYPE, optional
        DESCRIPTION. The default is 'Poluente'.

    Returns
    -------
    None.

    """
    
    pasta_saida = Path(pasta_saida)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # Agrupar os poluentes por estação
    poluentes_por_estacao = (
        df.groupby(station)[coluna_poluente]
        .apply(lambda x: ';'.join(sorted(set(x))))
        .to_dict()
    )

    # Remover duplicatas mantendo uma linha por estação
    df_resultado = df.drop_duplicates(subset=[station]).copy()

    # Substituir a coluna "Poluente" pelos poluentes agrupados
    df_resultado[coluna_poluente] = df_resultado[station].map(poluentes_por_estacao)

    # Renomear a coluna "Poluente" para "Poluentes"
    df_resultado = df_resultado.rename(columns={coluna_poluente:'Poluentes'})

    # Ordenar pelo código da estação (ou outra coluna que quiser)
    df_resultado = df_resultado.sort_values(by=station)

    # Salvar um CSV por estação
    for _, row in df_resultado.iterrows():
        codigo = row[station]
        caminho_csv = pasta_saida / f"{codigo}.csv"
        row.to_frame().T.to_csv(caminho_csv, index=False, encoding='utf-8')

    # Salvar um único CSV com todas as estações
    caminho_geral = pasta_saida / "estacoes.csv"
    df_resultado.to_csv(caminho_geral, index=False, encoding='utf-8')

estacoes(df_estacoes)

# fazendo um dataframe com as estações com apenas uma coluna contendo todos os poluente monitorados naquela estação 
StationPath = r"C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\estacoes.csv"
df_column = pd.read_csv(StationPath)


#%%
# FAZER ESTA FUNÇÃO PARA PREENCHER DATA DE INÍCIO E FIM NO estacoes_poluentes.csv
def timeStartEnd(df_column, df_pol_SP):
    """
    Função utilizada para adicionar duas colunas no csv, uma da data que começa a medir 
    e a outra da ultima data do monitoramento de cada estacao, seguindo a ordem de cada 
    poluente monitorado.

    Parameters
    ----------
    df_column : TYPE
        DESCRIPTION.
    df_pol_SP : TYPE
        DESCRIPTION.

    Returns
    -------
    df_column : TYPE
        DESCRIPTION.

    """
   
    # Agrupa para pegar as datas mínimas e máximas
    df_datas = df_pol_SP.groupby('Codigo')['datetime'].agg(['min', 'max']).reset_index()
    df_datas.columns = ['Codigo', 'DateStart', 'DateEnd']
    
    # Faz merge com o df_column original
    df_column = df_column.merge(df_datas, on='Codigo', how='left')
    
    # Mostrar resultado
    
   # print(df_column.head())
    return df_column

def addCount(df_column, df_pol_SP):
    """
    Essa função cria uma coluna com a somatoria de quantas linhas cada poluente de cada estação monitorou,
    isso resulta em quantos dias há um monitoramento naquele ano de cada poluente em cada estaçaõ.

    Parameters
    ----------
    df_column : data frame que queremos como resultado do tratamento de dados 
        DESCRIPTION.
    df_pol_SP : dataframe que contem as informações de monitoramento das estações de SP em 2023 
        DESCRIPTION.

    Returns
    -------
    df_column : df com os dados tratados que serao utilizados nas funções que queremos aplicar 
        DESCRIPTION.

    """
    
    # Conta quantos registros existem por Codigo
    df_count = df_pol_SP.groupby(['Codigo','Poluente']).size().reset_index(name='Count')
    df_count = (
        df_count.groupby('Codigo')
        .apply(lambda x: ';'.join(f"{row['Count']}" for _, row in x.iterrows()))
        .reset_index(name='Count')
    )

    # Faz merge com o df_column original
    df_column = df_column.merge(df_count, on='Codigo', how='left')
#    df_column = df_column.drop(['Poluentes e Contagem_x', 'Poluentes e Contagem_y', 'Count_x', 'Count_y'], axis=1)
    return df_column
#salvando um csv
df_column.to_csv(r'C:\PYTHON\ENS5132\ENS5132\Trabalho_03\outputs\estacoesInput.csv', index=False, encoding='utf-8')

#salvando df_column como estacoes.csv nos outputs 


# data a ser escolhida : DateStart 2023-01-01 00:00:00  DateEnd2023-12-31 23:00:00
# TO DO 
#fazer mais uma coluna em df_column e dentro dessa coluna vai ter a quantidae
# de linhas para aquela estacação e para cada poluente sendo:
    # 9233;445;... 
#ler as pastas outputs 
#para cada lin 